"""Multidimensional psychometric evaluator and adaptive item policy."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable

import numpy as np

from .math_utils import sigmoid, stable_inverse
from .metrics import (
    calibration_metrics,
    effective_response,
    efficiency_metrics,
    reliability_metrics,
    safety_metrics,
)
from .models import (
    AbilityEstimate,
    AgentProfile,
    CapabilityDimension,
    EvalItem,
    SelectionDecision,
    TrialOutcome,
)


class SpectraEvaluator:
    """Adaptive multidimensional IRT evaluator for one agent.

    Item parameters are assumed to have been calibrated on a reference panel.
    The agent posterior is fitted by MAP estimation under a Gaussian prior and
    a fractional-response 2PL likelihood.
    """

    def __init__(
        self,
        dimensions: Iterable[CapabilityDimension],
        item_bank: Iterable[EvalItem],
        *,
        prior_variance: float = 2.25,
        progress_weight: float = 0.20,
        safety_penalty: float = 0.75,
        information_weight: float = 1.0,
        coverage_weight: float = 0.30,
        reliability_weight: float = 0.12,
        cost_weight: float = 0.16,
        contamination_weight: float = 0.45,
    ) -> None:
        self.dimensions = tuple(dimensions)
        if not self.dimensions:
            raise ValueError("at least one capability dimension is required")
        names = [dimension.name for dimension in self.dimensions]
        if len(names) != len(set(names)):
            raise ValueError("capability dimension names must be unique")
        self.dimension_names = tuple(names)
        self.dimension_index = {name: index for index, name in enumerate(names)}

        items = tuple(item_bank)
        self.items = {item.item_id: item for item in items}
        if not self.items:
            raise ValueError("item bank must not be empty")
        if len(self.items) != len(items):
            raise ValueError("item ids must be unique")
        for item in items:
            unknown = set(item.loadings) - set(self.dimension_names)
            if unknown:
                raise ValueError(f"item {item.item_id} uses unknown dimensions: {sorted(unknown)}")

        if prior_variance <= 0 or not 0 <= progress_weight <= 1 or safety_penalty < 0:
            raise ValueError("invalid posterior or response configuration")
        self.prior_variance = prior_variance
        self.progress_weight = progress_weight
        self.safety_penalty = safety_penalty
        self.information_weight = information_weight
        self.coverage_weight = coverage_weight
        self.reliability_weight = reliability_weight
        self.cost_weight = cost_weight
        self.contamination_weight = contamination_weight

        size = len(self.dimensions)
        self.mean = np.zeros(size, dtype=float)
        self.covariance = np.eye(size, dtype=float) * prior_variance
        self.outcomes: list[TrialOutcome] = []
        self.fit_diagnostics: dict[str, float | int | bool] = {
            "converged": True,
            "iterations": 0,
            "objective": 0.0,
            "gradient_norm": 0.0,
            "line_search_backtracks": 0,
        }
        self._median_cost = float(np.median([item.cost for item in items]))
        self._median_latency = float(np.median([item.expected_latency_ms for item in items]))

    def _design(self, item: EvalItem) -> np.ndarray:
        loadings = np.array([item.loadings.get(name, 0.0) for name in self.dimension_names], dtype=float)
        norm = np.linalg.norm(loadings)
        return item.discrimination * loadings / max(norm, 1e-12)

    def _response(self, outcome: TrialOutcome) -> float:
        return effective_response(
            outcome,
            self.items[outcome.item_id],
            self.progress_weight,
            self.safety_penalty,
        )

    def _evidence_weight(self, outcome: TrialOutcome) -> float:
        item = self.items[outcome.item_id]
        return outcome.evidence_weight * max(0.05, (1.0 - item.contamination_risk) ** 2)

    def add_outcome(self, outcome: TrialOutcome, *, refit: bool = True) -> None:
        if outcome.item_id not in self.items:
            raise ValueError(f"unknown item: {outcome.item_id}")
        count = sum(existing.item_id == outcome.item_id for existing in self.outcomes)
        if count >= self.items[outcome.item_id].max_repeats:
            raise ValueError(f"item {outcome.item_id} exceeded max_repeats")
        if outcome.sequence is None:
            outcome = TrialOutcome(
                item_id=outcome.item_id,
                score=outcome.score,
                progress=outcome.progress,
                confidence=outcome.confidence,
                latency_ms=outcome.latency_ms,
                cost=outcome.cost,
                policy_violations=outcome.policy_violations,
                evidence_weight=outcome.evidence_weight,
                sequence=len(self.outcomes),
                metadata=outcome.metadata,
            )
        self.outcomes.append(outcome)
        if refit:
            self.fit()

    def _posterior_terms(
        self,
        theta: np.ndarray,
        outcomes: list[TrialOutcome],
    ) -> tuple[float, np.ndarray, np.ndarray]:
        size = len(self.dimensions)
        prior_precision = np.eye(size) / self.prior_variance
        objective = 0.5 * float(theta @ prior_precision @ theta)
        gradient = prior_precision @ theta
        hessian = prior_precision.copy()
        for outcome in outcomes:
            item = self.items[outcome.item_id]
            design = self._design(item)
            probability = float(sigmoid(design @ theta - item.discrimination * item.difficulty))
            response = self._response(outcome)
            weight = self._evidence_weight(outcome)
            clipped_probability = float(np.clip(probability, 1e-12, 1.0 - 1e-12))
            objective -= weight * (
                response * math.log(clipped_probability)
                + (1.0 - response) * math.log(1.0 - clipped_probability)
            )
            gradient += weight * design * (probability - response)
            curvature = max(1e-8, probability * (1.0 - probability))
            hessian += weight * curvature * np.outer(design, design)
        return objective, gradient, hessian

    def _fit_outcomes(
        self,
        outcomes: list[TrialOutcome],
        *,
        initial: np.ndarray | None = None,
        max_iterations: int = 80,
        tolerance: float = 1e-7,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, float | int | bool]]:
        size = len(self.dimensions)
        theta = np.zeros(size, dtype=float) if initial is None else initial.astype(float).copy()
        hessian = np.eye(size) / self.prior_variance
        backtracks = 0
        converged = False
        gradient_norm = 0.0

        for iteration in range(max_iterations):
            objective, gradient, hessian = self._posterior_terms(theta, outcomes)
            gradient_norm = float(np.linalg.norm(gradient))
            step = stable_inverse(hessian) @ gradient
            descent = float(gradient @ step)
            step_scale = 1.0
            accepted = False
            for _ in range(24):
                candidate = theta - step_scale * step
                candidate_objective, _, _ = self._posterior_terms(candidate, outcomes)
                if candidate_objective <= objective - 1e-4 * step_scale * descent:
                    theta = candidate
                    accepted = True
                    break
                step_scale *= 0.5
                backtracks += 1
            if not accepted:
                break
            if float(np.linalg.norm(step_scale * step)) < tolerance or gradient_norm < tolerance:
                converged = True
                break

        objective, gradient, hessian = self._posterior_terms(theta, outcomes)
        diagnostics: dict[str, float | int | bool] = {
            "converged": converged or not outcomes,
            "iterations": iteration + 1,
            "objective": objective,
            "gradient_norm": float(np.linalg.norm(gradient)),
            "line_search_backtracks": backtracks,
        }
        return theta, stable_inverse(hessian), diagnostics

    def fit(self) -> tuple[np.ndarray, np.ndarray]:
        self.mean, self.covariance, self.fit_diagnostics = self._fit_outcomes(self.outcomes, initial=self.mean)
        return self.mean.copy(), self.covariance.copy()

    def predict(self, item_id: str) -> float:
        item = self.items[item_id]
        design = self._design(item)
        return float(sigmoid(design @ self.mean - item.discrimination * item.difficulty))

    def rank_candidates(self, *, limit: int | None = None) -> tuple[SelectionDecision, ...]:
        """Return the auditable candidate ranking used by the adaptive policy."""

        counts = Counter(outcome.item_id for outcome in self.outcomes)
        administered_dimensions = Counter()
        for outcome in self.outcomes:
            for name, loading in self.items[outcome.item_id].loadings.items():
                if loading > 0:
                    administered_dimensions[name] += 1
        unique_count = len(counts)
        posterior_std = np.sqrt(np.clip(np.diag(self.covariance), 0.0, None))
        candidates: list[SelectionDecision] = []

        for item in self.items.values():
            repeats = counts[item.item_id]
            if repeats >= item.max_repeats:
                continue
            design = self._design(item)
            probability = self.predict(item.item_id)
            evidence_quality = max(0.05, (1.0 - item.contamination_risk) ** 2)
            fisher = evidence_quality * probability * (1.0 - probability) * np.outer(design, design)
            sign, logdet = np.linalg.slogdet(np.eye(len(self.dimensions)) + self.covariance @ fisher)
            information_gain = 0.5 * logdet if sign > 0 else 0.0

            normalized_loading = np.abs(design) / max(np.sum(np.abs(design)), 1e-12)
            uncertainty_coverage = float(normalized_loading @ posterior_std)
            undercovered = sum(
                normalized_loading[index] / (1.0 + administered_dimensions[name])
                for index, name in enumerate(self.dimension_names)
            )
            safety_bonus = 0.12 if item.safety_critical and not any(
                self.items[outcome.item_id].safety_critical for outcome in self.outcomes
            ) else 0.0
            coverage_bonus = uncertainty_coverage + undercovered + safety_bonus

            reliability_bonus = 0.0
            if repeats == 1 and unique_count >= 2 * len(self.dimensions):
                reliability_bonus = 1.0
            cost_penalty = 0.65 * item.cost / self._median_cost + 0.35 * item.expected_latency_ms / self._median_latency
            contamination_penalty = item.contamination_risk ** 2
            objective = (
                self.information_weight * information_gain
                + self.coverage_weight * coverage_bonus
                + self.reliability_weight * reliability_bonus
                - self.cost_weight * cost_penalty
                - self.contamination_weight * contamination_penalty
            )
            candidates.append(
                SelectionDecision(
                    item_id=item.item_id,
                    objective=objective,
                    information_gain=information_gain,
                    coverage_bonus=coverage_bonus,
                    reliability_bonus=reliability_bonus,
                    cost_penalty=cost_penalty,
                    contamination_penalty=contamination_penalty,
                    predicted_success=probability,
                    explanation=(
                        f"Select {item.item_id}: EIG={information_gain:.3f}, coverage={coverage_bonus:.3f}, "
                        f"predicted success={probability:.3f}, cost penalty={cost_penalty:.3f}."
                    ),
                )
            )
        ranked = tuple(sorted(candidates, key=lambda candidate: (-candidate.objective, candidate.item_id)))
        return ranked if limit is None else ranked[:limit]

    def select_next(self) -> SelectionDecision:
        ranked = self.rank_candidates(limit=1)
        if not ranked:
            raise RuntimeError("no selectable items remain")
        return ranked[0]

    def stopping_status(
        self,
        *,
        target_interval_width: float = 1.20,
        min_trials: int | None = None,
        max_trials: int | None = None,
        max_cost: float | None = None,
    ) -> tuple[bool, str]:
        min_trials = min_trials or 2 * len(self.dimensions)
        if max_trials is not None and len(self.outcomes) >= max_trials:
            return True, "maximum trial budget reached"
        if max_cost is not None and sum(outcome.cost for outcome in self.outcomes) >= max_cost:
            return True, "maximum monetary budget reached"
        if len(self.outcomes) < min_trials:
            return False, "minimum evidence requirement not reached"
        widths = 2.0 * 1.96 * np.sqrt(np.clip(np.diag(self.covariance), 0.0, None))
        if float(np.max(widths)) <= target_interval_width:
            return True, "all capability intervals reached target precision"
        return False, "additional information is still valuable"

    def _drift_metrics(self) -> dict[str, float | bool]:
        minimum = max(12, 2 * len(self.dimensions))
        if len(self.outcomes) < minimum:
            return {"detected": False, "max_standardized_shift": 0.0, "evidence_sufficient": False}
        ordered = sorted(self.outcomes, key=lambda outcome: outcome.sequence or 0)
        midpoint = len(ordered) // 2
        first_mean, first_covariance, _ = self._fit_outcomes(ordered[:midpoint])
        second_mean, second_covariance, _ = self._fit_outcomes(ordered[midpoint:])
        variance = np.clip(np.diag(first_covariance + second_covariance), 1e-8, None)
        standardized = np.abs(second_mean - first_mean) / np.sqrt(variance)
        maximum = float(np.max(standardized))
        return {"detected": maximum >= 2.0, "max_standardized_shift": maximum, "evidence_sufficient": True}

    def profile(self, agent_id: str) -> AgentProfile:
        if not self.outcomes:
            raise RuntimeError("at least one outcome is required to build a profile")
        self.fit()
        estimates: list[AbilityEstimate] = []
        for index, name in enumerate(self.dimension_names):
            standard_deviation = math.sqrt(max(0.0, self.covariance[index, index]))
            lower_latent = self.mean[index] - 1.96 * standard_deviation
            upper_latent = self.mean[index] + 1.96 * standard_deviation
            evidence = sum(self.items[outcome.item_id].loadings.get(name, 0.0) > 0 for outcome in self.outcomes)
            estimates.append(
                AbilityEstimate(
                    dimension=name,
                    latent_mean=float(self.mean[index]),
                    latent_std=standard_deviation,
                    score=100.0 * float(sigmoid(self.mean[index])),
                    lower=100.0 * float(sigmoid(lower_latent)),
                    upper=100.0 * float(sigmoid(upper_latent)),
                    evidence=evidence,
                )
            )

        by_strength = sorted(estimates, key=lambda estimate: (estimate.lower, estimate.score), reverse=True)
        by_weakness = sorted(estimates, key=lambda estimate: (estimate.upper, estimate.score))
        responses = [self._response(outcome) for outcome in self.outcomes]
        posterior_sign, posterior_logdet = np.linalg.slogdet(self.covariance)
        diagnostics: dict[str, float | int | str | bool] = {
            "trials": len(self.outcomes),
            "unique_items": len({outcome.item_id for outcome in self.outcomes}),
            "posterior_log_volume": float(posterior_logdet) if posterior_sign > 0 else float("inf"),
            "mean_interval_width": sum(estimate.upper - estimate.lower for estimate in estimates) / len(estimates),
            "model": "multidimensional-2PL-MAP",
            "optimizer_converged": bool(self.fit_diagnostics["converged"]),
            "optimizer_iterations": int(self.fit_diagnostics["iterations"]),
            "optimizer_gradient_norm": float(self.fit_diagnostics["gradient_norm"]),
            "optimizer_line_search_backtracks": int(self.fit_diagnostics["line_search_backtracks"]),
        }
        return AgentProfile(
            agent_id=agent_id,
            abilities=tuple(estimates),
            strengths=tuple(estimate.dimension for estimate in by_strength[:3]),
            development_areas=tuple(estimate.dimension for estimate in by_weakness[:3]),
            overall_score=sum(estimate.score for estimate in estimates) / len(estimates),
            reliability=reliability_metrics(self.outcomes, responses),
            calibration=calibration_metrics(self.outcomes, responses),
            efficiency=efficiency_metrics(self.outcomes, responses),
            safety=safety_metrics(self.outcomes, self.items),
            drift=self._drift_metrics(),
            diagnostics=diagnostics,
        )
