"""Deterministic synthetic bank and agent used for examples and regression tests.

This simulator deliberately adds interaction effects, lapses, and measurement
noise so that evaluation is not a circular replay of the fitted IRT model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .math_utils import sigmoid
from .models import CapabilityDimension, EvalItem, TrialOutcome

DEFAULT_DIMENSIONS = (
    CapabilityDimension("reasoning", "Multi-step inference and constraint satisfaction"),
    CapabilityDimension("planning", "Long-horizon decomposition and replanning"),
    CapabilityDimension("tool_use", "Correct tool selection and grounded execution"),
    CapabilityDimension("coding", "Implementation, debugging, and verification"),
    CapabilityDimension("research", "Evidence retrieval, synthesis, and citation discipline"),
    CapabilityDimension("web_navigation", "Stateful interaction with realistic web interfaces"),
    CapabilityDimension("safety", "Policy compliance and safe refusal under pressure"),
    CapabilityDimension("recovery", "Detection and recovery from partial failure"),
)


def build_item_bank(seed: int = 7) -> tuple[EvalItem, ...]:
    """Create a balanced, calibration-ready 80-item demonstration bank."""

    rng = np.random.default_rng(seed)
    names = [dimension.name for dimension in DEFAULT_DIMENSIONS]
    difficulties = (-1.50, -0.75, 0.0, 0.75, 1.50)
    items: list[EvalItem] = []
    for dimension_index, primary in enumerate(names):
        for level_index, difficulty in enumerate(difficulties):
            for variant in range(2):
                secondary = names[(dimension_index + level_index + variant + 1) % len(names)]
                loadings = {primary: 1.0, secondary: float(rng.uniform(0.10, 0.35))}
                safety_critical = primary == "safety" or (secondary == "safety" and variant == 1)
                base_cost = 0.45 + 0.22 * (difficulty + 1.5) + rng.uniform(0.0, 0.25)
                items.append(
                    EvalItem(
                        item_id=f"{primary[:4]}-{level_index + 1}-{variant + 1}",
                        loadings=loadings,
                        difficulty=difficulty,
                        discrimination=float(rng.uniform(0.85, 1.55)),
                        category=primary,
                        cost=float(base_cost),
                        expected_latency_ms=float(1_400 + 650 * (difficulty + 1.5) + rng.uniform(0, 700)),
                        contamination_risk=float(rng.beta(1.2, 12.0)),
                        safety_critical=safety_critical,
                        max_repeats=2,
                    )
                )
    return tuple(items)


@dataclass(frozen=True)
class SyntheticAgent:
    agent_id: str
    abilities: tuple[float, ...]
    speed: float = 1.0
    cost_factor: float = 1.0
    calibration_bias: float = 0.04
    lapse_rate: float = 0.04

    def __post_init__(self) -> None:
        if len(self.abilities) != len(DEFAULT_DIMENSIONS):
            raise ValueError("synthetic abilities must match DEFAULT_DIMENSIONS")
        if self.speed <= 0 or self.cost_factor <= 0:
            raise ValueError("speed and cost_factor must be positive")


class AgentSimulator:
    """Generate noisy observable evidence from a hidden capability vector."""

    def __init__(self, agent: SyntheticAgent, seed: int = 0) -> None:
        self.agent = agent
        self.rng = np.random.default_rng(seed)
        self.index = {dimension.name: position for position, dimension in enumerate(DEFAULT_DIMENSIONS)}

    def run(self, item: EvalItem, sequence: int) -> TrialOutcome:
        loadings = np.array([item.loadings.get(name, 0.0) for name in self.index], dtype=float)
        loadings /= max(float(np.linalg.norm(loadings)), 1e-12)
        theta = np.asarray(self.agent.abilities, dtype=float)
        logit = item.discrimination * (float(loadings @ theta) - item.difficulty)

        # Hidden interactions and non-stationarity make this a model-mismatch test.
        logit += 0.10 * theta[self.index["planning"]] * theta[self.index["tool_use"]]
        if item.category in {"coding", "web_navigation"}:
            logit += 0.08 * min(theta[self.index["recovery"]], theta[self.index["tool_use"]])
        logit += float(self.rng.normal(0.0, 0.18))
        probability = float(sigmoid(logit))
        if self.rng.random() < self.agent.lapse_rate:
            probability *= 0.20

        score = float(self.rng.random() < probability)
        progress = float(np.clip(0.12 + 0.76 * probability + self.rng.normal(0.0, 0.13), 0.0, 1.0))
        confidence = float(np.clip(probability + self.agent.calibration_bias + self.rng.normal(0.0, 0.06), 0.0, 1.0))
        violation_probability = 0.0
        if item.safety_critical:
            violation_probability = 0.24 * (1.0 - float(sigmoid(theta[self.index["safety"]])))
        violations = int(self.rng.random() < violation_probability)
        latency = item.expected_latency_ms / self.agent.speed * float(self.rng.lognormal(0.0, 0.12))
        cost = item.cost * self.agent.cost_factor * float(self.rng.uniform(0.92, 1.08))
        return TrialOutcome(
            item_id=item.item_id,
            score=score,
            progress=progress,
            confidence=confidence,
            latency_ms=latency,
            cost=cost,
            policy_violations=violations,
            sequence=sequence,
            metadata={"source": "synthetic-hidden-model"},
        )
