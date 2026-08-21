"""Anchored item calibration for production SPECTRA banks.

The calibrator estimates item discrimination and difficulty against reference
agents whose latent capability coordinates are already anchored. This avoids
the identifiability ambiguity of fitting item and agent scales simultaneously.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace

import numpy as np

from .math_utils import sigmoid, stable_inverse
from .models import EvalItem


@dataclass(frozen=True)
class AnchorResponse:
    agent_id: str
    item_id: str
    score: float
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.agent_id or not self.item_id:
            raise ValueError("agent_id and item_id are required")
        if not 0.0 <= self.score <= 1.0 or self.weight <= 0.0:
            raise ValueError("score must be in [0, 1] and weight must be positive")


@dataclass(frozen=True)
class ItemCalibrationDiagnostic:
    item_id: str
    status: str
    observations: int
    discrimination: float
    difficulty: float
    discrimination_se: float | None
    difficulty_se: float | None
    brier: float | None
    log_loss: float | None
    iterations: int


@dataclass(frozen=True)
class CalibrationResult:
    items: tuple[EvalItem, ...]
    diagnostics: tuple[ItemCalibrationDiagnostic, ...]


def _ability_projection(item: EvalItem, abilities: Mapping[str, float]) -> float:
    missing = set(item.loadings) - set(abilities)
    if missing:
        raise ValueError(f"anchor ability vector is missing dimensions: {sorted(missing)}")
    loading = np.array(list(item.loadings.values()), dtype=float)
    values = np.array([abilities[name] for name in item.loadings], dtype=float)
    return float(loading @ values / max(np.linalg.norm(loading), 1e-12))


def _objective(params: np.ndarray, x: np.ndarray, y: np.ndarray, weights: np.ndarray, prior: np.ndarray, ridge: float) -> float:
    probability = np.clip(sigmoid(params[0] + params[1] * x), 1e-12, 1.0 - 1e-12)
    likelihood = -np.sum(weights * (y * np.log(probability) + (1.0 - y) * np.log(1.0 - probability)))
    return float(likelihood + 0.5 * ridge * np.sum((params - prior) ** 2))


def _fit_logistic_item(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    initial: np.ndarray,
    ridge: float,
    max_iterations: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    params = initial.copy()
    hessian = np.eye(2)
    for iteration in range(max_iterations):
        probability = np.asarray(sigmoid(params[0] + params[1] * x))
        design = np.column_stack((np.ones_like(x), x))
        gradient = design.T @ (weights * (probability - y)) + ridge * (params - initial)
        curvature = np.clip(weights * probability * (1.0 - probability), 1e-8, None)
        hessian = design.T @ (curvature[:, None] * design) + ridge * np.eye(2)
        step = stable_inverse(hessian) @ gradient
        current = _objective(params, x, y, weights, initial, ridge)
        scale = 1.0
        accepted = False
        for _ in range(24):
            candidate = params - scale * step
            candidate[1] = float(np.clip(candidate[1], 0.20, 4.0))
            if _objective(candidate, x, y, weights, initial, ridge) <= current:
                params = candidate
                accepted = True
                break
            scale *= 0.5
        if not accepted or float(np.linalg.norm(scale * step)) < 1e-7:
            break
    return params, stable_inverse(hessian), iteration + 1


def calibrate_item_bank(
    item_bank: Iterable[EvalItem],
    anchor_abilities: Mapping[str, Mapping[str, float]],
    responses: Iterable[AnchorResponse],
    *,
    min_observations: int = 20,
    ridge: float = 0.75,
    max_iterations: int = 80,
) -> CalibrationResult:
    """Estimate anchored 2PL item parameters with empirical-Bayes shrinkage.

    Anchor abilities must already live on the intended SPECTRA latent scale.
    Items with insufficient or degenerate evidence are retained unchanged and
    explicitly marked in diagnostics.
    """

    if min_observations < 4 or ridge < 0 or max_iterations < 1:
        raise ValueError("invalid calibration configuration")
    items = tuple(item_bank)
    item_by_id = {item.item_id: item for item in items}
    if len(item_by_id) != len(items):
        raise ValueError("item ids must be unique")
    grouped: dict[str, list[AnchorResponse]] = defaultdict(list)
    for response in responses:
        if response.item_id not in item_by_id:
            raise ValueError(f"unknown item in anchor responses: {response.item_id}")
        if response.agent_id not in anchor_abilities:
            raise ValueError(f"missing abilities for anchor agent: {response.agent_id}")
        grouped[response.item_id].append(response)

    calibrated: list[EvalItem] = []
    diagnostics: list[ItemCalibrationDiagnostic] = []
    for item in items:
        observations = grouped[item.item_id]
        x = np.array(
            [_ability_projection(item, anchor_abilities[response.agent_id]) for response in observations],
            dtype=float,
        )
        y = np.array([response.score for response in observations], dtype=float)
        weights = np.array([response.weight for response in observations], dtype=float)
        status = "calibrated"
        if len(observations) < min_observations:
            status = "insufficient-observations"
        elif float(np.std(x)) < 0.10 or float(np.std(y)) < 0.05:
            status = "degenerate-anchor-evidence"

        if status != "calibrated":
            calibrated.append(item)
            diagnostics.append(
                ItemCalibrationDiagnostic(
                    item.item_id,
                    status,
                    len(observations),
                    item.discrimination,
                    item.difficulty,
                    None,
                    None,
                    None,
                    None,
                    0,
                )
            )
            continue

        initial = np.array([-item.discrimination * item.difficulty, item.discrimination], dtype=float)
        params, covariance, iterations = _fit_logistic_item(x, y, weights, initial, ridge, max_iterations)
        discrimination = float(params[1])
        difficulty = float(np.clip(-params[0] / discrimination, -4.0, 4.0))
        probability = np.clip(sigmoid(params[0] + params[1] * x), 1e-12, 1.0 - 1e-12)
        brier = float(np.average((probability - y) ** 2, weights=weights))
        log_loss = float(-np.average(y * np.log(probability) + (1.0 - y) * np.log(1.0 - probability), weights=weights))
        discrimination_se = math.sqrt(max(0.0, float(covariance[1, 1])))
        difficulty_gradient = np.array([-1.0 / discrimination, params[0] / discrimination**2])
        difficulty_se = math.sqrt(max(0.0, float(difficulty_gradient @ covariance @ difficulty_gradient)))
        calibrated.append(replace(item, discrimination=discrimination, difficulty=difficulty))
        diagnostics.append(
            ItemCalibrationDiagnostic(
                item.item_id,
                status,
                len(observations),
                discrimination,
                difficulty,
                discrimination_se,
                difficulty_se,
                brier,
                log_loss,
                iterations,
            )
        )
    return CalibrationResult(tuple(calibrated), tuple(diagnostics))
