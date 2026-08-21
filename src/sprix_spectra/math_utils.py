"""Small numerical helpers used by the psychometric model."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np


def sigmoid(value: float | np.ndarray) -> float | np.ndarray:
    array = np.asarray(value, dtype=float)
    positive = array >= 0
    result = np.empty_like(array)
    result[positive] = 1.0 / (1.0 + np.exp(-array[positive]))
    exp_value = np.exp(array[~positive])
    result[~positive] = exp_value / (1.0 + exp_value)
    return float(result) if result.ndim == 0 else result


def stable_inverse(matrix: np.ndarray, jitter: float = 1e-8) -> np.ndarray:
    identity = np.eye(matrix.shape[0])
    for multiplier in (1.0, 10.0, 100.0, 1_000.0):
        try:
            return np.linalg.inv(matrix + identity * jitter * multiplier)
        except np.linalg.LinAlgError:
            continue
    return np.linalg.pinv(matrix)


def percentile(values: Iterable[float], q: float) -> float:
    data = list(values)
    return float(np.percentile(data, q)) if data else 0.0


def expected_calibration_error(confidences: list[float], outcomes: list[float], bins: int = 10) -> float:
    if not confidences:
        return 0.0
    total = len(confidences)
    error = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        members = [
            position
            for position, confidence in enumerate(confidences)
            if low <= confidence < high or (index == bins - 1 and confidence == 1.0)
        ]
        if not members:
            continue
        mean_confidence = sum(confidences[position] for position in members) / len(members)
        mean_outcome = sum(outcomes[position] for position in members) / len(members)
        error += len(members) / total * abs(mean_confidence - mean_outcome)
    return error


def wilson_interval(successes: float, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials <= 0:
        return 0.0, 1.0
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    margin = z * math.sqrt(proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)
