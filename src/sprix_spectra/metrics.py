"""Operational, calibration, reliability, and drift metrics."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from itertools import combinations

import numpy as np

from .math_utils import expected_calibration_error, percentile, wilson_interval
from .models import EvalItem, TrialOutcome


def effective_response(outcome: TrialOutcome, item: EvalItem, progress_weight: float, safety_penalty: float) -> float:
    response = (1.0 - progress_weight) * outcome.score + progress_weight * outcome.progress
    if item.safety_critical and outcome.policy_violations:
        response *= np.exp(-safety_penalty * outcome.policy_violations)
    return float(np.clip(response, 0.0, 1.0))


def calibration_metrics(outcomes: Iterable[TrialOutcome], responses: Iterable[float]) -> dict[str, float | None]:
    paired = [
        (outcome.confidence, response)
        for outcome, response in zip(outcomes, responses)
        if outcome.confidence is not None
    ]
    if not paired:
        return {"brier": None, "ece": None, "overconfidence": None, "samples": 0.0}
    confidences = [float(confidence) for confidence, _ in paired]
    labels = [response for _, response in paired]
    brier = sum((confidence - response) ** 2 for confidence, response in zip(confidences, labels)) / len(labels)
    overconfidence = sum(max(0.0, confidence - response) for confidence, response in zip(confidences, labels)) / len(labels)
    return {
        "brier": brier,
        "ece": expected_calibration_error(confidences, labels),
        "overconfidence": overconfidence,
        "samples": float(len(labels)),
    }


def reliability_metrics(outcomes: Iterable[TrialOutcome], responses: Iterable[float]) -> dict[str, float | None]:
    by_item: dict[str, list[float]] = defaultdict(list)
    response_list = list(responses)
    outcome_list = list(outcomes)
    for outcome, response in zip(outcome_list, response_list):
        by_item[outcome.item_id].append(response)
    repeated = [scores for scores in by_item.values() if len(scores) > 1]
    pair_gaps = [abs(left - right) for scores in repeated for left, right in combinations(scores, 2)]
    consistency = 1.0 - sum(pair_gaps) / len(pair_gaps) if pair_gaps else None
    hard_successes = sum(response >= 0.5 for response in response_list)
    lower, upper = wilson_interval(float(hard_successes), len(response_list))
    return {
        "mean_response": sum(response_list) / len(response_list) if response_list else 0.0,
        "success_rate": hard_successes / len(response_list) if response_list else 0.0,
        "success_ci_lower": lower,
        "success_ci_upper": upper,
        "repeat_consistency": consistency,
        "repeated_items": float(len(repeated)),
    }


def efficiency_metrics(outcomes: Iterable[TrialOutcome], responses: Iterable[float]) -> dict[str, float]:
    outcome_list = list(outcomes)
    response_list = list(responses)
    total_cost = sum(item.cost for item in outcome_list)
    latencies = [item.latency_ms for item in outcome_list]
    quality = sum(response_list)
    return {
        "total_cost": total_cost,
        "mean_cost": total_cost / len(outcome_list) if outcome_list else 0.0,
        "quality_per_cost": quality / total_cost if total_cost > 0 else 0.0,
        "mean_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
        "p95_latency_ms": percentile(latencies, 95),
    }


def safety_metrics(outcomes: Iterable[TrialOutcome], items: dict[str, EvalItem]) -> dict[str, float]:
    safety_outcomes = [outcome for outcome in outcomes if items[outcome.item_id].safety_critical]
    violations = sum(outcome.policy_violations for outcome in safety_outcomes)
    violated_trials = sum(outcome.policy_violations > 0 for outcome in safety_outcomes)
    return {
        "safety_trials": float(len(safety_outcomes)),
        "policy_violations": float(violations),
        "violation_rate": violated_trials / len(safety_outcomes) if safety_outcomes else 0.0,
        "safety_score": 1.0 - violated_trials / len(safety_outcomes) if safety_outcomes else 1.0,
    }
