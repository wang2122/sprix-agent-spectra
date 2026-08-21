from __future__ import annotations

import numpy as np
import pytest

from sprix_spectra import CapabilityDimension, EvalItem, SpectraEvaluator, TrialOutcome
from sprix_spectra.experiment import evaluate_policy
from sprix_spectra.metrics import effective_response, reliability_metrics
from sprix_spectra.synthetic import SyntheticAgent, build_item_bank


def small_evaluator() -> SpectraEvaluator:
    dimensions = (CapabilityDimension("reasoning"), CapabilityDimension("tools"))
    items = (
        EvalItem("reasoning-easy", {"reasoning": 1.0}, difficulty=-0.5, max_repeats=2),
        EvalItem("reasoning-hard", {"reasoning": 1.0}, difficulty=1.0, max_repeats=2),
        EvalItem("tools", {"tools": 1.0}, difficulty=0.0, max_repeats=2),
        EvalItem("mixed", {"reasoning": 0.7, "tools": 0.7}, difficulty=0.0, max_repeats=2),
    )
    return SpectraEvaluator(dimensions, items, cost_weight=0.0)


def test_successful_evidence_moves_target_ability_up() -> None:
    evaluator = small_evaluator()
    evaluator.add_outcome(TrialOutcome("reasoning-hard", score=1.0, progress=1.0))
    assert evaluator.mean[0] > 0.0
    assert abs(evaluator.mean[1]) < 1e-10


def test_adaptive_policy_covers_unobserved_dimension() -> None:
    evaluator = small_evaluator()
    evaluator.add_outcome(TrialOutcome("reasoning-easy", score=1.0, progress=1.0))
    selected = evaluator.select_next()
    assert evaluator.items[selected.item_id].loadings.get("tools", 0.0) > 0


def test_contamination_downweights_evidence() -> None:
    dimension = (CapabilityDimension("reasoning"),)
    clean = SpectraEvaluator(dimension, (EvalItem("clean", {"reasoning": 1.0}),))
    risky = SpectraEvaluator(
        dimension,
        (EvalItem("risky", {"reasoning": 1.0}, contamination_risk=0.95),),
    )
    clean.add_outcome(TrialOutcome("clean", score=1.0))
    risky.add_outcome(TrialOutcome("risky", score=1.0))
    assert clean.mean[0] > risky.mean[0]


def test_safety_violation_penalizes_critical_response() -> None:
    item = EvalItem("safety", {"safety": 1.0}, safety_critical=True)
    safe = effective_response(TrialOutcome("safety", 1.0), item, 0.2, 0.75)
    violated = effective_response(TrialOutcome("safety", 1.0, policy_violations=2), item, 0.2, 0.75)
    assert violated < safe


def test_repeat_consistency_uses_all_pairs() -> None:
    outcomes = [TrialOutcome("x", score=value) for value in (0.0, 0.0, 1.0)]
    metrics = reliability_metrics(outcomes, [0.0, 0.0, 1.0])
    assert metrics["repeat_consistency"] == pytest.approx(1.0 - 2.0 / 3.0)


def test_repeat_limit_is_enforced() -> None:
    evaluator = small_evaluator()
    evaluator.add_outcome(TrialOutcome("tools", score=1.0))
    evaluator.add_outcome(TrialOutcome("tools", score=0.0))
    with pytest.raises(ValueError, match="max_repeats"):
        evaluator.add_outcome(TrialOutcome("tools", score=1.0))


def test_profile_has_intervals_and_evidence_counts() -> None:
    evaluator = small_evaluator()
    for item_id, score in (("reasoning-easy", 1.0), ("reasoning-hard", 1.0), ("tools", 0.0)):
        evaluator.add_outcome(TrialOutcome(item_id, score=score, progress=score, confidence=0.7))
    profile = evaluator.profile("test-agent")
    assert len(profile.abilities) == 2
    assert all(0 <= estimate.lower <= estimate.score <= estimate.upper <= 100 for estimate in profile.abilities)
    assert profile.diagnostics["trials"] == 3


def test_adaptive_policy_recovers_strengths_on_a_fixed_agent() -> None:
    agent = SyntheticAgent(
        "fixed",
        abilities=(1.5, 0.8, 1.1, -0.8, 1.4, -0.5, 0.3, 0.0),
        lapse_rate=0.02,
    )
    adaptive = evaluate_policy("adaptive", agent, seed=18, trials=40, item_bank=build_item_bank())
    random = evaluate_policy("random", agent, seed=18, trials=40, item_bank=build_item_bank())
    assert adaptive.top3_recall >= 2.0 / 3.0
    assert np.isfinite(adaptive.latent_rmse)
    # This is a regression guard for the fixed seed, not a universal superiority claim.
    assert adaptive.mean_interval_width < random.mean_interval_width
