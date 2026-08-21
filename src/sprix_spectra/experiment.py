"""Reproducible adaptive-vs-baseline evaluation experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from .evaluator import SpectraEvaluator
from .models import EvalItem
from .synthetic import DEFAULT_DIMENSIONS, AgentSimulator, SyntheticAgent, build_item_bank


@dataclass(frozen=True)
class ExperimentResult:
    policy: str
    latent_rmse: float
    top3_recall: float
    rank_correlation: float
    total_cost: float
    mean_interval_width: float
    latent_interval_coverage: float

    def to_dict(self) -> dict[str, float | str]:
        return self.__dict__.copy()


def _rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = np.argsort(np.argsort(left)).astype(float)
    right_rank = np.argsort(np.argsort(right)).astype(float)
    if np.std(left_rank) == 0 or np.std(right_rank) == 0:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _baseline_item(policy: str, evaluator: SpectraEvaluator, rng: np.random.Generator) -> EvalItem:
    available = [
        item
        for item in evaluator.items.values()
        if sum(outcome.item_id == item.item_id for outcome in evaluator.outcomes) < item.max_repeats
    ]
    if policy == "random":
        return available[int(rng.integers(0, len(available)))]
    if policy == "round_robin":
        counts = {
            name: sum(item.loadings.get(name, 0.0) > 0 for item in (evaluator.items[o.item_id] for o in evaluator.outcomes))
            for name in evaluator.dimension_names
        }
        target = min(counts, key=counts.get)
        targeted = [item for item in available if item.loadings.get(target, 0.0) > 0]
        return min(targeted or available, key=lambda item: item.cost)
    raise ValueError(f"unknown baseline policy: {policy}")


def evaluate_policy(
    policy: str,
    agent: SyntheticAgent,
    *,
    seed: int,
    trials: int = 32,
    item_bank: tuple[EvalItem, ...] | None = None,
) -> ExperimentResult:
    bank = item_bank or build_item_bank()
    evaluator = SpectraEvaluator(DEFAULT_DIMENSIONS, bank)
    simulator = AgentSimulator(agent, seed=seed + 100_000)
    rng = np.random.default_rng(seed)
    for sequence in range(trials):
        if policy == "adaptive":
            item = evaluator.items[evaluator.select_next().item_id]
        else:
            item = _baseline_item(policy, evaluator, rng)
        evaluator.add_outcome(simulator.run(item, sequence))

    profile = evaluator.profile(agent.agent_id)
    estimate = np.array([ability.latent_mean for ability in profile.abilities])
    truth = np.asarray(agent.abilities, dtype=float)
    estimated_top = set(np.argsort(estimate)[-3:])
    truth_top = set(np.argsort(truth)[-3:])
    return ExperimentResult(
        policy=policy,
        latent_rmse=float(np.sqrt(np.mean((estimate - truth) ** 2))),
        top3_recall=len(estimated_top & truth_top) / 3.0,
        rank_correlation=_rank_correlation(estimate, truth),
        total_cost=float(profile.efficiency["total_cost"]),
        mean_interval_width=float(profile.diagnostics["mean_interval_width"]),
        latent_interval_coverage=float(
            np.mean(
                [
                    ability.latent_mean - 1.96 * ability.latent_std
                    <= truth[index]
                    <= ability.latent_mean + 1.96 * ability.latent_std
                    for index, ability in enumerate(profile.abilities)
                ]
            )
        ),
    )


def _mean_and_standard_error(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    mean = float(np.mean(array))
    standard_error = float(np.std(array, ddof=1) / np.sqrt(len(array))) if len(array) > 1 else 0.0
    return mean, standard_error


def run_benchmark(*, seeds: int = 12, trials: int = 32) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(20260821)
    policies = ("adaptive", "round_robin", "random")
    collected: dict[str, list[ExperimentResult]] = {policy: [] for policy in policies}
    bank = build_item_bank()
    for seed in range(seeds):
        agent = SyntheticAgent(
            agent_id=f"agent-{seed}",
            abilities=tuple(float(value) for value in rng.normal(0.0, 1.0, len(DEFAULT_DIMENSIONS))),
            speed=float(rng.uniform(0.75, 1.30)),
            cost_factor=float(rng.uniform(0.85, 1.20)),
            calibration_bias=float(rng.normal(0.04, 0.04)),
            lapse_rate=float(rng.uniform(0.02, 0.08)),
        )
        # Common random numbers reduce variance in paired policy comparisons.
        evaluation_seed = seed * 97
        for policy in policies:
            collected[policy].append(evaluate_policy(policy, agent, seed=evaluation_seed, trials=trials, item_bank=bank))

    summary: dict[str, dict[str, float]] = {}
    fields = (
        "latent_rmse",
        "top3_recall",
        "rank_correlation",
        "total_cost",
        "mean_interval_width",
        "latent_interval_coverage",
    )
    for policy, results in collected.items():
        summary[policy] = {}
        for field in fields:
            mean, standard_error = _mean_and_standard_error([getattr(result, field) for result in results])
            summary[policy][field] = mean
            summary[policy][f"{field}_se"] = standard_error
        summary[policy]["runs"] = float(len(results))
    for policy in ("adaptive", "round_robin"):
        for field in ("latent_rmse", "top3_recall", "rank_correlation", "total_cost", "mean_interval_width"):
            differences = [
                getattr(left, field) - getattr(right, field)
                for left, right in zip(collected[policy], collected["random"], strict=True)
            ]
            mean, standard_error = _mean_and_standard_error(differences)
            summary[policy][f"{field}_delta_vs_random"] = mean
            summary[policy][f"{field}_delta_vs_random_se"] = standard_error
    return summary


def benchmark_markdown(summary: dict[str, dict[str, float]]) -> str:
    lines = [
        "| Policy | Latent RMSE ↓ | Top-3 recall ↑ | Rank corr. ↑ | Cost ↓ | Width ↓ | Coverage ↑ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for policy in ("adaptive", "round_robin", "random"):
        result = summary[policy]
        lines.append(
            f"| {policy} | {result['latent_rmse']:.3f}±{result['latent_rmse_se']:.3f} | "
            f"{result['top3_recall']:.3f}±{result['top3_recall_se']:.3f} | "
            f"{result['rank_correlation']:.3f}±{result['rank_correlation_se']:.3f} | "
            f"{result['total_cost']:.2f}±{result['total_cost_se']:.2f} | "
            f"{result['mean_interval_width']:.2f}±{result['mean_interval_width_se']:.2f} | "
            f"{result['latent_interval_coverage']:.3f}±{result['latent_interval_coverage_se']:.3f} |"
        )
    return "\n".join(lines)


def main() -> None:
    summary = run_benchmark()
    print(benchmark_markdown(summary))
    print("\nJSON:\n" + json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
