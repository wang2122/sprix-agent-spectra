"""Runnable SPECTRA demonstration."""

from __future__ import annotations

from .evaluator import SpectraEvaluator
from .report import profile_to_markdown
from .synthetic import DEFAULT_DIMENSIONS, AgentSimulator, SyntheticAgent, build_item_bank


def main() -> None:
    agent = SyntheticAgent(
        "demo-agent",
        abilities=(0.65, 1.10, 0.85, 0.30, 1.35, -0.25, 0.75, 0.45),
        speed=1.15,
    )
    evaluator = SpectraEvaluator(DEFAULT_DIMENSIONS, build_item_bank())
    simulator = AgentSimulator(agent, seed=42)
    for sequence in range(32):
        selected = evaluator.select_next()
        evaluator.add_outcome(simulator.run(evaluator.items[selected.item_id], sequence))
    print(profile_to_markdown(evaluator.profile(agent.agent_id)))


if __name__ == "__main__":
    main()
