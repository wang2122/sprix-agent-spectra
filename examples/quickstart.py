from sprix_spectra import (
    DEFAULT_DIMENSIONS,
    AgentSimulator,
    EvidenceLedger,
    SpectraEvaluator,
    SyntheticAgent,
    build_item_bank,
)
from sprix_spectra.report import profile_to_markdown

agent = SyntheticAgent(
    "research-copilot-v1",
    abilities=(0.7, 1.1, 0.9, 0.3, 1.4, -0.2, 0.8, 0.5),
)
evaluator = SpectraEvaluator(DEFAULT_DIMENSIONS, build_item_bank())
simulator = AgentSimulator(agent, seed=42)
ledger = EvidenceLedger()

for sequence in range(32):
    decision = evaluator.select_next()
    outcome = simulator.run(evaluator.items[decision.item_id], sequence)
    evaluator.add_outcome(outcome)
    ledger.append(outcome, decision)

print(profile_to_markdown(evaluator.profile(agent.agent_id)))
print(f"\nEvidence root: {ledger.root_hash}")
