# Research protocol

This protocol defines the minimum evidence required before making claims about a real Agent with SPECTRA.

## 1. Pre-register the measurement contract

- Freeze the capability ontology, scoring rubrics, stopping rules, and primary metrics before evaluation.
- Record Agent version, model, prompts, tools, permissions, environment, task-bank commit, evaluator version, and seeds.
- State the target population and deployment context. A capability profile is conditional on these choices.
- Identify safety-critical dimensions that cannot be compensated by a high average score.

## 2. Separate data roles

Use disjoint partitions:

- **Calibration panel** — estimates item difficulty and discrimination against anchored reference Agents.
- **Policy-development bank** — tunes adaptive-selection weights and stopping rules.
- **Locked audit bank** — estimates final performance and remains hidden from Agent developers.
- **Contamination canaries** — detect memorization or public-task leakage.

Do not report development-bank results as held-out evidence.

## 3. Calibrate the bank

Reference Agents should span the intended capability range and tool configurations. For each item, inspect:

- observation count and anchor coverage;
- discrimination and difficulty standard errors;
- Brier score and log loss;
- item characteristic curves and residuals;
- local dependence between related tasks;
- differential item functioning across model families, languages, and tool stacks.

SPECTRA v0.2 implements anchored 2PL calibration. It intentionally retains items with insufficient evidence unchanged and marks their diagnostic status instead of silently fitting unstable parameters.

## 4. Run the evaluation

- Use multiple independent seeds for stochastic Agents.
- Log the full candidate ranking, selected item, selection objective, outcome, partial progress, confidence, latency, cost, and policy violations.
- Export the evidence ledger and store its root hash with the report. The hash chain detects modification; sign the root hash externally when evaluator authenticity matters.
- Preserve selection propensities if a randomized policy is used, enabling off-policy analysis.

## 5. Report uncertainty and reliability

Always report capability intervals, number of observations per dimension, repeat consistency, calibration, safety violations, cost, and latency. For policy comparisons, use paired seeds or common random numbers and report standard errors or bootstrap intervals.

Do not claim that two Agents differ when intervals and paired evidence do not support the distinction. Do not use the scalar overall score to override a safety-critical weakness.

## 6. Audit and update

- Re-run a random hidden subset to audit the adaptive policy.
- Investigate drift signals rather than automatically attributing them to the Agent.
- Recalibrate after material task, environment, model, tool, or permission changes.
- Version every profile and retain reproducible evidence artifacts.

## Claim language

Acceptable: “Under task bank v2.1 and the recorded tool permissions, Agent A has a higher posterior lower bound on research synthesis than Agent B.”

Not acceptable: “Agent A is universally better” or “95% posterior interval means 95% of repeated experiments will contain the truth.”
