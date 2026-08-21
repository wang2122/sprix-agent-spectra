# Sprix SPECTRA

**Skill Profiling via Evidence-Calibrated Testing and Reliability Analysis**

[![CI](https://github.com/wang2122/sprix-agent-spectra/actions/workflows/ci.yml/badge.svg)](https://github.com/wang2122/sprix-agent-spectra/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-244b73)](https://www.python.org/)
[![Version 0.2.0](https://img.shields.io/badge/version-0.2.0-355c7d)](CHANGELOG.md)
[![License: MIT](https://img.shields.io/badge/license-MIT-2f6f5e)](LICENSE)
[![Research prototype](https://img.shields.io/badge/status-research%20prototype-b7791f)](#research-status-and-scope)

SPECTRA is an adaptive psychometric evaluation engine for autonomous AI agents. It estimates **what an agent is good at**, **how certain that estimate is**, and **whether the capability is reliable, calibrated, safe, and cost-efficient**. Instead of averaging a fixed benchmark, SPECTRA chooses the next trial by expected information gain and produces an evidence-backed multidimensional Agent Profile.

This repository is an open-source research output of **Sprix AI at 屿智同行**. The project is led by **Yichen Wang (CTO)** with **Yonghao Zhang (CEO, M.Eng. in Computer Science, Tsinghua University)** as a core contributor.

> Research status: the evaluator, simulator, tests, and benchmark harness are implemented. The bundled item bank is synthetic and intended for estimator validation. Real deployment claims require a separately calibrated, versioned task bank.

## Research status and scope

Version 0.2 adds anchored item calibration, robust posterior optimization, paired policy experiments with uncertainty, and a tamper-evident evidence ledger. SPECTRA is a working research prototype—not a peer-reviewed universal leaderboard. Follow the [`RESEARCH_PROTOCOL.md`](RESEARCH_PROTOCOL.md) before making claims about real Agents.

## Why SPECTRA

A single pass rate cannot tell a team whether an agent is a strong researcher but a weak operator, whether its apparent success is unstable, or whether it becomes unsafe under pressure. SPECTRA models eight separable dimensions by default:

`reasoning` · `planning` · `tool_use` · `coding` · `research` · `web_navigation` · `safety` · `recovery`

The ontology and bank are replaceable. Teams can define domain-specific dimensions such as customer support, quantitative finance, procurement, or software release engineering.

## Core contributions

- **Multidimensional capability inference** — fractional-response 2PL IRT with a Gaussian prior, line-searched MAP fitting, Laplace posterior intervals, and convergence diagnostics.
- **Adaptive, cost-aware testing** — D-optimal expected information gain balanced against capability coverage, repeated-trial value, latency, monetary cost, and contamination risk.
- **Anchored bank calibration** — estimates difficulty and discrimination from reference Agents with shrinkage, standard errors, Brier score, and log-loss diagnostics.
- **Evidence beyond final accuracy** — partial progress, safety violations, confidence calibration, repeat consistency, efficiency, and temporal drift.
- **Conservative strengths** — strengths are ranked by posterior lower bounds, not point estimates; weak spots use upper bounds so sparse evidence is not overstated.
- **Auditability** — ranked candidate decisions and trial outcomes can be exported as a hash-chained JSONL evidence ledger.

```mermaid
flowchart LR
    A["Versioned item bank<br/>difficulty, loadings, cost, risk"] --> B["Adaptive selector<br/>EIG + coverage - cost - contamination"]
    B --> C["Agent trial<br/>task, tools, environment"]
    C --> D["Hash-chained evidence ledger<br/>decision, score, progress, confidence, cost, safety"]
    D --> E["MIRT posterior<br/>MAP + covariance"]
    E --> B
    E --> F["Agent Profile<br/>strengths, intervals, reliability, calibration, drift"]
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
spectra-demo
```

Minimal API:

```python
from sprix_spectra import EvidenceLedger, SpectraEvaluator, TrialOutcome

evaluator = SpectraEvaluator(dimensions, calibrated_item_bank)
ledger = EvidenceLedger()

while True:
    decision = evaluator.select_next()
    result = run_agent_on_item(decision.item_id)
    outcome = TrialOutcome(
        item_id=decision.item_id,
        score=result.score,
        progress=result.progress,
        confidence=result.confidence,
        latency_ms=result.latency_ms,
        cost=result.cost,
        policy_violations=result.policy_violations,
    )
    evaluator.add_outcome(outcome)
    ledger.append(outcome, decision)
    stop, reason = evaluator.stopping_status(max_trials=40, max_cost=50)
    if stop:
        break

profile = evaluator.profile("agent://production/researcher-v3")
assert ledger.verify()[0]
print(ledger.root_hash)
```

See [`examples/quickstart.py`](examples/quickstart.py) for an end-to-end profile, [`examples/calibrate_bank.py`](examples/calibrate_bank.py) for anchored calibration, and [`ALGORITHM.md`](ALGORITHM.md) for equations and assumptions.

## Profile semantics

| Output | Interpretation |
|---|---|
| Capability score | Logistic transform of the estimated latent trait; useful for profiles, not a universal percentage |
| 95% interval | Laplace posterior interval; wider means less evidence or weaker identifiability |
| Strengths | Dimensions with the strongest conservative lower bounds |
| Repeat consistency | Agreement on controlled repeat items |
| Calibration | Brier score, expected calibration error, and overconfidence when confidence is available |
| Safety | Violation rate on designated safety-critical trials |
| Drift | Standardized split-window shift; a monitoring signal, not a causal diagnosis |

## Reproducible synthetic benchmark

Run:

```bash
spectra-benchmark
pytest
```

The benchmark compares adaptive SPECTRA with random and round-robin selection across 12 hidden synthetic Agents and 32 trials per Agent. Policies share seeds under a common-random-numbers design. Values are mean ± standard error.

| Policy | Latent RMSE ↓ | Top-3 recall ↑ | Rank correlation ↑ | Cost ↓ | Interval width ↓ | Coverage ↑ |
|---|---:|---:|---:|---:|---:|---:|
| **SPECTRA adaptive** | **0.655±0.064** | **0.889±0.063** | **0.776±0.050** | 26.26±0.68 | **58.45±0.66** | 0.979±0.021 |
| Round robin | 0.872±0.052 | 0.611±0.080 | 0.534±0.090 | **18.59±0.51** | 68.71±1.24 | 0.990±0.010 |
| Random | 0.819±0.079 | 0.583±0.083 | 0.540±0.128 | 29.22±0.68 | 67.12±0.60 | 0.958±0.032 |

Against random selection, adaptive SPECTRA reduces latent RMSE by `0.164±0.084`, improves Top-3 strength recall by `0.306±0.087`, narrows intervals by `8.67±0.94` points, and uses `2.96±0.64` less simulated cost. These are deterministic synthetic regression results, not real-world SOTA evidence; CI recomputes them from source.

## Research foundations

SPECTRA combines ideas from several research lines while adding a deployable Agent Profile and evidence ledger:

- **Interactive agent evaluation:** [AgentBench (ICLR 2024)](https://openreview.net/forum?id=zAdUB0aCTQ), [AgentBoard](https://openreview.net/forum?id=09Y7J22N9c), [GAIA (ICLR 2024)](https://openreview.net/forum?id=fibxvahvs3), and [WebArena (ICLR 2024)](https://openreview.net/forum?id=oKn9c6ytLx).
- **Reliability and tool-agent-user interaction:** [τ-bench (ICLR 2025)](https://openreview.net/forum?id=roNSXZpUDN) motivates repeated-trial reliability rather than one-shot pass rates.
- **Safety profiling:** [Agent-SafetyBench](https://arxiv.org/abs/2412.14470) demonstrates the need for agent-specific safety measurement.
- **Psychometric comparability:** [Comparing Test Sets with Item Response Theory (ACL 2021)](https://aclanthology.org/2021.acl-long.92/) motivates item-level difficulty/discrimination modeling.
- **Adaptive evaluation:** [ATLAS](https://arxiv.org/abs/2511.04689), [BanditCAT](https://proceedings.mlr.press/v264/sharpnack25a.html), and recent work on [cost-efficient general-ability estimation](https://arxiv.org/abs/2604.01418) motivate sequential value-of-information selection.
- **Agent-native evaluation infrastructure:** [AgentBeats](https://docs.agentbeats.org/) provides Green/Purple Agent evaluation roles over A2A; SPECTRA can serve as a profiling policy and evidence layer rather than redefining the protocol.

SPECTRA is an independent implementation. It does not claim reproduction or endorsement by the referenced authors.

## What SPECTRA is not

- It is not an A2A routing or scheduling protocol. The profile may inform a router, but evaluation and delegation are separate decisions.
- It is not a substitute for hidden, contamination-resistant tasks or human auditing.
- It does not infer a stable human-like personality from a few trajectories.
- It does not make a synthetic score comparable across organizations without shared calibration anchors.

## Roadmap

- [x] Anchored reference-panel calibration and item diagnostics
- [x] Tamper-evident decision/outcome evidence ledger
- [x] Paired policy benchmark with uncertainty reporting
- [ ] AgentBeats Green Agent adapter and A2A evidence envelopes
- [ ] Hierarchical Bayesian model for agent families and tool stacks
- [ ] Differential item functioning and contamination canaries
- [ ] Online drift/change-point detector
- [ ] Signed JSON-LD Agent Profile for Sprix A2A routing

## Team and contribution

- **Yichen Wang** — project lead, evaluation algorithm and Sprix integration; CTO, 屿智同行
- **Yonghao Zhang** — product and systems direction; CEO, 屿智同行; M.Eng. in Computer Science, Tsinghua University

Contributions from researchers and engineers are welcome. Please read [`CONTRIBUTING.md`](CONTRIBUTING.md). GitHub commits, pull requests, reviews, and the repository Contributors graph remain the source of record for individual contributions.

## Citation

If this research prototype is useful, cite the software metadata in [`CITATION.cff`](CITATION.cff). Please do not describe the current alpha release as a peer-reviewed method.

## License

MIT © 2026 Sprix AI at 屿智同行 and contributors.
