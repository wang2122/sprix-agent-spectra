# SPECTRA algorithm specification

SPECTRA estimates what an agent can do, how certain that estimate is, and how reliably the capability survives repeated and adversarial trials. It separates latent capability from observed task outcomes instead of treating a benchmark average as an intrinsic property.

## 1. Measurement model

Let an agent have latent capability vector `θ ∈ R^K`. A calibrated item `i` has discrimination `a_i`, difficulty `b_i`, and sparse non-negative capability loadings `q_i`. SPECTRA normalizes the loading vector and predicts:

```text
x_i = a_i q_i / ||q_i||
p_i(θ) = sigmoid(x_iᵀ θ - a_i b_i)
```

The observed fractional response combines terminal quality and partial progress:

```text
y_i = (1 - λ) · final_score + λ · progress
```

Safety-critical policy violations multiply `y_i` by `exp(-κv)`. Evidence from a high-contamination item receives weight `w_i(1-c_i)²`, where `c_i` is the estimated contamination risk.

With Gaussian prior `θ ~ N(0, σ²I)`, SPECTRA obtains a MAP estimate by Newton updates on the fractional Bernoulli log-likelihood. The inverse observed Hessian is a Laplace approximation of posterior covariance. Reported intervals are posterior uncertainty intervals, not repeated-sampling guarantees.

## 2. Adaptive selection

For every available item, SPECTRA computes expected Fisher information:

```text
F_i = (1-c_i)² p_i(1-p_i) x_i x_iᵀ
EIG_i = 1/2 log det(I + ΣF_i)
```

The policy then maximizes:

```text
utility(i) = EIG_i
           + α · posterior-uncertainty coverage
           + β · repeat reliability value
           - γ · normalized monetary/latency cost
           - δ · contamination risk²
```

This is a cost-aware D-optimal design policy. Coverage prevents a high-information cluster from starving less-tested dimensions. Controlled repeats estimate test-retest consistency.

## 3. Profile outputs

- Capability score and 95% posterior interval per dimension
- Strengths ranked by conservative lower bounds
- Development areas ranked by conservative upper bounds
- Mean response and Wilson success interval
- Repeated-item consistency
- Confidence calibration: Brier score, ECE, overconfidence
- Cost, latency, and quality-per-cost
- Safety violation rate
- Split-window standardized drift signal

## 4. Required calibration protocol

The demonstration bank is synthetic. A production bank should be calibrated on a diverse reference panel:

1. Freeze rubrics and task environments; record task versions and tool permissions.
2. Collect multiple runs per agent/task/seed, including failures and partial progress.
3. Fit item difficulty, discrimination, and capability loadings on a training panel.
4. Check item fit, local dependence, differential item functioning, and held-out predictive calibration.
5. Retire leaked or saturated items; keep a hidden canary pool for contamination audits.
6. Recalibrate after material environment, model, or tool changes.

## 5. Threats to validity

- The chosen capability ontology is a modeling decision, not ground truth.
- IRT local-independence assumptions can fail for related tasks or shared trajectories.
- Adaptive evaluation creates policy-dependent missingness; retain selection propensities and complete random audits.
- A scalar overall score can hide critical weaknesses. Use the multidimensional profile for deployment decisions.
- Self-reported confidence is only meaningful when the evaluated interface exposes it consistently.
- Synthetic experiments test software behavior and estimator recovery; they are not evidence of superiority on real agents.

## 6. Research extensions

- Hierarchical Bayesian calibration across model families and tool stacks
- Contextual item response models for environment and permission conditions
- Sequential change-point detection for online capability drift
- Constrained selection guaranteeing safety and domain coverage
- AgentBeats Green Agent adapter and signed evidence artifacts over A2A
- Counterfactual value-of-information analysis for routing and delegation
