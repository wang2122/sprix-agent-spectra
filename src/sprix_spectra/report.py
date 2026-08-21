"""Human- and machine-readable SPECTRA profile rendering."""

from __future__ import annotations

import json

from .models import AgentProfile


def profile_to_json(profile: AgentProfile, *, indent: int = 2) -> str:
    return json.dumps(profile.to_dict(), indent=indent, ensure_ascii=False, sort_keys=True)


def profile_to_markdown(profile: AgentProfile) -> str:
    rows = [
        "| Capability | Score | 95% interval | Evidence |",
        "|---|---:|---:|---:|",
    ]
    for estimate in sorted(profile.abilities, key=lambda item: item.score, reverse=True):
        rows.append(
            f"| {estimate.dimension} | {estimate.score:.1f} | "
            f"[{estimate.lower:.1f}, {estimate.upper:.1f}] | {estimate.evidence} |"
        )
    reliability = profile.reliability
    calibration = profile.calibration
    consistency = reliability.get("repeat_consistency")
    consistency_text = "n/a" if consistency is None else f"{100 * float(consistency):.1f}%"
    brier = calibration.get("brier")
    brier_text = "n/a" if brier is None else f"{float(brier):.3f}"
    return "\n".join(
        [
            f"# SPECTRA profile: {profile.agent_id}",
            "",
            f"**Overall latent capability score:** {profile.overall_score:.1f}/100",
            "",
            *rows,
            "",
            f"**Strengths:** {', '.join(profile.strengths)}",
            "",
            f"**Development areas:** {', '.join(profile.development_areas)}",
            "",
            "## Operational evidence",
            "",
            f"- Mean response: {100 * float(reliability['mean_response']):.1f}%",
            f"- Repeat consistency: {consistency_text}",
            f"- Calibration Brier score: {brier_text}",
            f"- P95 latency: {profile.efficiency['p95_latency_ms']:.0f} ms",
            f"- Safety score: {100 * profile.safety['safety_score']:.1f}%",
            f"- Drift detected: {profile.drift['detected']}",
            "",
            "_Scores are posterior estimates, not self-reported capabilities. Intervals widen when evidence is sparse._",
        ]
    )
