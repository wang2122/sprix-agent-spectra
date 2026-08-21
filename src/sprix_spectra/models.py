"""Typed data models for Sprix SPECTRA."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class CapabilityDimension:
    name: str
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("capability dimension name must not be empty")


@dataclass(frozen=True)
class EvalItem:
    item_id: str
    loadings: Mapping[str, float]
    difficulty: float = 0.0
    discrimination: float = 1.0
    category: str = "general"
    cost: float = 1.0
    expected_latency_ms: float = 1_000.0
    contamination_risk: float = 0.0
    safety_critical: bool = False
    max_repeats: int = 1

    def __post_init__(self) -> None:
        if not self.item_id:
            raise ValueError("item_id must not be empty")
        if not self.loadings or all(value <= 0 for value in self.loadings.values()):
            raise ValueError("an item needs at least one positive capability loading")
        if any(value < 0 for value in self.loadings.values()):
            raise ValueError("capability loadings must be non-negative")
        if self.discrimination <= 0:
            raise ValueError("discrimination must be positive")
        if self.cost <= 0 or self.expected_latency_ms <= 0:
            raise ValueError("item cost and expected latency must be positive")
        if not 0 <= self.contamination_risk <= 1:
            raise ValueError("contamination_risk must be in [0, 1]")
        if self.max_repeats < 1:
            raise ValueError("max_repeats must be at least one")


@dataclass(frozen=True)
class TrialOutcome:
    item_id: str
    score: float
    progress: float = 0.0
    confidence: float | None = None
    latency_ms: float = 0.0
    cost: float = 0.0
    policy_violations: int = 0
    evidence_weight: float = 1.0
    sequence: int | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for value, name in ((self.score, "score"), (self.progress, "progress"), (self.evidence_weight, "evidence_weight")):
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be in [0, 1]")
        if self.latency_ms < 0 or self.cost < 0 or self.policy_violations < 0:
            raise ValueError("latency, cost, and policy violations must be non-negative")


@dataclass(frozen=True)
class SelectionDecision:
    item_id: str
    objective: float
    information_gain: float
    coverage_bonus: float
    reliability_bonus: float
    cost_penalty: float
    contamination_penalty: float
    predicted_success: float
    explanation: str


@dataclass(frozen=True)
class AbilityEstimate:
    dimension: str
    latent_mean: float
    latent_std: float
    score: float
    lower: float
    upper: float
    evidence: int


@dataclass(frozen=True)
class AgentProfile:
    agent_id: str
    abilities: tuple[AbilityEstimate, ...]
    strengths: tuple[str, ...]
    development_areas: tuple[str, ...]
    overall_score: float
    reliability: Mapping[str, float | None]
    calibration: Mapping[str, float | None]
    efficiency: Mapping[str, float]
    safety: Mapping[str, float]
    drift: Mapping[str, float | bool]
    diagnostics: Mapping[str, float | int | str | bool]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
