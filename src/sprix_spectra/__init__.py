"""Sprix SPECTRA public API."""

from .evaluator import SpectraEvaluator
from .models import (
    AbilityEstimate,
    AgentProfile,
    CapabilityDimension,
    EvalItem,
    SelectionDecision,
    TrialOutcome,
)
from .report import profile_to_json, profile_to_markdown
from .synthetic import DEFAULT_DIMENSIONS, AgentSimulator, SyntheticAgent, build_item_bank

__all__ = [
    "DEFAULT_DIMENSIONS",
    "AbilityEstimate",
    "AgentProfile",
    "AgentSimulator",
    "CapabilityDimension",
    "EvalItem",
    "SelectionDecision",
    "SpectraEvaluator",
    "SyntheticAgent",
    "TrialOutcome",
    "build_item_bank",
    "profile_to_json",
    "profile_to_markdown",
]
