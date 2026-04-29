"""Mechanistic failure mode taxonomy for clinical trial analysis."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

from src.graph.models import EdgeType


# ── Failure modes ────────────────────────────────────────────────────────


class FailureMode(str, Enum):
    NO_TARGET_ENGAGEMENT = "no_target_engagement"
    TARGET_ENGAGED_BIOLOGY_NOT_MOVED = "target_engaged_biology_not_moved"
    BIOLOGY_MOVED_ENDPOINT_FLAT = "biology_moved_endpoint_flat"
    EFFICACY_IN_SUBGROUP_ONLY = "efficacy_in_subgroup_only"
    DOSE_LIMITING_TOXICITY = "dose_limiting_toxicity"
    WRONG_TIMEFRAME = "wrong_timeframe"
    HIGH_PLACEBO_RESPONSE = "high_placebo_response"
    WRONG_POPULATION = "wrong_population"
    UNDERPOWERED = "underpowered"
    MANUFACTURING_OR_DELIVERY = "manufacturing_or_delivery"
    COMMERCIAL_NOT_SCIENTIFIC = "commercial_not_scientific"
    INSUFFICIENT_INFORMATION = "insufficient_information"
    MULTIPLE_FACTORS = "multiple_factors"


# ── Edge update rules ────────────────────────────────────────────────────

Scope = Literal[
    "compound_specific", "target_level", "mechanism_level", "trial_design"
]


class EdgeUpdateRule(BaseModel):
    failure_mode: FailureMode
    edges_to_weaken: list[EdgeType] = Field(default_factory=list)
    edges_to_strengthen: list[EdgeType] = Field(default_factory=list)
    edges_neutral: list[EdgeType] = Field(default_factory=list)
    scope: Scope
    description: str


FAILURE_MODE_RULES: dict[FailureMode, EdgeUpdateRule] = {
    FailureMode.NO_TARGET_ENGAGEMENT: EdgeUpdateRule(
        failure_mode=FailureMode.NO_TARGET_ENGAGEMENT,
        edges_to_weaken=[EdgeType.BINDS_TO, EdgeType.MODULATES_VIA],
        edges_neutral=[
            EdgeType.MECHANISM_AFFECTS,
            EdgeType.BIOLOGY_DRIVES,
            EdgeType.ENDPOINT_CAPTURES,
        ],
        scope="compound_specific",
        description=(
            "Drug failed to engage its intended target. Weakens compound-target "
            "binding and modulation edges but says nothing about whether the "
            "target itself is valid."
        ),
    ),
    FailureMode.TARGET_ENGAGED_BIOLOGY_NOT_MOVED: EdgeUpdateRule(
        failure_mode=FailureMode.TARGET_ENGAGED_BIOLOGY_NOT_MOVED,
        edges_to_weaken=[EdgeType.MECHANISM_AFFECTS],
        edges_to_strengthen=[EdgeType.BINDS_TO],
        edges_neutral=[EdgeType.BIOLOGY_DRIVES, EdgeType.ENDPOINT_CAPTURES],
        scope="target_level",
        description=(
            "Drug hit the target but downstream biology didn't change. "
            "Strengthens binding evidence, weakens the mechanism-to-biology link. "
            "This is target-level: other drugs hitting the same target may also fail."
        ),
    ),
    FailureMode.BIOLOGY_MOVED_ENDPOINT_FLAT: EdgeUpdateRule(
        failure_mode=FailureMode.BIOLOGY_MOVED_ENDPOINT_FLAT,
        edges_to_weaken=[EdgeType.ENDPOINT_CAPTURES, EdgeType.BIOLOGY_DRIVES],
        edges_to_strengthen=[EdgeType.BINDS_TO, EdgeType.MECHANISM_AFFECTS],
        scope="mechanism_level",
        description=(
            "Biology moved as expected but the clinical endpoint didn't improve. "
            "Either the endpoint doesn't capture the biology, or the biology "
            "doesn't actually drive the disease."
        ),
    ),
    FailureMode.EFFICACY_IN_SUBGROUP_ONLY: EdgeUpdateRule(
        failure_mode=FailureMode.EFFICACY_IN_SUBGROUP_ONLY,
        edges_to_strengthen=[
            EdgeType.BINDS_TO,
            EdgeType.MECHANISM_AFFECTS,
            EdgeType.RESPONDS_DIFFERENTLY,
        ],
        edges_to_weaken=[EdgeType.BIOLOGY_DRIVES],
        scope="mechanism_level",
        description=(
            "Drug works in a biomarker-defined subgroup but not the full "
            "population. Strengthens the mechanism and subgroup-specific edges, "
            "weakens unselected biology_drives."
        ),
    ),
    FailureMode.DOSE_LIMITING_TOXICITY: EdgeUpdateRule(
        failure_mode=FailureMode.DOSE_LIMITING_TOXICITY,
        edges_to_weaken=[EdgeType.BINDS_TO],
        edges_neutral=[
            EdgeType.MECHANISM_AFFECTS,
            EdgeType.BIOLOGY_DRIVES,
            EdgeType.ENDPOINT_CAPTURES,
        ],
        scope="compound_specific",
        description=(
            "Toxicity prevented adequate dosing. Weakens this compound's "
            "binding edge (therapeutic window too narrow) but doesn't "
            "invalidate the target or mechanism."
        ),
    ),
    FailureMode.WRONG_TIMEFRAME: EdgeUpdateRule(
        failure_mode=FailureMode.WRONG_TIMEFRAME,
        edges_neutral=[
            EdgeType.BINDS_TO,
            EdgeType.MODULATES_VIA,
            EdgeType.MECHANISM_AFFECTS,
            EdgeType.BIOLOGY_DRIVES,
            EdgeType.ENDPOINT_CAPTURES,
        ],
        scope="trial_design",
        description=(
            "Trial duration or measurement timing was inappropriate. "
            "No mechanistic edges are updated — this is purely a "
            "trial design issue."
        ),
    ),
    FailureMode.HIGH_PLACEBO_RESPONSE: EdgeUpdateRule(
        failure_mode=FailureMode.HIGH_PLACEBO_RESPONSE,
        edges_neutral=[
            EdgeType.BINDS_TO,
            EdgeType.MODULATES_VIA,
            EdgeType.MECHANISM_AFFECTS,
            EdgeType.BIOLOGY_DRIVES,
            EdgeType.ENDPOINT_CAPTURES,
        ],
        scope="trial_design",
        description=(
            "Unusually high placebo response eroded the treatment effect. "
            "No mechanistic edges are updated — the signal may exist but "
            "was masked by trial design."
        ),
    ),
    FailureMode.WRONG_POPULATION: EdgeUpdateRule(
        failure_mode=FailureMode.WRONG_POPULATION,
        edges_to_strengthen=[EdgeType.MECHANISM_AFFECTS],
        edges_to_weaken=[EdgeType.RESPONDS_DIFFERENTLY],
        edges_neutral=[EdgeType.BINDS_TO, EdgeType.BIOLOGY_DRIVES],
        scope="mechanism_level",
        description=(
            "The enrolled population was not the right one for this mechanism. "
            "Strengthens mechanism validity (it may work in the right patients), "
            "weakens population selection edges."
        ),
    ),
    FailureMode.UNDERPOWERED: EdgeUpdateRule(
        failure_mode=FailureMode.UNDERPOWERED,
        edges_neutral=[
            EdgeType.BINDS_TO,
            EdgeType.MODULATES_VIA,
            EdgeType.MECHANISM_AFFECTS,
            EdgeType.BIOLOGY_DRIVES,
            EdgeType.ENDPOINT_CAPTURES,
        ],
        scope="trial_design",
        description=(
            "Trial was underpowered to detect the treatment effect. "
            "No mechanistic edges are updated — insufficient statistical "
            "power is not evidence of absence."
        ),
    ),
    FailureMode.MANUFACTURING_OR_DELIVERY: EdgeUpdateRule(
        failure_mode=FailureMode.MANUFACTURING_OR_DELIVERY,
        edges_to_weaken=[EdgeType.BINDS_TO],
        edges_neutral=[
            EdgeType.MECHANISM_AFFECTS,
            EdgeType.BIOLOGY_DRIVES,
            EdgeType.ENDPOINT_CAPTURES,
        ],
        scope="compound_specific",
        description=(
            "Manufacturing, formulation, or delivery issues prevented "
            "adequate drug exposure. Weakens compound-level binding "
            "but doesn't affect target or mechanism validity."
        ),
    ),
    FailureMode.COMMERCIAL_NOT_SCIENTIFIC: EdgeUpdateRule(
        failure_mode=FailureMode.COMMERCIAL_NOT_SCIENTIFIC,
        edges_neutral=[
            EdgeType.BINDS_TO,
            EdgeType.MODULATES_VIA,
            EdgeType.MECHANISM_AFFECTS,
            EdgeType.BIOLOGY_DRIVES,
            EdgeType.ENDPOINT_CAPTURES,
        ],
        scope="trial_design",
        description=(
            "Trial stopped for commercial/strategic reasons, not scientific "
            "failure. No mechanistic edges are updated."
        ),
    ),
    FailureMode.INSUFFICIENT_INFORMATION: EdgeUpdateRule(
        failure_mode=FailureMode.INSUFFICIENT_INFORMATION,
        edges_neutral=[
            EdgeType.BINDS_TO,
            EdgeType.MODULATES_VIA,
            EdgeType.MECHANISM_AFFECTS,
            EdgeType.BIOLOGY_DRIVES,
            EdgeType.ENDPOINT_CAPTURES,
        ],
        scope="trial_design",
        description=(
            "Not enough information to classify the failure mode. "
            "No edges are updated."
        ),
    ),
    FailureMode.MULTIPLE_FACTORS: EdgeUpdateRule(
        failure_mode=FailureMode.MULTIPLE_FACTORS,
        edges_to_weaken=[EdgeType.BINDS_TO, EdgeType.MECHANISM_AFFECTS],
        edges_neutral=[EdgeType.BIOLOGY_DRIVES, EdgeType.ENDPOINT_CAPTURES],
        scope="mechanism_level",
        description=(
            "Failure attributed to multiple overlapping factors. "
            "Applies mild weakening to compound and mechanism edges."
        ),
    ),
}


# ── Output models ────────────────────────────────────────────────────────


class TrialExtraction(BaseModel):
    """Structured data extracted from a trial's results."""

    trial_id: str = Field(min_length=1)
    compound_name: str = ""
    target_name: str = ""
    indication: str = ""
    phase: str = ""
    primary_endpoint: str = ""
    primary_endpoint_met: bool | None = None
    effect_size: float | None = None
    p_value: float | None = None
    biomarker_data: dict[str, Any] = Field(default_factory=dict)
    safety_signals: list[str] = Field(default_factory=list)
    subgroup_findings: list[str] = Field(default_factory=list)
    summary: str = ""


class FailureClassification(BaseModel):
    """Classification of a trial's failure mode."""

    trial_id: str = Field(min_length=1)
    primary_failure_mode: FailureMode
    secondary_failure_modes: list[FailureMode] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""
    evidence_quotes: list[str] = Field(default_factory=list)


class EdgeAttribution(BaseModel):
    """Maps a failure classification to specific edge updates."""

    trial_id: str = Field(min_length=1)
    failure_mode: FailureMode
    edge_updates: list[EdgeUpdate] = Field(default_factory=list)
    scope: Scope
    reasoning: str = ""


class EdgeUpdate(BaseModel):
    """A single edge update resulting from a failure attribution."""

    source_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    edge_type: EdgeType
    direction: Literal["weaken", "strengthen", "neutral"]
    magnitude: float = Field(ge=0.0, default=1.0)
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)


# Fix forward reference — EdgeAttribution references EdgeUpdate
EdgeAttribution.model_rebuild()
