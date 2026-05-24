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
        edges_to_weaken=[EdgeType.AFFECTS, EdgeType.MODULATES_VIA],
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
        edges_to_strengthen=[EdgeType.AFFECTS],
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
        edges_to_strengthen=[EdgeType.AFFECTS, EdgeType.MECHANISM_AFFECTS],
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
            EdgeType.AFFECTS,
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
        edges_to_weaken=[EdgeType.AFFECTS],
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
            EdgeType.AFFECTS,
            EdgeType.MODULATES_VIA,
            EdgeType.MECHANISM_AFFECTS,
            EdgeType.BIOLOGY_DRIVES,
            EdgeType.ENDPOINT_CAPTURES,
        ],
        scope="trial_design",
        description=(
            "Trial duration or measurement timing was inappropriate. "
            "No mechanistic edges are updated—this is purely a "
            "trial design issue."
        ),
    ),
    FailureMode.HIGH_PLACEBO_RESPONSE: EdgeUpdateRule(
        failure_mode=FailureMode.HIGH_PLACEBO_RESPONSE,
        edges_neutral=[
            EdgeType.AFFECTS,
            EdgeType.MODULATES_VIA,
            EdgeType.MECHANISM_AFFECTS,
            EdgeType.BIOLOGY_DRIVES,
            EdgeType.ENDPOINT_CAPTURES,
        ],
        scope="trial_design",
        description=(
            "Unusually high placebo response eroded the treatment effect. "
            "No mechanistic edges are updated—the signal may exist but "
            "was masked by trial design."
        ),
    ),
    FailureMode.WRONG_POPULATION: EdgeUpdateRule(
        failure_mode=FailureMode.WRONG_POPULATION,
        edges_to_strengthen=[EdgeType.MECHANISM_AFFECTS],
        edges_to_weaken=[EdgeType.RESPONDS_DIFFERENTLY],
        edges_neutral=[EdgeType.AFFECTS, EdgeType.BIOLOGY_DRIVES],
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
            EdgeType.AFFECTS,
            EdgeType.MODULATES_VIA,
            EdgeType.MECHANISM_AFFECTS,
            EdgeType.BIOLOGY_DRIVES,
            EdgeType.ENDPOINT_CAPTURES,
        ],
        scope="trial_design",
        description=(
            "Trial was underpowered to detect the treatment effect. "
            "No mechanistic edges are updated—insufficient statistical "
            "power is not evidence of absence."
        ),
    ),
    FailureMode.MANUFACTURING_OR_DELIVERY: EdgeUpdateRule(
        failure_mode=FailureMode.MANUFACTURING_OR_DELIVERY,
        edges_to_weaken=[EdgeType.AFFECTS],
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
            EdgeType.AFFECTS,
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
            EdgeType.AFFECTS,
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
        edges_to_weaken=[EdgeType.AFFECTS, EdgeType.MECHANISM_AFFECTS],
        edges_neutral=[EdgeType.BIOLOGY_DRIVES, EdgeType.ENDPOINT_CAPTURES],
        scope="mechanism_level",
        description=(
            "Failure attributed to multiple overlapping factors. "
            "Applies mild weakening to compound and mechanism edges."
        ),
    ),
}


# ── Output models ────────────────────────────────────────────────────────


class ExtractedArm(BaseModel):
    """An LLM-emitted arm definition from the extraction call."""

    arm_id: str = Field(min_length=1)
    compounds: list[str] = Field(default_factory=list)


class ExtractedSubgroup(BaseModel):
    """An LLM-canonicalized subgroup with its raw descriptor + features."""

    raw_descriptor: str = Field(min_length=1)
    # Free-form list of {axis, key, level} dicts; the populator runs each
    # through ``subgroup_taxonomy.canonicalize_feature`` to produce the
    # SubgroupFeatures that compose the PopulationNode id.
    features: list[dict[str, str]] = Field(default_factory=list)


class ChainResult(BaseModel):
    """Per-(arm × subgroup) result the LLM emits, used to populate chains."""

    arm_id: str = Field(min_length=1)
    # null subgroup_descriptor → result applies to the parent enrollment population
    subgroup_descriptor: str | None = None
    endpoint: str = ""
    effect_size: float | None = None
    p_value: float | None = None
    outcome: str = "unknown"  # success | failure | partial | unknown
    # A.0b: per-chain *contextualized* free-text descriptions, emitted by the
    # extractor for sharper manifold-2 (s,t) localization — the same edge gets
    # distinct evidence points (e.g. "VEGFR2 inhibition in tumor vasculature"
    # vs a different chain's framing). Default "" so pre-A.0b cached
    # extractions parse unchanged; A.3 uses them (falling back to the
    # trial-level descriptions) to place each evidence record on the edge
    # belief surface. See future_ideas/manifold_learning.md.
    mechanism_description: str = ""
    biology_description: str = ""
    population_description: str = ""


class DoseInfo(BaseModel):
    """Structured dose / schedule extracted from a trial.

    All fields are optional because trial reports vary widely in how
    much dosing detail they include; downstream consumers must handle
    None gracefully. ``dose`` is a free string (the LLM emits whatever
    units the report uses, e.g. "150 mg" or "10 mg/kg") rather than a
    parsed numeric—calibration of dose-vs-response across trials needs
    unit normalization that we don't have yet.
    """

    dose: str = ""
    schedule: str = ""
    max_tolerated_dose: str = ""
    dose_modifications: str = ""


class ArmIncidence(BaseModel):
    """One arm's reported incidence of a single adverse event.

    Sourced from ``adverseEventsModule.seriousEvents[].stats[]`` on
    ClinicalTrials.gov, where each arm gets ``numAffected`` over
    ``numAtRisk``. ``arm_descriptor`` is the CT.gov eventGroup title
    (e.g. "Nivolumab + Ipilimumab") — the attributor matches that
    string against compound names in the graph to decide which compounds
    were active on the arm.
    """

    arm_descriptor: str = Field(min_length=1)
    n_affected: int = Field(ge=0)
    n_at_risk: int = Field(ge=0)
    pct: float | None = None  # derived: 100 * n_affected / n_at_risk


class StructuredAE(BaseModel):
    """One adverse event reported in a trial.

    ``term`` is the raw term as the trial reported it; ``meddra_term`` is
    the normalized MedDRA preferred term (populated by the MedDRA mapper
    before attribution). The two are kept separate so we never lose the
    original wording.

    Per-arm rates (``arm_incidences``) are populated directly from
    CT.gov's structured ``adverseEventsModule`` when a trial reports
    results. They are the source of truth for attribution: the attributor
    derives a per-compound ``(tx, ctrl)`` pair by partitioning arms on
    whether the compound was active. ``incidence_treatment_pct`` and
    ``incidence_control_pct`` remain as a fallback for trials whose
    safety data is only available as narrative LLM extraction (no
    structured results section).
    """

    term: str = Field(min_length=1)
    meddra_term: str = ""
    grade: str = ""  # CTCAE grade or range, e.g. "3" or "1-2"
    incidence_treatment_pct: float | None = None
    incidence_control_pct: float | None = None
    arm_incidences: list[ArmIncidence] = Field(default_factory=list)
    serious: bool = False


class ModulationEntry(BaseModel):
    """One modulator → primary-chain modulation relationship.

    The causal hypothesis chain has a fixed canonical structure for
    every trial: ``compound → target → mechanism → biology → indication``.
    A modulation has two compounds in scope: a *primary* (the lead
    compound whose chain the trial is testing) and a *modulator* (the
    supportive compound altering the primary's effect).

    The LLM identifies which compound is which and at which LAYER of
    the primary's chain the modulation acts. The specific graph node
    ids at each layer are something the populator already knows from
    having built the chain — the LLM doesn't need to (and shouldn't
    try to) name them. This is the v0.3 redesign: ask for canonical
    things the LLM can know (compound names, layer names), not for
    Reactome / GO ids the LLM has to guess at.

    Populator routing: look up the primary compound's chain in this
    trial, pull ``chain.target_id`` / ``chain.mechanism_id`` /
    ``chain.biology_id`` based on ``affects_layer``, and emit a
    MODULATES_EFFICACY_OF edge from the modulator's compound node to
    that chain node.

    ``direction`` + ``confidence`` map to a ``SupportBucket`` via
    ``src.inference.beliefs.modulation_bucket`` so modulation edges
    receive Beta-Binomial updates with the same machinery the rest of
    the system uses. N_eff comes from trial structure, not the LLM.
    """

    modulator_compound_id: str = Field(min_length=1)
    primary_compound_id: str = Field(min_length=1)
    affects_layer: Literal["target", "mechanism", "biology"]
    direction: Literal["amplifies", "suppresses", "neutral"]
    confidence: float = Field(ge=0.0, le=1.0)
    hypothesis: str = ""
    citation: str = ""


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
    # Patient/observation count for the trial (LLM-extracted sample size;
    # ~99% populated in practice). Feeds the precision-aware n_eff path.
    sample_size: int | None = None
    biomarker_data: dict[str, Any] = Field(default_factory=dict)
    safety_signals: list[str] = Field(default_factory=list)
    subgroup_findings: list[str] = Field(default_factory=list)
    summary: str = ""
    # Combinatorial + subgroup-aware extraction.
    # Empty lists are tolerable—a trial with one arm and no reported
    # subgroups produces a single chain at the parent population.
    arms: list[ExtractedArm] = Field(default_factory=list)
    subgroups: list[ExtractedSubgroup] = Field(default_factory=list)
    results_by_chain: list[ChainResult] = Field(default_factory=list)
    # Structured safety + exposure fields. Feed graph-native AE attribution
    # (causes_ae edges) and let the mitigating-factor checklist on the
    # contradict side reference real exposure data rather than narrative.
    duration_weeks: float | None = None
    dose_info: DoseInfo = Field(default_factory=DoseInfo)
    adverse_events: list[StructuredAE] = Field(default_factory=list)
    # v0.3.0: LLM-emitted modulation relationships. One entry per
    # modulator-affects-edge claim. Defaults to empty; trials with no
    # combination interactions simply emit []. See ``ModulationEntry``.
    modulation_entries: list[ModulationEntry] = Field(default_factory=list)
    # A.0: trial-level rich free-text descriptions, preserved from the
    # therapeutic hypothesis so the populator can attach them to the
    # Mechanism / Biology / parent-Population nodes as the BioLORD embedding
    # substrate (the canonical id is just a routing tag). Already present in
    # every cached extraction's raw JSON, so backfilling them is a populate
    # re-run with no new LLM call. See future_ideas/eroom_node_graph_kickoff.md.
    mechanism_description: str = ""
    biology_description: str = ""
    target_population_description: str = ""


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
    """A single edge update resulting from a failure attribution.

    ``affecting_arm_id`` (v1 of classifier per-arm emission) names which
    arm's outcome this edge update is decomposing. The attributor uses it
    to restrict chain routing to chains in that arm — without it, a
    multi-arm trial's classifier output gets routed to whichever chain
    happens to match the entity names first, which silently drops
    supportive-arm evidence (e.g. aldesleukin monotherapy outcome in
    NCT00019682 contributing nothing to aldesleukin's chain because the
    classifier emitted only the combo arm's hypothesis).

    Null = back-compat for cached classifications written before this
    field existed. Treated as "applies to whichever chain matches" — the
    pre-v1 behavior.
    """

    source_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    edge_type: EdgeType
    direction: Literal["weaken", "strengthen", "neutral"]
    magnitude: float = Field(ge=0.0, default=1.0)
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    affecting_arm_id: str | None = None


# Fix forward reference—EdgeAttribution references EdgeUpdate
EdgeAttribution.model_rebuild()
