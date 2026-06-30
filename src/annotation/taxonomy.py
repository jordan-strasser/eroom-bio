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


# ── Reason → competing-risks routing branch (Pillar A, A3) ────────────────
#
# The 13-category FailureMode taxonomy IS extracted by the classifier but,
# historically, thrown away by the attributor except for one leaky binary
# operational gate (``gate_weight_for`` below). FINDINGS P4: only ~4% of
# failures implicate the efficacy spine, yet 100% of failures downvote it;
# 68% fail for non-efficacy reasons but still smear the backbone; the 3 DLT
# safety deaths downvote at full weight. Net: the beliefs are trained on
# ~139 contaminating failures, then scored on the clean efficacy subset.
#
# This map routes the training update by failure reason, implementing the
# EM doc's competing-risks censoring (§3.2). It is consumed by the
# attributor whenever ``CONFIG.routing`` is True (src/config.py) — baked
# ON; formerly the ``EROOM_ROUTING`` flag (default off). Branch semantics:
#
#   EFFICACY     — the conjunctive causal spine (target engagement / mechanism
#                  translation / target validity) is implicated. Blame within
#                  the must-hold backbone via the principled responsibility.
#   MEASUREMENT  — the readout, not the biology, is implicated (wrong /
#                  insensitive endpoint, endpoint didn't capture biology, high
#                  placebo response, wrong population / enrichment). A missed
#                  readout is ambiguous between "no real effect" and "effect
#                  not detected," so EFFICACY and MEASUREMENT both blame within
#                  the same backbone set and let cross-trial sharing
#                  disambiguate over time — they share the responsibility path.
#   SAFETY       — dose-limiting toxicity / AE-driven stop. CENSOR the efficacy
#                  + measurement backbone (a trial killed on safety never
#                  reveals whether its biology would have worked); the AE edges
#                  move via the separate ``attribute_adverse_events`` path.
#   OPERATIONAL  — the chain was never properly tested (underpowered,
#                  manufacturing/delivery, commercial/strategic, insufficient
#                  information). CENSOR everything — applies zero virtual
#                  evidence to any mechanistic edge.
#   UNKNOWN      — genuinely unclassifiable (multiple overlapping factors). No
#                  metadata to route on, so fall back to the existing
#                  full-spread responsibility (the unrouted §3.1 path).
class RoutingBranch(str, Enum):
    EFFICACY = "efficacy"
    MEASUREMENT = "measurement"
    SAFETY = "safety"
    OPERATIONAL = "operational"
    UNKNOWN = "unknown"


FAILURE_MODE_BRANCH: dict[FailureMode, RoutingBranch] = {
    # EFFICACY — the conjunctive spine is what failed.
    FailureMode.NO_TARGET_ENGAGEMENT: RoutingBranch.EFFICACY,
    FailureMode.TARGET_ENGAGED_BIOLOGY_NOT_MOVED: RoutingBranch.EFFICACY,
    # MEASUREMENT — detection / validity / population, not the biology itself.
    FailureMode.BIOLOGY_MOVED_ENDPOINT_FLAT: RoutingBranch.MEASUREMENT,
    FailureMode.WRONG_TIMEFRAME: RoutingBranch.MEASUREMENT,
    FailureMode.HIGH_PLACEBO_RESPONSE: RoutingBranch.MEASUREMENT,
    FailureMode.WRONG_POPULATION: RoutingBranch.MEASUREMENT,
    FailureMode.EFFICACY_IN_SUBGROUP_ONLY: RoutingBranch.MEASUREMENT,
    # SAFETY — censor the backbone; AE edges handled separately.
    FailureMode.DOSE_LIMITING_TOXICITY: RoutingBranch.SAFETY,
    # OPERATIONAL — the chain was never properly tested; censor everything.
    FailureMode.UNDERPOWERED: RoutingBranch.OPERATIONAL,
    FailureMode.MANUFACTURING_OR_DELIVERY: RoutingBranch.OPERATIONAL,
    FailureMode.COMMERCIAL_NOT_SCIENTIFIC: RoutingBranch.OPERATIONAL,
    FailureMode.INSUFFICIENT_INFORMATION: RoutingBranch.OPERATIONAL,
    # UNKNOWN — genuinely unclassifiable; full-spread fallback.
    FailureMode.MULTIPLE_FACTORS: RoutingBranch.UNKNOWN,
}


def routing_branch_for(mode: "FailureMode | None") -> RoutingBranch:
    """The competing-risks routing branch for a failure mode (A3).

    Defaults to ``UNKNOWN`` when the mode is missing or unmapped, so an
    unrecognized reason falls back to the safe full-spread responsibility
    rather than silently censoring or blaming. Every member of the current
    ``FailureMode`` enum has an explicit entry in ``FAILURE_MODE_BRANCH``.
    """
    if mode is None:
        return RoutingBranch.UNKNOWN
    return FAILURE_MODE_BRANCH.get(mode, RoutingBranch.UNKNOWN)


# ── CT.gov stop-reason override (the terminated-trial misroute fix) ─────────
# The LLM classifier reads the trial's results / description text, NOT the
# structured CT.gov ``overallStatus`` / ``whyStopped``. So an EARLY STOP
# (TERMINATED / WITHDRAWN / SUSPENDED) for a non-efficacy reason — slow accrual,
# funding, a sponsor/portfolio decision, toxicity — can be classified as an
# efficacy/measurement failure and then WRONGLY downvote the biological backbone.
# These trials never properly tested the chain, so the correct routing is to
# CENSOR the backbone (OPERATIONAL / SAFETY), not blame the biology. This
# deterministic override consults the CT.gov status and reroutes such trials.
#
# Veto: any efficacy/futility language means the stop IS a real backbone signal
# (e.g. "terminated for futility"), so we leave the classifier's branch alone.
_EARLY_STOP_STATUSES = ("TERMINATED", "WITHDRAWN", "SUSPENDED")

# Efficacy/futility language → do NOT override (the stop is a genuine efficacy
# signal; trust the classifier's efficacy/measurement routing).
_STOP_EFFICACY_KW = (
    "futility", "futile", "efficac", "did not meet", "lack of response",
    "no benefit", "ineffective", "primary endpoint", "interim analysis",
    "lack of response", "insufficient response", "negative result",
)
# Operational / business / strategic stop language → OPERATIONAL (censor).
_STOP_OPERATIONAL_KW = (
    "accrual", "enroll", "enrol", "recruit", "fund", "financ", "budget",
    "sponsor", "business", "commercial", "strateg", "portfolio", "prioriti",
    "supply", "manufactur", "administrativ", "merger", "reorganiz", "company",
    "discontinu", "logistic", "feasib", "covid", "pandemic",
)
# Toxicity / safety stop language → SAFETY (also censors the backbone). Checked
# AFTER operational so a negated mention ("not driven by safety concerns") that
# co-occurs with a real operational reason lands as OPERATIONAL.
_STOP_SAFETY_KW = (
    "toxic", "adverse event", "tolerab", "unacceptable risk", "safety signal",
    "serious adverse",
)


def stop_reason_override(
    overall_status: str | None, why_stopped: str | None
) -> "tuple[RoutingBranch, str] | None":
    """Routing-branch override implied by CT.gov status, or ``None`` to leave the
    classifier's branch untouched.

    Fires only for an EARLY STOP (``overallStatus`` terminated/withdrawn/suspended)
    with a documented ``whyStopped`` that is clearly NON-efficacy. Returns
    ``(RoutingBranch.OPERATIONAL | SAFETY, category)`` — both censor the efficacy +
    measurement backbone downstream (and neither earns safety-survival credit), so
    the trial that never tested its biology stops wrongly downvoting it. Efficacy /
    futility language vetoes the override. Deterministic and keyword-based; absent
    or empty status returns ``None`` (default behaviour preserved).
    """
    status = (overall_status or "").strip().upper()
    why = (why_stopped or "").strip().lower()
    if not why or not any(s in status for s in _EARLY_STOP_STATUSES):
        return None
    if any(k in why for k in _STOP_EFFICACY_KW):
        return None  # documented futility/efficacy → trust the classifier
    if any(k in why for k in _STOP_OPERATIONAL_KW):
        return RoutingBranch.OPERATIONAL, "operational"
    if any(k in why for k in _STOP_SAFETY_KW):
        return RoutingBranch.SAFETY, "safety"
    return None  # ambiguous (e.g. "DSMB recommendation") → leave it alone


def _print_routing_branch_map() -> None:  # pragma: no cover - review helper
    """Print the FailureMode → RoutingBranch map for human review.

    Run with ``python -m src.annotation.taxonomy`` (see ``__main__`` below).
    """
    width = max(len(m.value) for m in FailureMode)
    print(f"{'FailureMode':<{width}}  ->  RoutingBranch")
    print("-" * (width + 22))
    for mode in FailureMode:
        print(f"{mode.value:<{width}}  ->  {routing_branch_for(mode).value}")


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
    # Combo-arm biology fix: a results_by_chain entry now describes ONE drug's
    # causal chain within the arm, not the whole arm. ``intervention`` names
    # that drug (a single compound name / slug, matching one of the arm's
    # ``compounds`` entries) so the populator can attach this entry's
    # mechanism/biology descriptions to the per-compound fanned-out chain
    # (``CausalChain.compound_id``) rather than stamping one arm-level
    # description onto every constituent. Default "" so (a) pre-fix cached
    # extractions parse unchanged and (b) mono-arm entries that omit it fall
    # back to per-arm keying. See feedback_simple_faithful / the combo-arm
    # biology-fusion fix.
    intervention: str = ""
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
    # Abstraction-ladder redesign: the drug-class functional ontology bucket
    # for THIS entry's ``intervention`` drug — a free-text label the extractor
    # maps to a ``MechanismCategory`` value (e.g. "checkpoint_blockade",
    # "kinase_inhibition", "dna_crosslinking"). The MechanismNode's IDENTITY is
    # the specific action (``mechanism_description``); this is the COARSE class
    # it rolls up into, carried onto ``MechanismNode.category`` as metadata so
    # audits + round-28 direction/type derivation keep a categorical handle.
    # Default "" so (a) pre-redesign cached extractions parse unchanged and
    # (b) entries that omit it simply contribute no category.
    mechanism_category: str = ""


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
    # Relative-effect fields for literature-derived AEs (PubMed enrichment).
    # Abstracts report a hazard/risk ratio + 95% CI rather than per-arm
    # incidences. When ``hazard_ratio`` is present the support bucket is graded
    # by effect size GATED on the CI excluding 1.0 (significance), which captures
    # rare-but-decisive endpoints (a trial-terminating mortality HR) that the
    # absolute-rate-delta path under-grades because the incidence gap is tiny.
    hazard_ratio: float | None = None
    hr_ci_low: float | None = None
    hr_ci_high: float | None = None
    # Literature can state that an AE *caused* the trial's termination (e.g.
    # torcetrapib's abstract: "terminated prematurely because of an increased
    # risk of death and cardiac events"). When True, this AE's evidence is tagged
    # failure_causing_tox so the round-30 DLT safety gate counts it fully rather
    # than flooring it as mere occurrence — even when the structured failure-mode
    # classification is insufficient_information (empty resultsSection).
    failure_causing: bool = False


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


# Soft gate weight applied to the trial's evidence when its outcome
# conditions the whole causal chain (outcome-conditioning attributor). A
# genuine TRIAL-CONDUCT failure (the chain was never properly tested:
# recruitment collapse, dosing/manufacturing error, administrative/funding
# termination, gross underpowering) should barely move the mechanistic
# beliefs — the trial isn't evidence about the biology. A valid test
# (whatever the outcome) gets full weight.
#
# CRUCIAL: wrong endpoint, wrong population SUBGROUP, wrong
# target/mechanism/biology are CHAIN failures, NOT operational — they DO
# condition the chain at full weight (gate 1.0). Only true trial-conduct
# problems gate down. Default conservatively to a valid test when unsure.
OPERATIONAL_GATE_WEIGHT: float = 0.2
VALID_TEST_GATE_WEIGHT: float = 1.0


def gate_weight_for(operational_failure: bool | None) -> float:
    """Derive the chain-conditioning gate weight from the coarse signal.

    ``True``  → the failure was operational (chain never properly tested)
                → low weight, so the trial barely perturbs the mechanism beliefs.
    ``False`` / ``None`` → valid test (or unknown) → full weight. Conservative
                by design: we only down-weight when the classifier is positive
                the failure was trial conduct, never on a guess.
    """
    if operational_failure is True:
        return OPERATIONAL_GATE_WEIGHT
    return VALID_TEST_GATE_WEIGHT


class FailureClassification(BaseModel):
    """Classification of a trial's failure mode."""

    trial_id: str = Field(min_length=1)
    primary_failure_mode: FailureMode
    secondary_failure_modes: list[FailureMode] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""
    evidence_quotes: list[str] = Field(default_factory=list)
    # Coarse per-trial gate (outcome-conditioning redesign). True ⇒ the
    # failure was a TRIAL-CONDUCT problem (the chain was never properly
    # tested: recruitment collapse, dosing/manufacturing error,
    # administrative/funding termination, gross underpowering) — the
    # outcome-conditioning attributor then applies the trial's evidence at
    # ``OPERATIONAL_GATE_WEIGHT`` so it barely moves the mechanism beliefs.
    # False / None ⇒ a valid test (full weight), whatever the outcome.
    # This is NOT edge attribution and NOT the 13-category failure mode —
    # it's one coarse "was the chain actually tested?" bit. Default None so
    # cached classifications written before this field parse unchanged and
    # fall back to full weight (valid test). See ``gate_weight_for``.
    operational_failure: bool | None = None

    @property
    def gate_weight(self) -> float:
        """Chain-conditioning weight in (0, 1] derived from
        ``operational_failure`` (see ``gate_weight_for``)."""
        return gate_weight_for(self.operational_failure)


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


if __name__ == "__main__":  # pragma: no cover
    _print_routing_branch_map()
