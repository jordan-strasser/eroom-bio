"""Translate failure classifications into graph edge updates."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src.graph.models import (
    AdverseEventNode,
    canonical_modulation_endpoints,
    CausalChain,
    EdgeBeliefState,
    EdgeType,
    EvidenceDirection,
    EvidenceRecord,
    EvidenceType,
    GraphEdge,
    TrialArm,
    TrialOutcome,
    TrialSubgraph,
    normalize_entity,
)
from src.graph.store import GraphStore
from src.inference.ae_propagation import propagate_to_target_associated_ae
from src.inference.beliefs import SupportBucket, bucket_to_direction, modulation_bucket
from src.annotation.meddra import MeddraCache, ae_node_id, normalize_ae_term
from src.annotation.taxonomy import (
    ArmIncidence,
    FAILURE_MODE_RULES,
    FailureClassification,
    FailureMode,
    ModulationEntry,
    StructuredAE,
    TrialExtraction,
)

logger = logging.getLogger(__name__)

_ANNOTATIONS_DIR = Path("data/annotations")
# Misrouted-update audit log—written when a classifier-emitted edge update
# can't be matched to any chain in the trial subgraph. The expected use is
# vocab/extraction-prompt review, not silent drop.
_UNROUTED_LOG_PATH = Path("data/dev/unrouted_attribution_updates.jsonl")
# v0.3.0 unrouted modulation log — separate file so unroutable LLM
# modulation_entries don't drown in the classifier unrouted log. A high
# count here signals the extraction prompt is emitting entity ids that
# don't normalize to graph nodes; treat it as a debug signal worth
# investigating before scaling the prompt to a wider corpus.
_UNROUTED_MOD_LOG_PATH = Path("data/dev/unrouted_modulation_entries.jsonl")

# Map trial phase string to EvidenceType
_PHASE_TO_EVIDENCE: dict[str, EvidenceType] = {
    "1": EvidenceType.CLINICAL_PHASE1,
    "early_1": EvidenceType.CLINICAL_PHASE1,
    "2": EvidenceType.CLINICAL_PHASE2,
    "2/3": EvidenceType.CLINICAL_PHASE3,
    "3": EvidenceType.CLINICAL_PHASE3,
    "4": EvidenceType.CLINICAL_PHASE3,
}

# Node-type pairs each edge type connects (source_type, target_type). Used to
# constrain free-text entity-name → canonical-id resolution to plausible
# node types.
_EDGE_TYPE_TO_NODE_TYPES: dict[EdgeType, tuple[str, str]] = {
    EdgeType.AFFECTS:              ("InterventionNode", "TargetNode"),
    EdgeType.MODULATES_VIA:        ("TargetNode", "MechanismNode"),
    EdgeType.MECHANISM_AFFECTS:    ("MechanismNode", "BiologyNode"),
    EdgeType.BIOLOGY_DRIVES:       ("BiologyNode", "IndicationNode"),
    EdgeType.REFLECTS_BIOLOGY:     ("BiologyNode", "EndpointNode"),
    EdgeType.ENDPOINT_CAPTURES:    ("EndpointNode", "IndicationNode"),
    EdgeType.RESPONDS_DIFFERENTLY: ("PopulationNode", "IndicationNode"),
    EdgeType.CAUSES_AE:            ("InterventionNode", "AdverseEventNode"),
    EdgeType.TARGET_ASSOCIATED_AE: ("TargetNode", "AdverseEventNode"),
    # Structural (no Beta belief, like COMPOSED_OF). Classifier never
    # emits these directly — they're added by the populator from the
    # _INDICATION_HIERARCHY table.
    EdgeType.SUBTYPE_OF:           ("IndicationNode", "IndicationNode"),
}


# Sentinel used in CausalChain fields when a graph id wasn't yet resolved
# (e.g. by populate.py before extraction filled in the biology id).
_UNKNOWN_PLACEHOLDER = "UNKNOWN"


# TrialOutcome → ordinal scale for arm-differential bucket mapping.
# UNKNOWN is excluded — pairs involving an unknown outcome yield no
# modulation evidence (returned as None below).
_OUTCOME_SCALE: dict[TrialOutcome, int] = {
    TrialOutcome.FAILURE: 0,
    TrialOutcome.PARTIAL: 1,
    TrialOutcome.SUCCESS: 2,
}


def _arm_differential_bucket(
    backbone_outcome: TrialOutcome,
    combo_outcome: TrialOutcome,
) -> SupportBucket | None:
    """Map (backbone arm, combo arm) outcomes to a support bucket.

    Bucket describes "adding the extra constituents to the backbone
    helps." A 2-step jump (e.g. failure → success) is strong; 1-step is
    moderate; equal outcomes is ambiguous; negative deltas contradict.
    Returns None when either arm's outcome is UNKNOWN — no differential
    can be computed.
    """
    if (
        backbone_outcome == TrialOutcome.UNKNOWN
        or combo_outcome == TrialOutcome.UNKNOWN
    ):
        return None
    delta = _OUTCOME_SCALE[combo_outcome] - _OUTCOME_SCALE[backbone_outcome]
    if delta == 2:
        return SupportBucket.STRONG_SUPPORT
    if delta == 1:
        return SupportBucket.MODERATE_SUPPORT
    if delta == 0:
        return SupportBucket.AMBIGUOUS
    if delta == -1:
        return SupportBucket.MODERATE_CONTRADICT
    if delta == -2:
        return SupportBucket.STRONG_CONTRADICT
    return None


def _single_arm_combo_bucket(
    outcome: TrialOutcome,
) -> SupportBucket | None:
    """Map a standalone combo-arm outcome to a weak pairwise bucket.

    No head-to-head signal, so even a clearly successful or failed combo
    only contributes a weak update to each pairwise modulation edge.
    """
    if outcome == TrialOutcome.SUCCESS:
        return SupportBucket.WEAK_SUPPORT
    if outcome == TrialOutcome.PARTIAL:
        return SupportBucket.AMBIGUOUS
    if outcome == TrialOutcome.FAILURE:
        return SupportBucket.WEAK_CONTRADICT
    return None


def _aggregate_arm_outcomes(
    trial: TrialSubgraph,
    extraction: "TrialExtraction | None" = None,
) -> dict[str, TrialOutcome]:
    """Per-arm authoritative outcome, keyed by graph arm_id.

    Reads from extraction.results_by_chain when provided, mapping the
    LLM-emitted arm_ids onto graph arm_ids by compound-set match. This
    is the primary path: chain.outcome is never written back from the
    extraction in the current populate flow, so it stays UNKNOWN even
    when the extraction reports per-arm outcomes.

    Falls back to ``chain.outcome`` when the extraction is unavailable
    or doesn't yield a mapping — exercised by unit tests that fixture
    chain outcomes directly.
    """
    by_arm: dict[str, TrialOutcome] = {}
    if extraction is not None:
        ext_to_graph = _map_extraction_arms_to_graph(trial, extraction)
        for cr in getattr(extraction, "results_by_chain", []) or []:
            # Parent-population results only (subgroup_descriptor is null).
            if cr.subgroup_descriptor is not None:
                continue
            graph_arm_id = ext_to_graph.get(cr.arm_id)
            if graph_arm_id is None:
                continue
            try:
                outcome = TrialOutcome(cr.outcome)
            except ValueError:
                continue
            if outcome == TrialOutcome.UNKNOWN:
                continue
            by_arm.setdefault(graph_arm_id, outcome)
        if by_arm:
            return by_arm

    # Fallback: chain.outcome (used by tests + as a defensive path).
    for chain in trial.chains:
        if chain.subgroup_population_id != trial.parent_population_id:
            continue
        if chain.outcome == TrialOutcome.UNKNOWN:
            continue
        by_arm.setdefault(chain.arm_id, chain.outcome)
    for chain in trial.chains:
        if chain.arm_id in by_arm:
            continue
        if chain.outcome == TrialOutcome.UNKNOWN:
            continue
        by_arm[chain.arm_id] = chain.outcome
    return by_arm


def _map_extraction_arms_to_graph(
    trial: TrialSubgraph,
    extraction: "TrialExtraction",
) -> dict[str, str]:
    """ext_arm_id → graph_arm_id by compound-set match.

    Extraction arms carry LLM-emitted arm_ids ("aldesleukin_alone") plus
    free-text compound names ("aldesleukin", "gp100 antigen"). Graph
    arms carry CT.gov group_ids ("arm_i_aldesleukin") plus canonical
    compound ids ("aldesleukin", "gp100_antigen"). The reconciliation
    normalizes the extraction's compound names through `normalize_entity`
    and matches by set equality.
    """
    from src.graph.models import normalize_entity

    graph_arm_by_compounds: dict[frozenset[str], str] = {
        frozenset(arm.compound_ids): arm.arm_id for arm in trial.arms
    }
    mapping: dict[str, str] = {}
    for ea in getattr(extraction, "arms", []) or []:
        compounds = ea.compounds or []
        if not compounds:
            continue
        normalized = frozenset(
            normalize_entity(c, "InterventionNode") for c in compounds
        )
        graph_arm_id = graph_arm_by_compounds.get(normalized)
        if graph_arm_id is None:
            continue
        mapping[ea.arm_id] = graph_arm_id
    return mapping


# ── Routing helpers ──────────────────────────────────────────────────────


_NON_ALNUM_RE = __import__("re").compile(r"[^a-z0-9]+")


def _norm_name(text: str) -> str:
    """Lowercase, strip non-alphanumerics. PD-1 / PD1 / pd_1 → 'pd1'."""
    return _NON_ALNUM_RE.sub("", (text or "").lower())


class _NameIndex:
    """Case-insensitive, punctuation-insensitive name → node-id index.

    Built once per ``attribute()`` call. ``matches`` returns True for an
    exact normalized match or a substring containment (length-gated to
    avoid 1-2 char noise). Normalization strips dashes and spaces so
    "CTLA-4" / "CTLA4" / "ctla 4" all collapse to "ctla4".
    """
    def __init__(self) -> None:
        # node_id -> list of normalized names
        self._names_by_id: dict[str, list[str]] = {}

    def add(self, node_type: str, node_id: str, names: list[str]) -> None:
        normed: list[str] = []
        for name in names:
            n = _norm_name(name)
            if n:
                normed.append(n)
        # Always include the id itself as a fallback name to match against
        #—some classifier emissions reuse the canonical id directly.
        normed.append(_norm_name(node_id))
        self._names_by_id.setdefault(node_id, []).extend(normed)

    def matches(self, node_id: str, query: str) -> bool:
        q = _norm_name(query)
        if not q:
            return False
        names = self._names_by_id.get(node_id, [])
        for name in names:
            if name == q:
                return True
        if len(q) < 3:
            return False
        for name in names:
            if len(name) < 3:
                continue
            if q in name or name in q:
                return True
        return False


def _build_name_index(
    graph: GraphStore, *, node_types: set[str]
) -> _NameIndex:
    """Build a name → node-id index for lookup at attribution time.

    For TargetNodes, also seed the index with HGNC aliases of the
    canonical gene_symbol (so a classifier emitting "PD-1" routes to
    the same node as its HUGO canonical "PDCD1"). HGNC lookup is
    best-effort—when the resolver isn't loaded the index falls back
    to name + gene_symbol only.
    """
    from src.graph.hgnc_resolver import (
        _ALIAS_TO_CANONICAL,
        canonical_symbol,
        is_loaded as hgnc_loaded,
    )

    idx = _NameIndex()

    # Reverse the HGNC dict once: canonical → list[alias]. Cheap because
    # HGNC has ~50k canonicals and we only iterate Targets here.
    canonical_to_aliases: dict[str, list[str]] = {}
    if hgnc_loaded() and _ALIAS_TO_CANONICAL is not None:
        for alias_norm, canonical in _ALIAS_TO_CANONICAL.items():
            canonical_to_aliases.setdefault(canonical, []).append(alias_norm)

    for node_type in node_types:
        for node in graph.get_nodes_by_type(node_type):
            names = [node.get("name", "")]
            if node_type == "TargetNode":
                gs = node.get("gene_symbol", "") or ""
                if gs:
                    names.append(gs)
                # Alias expansion: if gene_symbol resolves through HGNC,
                # add every known alias so any classifier-emitted variant
                # ("PD-1", "PDL1", "B7-H1") matches the same TargetNode.
                if gs:
                    canonical = canonical_symbol(gs) or gs.upper()
                    for alias_norm in canonical_to_aliases.get(canonical, []):
                        names.append(alias_norm)
            idx.add(node_type, node["id"], names)
    return idx


def _chain_edges_for_type(
    chain: CausalChain,
    arm: TrialArm,
    edge_type: EdgeType,
) -> list[tuple[str, str]]:
    """All (source_id, target_id) candidates this chain implies for the edge type.

    binds_to gets one candidate per constituent compound on the arm—that
    way the classifier-emitted ``Ipilimumab → CTLA-4`` update routes to the
    ipi mono chain (or the combo chain's ipi side), never to the nivo→PD-1
    pair.
    """
    if edge_type == EdgeType.AFFECTS:
        return [(cid, chain.target_id) for cid in arm.compound_ids]
    if edge_type == EdgeType.MODULATES_VIA:
        return [(chain.target_id, chain.mechanism_id)]
    if edge_type == EdgeType.MECHANISM_AFFECTS:
        return [(chain.mechanism_id, chain.biology_id)]
    if edge_type == EdgeType.BIOLOGY_DRIVES:
        return [(chain.biology_id, chain.indication_id)]
    if edge_type == EdgeType.REFLECTS_BIOLOGY:
        return [(chain.biology_id, chain.endpoint_id)]
    if edge_type == EdgeType.ENDPOINT_CAPTURES:
        return [(chain.endpoint_id, chain.indication_id)]
    if edge_type == EdgeType.RESPONDS_DIFFERENTLY:
        return [(chain.subgroup_population_id, chain.indication_id)]
    return []


def _score_pair_against_names(
    src_id: str,
    tgt_id: str,
    src_name: str,
    tgt_name: str,
    name_index: _NameIndex,
) -> tuple[int, bool]:
    """Returns ``(score, explicit_mismatch)``.

    Score 2: both source and target match the classifier-emitted names.
    Score 1: one side matches, other side has no name to check (empty
             classifier emission). Falls back rather than dropping.
    Score 0: either at least one side has a name that *doesn't* match
             (``explicit_mismatch=True``), or both sides are empty
             (``explicit_mismatch=False``).
    """
    src_match = name_index.matches(src_id, src_name) if src_name else None
    tgt_match = name_index.matches(tgt_id, tgt_name) if tgt_name else None

    if src_match is False or tgt_match is False:
        return 0, True

    score = 0
    if src_match:
        score += 1
    if tgt_match:
        score += 1
    return score, False


def _log_unrouted(trial_id: str, item: dict[str, Any], *, reason: str) -> None:
    _UNROUTED_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "trial_id": trial_id,
        "edge_type": item.get("edge_type"),
        "source_entity": item.get("source_entity"),
        "target_entity": item.get("target_entity"),
        "support": item.get("support"),
        "reason": reason,
        "logged_at": datetime.now(timezone.utc).isoformat(),
    }
    with _UNROUTED_LOG_PATH.open("a") as fh:
        fh.write(json.dumps(record) + "\n")


def _resolve_arm_id(
    llm_arm_id: str,
    arm_by_id: dict[str, TrialArm],
) -> str | None:
    """Map an LLM-emitted arm_id to a real arm_id in the trial subgraph.

    Both extractor and classifier prompts enumerate the trial's
    CT.gov-derived `group_id` values and instruct the LLM to use them
    verbatim (see `_format_arm_id_menu` in classifier.py and the
    Arm Groups section of extraction_user.txt). Resolution is therefore
    an exact dict lookup — no fuzzy matching, no token overlap, no
    compound-set heuristics. When the LLM ignores the constraint and
    invents a slug anyway, the standard unrouted log surfaces it as a
    prompt-regression signal instead of silently routing the wrong arm.
    """
    if not llm_arm_id:
        return None
    if llm_arm_id in arm_by_id:
        return llm_arm_id
    return None


def _log_unrouted_modulation(
    trial_id: str, entry: ModulationEntry, *, reason: str,
) -> None:
    """Append a record to the v0.3.0 unrouted-modulation log.

    A high volume of records here indicates the extraction prompt is
    emitting compound names that aren't in the trial (or the primary
    chain layer being named can't be resolved). Surface for debugging
    rather than silently dropping the modulation.
    """
    _UNROUTED_MOD_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "trial_id": trial_id,
        "modulator_compound_id": entry.modulator_compound_id,
        "primary_compound_id": entry.primary_compound_id,
        "affects_layer": entry.affects_layer,
        "direction": entry.direction,
        "confidence": entry.confidence,
        "reason": reason,
        "logged_at": datetime.now(timezone.utc).isoformat(),
    }
    with _UNROUTED_MOD_LOG_PATH.open("a") as fh:
        fh.write(json.dumps(record) + "\n")




# ── AE attribution helpers ──────────────────────────────────────────────


def _ae_support_bucket(
    treatment_pct: float | None,
    control_pct: float | None,
    treatment_n: int | None = None,
) -> SupportBucket:
    """Map per-arm AE incidence to a SupportBucket for the causes_ae edge.

    No treatment-arm rate → AMBIGUOUS (we can't even say the trial saw it).
    Control rate missing is treated as 0—a conservative read that says
    "no reported background", which lets unilateral safety signals from
    single-arm Phase 1s contribute (with the bucket downgrade reflecting
    the missing comparator).

    Rate thresholds (deliberately coarse):
      - delta ≥ 20pp OR RR ≥ 3 → strong_support
      - delta ≥ 10pp OR RR ≥ 2 → moderate_support
      - delta ≥ 5pp  OR RR ≥ 1.5 → weak_support
      - delta ≤ -5pp           → weak_contradict (drug arm safer than control)
      - otherwise              → ambiguous (background rate)

    Absolute-count gate (fixes.md #9): a 1.2% rate in an n=85 arm is
    ~1 patient, which the rate-only path would label moderate_support
    via RR ≥ 2. That's noise being graded as evidence. With
    ``treatment_n`` available we additionally require:
      - ≥ 5 affected patients to remain at strong / moderate support
      - ≥ 3 affected patients to remain at weak support
      - otherwise downgrade to AMBIGUOUS

    Calibration of these cutoffs is downstream of the calibration harness
    (NEXT_SESSION follow-up #1) just like the bucket→p_obs table.
    """
    if treatment_pct is None:
        return SupportBucket.AMBIGUOUS
    c = control_pct if control_pct is not None else 0.0
    delta = treatment_pct - c
    # 0.5pp floor on the denominator avoids div-by-zero and keeps RR
    # finite when control = 0; an AE seen in 30% of treated patients
    # with 0% control still gets RR = 60, well into strong territory.
    rr = treatment_pct / max(c, 0.5)

    if delta >= 20 or rr >= 3:
        bucket = SupportBucket.STRONG_SUPPORT
    elif delta >= 10 or rr >= 2:
        bucket = SupportBucket.MODERATE_SUPPORT
    elif delta >= 5 or rr >= 1.5:
        bucket = SupportBucket.WEAK_SUPPORT
    elif delta <= -5:
        bucket = SupportBucket.WEAK_CONTRADICT
    else:
        bucket = SupportBucket.AMBIGUOUS

    if treatment_n is None or bucket == SupportBucket.AMBIGUOUS:
        return bucket

    # Absolute-count gate. round to ensure 1.2% × 85 = 1.02 → 1 patient,
    # not 1.02 → moderate_support survives the float comparison.
    abs_count = round(treatment_pct * treatment_n / 100.0)
    if bucket in (SupportBucket.STRONG_SUPPORT, SupportBucket.MODERATE_SUPPORT):
        if abs_count < 5:
            if abs_count >= 3:
                return SupportBucket.WEAK_SUPPORT
            return SupportBucket.AMBIGUOUS
    elif bucket == SupportBucket.WEAK_SUPPORT:
        if abs_count < 3:
            return SupportBucket.AMBIGUOUS
    elif bucket == SupportBucket.WEAK_CONTRADICT:
        # Drop in incidence: gate on absolute affected count in the
        # CONTROL arm so a 5pp drop from "5% of 5 control patients" to
        # 0% in treatment isn't graded as a real safety improvement.
        ctrl_abs = round(c * treatment_n / 100.0)
        if ctrl_abs < 3:
            return SupportBucket.AMBIGUOUS
    return bucket


def _per_compound_rates_from_arms(
    arm_incidences: list[ArmIncidence],
    compound_name: str,
) -> tuple[float | None, float | None, int | None]:
    """Partition per-arm AE counts on whether ``compound_name`` was active.

    Arms whose ``arm_descriptor`` contains the compound name (word-boundary,
    case-insensitive) contribute to the treatment pool; arms without it
    contribute to the comparator pool. Returns ``(tx_pct, ctrl_pct, tx_n)``
    where the pcts are pooled rates and tx_n is the total at-risk
    population for the compound — fed to ``_ae_support_bucket`` for the
    absolute-count gate.

    Returns (None, None, None) when no arm matched the compound name —
    callers should fall back to the legacy flat ``tx/ctrl`` pair in that
    case.
    """
    if not compound_name:
        return (None, None, None)
    pattern = re.compile(rf"\b{re.escape(compound_name)}\b", re.IGNORECASE)
    tx_affected = tx_at_risk = 0
    ctrl_affected = ctrl_at_risk = 0
    for ai in arm_incidences:
        if pattern.search(ai.arm_descriptor):
            tx_affected += ai.n_affected
            tx_at_risk += ai.n_at_risk
        else:
            ctrl_affected += ai.n_affected
            ctrl_at_risk += ai.n_at_risk
    if tx_at_risk == 0:
        return (None, None, None)
    tx_pct = 100.0 * tx_affected / tx_at_risk
    ctrl_pct = 100.0 * ctrl_affected / ctrl_at_risk if ctrl_at_risk > 0 else None
    return (tx_pct, ctrl_pct, tx_at_risk)


def _format_ae_note(
    ae: StructuredAE,
    preferred_term: str,
    *,
    tx_pct: float | None = None,
    ctrl_pct: float | None = None,
) -> str:
    bits = [f"AE: {preferred_term} (raw: {ae.term!r})"]
    if ae.grade:
        bits.append(f"grade {ae.grade}")
    # Per-compound rates (computed by attributor from arm_incidences)
    # win over the legacy flat fields; only fall back when missing.
    tx = tx_pct if tx_pct is not None else ae.incidence_treatment_pct
    ctrl = ctrl_pct if ctrl_pct is not None else ae.incidence_control_pct
    if tx is not None:
        bits.append(f"tx {tx:g}%")
    if ctrl is not None:
        bits.append(f"ctrl {ctrl:g}%")
    if ae.serious:
        bits.append("SAE")
    return "; ".join(bits)


# ── Output model ─────────────────────────────────────────────────────────


class AppliedEdgeUpdate(BaseModel):
    """A single edge update that was applied to the graph."""

    source_id: str
    target_id: str
    edge_type: EdgeType
    evidence: EvidenceRecord
    pre_update_belief: EdgeBeliefState
    post_update_belief: EdgeBeliefState

    @property
    def probability_change(self) -> float:
        return (
            self.post_update_belief.expected_probability
            - self.pre_update_belief.expected_probability
        )


# ── Attributor ───────────────────────────────────────────────────────────


class Attributor:
    def __init__(self, graph: GraphStore) -> None:
        self.graph = graph
        # Round-16 observability: per-call counters that the orchestrator
        # can read after attribute() to decide whether the classifier's
        # output landed cleanly on the trial subgraph or got dropped
        # excessively (LLM hallucinated entities, schema mismatches,
        # bad arm ids, etc.). The drop count is otherwise only visible
        # as logged warnings, which means a build-level threshold
        # ("abort if > 30% of classifier edges drop") couldn't be
        # enforced without scraping the log.
        self.last_attempted_updates: int = 0
        self.last_dropped_updates: int = 0

    def attribute(
        self,
        classification: FailureClassification,
        trial: TrialSubgraph,
        extraction: "TrialExtraction | None" = None,
    ) -> list[AppliedEdgeUpdate]:
        """Translate a classification into concrete edge updates.

        Each classifier-emitted edge update is routed to the specific chain
        whose canonical ids match the classifier's free-text source/target
        entity names—preventing the misrouting bug where (e.g.)
        Ipilimumab→CTLA-4 evidence lands on the nivolumab→PD-1 edge in a
        combo trial. Updates that don't match any chain are logged to
        ``data/dev/unrouted_attribution_updates.jsonl`` rather than
        silently misapplied.

        ``extraction`` (optional) is the trial's structured extraction; if
        present, modulation-edge emission reads per-arm outcomes from
        ``results_by_chain`` (mapping LLM arm_ids to graph arm_ids by
        compound-set match). Without it, modulation emission falls back
        to ``chain.outcome``, which is UNKNOWN in current populate flows
        — so production callers should always pass the extraction.
        """
        raw = getattr(classification, "_raw", {})
        raw_edges = raw.get("edges_to_update", [])
        rule = FAILURE_MODE_RULES.get(classification.primary_failure_mode)
        # For trial_outcome=success, the failure-mode label is descriptive
        # only—the schema forces a pick but mechanistic-failure rules
        # don't apply when nothing failed. Disabling the cross-check lets
        # the per-edge bucket drive the update directly; otherwise a
        # successful subgroup trial labeled efficacy_in_subgroup_only loses
        # its biology_drives validation to the cross-check.
        if raw.get("trial_outcome") == "success":
            rule = None
        phase = trial.phase
        evidence_type = _PHASE_TO_EVIDENCE.get(phase, EvidenceType.LITERATURE)

        # Pre-compute name-resolution helpers from the graph's nodes (one
        # pass per node type used here).
        name_index = _build_name_index(
            self.graph,
            node_types={
                "InterventionNode", "CompoundNode",  # accept both names
                "TargetNode", "MechanismNode",
                "BiologyNode", "EndpointNode", "IndicationNode",
                "PopulationNode",
            },
        )
        arm_by_id = {arm.arm_id: arm for arm in trial.arms}

        updates: list[AppliedEdgeUpdate] = []
        # Per-trial dedup: an (edge_type, src_id, tgt_id) triple may be
        # named by multiple classifier emissions (e.g. nivolumab → PD-1
        # surfacing on both the mono-nivo and combo-arm chains). Apply
        # the first matching update and skip the rest so a single trial
        # never delivers 2× the conjugate evidence to the same edge.
        #
        # Phase B of v1 classifier-per-arm: the dedupe key includes
        # arm_id as a 4th slot. Two updates that hit the same
        # (edge_type, src, tgt) under DIFFERENT arm_ids are treated as
        # separate evidence records — each arm's outcome about the same
        # edge is its own signal. Null arm_id (back-compat) collapses
        # all unsubscripted updates to one slot.
        applied_edges: set[tuple[str, str, str, str | None]] = set()

        # Round-16: reset per-call counters. Orchestrator reads
        # last_attempted_updates + last_dropped_updates after each
        # attribute() call to enforce build-level drop thresholds.
        self.last_attempted_updates = len(raw_edges)
        self.last_dropped_updates = 0

        for item in raw_edges:
            edge_type_str = item.get("edge_type", "")
            # Round 3.3 renamed `binds_to` → `affects`. Cached classifier
            # outputs from earlier rounds still emit `binds_to`; accept it
            # as an alias so we don't need to re-classify every trial just
            # for the rename.
            if edge_type_str == "binds_to":
                edge_type_str = "affects"
            try:
                edge_type = EdgeType(edge_type_str)
            except ValueError:
                logger.warning("Unknown edge type '%s', skipping", edge_type_str)
                self.last_dropped_updates += 1
                continue
            if edge_type in (EdgeType.COMPOSED_OF, EdgeType.SUBTYPE_OF):
                # Structural edges aren't classifier-modulable.
                self.last_dropped_updates += 1
                continue

            support_str = item.get("support", "ambiguous")
            try:
                bucket = SupportBucket(support_str)
            except ValueError:
                logger.warning(
                    "Unknown support bucket %r, defaulting to ambiguous", support_str,
                )
                bucket = SupportBucket.AMBIGUOUS

            src_id, tgt_id, route_reason = self._route_to_chain_edge(
                edge_type, trial, arm_by_id, name_index, item
            )
            if not src_id or not tgt_id:
                reason = route_reason or "no_chain_match"
                if reason == "entity_not_in_trial":
                    logger.warning(
                        "Dropping classifier update for trial %s: "
                        "%s %r → %r is not in the trial subgraph "
                        "(possible LLM hallucination of off-trial entity)",
                        trial.trial_id,
                        edge_type_str,
                        item.get("source_entity"),
                        item.get("target_entity"),
                    )
                _log_unrouted(trial.trial_id, item, reason=reason)
                self.last_dropped_updates += 1
                continue

            arm_tag = item.get("affecting_arm_id")
            edge_key = (edge_type_str, src_id, tgt_id, arm_tag)
            if edge_key in applied_edges:
                logger.debug(
                    "Skipping duplicate update for trial %s: %s %s → %s "
                    "(arm=%s)",
                    trial.trial_id, edge_type_str, src_id, tgt_id, arm_tag,
                )
                continue
            applied_edges.add(edge_key)

            # Cross-check with taxonomy rule. If the classifier picked a
            # bucket whose coarse direction disagrees with the taxonomy's
            # expectation for this failure mode, downgrade to AMBIGUOUS —
            # the conjugate update then contributes only neutral pseudocounts.
            ev_direction = bucket_to_direction(bucket)
            if rule:
                if ev_direction == EvidenceDirection.CONTRADICTING and edge_type in rule.edges_to_strengthen:
                    logger.debug(
                        "Classifier says contradict %s but taxonomy says strengthen—using ambiguous",
                        edge_type_str,
                    )
                    bucket = SupportBucket.AMBIGUOUS
                elif ev_direction == EvidenceDirection.SUPPORTING and edge_type in rule.edges_to_weaken:
                    logger.debug(
                        "Classifier says support %s but taxonomy says weaken—using ambiguous",
                        edge_type_str,
                    )
                    bucket = SupportBucket.AMBIGUOUS

            evidence = EvidenceRecord(
                source_id=trial.trial_id,
                source_type=evidence_type,
                support=bucket.value,
                quality_score=min(classification.confidence, 1.0),
                timestamp=datetime.now(timezone.utc),
                notes=item.get("reasoning", ""),
            )

            # Get pre-update belief
            try:
                pre_belief = self.graph.get_edge_belief(src_id, tgt_id, edge_type)
            except KeyError:
                logger.debug(
                    "Edge %s -> %s (%s) not in graph, skipping",
                    src_id, tgt_id, edge_type_str,
                )
                continue

            # Apply update
            post_belief = self.graph.update_edge_belief(
                src_id, tgt_id, edge_type, evidence
            )

            updates.append(AppliedEdgeUpdate(
                source_id=src_id,
                target_id=tgt_id,
                edge_type=edge_type,
                evidence=evidence,
                pre_update_belief=pre_belief,
                post_update_belief=post_belief,
            ))

        # Arm-differential modulation edges (round 8 v0.2.0). For trials
        # with arm pairs where one is a strict subset of another, emit
        # MODULATES_EFFICACY_OF edges between added constituents and
        # backbone constituents. Compound→compound layer only at v0.2.0;
        # v0.2.1 / v0.3.0 will promote to chain-node layers.
        updates.extend(self._emit_arm_differential_modulations(
            trial, classification, evidence_type, applied_edges,
            extraction=extraction,
        ))

        # Single-arm-combo emission. For each combo arm whose compound
        # set has no subset comparator in the trial, fall back to a
        # weak pairwise signal across all C(n, 2) constituent pairs based
        # on the arm's standalone outcome. Weaker than the differential
        # path because we have no head-to-head signal.
        updates.extend(self._emit_single_arm_combo_modulations(
            trial, classification, evidence_type, applied_edges,
            extraction=extraction,
        ))

        # v0.3.0 LLM-anchored modulation edges. The extractor identifies
        # the specific edge in the primary chain that each modulator
        # acts on; this emission routes them into MODULATES_EFFICACY_OF
        # edges anchored at the layer the LLM named (not the compound
        # layer v0.2.0 fixes them to). Unroutable entries land in
        # data/dev/unrouted_modulation_entries.jsonl.
        updates.extend(self._emit_llm_modulations(
            trial, classification, evidence_type, applied_edges,
            extraction=extraction,
        ))

        # Failure-trial backstop. The classifier prompt explicitly requires
        # at least one upstream causal-chain edge update on a failure trial
        # (a missed endpoint refutes part of the chain even when biomarker
        # data is absent), but the LLM occasionally returns zero edges
        # anyway, especially at confidence_overall <0.5. Without a backstop
        # the trial is silent — 0 evidence on every chain, no learning from
        # the failure. Inject a default `biology_drives weak_contradict`
        # against the parent chain so the failure signal lands somewhere.
        outcome = raw.get("trial_outcome")
        if outcome == "failure":
            biology_drives_applied = any(
                et == EdgeType.BIOLOGY_DRIVES.value
                for et, _, _, _ in applied_edges
            )
            if not biology_drives_applied and trial.chains:
                default_update = self._emit_failure_backstop(
                    trial, classification, evidence_type,
                )
                if default_update is not None:
                    updates.append(default_update)

        return updates

    def _emit_failure_backstop(
        self,
        trial: TrialSubgraph,
        classification: FailureClassification,
        evidence_type: EvidenceType,
    ) -> AppliedEdgeUpdate | None:
        """Auto-emit `biology_drives weak_contradict` for a silent failure trial.

        Used only when the classifier returned zero `biology_drives` edges
        on a failure trial — the prompt rule was violated, and the graph
        otherwise gets no signal from the failure. Targets the parent
        chain's (biology_id → indication_id) edge with a deliberately
        low-strength bucket because the classifier didn't independently
        reason about the contradict; we're back-filling the structural
        expectation, not adding new mechanistic information.
        """
        parent_chain = trial.chains[0]
        src_id = parent_chain.biology_id
        tgt_id = parent_chain.indication_id
        try:
            pre = self.graph.get_edge_belief(src_id, tgt_id, EdgeType.BIOLOGY_DRIVES)
        except KeyError:
            logger.debug(
                "Failure backstop skipped for %s: biology_drives %s → %s not in graph",
                trial.trial_id, src_id, tgt_id,
            )
            return None
        evidence = EvidenceRecord(
            source_id=trial.trial_id,
            source_type=evidence_type,
            support=SupportBucket.WEAK_CONTRADICT.value,
            quality_score=min(classification.confidence, 1.0),
            timestamp=datetime.now(timezone.utc),
            notes=(
                "Failure-trial backstop: classifier returned zero "
                "biology_drives edges on a failure outcome, so a default "
                "weak_contradict is emitted on the parent chain."
            ),
        )
        post = self.graph.update_edge_belief(
            src_id, tgt_id, EdgeType.BIOLOGY_DRIVES, evidence,
        )
        logger.warning(
            "Failure-trial backstop fired for %s: classifier emitted 0 "
            "biology_drives on outcome=failure; auto-emitted weak_contradict "
            "on %s → %s",
            trial.trial_id, src_id, tgt_id,
        )
        return AppliedEdgeUpdate(
            source_id=src_id,
            target_id=tgt_id,
            edge_type=EdgeType.BIOLOGY_DRIVES,
            evidence=evidence,
            pre_update_belief=pre,
            post_update_belief=post,
        )

    def _emit_arm_differential_modulations(
        self,
        trial: TrialSubgraph,
        classification: FailureClassification,
        evidence_type: EvidenceType,
        applied_edges: set[tuple[str, str, str, str | None]],
        *,
        extraction: "TrialExtraction | None" = None,
    ) -> list[AppliedEdgeUpdate]:
        """Emit MODULATES_EFFICACY_OF edges from arm-pair differentials.

        For each ordered pair of arms (a, b) where a's compound set is a
        strict subset of b's, the outcome differential is evidence about
        adding the extra constituents to a's backbone. Emits one edge
        per (added constituent × backbone constituent) pair, with the
        endpoints lex-canonicalized so symmetric (compound-compound)
        relations accumulate on a single edge across emissions.

        v0.2.0 emits at the compound→compound layer only; the schema
        supports cross-layer edges (compound → target, mechanism →
        biology, etc.) but layer resolution is deferred to v0.2.1 and
        v0.3.0. See `audit/round_8_architecture_design.md`.
        """
        if len(trial.arms) < 2:
            return []
        outcomes = _aggregate_arm_outcomes(trial, extraction)
        if len(outcomes) < 2:
            return []

        emitted: list[AppliedEdgeUpdate] = []
        for arm_a in trial.arms:
            outcome_a = outcomes.get(arm_a.arm_id)
            if outcome_a is None:
                continue
            a_set = set(arm_a.compound_ids)
            for arm_b in trial.arms:
                if arm_a.arm_id == arm_b.arm_id:
                    continue
                outcome_b = outcomes.get(arm_b.arm_id)
                if outcome_b is None:
                    continue
                b_set = set(arm_b.compound_ids)
                if not a_set < b_set:  # require strict subset
                    continue
                added = b_set - a_set
                bucket = _arm_differential_bucket(outcome_a, outcome_b)
                if bucket is None:
                    continue
                for new_c in sorted(added):
                    for backbone_c in sorted(a_set):
                        src, tgt = canonical_modulation_endpoints(
                            new_c, backbone_c,
                        )
                        # Modulation edges are trial-level (not arm-scoped);
                        # use None for the arm slot of the dedupe key.
                        edge_key = (
                            EdgeType.MODULATES_EFFICACY_OF.value,
                            src, tgt, None,
                        )
                        if edge_key in applied_edges:
                            continue
                        applied_edges.add(edge_key)
                        update = self._upsert_modulation_edge(
                            src=src,
                            tgt=tgt,
                            bucket=bucket,
                            evidence_type=evidence_type,
                            trial=trial,
                            classification=classification,
                            note=(
                                f"Arm differential: {arm_a.arm_id} "
                                f"(outcome={outcome_a.value}) vs "
                                f"{arm_b.arm_id} (outcome={outcome_b.value}); "
                                f"added={sorted(added)}"
                            ),
                        )
                        emitted.append(update)
        return emitted

    def _emit_single_arm_combo_modulations(
        self,
        trial: TrialSubgraph,
        classification: FailureClassification,
        evidence_type: EvidenceType,
        applied_edges: set[tuple[str, str, str, str | None]],
        *,
        extraction: "TrialExtraction | None" = None,
    ) -> list[AppliedEdgeUpdate]:
        """Pairwise weak emission for combo arms with no subset comparator.

        When a trial reports a combo arm's outcome but no monotherapy (or
        smaller-subset) arm to compare against, we can't compute a
        differential — but the combo arm's outcome is still evidence at
        a much weaker level: each pair of constituents accumulates one
        weak update in the direction of the arm's standalone outcome.
        Triggered per-arm: only fires for combo arms (≥2 constituents)
        whose compound set has no strict subset in any other arm.
        """
        outcomes = _aggregate_arm_outcomes(trial, extraction)
        if not outcomes:
            return []
        all_compound_sets = [set(a.compound_ids) for a in trial.arms]

        emitted: list[AppliedEdgeUpdate] = []
        for arm in trial.arms:
            if len(arm.compound_ids) < 2:
                continue
            outcome = outcomes.get(arm.arm_id)
            if outcome is None:
                continue
            bucket = _single_arm_combo_bucket(outcome)
            if bucket is None:
                continue
            arm_set = set(arm.compound_ids)
            has_subset_comparator = any(
                other < arm_set for other in all_compound_sets
            )
            if has_subset_comparator:
                continue
            sorted_compounds = sorted(arm_set)
            for i, c1 in enumerate(sorted_compounds):
                for c2 in sorted_compounds[i + 1:]:
                    src, tgt = canonical_modulation_endpoints(c1, c2)
                    edge_key = (
                        EdgeType.MODULATES_EFFICACY_OF.value, src, tgt, None,
                    )
                    if edge_key in applied_edges:
                        continue
                    applied_edges.add(edge_key)
                    emitted.append(self._upsert_modulation_edge(
                        src=src,
                        tgt=tgt,
                        bucket=bucket,
                        evidence_type=evidence_type,
                        trial=trial,
                        classification=classification,
                        note=(
                            f"Single-arm combo: {arm.arm_id} "
                            f"(outcome={outcome.value}); no subset "
                            f"comparator in trial."
                        ),
                    ))
        return emitted

    def _resolve_compound_id(
        self,
        name: str,
    ) -> str | None:
        """Resolve a compound name to a graph InterventionNode id.

        Three paths in priority order:
          1. Direct id lookup — the LLM often emits the canonical slug.
          2. ``normalize_entity`` slugification — handles capitalized /
             punctuated forms like "Nivolumab" or "CMP-001".
          3. Separator-insensitive match — the LLM sometimes drops
             hyphens/underscores that the graph kept ("cmp001" vs
             "cmp_001"). Compare alnum-only-lowercased forms across
             all Intervention/Compound nodes.

        Returns None on no match; the caller logs to the unrouted log.
        """
        if not name:
            return None
        try:
            node = self.graph.get_node(name)
            if node.get("node_type") in ("InterventionNode", "CompoundNode"):
                return name
        except KeyError:
            pass
        try:
            slug = normalize_entity(name, "InterventionNode")
        except ValueError:
            slug = None
        if slug:
            try:
                node = self.graph.get_node(slug)
                if node.get("node_type") in ("InterventionNode", "CompoundNode"):
                    return slug
            except KeyError:
                pass
        # Separator-insensitive fallback.
        target_norm = _norm_name(name)
        if not target_norm:
            return None
        for node_id, data in self.graph._graph.nodes(data=True):  # noqa: SLF001
            if data.get("node_type") not in ("InterventionNode", "CompoundNode"):
                continue
            if _norm_name(node_id) == target_norm:
                return node_id
        return None

    def _find_primary_chain(
        self,
        trial: TrialSubgraph,
        primary_compound_id: str,
    ) -> CausalChain | None:
        """Find a chain in the trial whose compound_id matches the LLM's
        primary_compound_id. Returns the first matching chain (chains for
        the same compound share the upstream backbone, so target /
        mechanism / biology are equivalent across them).
        """
        for chain in trial.chains:
            if chain.compound_id == primary_compound_id:
                return chain
        # Try to match via arm membership — combo regimens encode the
        # compound list separately, and the chain's compound_id is the
        # regimen slug (e.g. "ipilimumab+nivolumab") not a single
        # constituent. The LLM emits constituent compound ids, so check
        # the arm's compound list.
        for chain in trial.chains:
            for arm in trial.arms:
                if (
                    chain.arm_id == arm.arm_id
                    and primary_compound_id in arm.compound_ids
                ):
                    return chain
        return None

    def _emit_llm_modulations(
        self,
        trial: TrialSubgraph,
        classification: FailureClassification,
        evidence_type: EvidenceType,
        applied_edges: set[tuple[str, str, str, str | None]],
        *,
        extraction: "TrialExtraction | None" = None,
    ) -> list[AppliedEdgeUpdate]:
        """Emit v0.3.0 LLM-anchored MODULATES_EFFICACY_OF edges.

        For each ``ModulationEntry`` in the extraction:
          1. Resolve modulator + primary compound names to graph node ids.
          2. Find the primary's chain in this trial.
          3. Pull the chain node at the LLM-specified layer
             (``chain.target_id`` / ``mechanism_id`` / ``biology_id``).
          4. Emit a MODULATES_EFFICACY_OF edge from modulator → that
             node, with the layer + direction + confidence in the
             evidence context for downstream prediction.

        The LLM doesn't name internal node ids — those are populator
        outputs the LLM can't know at extraction time. The LLM names
        canonical things (compound names, layer names); the populator
        does the chain walk.

        Unroutable entries go to ``data/dev/unrouted_modulation_entries.jsonl``
        with a ``reason`` field. A high volume there means the prompt
        is producing modulator/primary names that aren't in the trial,
        which is a real debug signal.
        """
        if extraction is None or not extraction.modulation_entries:
            return []

        emitted: list[AppliedEdgeUpdate] = []
        for entry in extraction.modulation_entries:
            modulator_id = self._resolve_compound_id(entry.modulator_compound_id)
            if modulator_id is None:
                _log_unrouted_modulation(
                    trial.trial_id, entry, reason="modulator_not_in_graph",
                )
                continue

            primary_id = self._resolve_compound_id(entry.primary_compound_id)
            if primary_id is None:
                _log_unrouted_modulation(
                    trial.trial_id, entry, reason="primary_not_in_graph",
                )
                continue

            chain = self._find_primary_chain(trial, primary_id)
            if chain is None:
                _log_unrouted_modulation(
                    trial.trial_id, entry, reason="primary_chain_not_in_trial",
                )
                continue

            anchor_id: str | None
            if entry.affects_layer == "target":
                anchor_id = chain.target_id
            elif entry.affects_layer == "mechanism":
                anchor_id = chain.mechanism_id
            elif entry.affects_layer == "biology":
                anchor_id = chain.biology_id
            else:
                _log_unrouted_modulation(
                    trial.trial_id, entry,
                    reason=f"unknown_affects_layer:{entry.affects_layer}",
                )
                continue

            if not anchor_id or anchor_id == _UNKNOWN_PLACEHOLDER:
                _log_unrouted_modulation(
                    trial.trial_id, entry,
                    reason=f"primary_chain_layer_unresolved:{entry.affects_layer}",
                )
                continue

            bucket = modulation_bucket(entry.direction, entry.confidence)

            edge_key = (
                EdgeType.MODULATES_EFFICACY_OF.value,
                modulator_id, anchor_id, None,
            )
            if edge_key in applied_edges:
                continue
            applied_edges.add(edge_key)

            update = self._upsert_modulation_edge(
                src=modulator_id,
                tgt=anchor_id,
                bucket=bucket,
                evidence_type=evidence_type,
                trial=trial,
                classification=classification,
                note=(
                    f"LLM modulation: {entry.modulator_compound_id} "
                    f"{entry.direction} primary={entry.primary_compound_id} "
                    f"at {entry.affects_layer} layer "
                    f"(anchor={anchor_id}); conf={entry.confidence:.2f}"
                ),
                source_label="llm_modulation_entry",
                evidence_context_extras={
                    "primary_compound": primary_id,
                    "affects_layer": entry.affects_layer,
                    "modulation_direction": entry.direction,
                    "modulation_confidence": entry.confidence,
                    "hypothesis": entry.hypothesis,
                    "citation": entry.citation,
                },
            )
            emitted.append(update)
        return emitted

    def _upsert_modulation_edge(
        self,
        *,
        src: str,
        tgt: str,
        bucket: SupportBucket,
        evidence_type: EvidenceType,
        trial: TrialSubgraph,
        classification: FailureClassification,
        note: str,
        source_label: str = "arm_differential",
        evidence_context_extras: dict[str, Any] | None = None,
    ) -> AppliedEdgeUpdate:
        """Create the modulation edge with a neutral prior if absent, then
        apply one evidence record. Returns the AppliedEdgeUpdate.

        Every modulation evidence record carries ``context["indication"]``
        so a future v0.2.x refinement can apply Path-2-style conditioning
        on this edge type (sample beliefs only under the queried
        indication, downweight off-context records). v0.2.0 doesn't read
        the context yet — it's tagged forward-compatibly.

        ``source_label`` flags how the edge was first created
        ("arm_differential" for v0.2.0 attributor emissions,
        "llm_modulation_entry" for v0.3.0 LLM emissions). Edge metadata
        is only set at creation; subsequent evidence records carry their
        own per-trial provenance via ``context``.

        ``evidence_context_extras`` extends the per-record context dict —
        used by LLM modulations to record the specific affects_edge
        triple they were claiming about, alongside direction/confidence.
        """
        if not self.graph._graph.has_edge(  # noqa: SLF001
            src, tgt, key=EdgeType.MODULATES_EFFICACY_OF.value,
        ):
            self.graph.add_edge(GraphEdge(
                source_id=src,
                target_id=tgt,
                edge_type=EdgeType.MODULATES_EFFICACY_OF,
                belief=EdgeBeliefState(alpha=1.0, beta=1.0),
                metadata={"source": source_label},
            ))
        pre_belief = self.graph.get_edge_belief(
            src, tgt, EdgeType.MODULATES_EFFICACY_OF,
        )
        indication_id = trial.chains[0].indication_id if trial.chains else None
        evidence_context: dict[str, Any] = {}
        if indication_id and indication_id != _UNKNOWN_PLACEHOLDER:
            evidence_context["indication"] = indication_id
        if evidence_context_extras:
            evidence_context.update(evidence_context_extras)
        evidence = EvidenceRecord(
            source_id=trial.trial_id,
            source_type=evidence_type,
            support=bucket.value,
            quality_score=min(classification.confidence, 1.0),
            timestamp=datetime.now(timezone.utc),
            notes=note,
            context=evidence_context,
        )
        post_belief = self.graph.update_edge_belief(
            src, tgt, EdgeType.MODULATES_EFFICACY_OF, evidence,
        )
        return AppliedEdgeUpdate(
            source_id=src,
            target_id=tgt,
            edge_type=EdgeType.MODULATES_EFFICACY_OF,
            evidence=evidence,
            pre_update_belief=pre_belief,
            post_update_belief=post_belief,
        )

    async def attribute_adverse_events(
        self,
        trial: TrialSubgraph,
        extraction: TrialExtraction,
        client: Any,  # anthropic.AsyncAnthropic—kept loose to avoid import cost in attributor
        meddra_cache: MeddraCache | None = None,
    ) -> list[AppliedEdgeUpdate]:
        """Update causes_ae edges from a trial's structured adverse events.

        For each AE the extractor pulled from the trial:
          1. Normalize the term to a MedDRA preferred term (cached LLM call).
          2. Ensure an AdverseEventNode exists in the graph (create on miss).
          3. Choose a SupportBucket from the per-arm incidence rates.
          4. Update causes_ae from each treatment-arm compound to the AE.

        Skips arms that look like placebo/sham/vehicle so the comparator's
        background-rate mentions don't accumulate causes_ae evidence on
        the wrong compound.
        """
        cache = meddra_cache or MeddraCache()
        evidence_type = _PHASE_TO_EVIDENCE.get(trial.phase, EvidenceType.LITERATURE)
        treatment_compound_ids = self._treatment_compound_ids(trial)
        if not treatment_compound_ids:
            return []

        # Estimate per-arm patient count from total enrollment / arm
        # count. Trials rarely report exact per-arm n in CT.gov, but
        # the trial-level enrollment IS reliable. Used by
        # _ae_support_bucket to gate small-count AE signals (fixes.md
        # #9 — a 1.2% rate in an n=85 arm is ~1 patient, which the
        # rate-only path would still label moderate_support).
        treatment_n: int | None = None
        total_n = trial.metadata.get("enrollment") if trial.metadata else None
        if isinstance(total_n, int) and total_n > 0 and trial.arms:
            treatment_n = max(1, total_n // len(trial.arms))

        updates: list[AppliedEdgeUpdate] = []
        for ae in extraction.adverse_events:
            normalized = await normalize_ae_term(client, ae.term, cache)
            if normalized is None:
                # Meta / summary row (e.g. "Grade 3-5 adverse events") —
                # no single clinical concept to attribute. Skip rather
                # than collapse onto the meaningless "Unspecified" node.
                logger.debug(
                    "Skipping meta AE term %r for trial %s",
                    ae.term, trial.trial_id,
                )
                continue
            preferred_term = normalized["preferred_term"]
            soc = normalized.get("system_organ_class", "")
            ae_id = ae_node_id(preferred_term)
            self._ensure_ae_node(ae_id, preferred_term, soc, ae.grade)

            for compound_id in treatment_compound_ids:
                # When arm_incidences is populated (CT.gov-structured path),
                # compute per-compound tx/ctrl by partitioning arms on whether
                # this compound was active. Otherwise fall back to the flat
                # tx/ctrl pair from LLM-extracted narrative-only trials.
                tx_pct, ctrl_pct, tx_n = (None, None, None)
                if ae.arm_incidences:
                    compound_name = self._compound_display_name(compound_id)
                    tx_pct, ctrl_pct, tx_n = _per_compound_rates_from_arms(
                        ae.arm_incidences, compound_name,
                    )
                if tx_pct is None:
                    tx_pct = ae.incidence_treatment_pct
                    ctrl_pct = ae.incidence_control_pct
                    tx_n = treatment_n

                bucket = _ae_support_bucket(
                    tx_pct, ctrl_pct, treatment_n=tx_n,
                )
                note = _format_ae_note(
                    ae, preferred_term, tx_pct=tx_pct, ctrl_pct=ctrl_pct,
                )

                self._ensure_causes_ae_edge(compound_id, ae_id)
                evidence = EvidenceRecord(
                    source_id=trial.trial_id,
                    source_type=evidence_type,
                    support=bucket.value,
                    quality_score=1.0,  # incidence-rate evidence is structured, not LLM-judgment
                    timestamp=datetime.now(timezone.utc),
                    notes=note,
                    context={"ae_term_raw": ae.term, "ae_grade": ae.grade},
                )
                pre = self.graph.get_edge_belief(
                    compound_id, ae_id, EdgeType.CAUSES_AE
                )
                post = self.graph.update_edge_belief(
                    compound_id, ae_id, EdgeType.CAUSES_AE, evidence
                )
                updates.append(AppliedEdgeUpdate(
                    source_id=compound_id,
                    target_id=ae_id,
                    edge_type=EdgeType.CAUSES_AE,
                    evidence=evidence,
                    pre_update_belief=pre,
                    post_update_belief=post,
                ))

        # After all causes_ae updates land, refresh target_associated_ae for
        # each touched (compound, AE) pair. Propagation is idempotent —
        # running it once per touched pair is enough even when the same
        # AE was attributed to multiple compounds in this trial.
        touched_pairs = {(u.source_id, u.target_id) for u in updates}
        for compound_id, ae_id in touched_pairs:
            propagate_to_target_associated_ae(self.graph, compound_id, ae_id)

        return updates

    def _compound_display_name(self, compound_id: str) -> str:
        """Return the CompoundNode's display name, or '' if missing.

        Used to match against CT.gov eventGroup titles (e.g. compound
        name 'Nivolumab' against descriptor 'Nivolumab + Ipilimumab')
        in ``_per_compound_rates_from_arms``.
        """
        try:
            node = self.graph.get_node(compound_id)
        except KeyError:
            return ""
        return str(node.get("name") or "")

    def _treatment_compound_ids(self, trial: TrialSubgraph) -> list[str]:
        """Compound ids on any non-placebo arm of the trial.

        Returns the *constituent* ids (not the synthesized regimen id for
        combos) so causes_ae attribution accumulates on the same Compound
        node a different trial would attribute to.
        """
        seen: set[str] = set()
        out: list[str] = []
        for arm in trial.arms:
            for cid in arm.compound_ids:
                if cid in seen:
                    continue
                try:
                    node = self.graph.get_node(cid)
                except KeyError:
                    continue
                name = (node.get("name") or "").lower()
                if any(token in name for token in ("placebo", "sham", "vehicle")):
                    continue
                seen.add(cid)
                out.append(cid)
        return out

    def _ensure_ae_node(
        self, ae_id: str, preferred_term: str, soc: str, grade: str
    ) -> None:
        try:
            existing = self.graph.get_node(ae_id)
        except KeyError:
            self.graph.add_node(AdverseEventNode(
                id=ae_id,
                name=preferred_term,
                system_organ_class=soc,
                severity_range=grade or "",
            ))
            return
        # Node exists—extend severity_range if this AE reported a new grade
        # we haven't seen for this term before. SOC is locked in on first
        # write; the normalizer should be deterministic for the same input.
        if grade and grade not in (existing.get("severity_range") or ""):
            existing_range = existing.get("severity_range") or ""
            merged = f"{existing_range},{grade}".strip(",")
            self.graph._graph.nodes[ae_id]["severity_range"] = merged

    def _ensure_causes_ae_edge(self, compound_id: str, ae_id: str) -> None:
        try:
            self.graph.get_edge_belief(
                compound_id, ae_id, EdgeType.CAUSES_AE
            )
        except KeyError:
            self.graph.add_edge(GraphEdge(
                source_id=compound_id,
                target_id=ae_id,
                edge_type=EdgeType.CAUSES_AE,
                belief=EdgeBeliefState(),
            ))

    def apply_updates(
        self, updates: list[AppliedEdgeUpdate]
    ) -> dict[str, Any]:
        """Summarize applied updates (already applied during attribute())."""
        if not updates:
            return {"edges_updated": 0, "largest_changes": []}

        sorted_updates = sorted(
            updates, key=lambda u: abs(u.probability_change), reverse=True
        )

        largest = [
            {
                "edge": f"{u.source_id} -> {u.target_id} ({u.edge_type.value})",
                "pre": round(u.pre_update_belief.expected_probability, 4),
                "post": round(u.post_update_belief.expected_probability, 4),
                "change": round(u.probability_change, 4),
            }
            for u in sorted_updates[:5]
        ]

        return {
            "edges_updated": len(updates),
            "largest_changes": largest,
        }

    def _route_to_chain_edge(
        self,
        edge_type: EdgeType,
        trial: TrialSubgraph,
        arm_by_id: dict[str, TrialArm],
        name_index: "_NameIndex",
        item: dict[str, Any],
    ) -> tuple[str | None, str | None, str | None]:
        """Pick the chain whose canonical ids best match the classifier's
        free-text source/target entity names, then return that chain's
        ``(source_id, target_id, reason)`` for the given edge type.

        Returns ``reason=None`` on success. On failure, ids are ``None``
        and reason is one of:
          - ``"no_chain_match"``—no chain in the trial subgraph has
            non-UNKNOWN candidates for this edge type. Means the trial
            is too sparsely populated to verify the update.
          - ``"entity_not_in_trial"``—candidate pairs exist, but the
            classifier's named source/target match nothing among them.
            This is the hallucination guard: graph-build trusts only
            entities derivable from the trial subgraph, so if the
            classifier invents an off-trial entity the update is
            dropped rather than misrouted.
          - ``"unknown_arm_id:<arm>"``—v1 of classifier-per-arm
            emission: the classifier tagged this update with an
            arm_id that isn't in the trial. The LLM hallucinated.

        Open Targets-seeded biology_drives gets a special pass-through:
        OT writes a single (target_id, indication_id) edge during populate
        and the trial-time biology id may be a different node, so we
        update the OT-keyed edge instead.

        ``affecting_arm_id`` on the item, when non-null, restricts the
        chain iteration to chains whose ``arm_id`` matches. This is the
        v1 fix for multi-arm trials where a constituent appears in
        several arms — the LLM's per-arm tag picks the right chain
        instead of entity-name matching to whichever chain came first.
        """
        if edge_type == EdgeType.BIOLOGY_DRIVES:
            ot_coords = trial.metadata.get("ot_biology_drives")
            if ot_coords:
                return ot_coords.get("source_id"), ot_coords.get("target_id"), None

        src_name = (item.get("source_entity") or "").strip()
        tgt_name = (item.get("target_entity") or "").strip()
        affecting_arm_id = item.get("affecting_arm_id")

        # Phase B: if the classifier tagged this update with an arm_id,
        # restrict chain iteration to that arm's chains. Falls back to
        # all-chains entity-name matching when the tag is null (back-
        # compat for cached classifications written pre-v1).
        #
        # The LLM tends to emit natural arm slugs ("ipilimumab_alone",
        # "cmp001_plus_nivo") that don't exact-match the populator's
        # long-form slugs ("arm_a_ipilimumab", "nivolumab_and_cmp_001_
        # combination"). Same "LLM doesn't know graph-internal ids"
        # problem we hit with v0.3 modulation. Two-stage match:
        #   1. Exact arm_id match.
        #   2. Normalized substring match: when exactly one graph arm
        #      contains (or is contained by) the LLM's slug after
        #      alnum-lowercasing, use it.
        # Only unroute as unknown_arm_id when neither pass resolves.
        if affecting_arm_id:
            resolved_arm_id = _resolve_arm_id(
                affecting_arm_id, arm_by_id,
            )
            if resolved_arm_id is None:
                return None, None, f"unknown_arm_id:{affecting_arm_id}"
            candidate_chains = [
                c for c in trial.chains if c.arm_id == resolved_arm_id
            ]
        else:
            candidate_chains = list(trial.chains)

        best: tuple[str, str] | None = None
        best_score = -1
        any_candidate = False
        any_explicit_mismatch = False
        for chain in candidate_chains:
            arm = arm_by_id.get(chain.arm_id)
            if arm is None:
                continue
            for src_id, tgt_id in _chain_edges_for_type(chain, arm, edge_type):
                if src_id == _UNKNOWN_PLACEHOLDER or tgt_id == _UNKNOWN_PLACEHOLDER:
                    continue
                any_candidate = True
                score, mismatched = _score_pair_against_names(
                    src_id, tgt_id, src_name, tgt_name, name_index,
                )
                if mismatched:
                    any_explicit_mismatch = True
                if score > best_score:
                    best_score = score
                    best = (src_id, tgt_id)

        if best is not None and best_score >= 1:
            return best[0], best[1], None

        if any_candidate and any_explicit_mismatch:
            return None, None, "entity_not_in_trial"
        return None, None, "no_chain_match"


# ── CLI ──────────────────────────────────────────────────────────────────


def _load_classifications(annotations_dir: Path) -> list[tuple[dict, dict]]:
    """Load paired extraction + classification JSONs."""
    pairs = []
    for clf_path in sorted(annotations_dir.glob("*_classification.json")):
        nct_id = clf_path.stem.replace("_classification", "")
        ext_path = annotations_dir / f"{nct_id}_extraction.json"
        if ext_path.exists():
            clf_data = json.loads(clf_path.read_text())
            ext_data = json.loads(ext_path.read_text())
            pairs.append((ext_data, clf_data))
    return pairs


async def _main(annotations_dir: str, graph_path: str, output_path: str) -> None:
    import anthropic
    from rich.console import Console

    from src.annotation.meddra import MeddraCache

    console = Console()

    # Load graph
    graph = GraphStore()
    graph_file = Path(graph_path)
    if graph_file.exists():
        console.print(f"[bold]Loading graph from {graph_path}...[/bold]")
        graph.import_snapshot(graph_path)
        stats = graph.stats()
        console.print(f"  Loaded: {stats['node_count']} nodes, {stats['edge_count']} edges")
    else:
        console.print(f"[yellow]Graph file not found: {graph_path}[/yellow]")
        return

    attributor = Attributor(graph)
    pairs = _load_classifications(Path(annotations_dir))
    console.print(f"\n[bold]Found {len(pairs)} annotated trials[/bold]")

    # Shared resources for AE attribution: one Anthropic client + one
    # MeddraCache reused across the whole batch so repeat AE terms hit
    # the cache rather than the LLM.
    client = anthropic.AsyncAnthropic(timeout=60.0)
    meddra_cache = MeddraCache()

    total_updates: list[AppliedEdgeUpdate] = []

    for ext_data, clf_data in pairs:
        trial_id = clf_data.get("nct_id", ext_data.get("nct_id", "unknown"))

        # Round-19: idempotency guard. Trials whose attribution has
        # already been folded into this snapshot must not have their
        # Beta-Binomial updates re-applied — that would double-count
        # evidence on every edge the trial touched. The set is added
        # to AFTER the trial's updates land successfully (transactional),
        # so a partial / failed attribution leaves the trial retryable.
        if trial_id in graph.applied_attribution_trial_ids:
            console.print(
                f"  [dim]Skipped {trial_id}:[/dim] already attributed in this snapshot"
            )
            continue

        # The trial subgraph (with arms + chains) must already exist in the
        # graph sidecar—produced by populate.build_trial_subgraphs and
        # extended by add_subgroup_chains during the extraction pipeline.
        try:
            trial = graph.get_trial_subgraph_by_id(trial_id)
        except KeyError:
            console.print(
                f"  [yellow]Skipped {trial_id}:[/yellow] no trial_subgraph in sidecar"
            )
            continue

        # Build classification
        modes = clf_data.get("failure_modes", [])
        primary_mode = FailureMode.INSUFFICIENT_INFORMATION
        if modes:
            sorted_modes = sorted(modes, key=lambda m: m.get("confidence", 0), reverse=True)
            try:
                primary_mode = FailureMode(sorted_modes[0]["mode"])
            except (ValueError, KeyError):
                pass

        classification = FailureClassification(
            trial_id=trial_id,
            primary_failure_mode=primary_mode,
            confidence=clf_data.get("confidence_overall", 0.5),
            reasoning=clf_data.get("reasoning", ""),
        )
        classification._raw = clf_data  # type: ignore[attr-defined]

        # Parse extraction once — passed to attribute() so modulation
        # emission can read per-arm outcomes, then reused for AE
        # attribution below.
        from src.annotation.extractor import _parse_extraction_response
        try:
            extraction = _parse_extraction_response(ext_data, trial_id)
        except Exception as exc:  # noqa: BLE001—pydantic ValidationError + others
            logger.warning(
                "Extraction JSON invalid for %s (%s); modulation emission "
                "and AE attribution will be skipped",
                trial_id, exc,
            )
            extraction = None

        updates = attributor.attribute(classification, trial, extraction)
        total_updates.extend(updates)

        if extraction is not None and extraction.adverse_events:
            ae_updates = await attributor.attribute_adverse_events(
                trial, extraction, client=client, meddra_cache=meddra_cache,
            )
            total_updates.extend(ae_updates)

        # Round-19: record successful attribution. Placed after both
        # efficacy + AE phases land so a mid-trial raise leaves the
        # set unchanged and the trial retryable on the next run.
        graph.applied_attribution_trial_ids.add(trial_id)

        if updates:
            console.print(f"  {trial_id}: {len(updates)} efficacy edge updates")

    # Summary
    summary = attributor.apply_updates(total_updates)
    console.print(f"\n[bold green]Processed {len(pairs)} trials. Updated {summary['edges_updated']} edges.[/bold green]")

    if summary["largest_changes"]:
        console.print("[bold]Largest changes:[/bold]")
        for change in summary["largest_changes"]:
            direction = "+" if change["change"] > 0 else ""
            console.print(
                f"  {change['edge']}: {change['pre']:.4f} → {change['post']:.4f} "
                f"({direction}{change['change']:.4f})"
            )

    # Save updated graph
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    graph.export_snapshot(output_path)
    console.print(f"\n[bold]Saved annotated graph to {output_path}[/bold]")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Apply failure attributions to knowledge graph")
    parser.add_argument("--input", default="data/annotations/", help="Annotations directory")
    parser.add_argument("--graph", default="data/exports/oncology_initial.json", help="Input graph snapshot")
    parser.add_argument("--output", default="data/exports/oncology_annotated.json", help="Output graph snapshot")
    args = parser.parse_args()

    asyncio.run(_main(args.input, args.graph, args.output))
