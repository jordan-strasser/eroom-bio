"""Cross-trial propagation: causes_ae → target_associated_ae.

When the same adverse event has been attributed to multiple compounds
that all bind the same target, that's evidence the AE is mechanism-related
(target-associated) rather than chemistry-specific to one molecule.
This module rebuilds ``target_associated_ae`` beliefs by aggregating the
``causes_ae`` beliefs of every compound binding the target.

Round-28 extension — SOC-tier roll-up. Sibling-compound `causes_ae`
extractions often land at DISJOINT PT-level terms even when they all
describe the same class of toxicity (e.g. CETP siblings reporting
atrial_fibrillation / myocardial_infarction / bradycardia individually
but no shared PT across the trio). PT-only propagation never fires the
"≥ 2 siblings share an AE" gate. The SOC tier aggregates each sibling's
causes_ae beliefs by their AE node's MedDRA SOC parent, so the gate
clears at the SOC level even when no two siblings share a PT.

Idempotent by design—the target_associated_ae belief is rebuilt from
scratch on every call rather than incrementally updated, so re-running
after every causes_ae attribution doesn't double-count.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from pydantic import BaseModel

from src.graph.models import (
    AdverseEventNode,
    EdgeBeliefState,
    EdgeType,
    EvidenceRecord,
    EvidenceType,
    GraphEdge,
)
from src.graph.store import GraphStore
from src.inference.beliefs import (
    SupportBucket,
    apply_virtual_evidence,
    p_obs_for_bucket,
)

logger = logging.getLogger(__name__)


# Round-28 SOC-tier AE node ids use this prefix so they're
# unambiguously distinct from PT-level AE nodes
# (``AE:atrial_fibrillation`` vs ``AE:soc:cardiac_disorders``).
SOC_AE_PREFIX = "AE:soc:"


def soc_ae_node_id(soc_id: str) -> str:
    """Compose the canonical SOC-tier AdverseEventNode id."""
    if not soc_id:
        return ""
    return f"{SOC_AE_PREFIX}{soc_id}"


def _union_grade_tokens(severity_ranges: list[str]) -> str:
    """Merge severity_range strings into a single comma-separated union.

    Each input is the PT-tier `severity_range` of a contributing sibling
    AE node — itself already a comma-separated list of grade tokens
    accumulated across trials (e.g. ``"1,2,3-5"``). The union preserves
    the existing PT-tier wire format so the existing
    ``_max_grade_from_severity_range`` parser in
    ``src/prediction/path_query.py`` can consume the result without
    changes.

    Tokens are de-duplicated. Order is stable across calls (first-seen
    wins) so the propagation stays idempotent. Empty inputs and empty
    tokens are skipped; the return is ``""`` iff every contributing PT
    had an empty ``severity_range``.

    Ranges like ``"3-5"`` are passed through as-is — the downstream
    parser already understands range tokens and picks the upper bound.
    """
    seen: list[str] = []
    seen_set: set[str] = set()
    for s in severity_ranges:
        if not s:
            continue
        for tok in s.split(","):
            tok = tok.strip()
            if not tok or tok in seen_set:
                continue
            seen.append(tok)
            seen_set.add(tok)
    return ",".join(seen)


# Each compound binding a target contributes one vote about the target's
# AE liability. The target-class hypothesis is one step removed from any
# single trial, so each compound's vote carries less weight than a single
# Phase-3 trial—anchor at GWAS-tier (N_eff = 4) by default.
_VOTE_N_EFF_PER_COMPOUND = 4.0
# Compound→AE beliefs with very little underlying evidence don't get a
# vote. Threshold = 1.0 of accumulated pseudo-counts (alpha+beta-2 ≥ 1)
# excludes pristine Beta(1,1) edges from contributing.
_MIN_EVIDENCE_STRENGTH_FOR_VOTE = 1.0
# Need at least this many compounds with meaningful evidence for the
# target_associated_ae hypothesis to be worth materializing as an edge.
_MIN_COMPOUNDS_FOR_TARGET_AE = 2


class AppliedTargetAEUpdate(BaseModel):
    """One target_associated_ae belief that was rebuilt by propagation."""

    target_id: str
    ae_id: str
    pre_update_belief: EdgeBeliefState
    post_update_belief: EdgeBeliefState
    contributing_compound_ids: list[str]


def propagate_to_target_associated_ae(
    graph: GraphStore,
    compound_id: str,
    ae_id: str,
    *,
    roll_up_tier: str = "soc",
) -> list[AppliedTargetAEUpdate]:
    """Refresh target_associated_ae beliefs implicated by a causes_ae update.

    For every target the compound binds_to, gather the causes_ae beliefs
    of all compounds binding that target for the given AE. If at least
    ``_MIN_COMPOUNDS_FOR_TARGET_AE`` of them carry meaningful evidence,
    rebuild the PT-tier target_associated_ae belief.

    Round-28: when ``roll_up_tier == "soc"`` (default), ALSO aggregate
    sibling causes_ae beliefs at the MedDRA SOC parent of ``ae_id`` and
    emit a ``target_associated_ae → AE:soc:<soc_id>`` edge when ≥
    ``_MIN_COMPOUNDS_FOR_TARGET_AE`` siblings have any AE under that SOC.
    Pass ``roll_up_tier=""`` to disable SOC-tier propagation (PT-only
    behavior, matching pre-round-28).
    """
    updates: list[AppliedTargetAEUpdate] = []

    # PT-tier propagation (unchanged behavior).
    targets = _binds_to_targets(graph, compound_id)
    for target_id in targets:
        sibling_compound_ids = _compounds_binding_target(graph, target_id)
        contributing: list[tuple[str, EdgeBeliefState]] = []
        for sib_id in sibling_compound_ids:
            try:
                belief = graph.get_edge_belief(
                    sib_id, ae_id, EdgeType.CAUSES_AE
                )
            except KeyError:
                continue
            if belief.evidence_strength < _MIN_EVIDENCE_STRENGTH_FOR_VOTE:
                continue
            contributing.append((sib_id, belief))

        if len(contributing) < _MIN_COMPOUNDS_FOR_TARGET_AE:
            continue

        update = _rebuild_target_ae_edge(graph, target_id, ae_id, contributing)
        updates.append(update)

    # Round-28 SOC-tier propagation. Run on the same targets and use the
    # MedDRA hierarchy parent of ``ae_id`` as the aggregation key, so
    # sibling compounds whose causes_ae beliefs sit on different PT nodes
    # still aggregate at the SOC parent.
    if roll_up_tier == "soc":
        soc_updates = _propagate_at_soc_tier(graph, compound_id, ae_id)
        updates.extend(soc_updates)

    return updates


def _propagate_at_soc_tier(
    graph: GraphStore,
    compound_id: str,
    ae_id: str,
) -> list[AppliedTargetAEUpdate]:
    """SOC-tier propagation. Returns updates against ``AE:soc:<soc_id>`` nodes."""
    # Resolve the SOC parent of the triggering AE node.
    try:
        ae_node = graph.get_node(ae_id)
    except KeyError:
        return []
    soc_id = (ae_node.get("soc_id") or "").strip()
    if not soc_id:
        return []
    soc_name = (ae_node.get("soc_name") or "").strip()
    soc_ae_id = soc_ae_node_id(soc_id)

    _ensure_soc_ae_node(graph, soc_ae_id, soc_id, soc_name)

    updates: list[AppliedTargetAEUpdate] = []
    for target_id in _binds_to_targets(graph, compound_id):
        contributing = _collect_soc_votes(graph, target_id, soc_id)
        if len(contributing) < _MIN_COMPOUNDS_FOR_TARGET_AE:
            continue
        # Round-29: aggregate severity_range AND the `serious` flag
        # across the contributing sibling PT-level AE nodes — both
        # TARGET-SCOPED so different targets routing through this SOC
        # carry their own severity. Stored on the target_associated_ae
        # edge so the shared SOC AE node doesn't have to choose one
        # target's severity over another. `serious=True` aggregated via
        # OR acts as a coarse severity floor downstream when grade data
        # is missing (CT.gov rarely posts grades but routinely flags
        # SAEs).
        pt_severities, any_serious = _collect_pt_severities_for_soc(
            graph, target_id, soc_id,
        )
        aggregated_severity = _union_grade_tokens(pt_severities)
        update = _rebuild_target_ae_edge(
            graph, target_id, soc_ae_id, contributing,
            severity_range=aggregated_severity,
            serious=any_serious,
        )
        updates.append(update)
    return updates


def _ensure_soc_ae_node(
    graph: GraphStore,
    soc_ae_id: str,
    soc_id: str,
    soc_name: str,
) -> None:
    """Create the SOC-tier AE node if missing.

    The node carries identity / hierarchy metadata only — severity is
    NOT stored on the node, because the same SOC AE node is referenced
    by ``target_associated_ae`` edges from MANY different targets and
    each target's contributing PT severities are different. Storing
    severity on the shared node would leak grade data from one target
    class to another (e.g. MEK-inhibitor cardiotoxicity grades would
    inflate CETP-inhibitor cardiotoxicity weight).

    Round-29 stores aggregated severity_range on the individual
    ``target_associated_ae`` EDGE instead — see
    ``_rebuild_target_ae_edge`` and the ``severity_range`` edge
    metadata field. ``_compute_safety_penalty`` reads from the edge
    when present, falling back to the node otherwise.
    """
    try:
        graph.get_node(soc_ae_id)
    except KeyError:
        graph.add_node(AdverseEventNode(
            id=soc_ae_id,
            name=soc_name or soc_id,
            system_organ_class=soc_name,
            soc_id=soc_id,
            soc_name=soc_name,
            metadata={"tier": "soc"},
        ))


def _collect_soc_votes(
    graph: GraphStore,
    target_id: str,
    soc_id: str,
) -> list[tuple[str, EdgeBeliefState]]:
    """For each sibling binding ``target_id``, return its strongest
    causes_ae belief whose target AE node rolls up to ``soc_id``.

    Returns one (compound_id, belief) tuple per CONTRIBUTING sibling.
    Siblings with no qualifying AE are omitted entirely (not present
    in the result), which is what the ≥_MIN_COMPOUNDS_FOR_TARGET_AE
    gate checks against.
    """
    sibling_compound_ids = _compounds_binding_target(graph, target_id)
    g = graph._graph
    contributing: list[tuple[str, EdgeBeliefState]] = []
    for sib_id in sibling_compound_ids:
        if sib_id not in g:
            continue
        strongest: EdgeBeliefState | None = None
        for _src, candidate_ae_id, key in g.out_edges(sib_id, keys=True):
            if key != EdgeType.CAUSES_AE.value:
                continue
            try:
                ae_node = graph.get_node(candidate_ae_id)
            except KeyError:
                continue
            if (ae_node.get("soc_id") or "") != soc_id:
                continue
            try:
                belief = graph.get_edge_belief(
                    sib_id, candidate_ae_id, EdgeType.CAUSES_AE,
                )
            except KeyError:
                continue
            if belief.evidence_strength < _MIN_EVIDENCE_STRENGTH_FOR_VOTE:
                continue
            if (
                strongest is None
                or belief.expected_probability > strongest.expected_probability
            ):
                strongest = belief
        if strongest is not None:
            contributing.append((sib_id, strongest))
    return contributing


def _collect_pt_severities_for_soc(
    graph: GraphStore,
    target_id: str,
    soc_id: str,
) -> tuple[list[str], bool]:
    """Return (severity_range strings, any-serious flag) across PT AE
    nodes under ``soc_id`` that have a causes_ae from any compound
    binding ``target_id`` whose belief passes the vote threshold.

    Round-29: also OR-aggregates the ``serious`` flag across qualifying
    PT nodes. CT.gov reports ``serious=True`` on ~90% of AEs even when
    CTCAE grade is unavailable, so this is the primary severity signal
    when ``severity_range`` is empty.

    Mirrors the sibling-traversal in ``_collect_soc_votes`` (same
    filter rules). Returns one severity string per qualifying PT node
    (empty strings preserved) plus a single bool for the SOC-tier
    serious-floor decision.
    """
    sibling_compound_ids = _compounds_binding_target(graph, target_id)
    g = graph._graph
    severities: list[str] = []
    any_serious = False
    for sib_id in sibling_compound_ids:
        if sib_id not in g:
            continue
        for _src, candidate_ae_id, key in g.out_edges(sib_id, keys=True):
            if key != EdgeType.CAUSES_AE.value:
                continue
            try:
                ae_node = graph.get_node(candidate_ae_id)
            except KeyError:
                continue
            if (ae_node.get("soc_id") or "") != soc_id:
                continue
            try:
                belief = graph.get_edge_belief(
                    sib_id, candidate_ae_id, EdgeType.CAUSES_AE,
                )
            except KeyError:
                continue
            if belief.evidence_strength < _MIN_EVIDENCE_STRENGTH_FOR_VOTE:
                continue
            severities.append(ae_node.get("severity_range") or "")
            if ae_node.get("serious"):
                any_serious = True
    return severities, any_serious


def _binds_to_targets(graph: GraphStore, compound_id: str) -> list[str]:
    """Targets the given compound binds_to (out-edges of type binds_to)."""
    g = graph._graph
    if compound_id not in g:
        return []
    out: list[str] = []
    for _src, tgt, key in g.out_edges(compound_id, keys=True):
        if key == EdgeType.AFFECTS.value:
            out.append(tgt)
    return out


def _compounds_binding_target(graph: GraphStore, target_id: str) -> list[str]:
    """Compounds that bind_to the given target (in-edges of type binds_to)."""
    g = graph._graph
    if target_id not in g:
        return []
    out: list[str] = []
    for src, _tgt, key in g.in_edges(target_id, keys=True):
        if key == EdgeType.AFFECTS.value:
            out.append(src)
    return out


def _compound_vote_bucket(causes_ae_belief: EdgeBeliefState) -> SupportBucket:
    """Map a compound's causes_ae posterior to a support-bucket vote.

    The aggregation treats each compound as one observation of the
    target-class hypothesis. Strong individual evidence (high posterior
    + tight credible interval, reflected in evidence_strength) gets a
    full strong_support vote; weak evidence gets demoted toward
    ambiguous so it doesn't push the target belief around.
    """
    p = causes_ae_belief.expected_probability
    if p >= 0.75:
        return SupportBucket.STRONG_SUPPORT
    if p >= 0.55:
        return SupportBucket.MODERATE_SUPPORT
    if p >= 0.40:
        return SupportBucket.WEAK_SUPPORT
    if p <= 0.25:
        return SupportBucket.WEAK_CONTRADICT
    return SupportBucket.AMBIGUOUS


def _rebuild_target_ae_edge(
    graph: GraphStore,
    target_id: str,
    ae_id: str,
    contributing: list[tuple[str, EdgeBeliefState]],
    *,
    severity_range: str = "",
    serious: bool = False,
) -> AppliedTargetAEUpdate:
    """Aggregate per-compound votes into a fresh target_associated_ae belief.

    Replaces the existing edge belief (if any) rather than appending,
    so the propagation is idempotent.

    Round-29: ``severity_range`` is an OPTIONAL per-edge aggregation of
    grade tokens collected from the contributing sibling PT-level AE
    nodes (passed in by ``_propagate_at_soc_tier`` for SOC-tier edges).
    For SOC-tier edges this is target-specific — different compounds
    binding different targets see different SOC severities. PT-tier
    edges leave severity_range empty (default) and the downstream
    safety-penalty reader falls back to the AE node's own
    ``severity_range``, which is already PT-specific.
    """
    # Capture the pre-update belief for diffing.
    try:
        pre = graph.get_edge_belief(
            target_id, ae_id, EdgeType.TARGET_ASSOCIATED_AE
        )
    except KeyError:
        pre = EdgeBeliefState()
        graph.add_edge(GraphEdge(
            source_id=target_id,
            target_id=ae_id,
            edge_type=EdgeType.TARGET_ASSOCIATED_AE,
            belief=EdgeBeliefState(),
        ))

    # Rebuild from the Beta(1,1) prior, applying one virtual-evidence
    # update per contributing compound.
    aggregated = EdgeBeliefState()
    contributing_records: list[EvidenceRecord] = []
    now = datetime.now(timezone.utc)
    for compound_id, belief in contributing:
        bucket = _compound_vote_bucket(belief)
        p_obs = p_obs_for_bucket(bucket)
        aggregated = apply_virtual_evidence(
            aggregated, n_eff=_VOTE_N_EFF_PER_COMPOUND, p_obs=p_obs,
        )
        # Round-30 DLT-gate: carry the contributing compound's
        # failure-causing-toxicity signal through propagation, so the
        # target-class (class-effect) AE belief knows whether the class's
        # toxicity was dose-limiting or merely occurred in tolerated trials.
        # Without this, class-effect toxicity (e.g. checkpoint-class irAEs)
        # bypasses the safety DLT-gate. The vote is failure-causing iff a
        # majority of the compound's underlying causes_ae records were.
        _recs = belief.evidence or []
        _fc = sum(
            1 for e in _recs if (e.context or {}).get("failure_causing_tox")
        )
        _vote_dlt = bool(_recs and _fc / len(_recs) >= 0.5)
        contributing_records.append(EvidenceRecord(
            source_id=compound_id,
            source_type=EvidenceType.LITERATURE,  # synthesis vote, not raw evidence
            support=bucket.value,
            quality_score=1.0,
            timestamp=now,
            notes=(
                f"target_associated_ae vote from causes_ae({compound_id}) "
                f"= {belief.expected_probability:.3f}"
            ),
            context={
                "propagation": "target_associated_ae_aggregation",
                "failure_causing_tox": _vote_dlt,
            },
        ))

    # Replace the stored belief in place. We bypass update_edge_belief
    # because that method *appends* one record + grows the posterior;
    # propagation needs a full rebuild instead.
    g = graph._graph
    aggregated.evidence = contributing_records
    edge_data = g.edges[target_id, ae_id, EdgeType.TARGET_ASSOCIATED_AE.value]
    edge_data["belief"] = aggregated.model_dump(mode="json")
    # Round-29: stash the target-scoped severity union AND `serious`
    # flag on the edge so _compute_safety_penalty reads per-target
    # severity (instead of the shared SOC AE node's globally-leaky
    # value) and per-target seriousness (for the grade-3 floor when
    # CTCAE grade is missing).
    edge_data["severity_range"] = severity_range
    edge_data["serious"] = bool(serious)

    return AppliedTargetAEUpdate(
        target_id=target_id,
        ae_id=ae_id,
        pre_update_belief=pre,
        post_update_belief=aggregated,
        contributing_compound_ids=[cid for cid, _ in contributing],
    )
