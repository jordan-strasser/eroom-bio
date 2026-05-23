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
        update = _rebuild_target_ae_edge(
            graph, target_id, soc_ae_id, contributing,
        )
        updates.append(update)
    return updates


def _ensure_soc_ae_node(
    graph: GraphStore,
    soc_ae_id: str,
    soc_id: str,
    soc_name: str,
) -> None:
    """Create the SOC-tier AE node if missing."""
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
) -> AppliedTargetAEUpdate:
    """Aggregate per-compound votes into a fresh target_associated_ae belief.

    Replaces the existing edge belief (if any) rather than appending,
    so the propagation is idempotent.
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
            context={"propagation": "target_associated_ae_aggregation"},
        ))

    # Replace the stored belief in place. We bypass update_edge_belief
    # because that method *appends* one record + grows the posterior;
    # propagation needs a full rebuild instead.
    g = graph._graph
    aggregated.evidence = contributing_records
    g.edges[target_id, ae_id, EdgeType.TARGET_ASSOCIATED_AE.value]["belief"] = (
        aggregated.model_dump(mode="json")
    )

    return AppliedTargetAEUpdate(
        target_id=target_id,
        ae_id=ae_id,
        pre_update_belief=pre,
        post_update_belief=aggregated,
        contributing_compound_ids=[cid for cid, _ in contributing],
    )
