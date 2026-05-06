"""Cross-trial propagation: causes_ae → target_associated_ae.

When the same adverse event has been attributed to multiple compounds
that all bind the same target, that's evidence the AE is mechanism-related
(target-associated) rather than chemistry-specific to one molecule.
This module rebuilds ``target_associated_ae`` beliefs by aggregating the
``causes_ae`` beliefs of every compound binding the target.

Idempotent by design — the target_associated_ae belief is rebuilt from
scratch on every call rather than incrementally updated, so re-running
after every causes_ae attribution doesn't double-count.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from pydantic import BaseModel

from src.graph.models import (
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


# Each compound binding a target contributes one vote about the target's
# AE liability. The target-class hypothesis is one step removed from any
# single trial, so each compound's vote carries less weight than a single
# Phase-3 trial — anchor at GWAS-tier (N_eff = 4) by default.
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
) -> list[AppliedTargetAEUpdate]:
    """Refresh target_associated_ae beliefs implicated by a causes_ae update.

    For every target the compound binds_to, gather the causes_ae beliefs
    of all compounds binding that target for the given AE. If at least
    ``_MIN_COMPOUNDS_FOR_TARGET_AE`` of them carry meaningful evidence,
    rebuild the target_associated_ae belief from those votes.
    """
    targets = _binds_to_targets(graph, compound_id)
    updates: list[AppliedTargetAEUpdate] = []
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
    return updates


def _binds_to_targets(graph: GraphStore, compound_id: str) -> list[str]:
    """Targets the given compound binds_to (out-edges of type binds_to)."""
    g = graph._graph
    if compound_id not in g:
        return []
    out: list[str] = []
    for _src, tgt, key in g.out_edges(compound_id, keys=True):
        if key == EdgeType.BINDS_TO.value:
            out.append(tgt)
    return out


def _compounds_binding_target(graph: GraphStore, target_id: str) -> list[str]:
    """Compounds that bind_to the given target (in-edges of type binds_to)."""
    g = graph._graph
    if target_id not in g:
        return []
    out: list[str] = []
    for src, _tgt, key in g.in_edges(target_id, keys=True):
        if key == EdgeType.BINDS_TO.value:
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
