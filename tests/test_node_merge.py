"""Tests for the generalized geometric + id node merge (Phase-2 assembly).

Covers the three merge tiers (ontology id, SapBERT name_id, BioLORD cosine with a
tunable threshold), lossless belief union + replay, belief_field anchor
concatenation, generalized chain-id rewrite across ALL id fields, provenance, and
idempotency.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.graph.models import (
    BiologyNode,
    CausalChain,
    EdgeBeliefState,
    EdgeType,
    EvidenceRecord,
    EvidenceType,
    GraphEdge,
    TargetNode,
    TrialSubgraph,
)
from src.graph.node_merge import MergeConfig, assemble, resolve_hierarchy
from src.graph.store import GraphStore
from src.inference.beliefs import SupportBucket


def _rec(source_id: str) -> EvidenceRecord:
    return EvidenceRecord(
        source_id=source_id,
        source_type=EvidenceType.CLINICAL_PHASE3,
        support=SupportBucket.STRONG_SUPPORT.value,
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def _belief(*source_ids: str, belief_field: dict | None = None) -> EdgeBeliefState:
    return EdgeBeliefState(
        alpha=1.0, beta=1.0,
        evidence=[_rec(s) for s in source_ids],
        belief_field=belief_field,
    )


def _bio(nid: str, *, ontology_id: str | None = None, name: str | None = None,
         name_id: str | None = None, desc: str = "") -> BiologyNode:
    meta: dict = {}
    if ontology_id:
        meta["ontology_id"] = ontology_id
    if name_id:
        meta["name_id"] = name_id
    return BiologyNode(id=nid, name=name or nid, metadata=meta, description=desc)


def _bio_only(types=("BiologyNode",), **kw) -> MergeConfig:
    return MergeConfig(node_types=types, enable_biolord=False, enable_name_id=False,
                       enable_id=False, **kw)


# ── Tier 1: ontology id ──────────────────────────────────────────────────────


def test_tier1_ontology_id_merges_distinct_node_ids():
    g = GraphStore()
    g.add_node(_bio("R-HSA-1", ontology_id="GO:42"))
    g.add_node(_bio("R-HSA-2", ontology_id="GO:42"))  # same underlying biology
    g.add_node(_bio("R-HSA-9", ontology_id="GO:99"))  # distinct
    cfg = _bio_only()
    cfg.enable_id = True
    report = assemble(g, cfg)
    assert report.nodes_removed == ["R-HSA-2"]          # winner = sorted-first
    assert report.winner_by_loser == {"R-HSA-2": "R-HSA-1"}
    assert "R-HSA-9" in g._graph and "R-HSA-2" not in g._graph  # noqa: SLF001


# ── Tier 2: SapBERT name_id ──────────────────────────────────────────────────


def test_tier2_name_id_separates_then_merges():
    g = GraphStore()
    g.add_node(_bio("slug_a", name_id="signaling_by_vegf"))
    g.add_node(_bio("R-HSA-194138", name_id="signaling_by_vegf"))
    g.add_node(_bio("R-HSA-389948", name_id="pd1_signaling"))  # distinct entity
    cfg = _bio_only()
    cfg.enable_name_id = True
    report = assemble(g, cfg)
    assert report.nodes_after == 2                      # the two vegf nodes merged
    assert report.chains_updated == 0
    assert "R-HSA-389948" in g._graph  # noqa: SLF001  (not pooled with vegf)


# ── Tier 3: BioLORD cosine, tunable threshold ────────────────────────────────


def _stub_embed(text: str) -> list[float]:
    return {
        "vegf signaling": [1.0, 0.0],
        "vegf angiogenesis": [0.97, 0.24],   # cos≈0.97 with vegf signaling
        "pd-1 checkpoint": [0.0, 1.0],       # orthogonal
    }[text]


def test_tier3_cosine_merge_respects_threshold():
    def build():
        g = GraphStore()
        g.add_node(_bio("a", desc="vegf signaling"))
        g.add_node(_bio("b", desc="vegf angiogenesis"))
        g.add_node(_bio("c", desc="pd-1 checkpoint"))
        return g

    loose = _bio_only()
    loose.enable_biolord = True
    loose.biolord_threshold = 0.85
    g = build()
    assemble(g, loose, embed_fn=_stub_embed)
    assert g._graph.number_of_nodes() == 2  # noqa: SLF001  (a+b merge, c stays)

    tight = _bio_only()
    tight.enable_biolord = True
    tight.biolord_threshold = 0.99          # above 0.97 → no merge
    g2 = build()
    assemble(g2, tight, embed_fn=_stub_embed)
    assert g2._graph.number_of_nodes() == 3  # noqa: SLF001


# ── Tier 2b: SapBERT separates siblings BioLORD merges ───────────────────────


def test_sapbert_tier_separates_siblings_biolord_merges():
    """Empirically (real models): BioLORD scores 'Co-inhibition by PD-1' vs
    '...CTLA4' at 1.000 (over-merge), SapBERT at 0.67 (separate). This pins that
    behaviour with stubs: BioLORD-on-description merges the pair, SapBERT-on-name
    does not."""
    def biolord_stub(_text):           # description embeds identically (1.000)
        return [1.0, 0.0]

    def sapbert_stub(name):            # name embeds distinctly per entity
        return {"Co-inhibition by PD-1": [1.0, 0.0],
                "Co-inhibition by CTLA4": [0.67, 0.74]}[name]  # cos≈0.67

    def build():
        g = GraphStore()
        g.add_node(_bio("R-HSA-389948", name="Co-inhibition by PD-1", desc="checkpoint co-inhibition"))
        g.add_node(_bio("R-HSA-389513", name="Co-inhibition by CTLA4", desc="checkpoint co-inhibition"))
        return g

    g = build()
    cfg = MergeConfig(node_types=("BiologyNode",), enable_id=False, enable_name_id=False,
                      enable_biolord=True, biolord_threshold=0.85)
    assemble(g, cfg, embed_fn=biolord_stub)
    assert g._graph.number_of_nodes() == 1  # noqa: SLF001  (BioLORD over-merges)

    g = build()
    cfg = MergeConfig(node_types=("BiologyNode",), enable_id=False, enable_name_id=False,
                      enable_biolord=False, enable_sapbert=True, sapbert_threshold=0.80)
    assemble(g, cfg, sapbert_embed_fn=sapbert_stub)
    assert g._graph.number_of_nodes() == 2  # noqa: SLF001  (SapBERT keeps them distinct)


# ── Lossless belief union + replay + belief_field anchors ────────────────────


def test_merge_unions_evidence_and_replays_belief():
    g = GraphStore()
    g.add_node(_bio("b1", ontology_id="GO:1"))
    g.add_node(_bio("b2", ontology_id="GO:1"))
    # same source edge to each → collision on redirect
    g.add_edge(GraphEdge(source_id="S", target_id="b1",
                         edge_type=EdgeType.MECHANISM_AFFECTS, belief=_belief("NCT_A")))
    g.add_edge(GraphEdge(source_id="S", target_id="b2",
                         edge_type=EdgeType.MECHANISM_AFFECTS, belief=_belief("NCT_B")))
    one_record_alpha = g.get_edge_belief("S", "b1", EdgeType.MECHANISM_AFFECTS).alpha

    cfg = _bio_only(); cfg.enable_id = True
    assemble(g, cfg)

    merged = g.get_edge_belief("S", "b1", EdgeType.MECHANISM_AFFECTS)
    assert {r.source_id for r in merged.evidence} == {"NCT_A", "NCT_B"}  # unioned
    assert merged.alpha > one_record_alpha                              # replayed up


def test_merge_concatenates_belief_field_anchors():
    anc = lambda v: {"s": v, "t": v, "alpha": 3.0, "beta": 1.0}  # noqa: E731
    g = GraphStore()
    g.add_node(_bio("b1", ontology_id="GO:1"))
    g.add_node(_bio("b2", ontology_id="GO:1"))
    g.add_edge(GraphEdge(source_id="S", target_id="b1", edge_type=EdgeType.MECHANISM_AFFECTS,
                         belief=_belief("NCT_A", belief_field={"anchors": [anc([1.0, 0.0])]})))
    g.add_edge(GraphEdge(source_id="S", target_id="b2", edge_type=EdgeType.MECHANISM_AFFECTS,
                         belief=_belief("NCT_B", belief_field={"anchors": [anc([0.0, 1.0])]})))
    cfg = _bio_only(); cfg.enable_id = True
    assemble(g, cfg)
    bf = g.get_edge_belief("S", "b1", EdgeType.MECHANISM_AFFECTS).belief_field
    assert len(bf["anchors"]) == 2                       # both kept, separable
    assert bf["marginal_alpha"] > 1.0                    # re-centered on replayed scalar


# ── Generalized chain-id rewrite across ALL fields ───────────────────────────


def _chain(**overrides) -> CausalChain:
    base = dict(arm_id="arm1", compound_id="c1", subgroup_population_id="pop1",
                target_id="t1", mechanism_id="m1", biology_id="bio1",
                indication_id="ind1", endpoint_id="ep1")
    base.update(overrides)
    return CausalChain(**base)


def test_chain_ids_rewritten_for_every_field():
    g = GraphStore()
    # a target merge and a biology merge in the same sweep. Winner = sorted-first
    # id, so "t_a"/"bio_a" win and the chain (referencing "t_z"/"bio_z") rewrites.
    g.add_node(TargetNode(id="t_a", gene_symbol="EGFR", name="EGFR",
                          metadata={"ontology_id": "ENSG1"}))
    g.add_node(TargetNode(id="t_z", gene_symbol="EGFR", name="EGFR",
                          metadata={"ontology_id": "ENSG1"}))
    g.add_node(_bio("bio_a", ontology_id="GO:7"))
    g.add_node(_bio("bio_z", ontology_id="GO:7"))
    g.set_trial_subgraph(TrialSubgraph(
        trial_id="NCT1", parent_population_id="pop1",
        chains=[_chain(target_id="t_z", biology_id="bio_z")],
    ))
    cfg = MergeConfig(node_types=("TargetNode", "BiologyNode"),
                      enable_id=True, enable_name_id=False, enable_biolord=False)
    report = assemble(g, cfg)
    ch = g.trial_subgraphs["NCT1"].chains[0]
    assert ch.target_id == "t_a"          # sorted-first winner
    assert ch.biology_id == "bio_a"
    assert report.chains_updated == 2     # two fields rewritten


# ── Provenance + idempotency ─────────────────────────────────────────────────


# ── Hierarchy resolution from box geometry ───────────────────────────────────


def test_resolve_hierarchy_emits_subtype_edges_from_containment():
    import numpy as np

    from src.graph.box_embeddings import Box
    from src.graph.models import EdgeType

    g = GraphStore()
    for nid in ("broad", "narrow", "sib"):
        g.add_node(_bio(nid))
    boxes = {
        "broad": Box(np.array([0.0, 0.0]), np.array([10.0, 10.0])),
        "narrow": Box(np.array([1.0, 1.0]), np.array([2.0, 2.0])),   # inside broad
        "sib": Box(np.array([8.0, 8.0]), np.array([18.0, 18.0])),    # overlaps, not contained
    }
    added = resolve_hierarchy(g, boxes, node_types=("BiologyNode",))
    assert added == {"BiologyNode": 1}
    # narrow is-a broad (child→parent), direction = specific→general
    assert g._graph.has_edge("narrow", "broad", key=EdgeType.SUBTYPE_OF.value)  # noqa: SLF001
    assert not g._graph.has_edge("broad", "narrow", key=EdgeType.SUBTYPE_OF.value)  # noqa: SLF001
    # sibling pair creates no hierarchy edge
    assert not g._graph.has_edge("sib", "broad", key=EdgeType.SUBTYPE_OF.value)  # noqa: SLF001
    # idempotent
    assert resolve_hierarchy(g, boxes, node_types=("BiologyNode",)) == {}


def test_provenance_and_idempotency():
    g = GraphStore()
    g.add_node(_bio("b1", ontology_id="GO:1"))
    g.add_node(_bio("b2", ontology_id="GO:1"))
    cfg = _bio_only(); cfg.enable_id = True
    assemble(g, cfg)
    assert "b2" in (g.get_node("b1").get("metadata") or {}).get("merged_from", [])
    # second sweep is a no-op
    report2 = assemble(g, cfg)
    assert report2.nodes_removed == []
    assert report2.nodes_after == report2.nodes_before
