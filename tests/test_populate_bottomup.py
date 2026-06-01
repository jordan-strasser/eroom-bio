"""Tests for the bottom-up (chains-first) build helpers: trial-scoped namespace
(belief-preserving), union, merge round-trip, and canonicalize."""

from __future__ import annotations

from datetime import datetime, timezone

from src.graph.models import (
    CausalChain,
    EdgeBeliefState,
    EdgeType,
    EvidenceRecord,
    EvidenceType,
    GraphEdge,
    TargetNode,
    TrialSubgraph,
)
from src.graph.node_merge import MergeConfig, assemble
from src.graph.populate_bottomup import _canonicalize_ids, _namespace_graph, _union_into
from src.graph.store import GraphStore
from src.inference.beliefs import SupportBucket


def _belief(*source_ids: str) -> EdgeBeliefState:
    return EdgeBeliefState(alpha=1.0, beta=1.0, evidence=[
        EvidenceRecord(source_id=s, source_type=EvidenceType.CLINICAL_PHASE3,
                       support=SupportBucket.STRONG_SUPPORT.value,
                       timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc))
        for s in source_ids])


def _tgt(nid: str) -> TargetNode:
    return TargetNode(id=nid, gene_symbol="EGFR", name="EGFR", metadata={})


def _chain(**ov) -> CausalChain:
    base = dict(arm_id="a", compound_id="c1", subgroup_population_id="pop1",
                target_id="t1", mechanism_id="m1", biology_id="bio1",
                indication_id="ind1", endpoint_id="ep1")
    base.update(ov)
    return CausalChain(**base)


def _ev_sids(belief) -> set[str]:
    ev = belief.get("evidence") if isinstance(belief, dict) else belief.evidence
    return {(e.get("source_id") if isinstance(e, dict) else e.source_id) for e in (ev or [])}


def _single_trial(nct: str) -> GraphStore:
    g = GraphStore()
    g.add_node(_tgt("ENSG1"))
    g.add_node(_tgt("ENSG2"))
    g.add_edge(GraphEdge(source_id="ENSG1", target_id="ENSG2",
                         edge_type=EdgeType.MODULATES_VIA, belief=_belief(nct)))
    g.set_trial_subgraph(TrialSubgraph(
        trial_id=nct, parent_population_id="pop1",
        chains=[_chain(target_id="ENSG1", mechanism_id="ENSG2")]))
    return g


def test_namespace_graph_scopes_ids_preserves_beliefs_and_rewrites_chains():
    ns = _namespace_graph(_single_trial("NCT1"), "NCT1")
    assert "ENSG1#NCT1" in ns._graph  # noqa: SLF001
    assert ns._graph.nodes["ENSG1#NCT1"]["ontology_id"] == "ENSG1"  # noqa: SLF001
    # edge + belief carried through (NOT dropped like the structural explode)
    data = ns._graph.get_edge_data("ENSG1#NCT1", "ENSG2#NCT1",  # noqa: SLF001
                                   key=EdgeType.MODULATES_VIA.value)
    assert _ev_sids(data["belief"]) == {"NCT1"}
    ch = ns.trial_subgraphs["NCT1"].chains[0]
    assert ch.target_id == "ENSG1#NCT1" and ch.mechanism_id == "ENSG2#NCT1"


def test_canonicalize_ids_restores_canonical_ids():
    ns = _namespace_graph(_single_trial("NCT1"), "NCT1")
    renamed = _canonicalize_ids(ns)
    assert renamed == 2  # ENSG1, ENSG2
    assert "ENSG1" in ns._graph and "ENSG1#NCT1" not in ns._graph  # noqa: SLF001
    ch = ns.trial_subgraphs["NCT1"].chains[0]
    assert ch.target_id == "ENSG1"  # chain rewritten back to canonical


def test_two_trials_round_trip_merges_concepts_and_unions_beliefs():
    """The faithfulness core: two trials resolving the same concepts in
    ISOLATION, then namespace -> union -> merge -> canonicalize, reconstruct one
    shared concept set with the per-trial evidence losslessly unioned."""
    merged = GraphStore()
    _union_into(merged, _namespace_graph(_single_trial("NCT1"), "NCT1"))
    _union_into(merged, _namespace_graph(_single_trial("NCT2"), "NCT2"))
    assert merged._graph.number_of_nodes() == 4  # 2 trials x 2 scoped instances  # noqa: SLF001

    assemble(merged, MergeConfig(node_types=("TargetNode",), enable_id=True,
                                 enable_name_id=False, enable_biolord=False))
    _canonicalize_ids(merged)

    assert merged._graph.number_of_nodes() == 2          # merged to 2 concepts  # noqa: SLF001
    assert {"ENSG1", "ENSG2"} <= set(merged._graph.nodes)  # noqa: SLF001
    data = merged._graph.get_edge_data("ENSG1", "ENSG2",  # noqa: SLF001
                                       key=EdgeType.MODULATES_VIA.value)
    assert _ev_sids(data["belief"]) == {"NCT1", "NCT2"}  # lossless belief union


def test_union_into_is_additive():
    dst = GraphStore()
    _union_into(dst, _namespace_graph(_single_trial("NCT1"), "NCT1"))
    n1 = dst._graph.number_of_nodes()  # noqa: SLF001
    _union_into(dst, _namespace_graph(_single_trial("NCT2"), "NCT2"))
    assert dst._graph.number_of_nodes() == 2 * n1  # noqa: SLF001
    assert set(dst.trial_subgraphs) == {"NCT1", "NCT2"}
