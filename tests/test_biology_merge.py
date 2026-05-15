"""Tests for src/graph/biology_merge.py.

These tests construct an in-memory ``GraphStore`` with deliberately-fragmented
BiologyNodes and verify the merge collapses them correctly. The
Reactome→GO crosswalk is supplied directly to ``find_equivalence_classes``
so we don't hit the live API.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from src.graph.biology_merge import (
    find_equivalence_classes,
    merge_equivalent_biology_nodes,
    pick_winner,
)
from src.graph.models import (
    BiologyNode,
    CausalChain,
    EdgeBeliefState,
    EdgeType,
    EvidenceRecord,
    EvidenceType,
    GraphEdge,
    MechanismNode,
    MechanismType,
    TargetNode,
    TrialArm,
    TrialSubgraph,
)
from src.graph.store import GraphStore
from src.inference.beliefs import SupportBucket


# ── Equivalence-class detection ──────────────────────────────────────────


class TestEquivalenceClasses:
    def test_reactome_and_go_pair_classed_together(self):
        # Both R-HSA-194138 and GO:0048010 are in the graph; the crosswalk
        # says R-HSA-194138 → GO:0048010, so they should cluster.
        classes = find_equivalence_classes(
            biology_ids=["R-HSA-194138", "GO:0048010", "R-HSA-OTHER"],
            crosswalk={"R-HSA-194138": "GO:0048010"},
        )
        assert len(classes) == 1
        assert {"R-HSA-194138", "GO:0048010"} == classes[0]

    def test_two_reactome_pathways_via_shared_go_term(self):
        """Two Reactome pathways that crosswalk to the same GO term cluster
        even when the GO term itself isn't a graph node. The GO term is the
        shared canonical ancestor — these pathways describe the same biology
        and should pool evidence.
        """
        classes = find_equivalence_classes(
            biology_ids=["R-HSA-X", "R-HSA-Y"],
            crosswalk={"R-HSA-X": "GO:0001525", "R-HSA-Y": "GO:0001525"},
        )
        assert len(classes) == 1
        assert classes[0] == {"R-HSA-X", "R-HSA-Y"}

    def test_three_way_cluster_with_go_in_graph(self):
        """Two Reactome pathways + their shared GO term, all in graph, all merge."""
        classes = find_equivalence_classes(
            biology_ids=["R-HSA-X", "R-HSA-Y", "GO:0001525"],
            crosswalk={"R-HSA-X": "GO:0001525", "R-HSA-Y": "GO:0001525"},
        )
        assert len(classes) == 1
        assert classes[0] == {"R-HSA-X", "R-HSA-Y", "GO:0001525"}

    def test_reactome_with_no_graph_go_counterpart_stays_singleton(self):
        classes = find_equivalence_classes(
            biology_ids=["R-HSA-194138"],
            crosswalk={"R-HSA-194138": "GO:0048010"},
        )
        assert classes == []  # singleton, not actionable

    def test_empty_crosswalk_returns_no_classes(self):
        classes = find_equivalence_classes(
            biology_ids=["R-HSA-X", "GO:0001"],
            crosswalk={},
        )
        assert classes == []


# ── Winner selection ─────────────────────────────────────────────────────


@pytest.fixture
def store() -> GraphStore:
    return GraphStore()


def _add_biology(store: GraphStore, *, id: str, name: str, primary_score: float = 0.0):
    store.add_node(BiologyNode(
        id=id,
        name=name,
        pathway_ids=[id],
        metadata={
            "source": "reactome_target_lookup",
            "primary_pathway": id,
            "primary_score": primary_score,
        },
    ))


class TestPickWinner:
    def test_higher_score_wins(self, store: GraphStore):
        _add_biology(store, id="R-HSA-194138", name="Signaling by VEGF", primary_score=0.50)
        _add_biology(store, id="GO:0001525", name="angiogenesis", primary_score=1.00)
        winner = pick_winner({"R-HSA-194138", "GO:0001525"}, store)
        assert winner == "GO:0001525"

    def test_score_tie_breaks_on_smallest_id(self, store: GraphStore):
        _add_biology(store, id="GO:0001525", name="angiogenesis", primary_score=0.50)
        _add_biology(store, id="R-HSA-194138", name="Signaling by VEGF", primary_score=0.50)
        winner = pick_winner({"R-HSA-194138", "GO:0001525"}, store)
        assert winner == "GO:0001525"  # 'GO:...' < 'R-HSA-...' lexicographically

    def test_missing_score_defaults_to_zero(self, store: GraphStore):
        _add_biology(store, id="A", name="alpha", primary_score=0.10)
        store.add_node(BiologyNode(id="B", name="beta", pathway_ids=["B"]))
        winner = pick_winner({"A", "B"}, store)
        assert winner == "A"


# ── End-to-end merge sweep ───────────────────────────────────────────────


def _trivial_evidence_record(source_id: str = "NCT00000001") -> EvidenceRecord:
    return EvidenceRecord(
        source_id=source_id,
        source_type=EvidenceType.CLINICAL_PHASE3,
        support=SupportBucket.STRONG_SUPPORT.value,
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )


class _FakeLincsClient:
    """Minimal stand-in for LINCSClient.get_pathway_go_term."""
    def __init__(self, crosswalk: dict[str, str]):
        self._crosswalk = crosswalk

    async def get_pathway_go_term(self, stable_id: str) -> str | None:
        return self._crosswalk.get(stable_id)


@pytest.mark.asyncio
async def test_merge_collapses_reactome_and_go(store: GraphStore):
    """End-to-end: two nodes whose Reactome→GO crosswalk says they're
    equivalent should collapse into one, edges redirected, chain
    biology_ids rewritten to the winner."""
    # Mechanism upstream + indication downstream of the to-be-merged bios.
    store.add_node(MechanismNode(
        id="kinase_inhibition", name="kinase_inhibition",
        mechanism_type=MechanismType.AGONISM,
    ))
    from src.graph.models import IndicationNode
    store.add_node(IndicationNode(id="melanoma", name="melanoma"))

    # Two equivalent biology nodes — Reactome with score 0.5, GO with 1.0.
    _add_biology(store, id="R-HSA-194138", name="Signaling by VEGF", primary_score=0.50)
    _add_biology(store, id="GO:0048010", name="VEGF receptor signaling", primary_score=1.00)

    # Edges into and out of each so we exercise the redirect path.
    ev = _trivial_evidence_record("NCT00000001")
    store.add_edge(GraphEdge(
        source_id="kinase_inhibition",
        target_id="R-HSA-194138",
        edge_type=EdgeType.MECHANISM_AFFECTS,
        belief=EdgeBeliefState(alpha=2.0, beta=1.0, evidence=[ev]),
    ))
    store.add_edge(GraphEdge(
        source_id="R-HSA-194138",
        target_id="melanoma",
        edge_type=EdgeType.BIOLOGY_DRIVES,
        belief=EdgeBeliefState(alpha=2.0, beta=1.0),
    ))

    # A trial subgraph with a chain pointing at the loser.
    store.add_node(TargetNode(id="ENSG00000112715", name="VEGFA", gene_symbol="VEGFA"))
    chain = CausalChain(
        arm_id="arm1", compound_id="bevacizumab",
        subgroup_population_id="melanoma__unselected",
        target_id="ENSG00000112715", mechanism_id="kinase_inhibition",
        biology_id="R-HSA-194138", indication_id="melanoma",
        endpoint_id="ORR_melanoma",
    )
    arm = TrialArm(
        arm_id="arm1", compound_ids=["bevacizumab"],
        regimen_compound_id="bevacizumab", is_combination=False,
    )
    store.set_trial_subgraph(TrialSubgraph(
        trial_id="NCT00000001", phase="3", arms=[arm], chains=[chain],
        parent_population_id="melanoma__unselected",
    ))

    fake_client: Any = _FakeLincsClient({"R-HSA-194138": "GO:0048010"})
    report = await merge_equivalent_biology_nodes(store, fake_client)

    assert report.biology_nodes_before == 2
    assert report.biology_nodes_after == 1
    assert report.nodes_removed == ["R-HSA-194138"]
    assert report.winner_by_loser == {"R-HSA-194138": "GO:0048010"}
    assert report.chains_updated == 1

    # Loser gone, winner remains
    with pytest.raises(KeyError):
        store.get_node("R-HSA-194138")
    assert store.get_node("GO:0048010")["name"] == "VEGF receptor signaling"

    # Edges redirected to winner
    assert store._graph.has_edge(  # noqa: SLF001
        "kinase_inhibition", "GO:0048010",
        key=EdgeType.MECHANISM_AFFECTS.value,
    )
    assert store._graph.has_edge(  # noqa: SLF001
        "GO:0048010", "melanoma", key=EdgeType.BIOLOGY_DRIVES.value,
    )

    # Chain biology_id rewritten
    ts = store.get_trial_subgraph_by_id("NCT00000001")
    assert ts.chains[0].biology_id == "GO:0048010"

    # Loser id preserved in winner's metadata audit trail
    winner_meta = store.get_node("GO:0048010").get("metadata") or {}
    assert "R-HSA-194138" in (winner_meta.get("merged_from") or [])
    assert winner_meta.get("alternate_pathway_names", {}).get("R-HSA-194138") == "Signaling by VEGF"


@pytest.mark.asyncio
async def test_merge_idempotent_on_rebuild(store: GraphStore):
    """Running the sweep twice on the same graph collapses on the first
    pass and is a no-op on the second."""
    _add_biology(store, id="R-HSA-X", name="alpha", primary_score=0.3)
    _add_biology(store, id="GO:0001234", name="beta", primary_score=0.5)

    fake_client: Any = _FakeLincsClient({"R-HSA-X": "GO:0001234"})

    r1 = await merge_equivalent_biology_nodes(store, fake_client)
    assert r1.nodes_collapsed == 1

    r2 = await merge_equivalent_biology_nodes(store, fake_client)
    assert r2.nodes_collapsed == 0


@pytest.mark.asyncio
async def test_no_crosswalk_means_no_merge(store: GraphStore):
    _add_biology(store, id="R-HSA-X", name="alpha", primary_score=0.3)
    _add_biology(store, id="GO:0001234", name="beta", primary_score=0.5)

    fake_client: Any = _FakeLincsClient({})  # no crosswalks at all
    report = await merge_equivalent_biology_nodes(store, fake_client)

    assert report.nodes_collapsed == 0
    assert report.biology_nodes_after == 2
