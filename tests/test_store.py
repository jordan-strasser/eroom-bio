"""Tests for the GraphStore."""

import json
from datetime import datetime, timezone

import pytest

from src.graph.models import (
    BiologyNode,
    CompoundNode,
    EdgeBeliefState,
    EdgeType,
    EndpointNode,
    EndpointType,
    EvidenceDirection,
    EvidenceRecord,
    EvidenceType,
    GraphEdge,
    IndicationNode,
    MechanismNode,
    MechanismType,
    Modality,
    PopulationNode,
    RegulatoryStatus,
    TargetNode,
    TrialOutcome,
    TrialSubgraph,
)
from src.graph.store import EVIDENCE_TYPE_WEIGHTS, GraphStore


@pytest.fixture
def store():
    return GraphStore()


@pytest.fixture
def compound():
    return CompoundNode(
        id="COMPOUND_IMA", name="Imatinib", modality=Modality.SMALL_MOLECULE
    )


@pytest.fixture
def target():
    return TargetNode(id="TARGET_ABL", name="ABL1", gene_symbol="ABL1")


@pytest.fixture
def edge(compound, target):
    return GraphEdge(
        source_id=compound.id,
        target_id=target.id,
        edge_type=EdgeType.BINDS_TO,
    )


def _make_evidence(
    direction: EvidenceDirection = EvidenceDirection.SUPPORTING,
    source_type: EvidenceType = EvidenceType.CLINICAL_PHASE3,
    quality: float = 1.0,
    magnitude: float = 1.0,
) -> EvidenceRecord:
    return EvidenceRecord(
        source_id="SRC_001",
        source_type=source_type,
        quality_score=quality,
        direction=direction,
        magnitude=magnitude,
        timestamp=datetime(2024, 6, 1, tzinfo=timezone.utc),
    )


# ── Node CRUD ────────────────────────────────────────────────────────────


class TestNodeCRUD:
    def test_add_and_get(self, store, compound):
        store.add_node(compound)
        data = store.get_node("COMPOUND_IMA")
        assert data["name"] == "Imatinib"
        assert data["node_type"] == "CompoundNode"

    def test_get_missing_raises(self, store):
        with pytest.raises(KeyError):
            store.get_node("NONEXISTENT")

    def test_get_nodes_by_type(self, store, compound, target):
        store.add_node(compound)
        store.add_node(target)
        compounds = store.get_nodes_by_type("CompoundNode")
        assert len(compounds) == 1
        assert compounds[0]["id"] == "COMPOUND_IMA"
        targets = store.get_nodes_by_type("TargetNode")
        assert len(targets) == 1


# ── Edge CRUD ────────────────────────────────────────────────────────────


class TestEdgeCRUD:
    def test_add_and_get_belief(self, store, compound, target, edge):
        store.add_node(compound)
        store.add_node(target)
        store.add_edge(edge)
        belief = store.get_edge_belief(
            compound.id, target.id, EdgeType.BINDS_TO
        )
        assert belief.alpha == 1.0
        assert belief.beta == 1.0

    def test_get_missing_edge_raises(self, store, compound, target):
        store.add_node(compound)
        store.add_node(target)
        with pytest.raises(KeyError):
            store.get_edge_belief(compound.id, target.id, EdgeType.BINDS_TO)

    def test_get_edges_by_type(self, store, compound, target, edge):
        store.add_node(compound)
        store.add_node(target)
        store.add_edge(edge)
        edges = store.get_edges_by_type(EdgeType.BINDS_TO)
        assert len(edges) == 1
        assert edges[0]["source_id"] == compound.id

    def test_neighboring_edges(self, store, compound, target, edge):
        store.add_node(compound)
        store.add_node(target)
        store.add_edge(edge)
        neighbors = store.get_neighboring_edges(compound.id)
        assert len(neighbors) == 1
        assert neighbors[0]["target_id"] == target.id

    def test_neighboring_edges_filtered(self, store, compound, target, edge):
        store.add_node(compound)
        store.add_node(target)
        store.add_edge(edge)
        assert len(
            store.get_neighboring_edges(
                compound.id, edge_types=[EdgeType.MODULATES_VIA]
            )
        ) == 0
        assert len(
            store.get_neighboring_edges(
                compound.id, edge_types=[EdgeType.BINDS_TO]
            )
        ) == 1


# ── Bayesian updates ────────────────────────────────────────────────────


class TestBayesianUpdates:
    def test_supporting_updates_alpha(self, store, compound, target, edge):
        store.add_node(compound)
        store.add_node(target)
        store.add_edge(edge)
        ev = _make_evidence(
            direction=EvidenceDirection.SUPPORTING,
            source_type=EvidenceType.CLINICAL_PHASE3,
            quality=0.9,
            magnitude=2.0,
        )
        belief = store.update_edge_belief(
            compound.id, target.id, EdgeType.BINDS_TO, ev
        )
        expected_delta = 5.0 * 0.9 * 2.0  # weight * quality * magnitude
        assert belief.alpha == pytest.approx(1.0 + expected_delta)
        assert belief.beta == 1.0
        assert len(belief.evidence) == 1

    def test_contradicting_updates_beta(self, store, compound, target, edge):
        store.add_node(compound)
        store.add_node(target)
        store.add_edge(edge)
        ev = _make_evidence(
            direction=EvidenceDirection.CONTRADICTING,
            source_type=EvidenceType.PRECLINICAL_IN_VITRO,
            quality=0.8,
            magnitude=1.5,
        )
        belief = store.update_edge_belief(
            compound.id, target.id, EdgeType.BINDS_TO, ev
        )
        expected_delta = 1.0 * 0.8 * 1.5
        assert belief.alpha == 1.0
        assert belief.beta == pytest.approx(1.0 + expected_delta)

    def test_ambiguous_updates_both(self, store, compound, target, edge):
        store.add_node(compound)
        store.add_node(target)
        store.add_edge(edge)
        ev = _make_evidence(
            direction=EvidenceDirection.AMBIGUOUS,
            source_type=EvidenceType.LITERATURE,
            quality=0.5,
            magnitude=1.0,
        )
        belief = store.update_edge_belief(
            compound.id, target.id, EdgeType.BINDS_TO, ev
        )
        expected_delta = 0.3 * 0.5 * 1.0 * 0.3
        assert belief.alpha == pytest.approx(1.0 + expected_delta)
        assert belief.beta == pytest.approx(1.0 + expected_delta)

    def test_sequential_updates_accumulate(self, store, compound, target, edge):
        store.add_node(compound)
        store.add_node(target)
        store.add_edge(edge)
        ev1 = _make_evidence(
            direction=EvidenceDirection.SUPPORTING,
            source_type=EvidenceType.GENETIC_MR,
            quality=1.0,
            magnitude=1.0,
        )
        ev2 = _make_evidence(
            direction=EvidenceDirection.CONTRADICTING,
            source_type=EvidenceType.CLINICAL_PHASE2,
            quality=0.7,
            magnitude=1.0,
        )
        store.update_edge_belief(compound.id, target.id, EdgeType.BINDS_TO, ev1)
        belief = store.update_edge_belief(
            compound.id, target.id, EdgeType.BINDS_TO, ev2
        )
        assert belief.alpha == pytest.approx(1.0 + 4.0)  # GENETIC_MR weight
        assert belief.beta == pytest.approx(1.0 + 3.0 * 0.7)  # PHASE2 weight * quality
        assert len(belief.evidence) == 2

    def test_all_evidence_weights_present(self):
        for et in EvidenceType:
            assert et in EVIDENCE_TYPE_WEIGHTS


# ── Path finding ─────────────────────────────────────────────────────────


class TestPathFinding:
    def _build_chain(self, store):
        """Build a simple A -> B -> C -> D chain."""
        nodes = [
            CompoundNode(id="A", name="A", modality=Modality.OTHER),
            TargetNode(id="B", name="B", gene_symbol="B"),
            MechanismNode(id="C", name="C", mechanism_type=MechanismType.OTHER),
            BiologyNode(id="D", name="D"),
        ]
        for n in nodes:
            store.add_node(n)
        store.add_edge(GraphEdge(source_id="A", target_id="B", edge_type=EdgeType.BINDS_TO))
        store.add_edge(GraphEdge(source_id="B", target_id="C", edge_type=EdgeType.MODULATES_VIA))
        store.add_edge(GraphEdge(source_id="C", target_id="D", edge_type=EdgeType.MECHANISM_AFFECTS))

    def test_find_direct_path(self, store):
        self._build_chain(store)
        paths = store.find_paths("A", "D")
        assert len(paths) == 1
        assert paths[0] == ["A", "B", "C", "D"]

    def test_find_paths_no_route(self, store):
        self._build_chain(store)
        paths = store.find_paths("D", "A")  # reverse direction, no edges
        assert paths == []

    def test_find_paths_missing_node(self, store):
        paths = store.find_paths("NOPE", "ALSO_NOPE")
        assert paths == []

    def test_max_length_cutoff(self, store):
        self._build_chain(store)
        paths = store.find_paths("A", "D", max_length=2)
        assert paths == []  # chain is length 3


# ── Trial subgraph ───────────────────────────────────────────────────────


class TestTrialSubgraph:
    def test_get_trial_subgraph(self, store):
        nodes = [
            CompoundNode(id="C1", name="C", modality=Modality.OTHER),
            TargetNode(id="T1", name="T", gene_symbol="T"),
            MechanismNode(id="M1", name="M", mechanism_type=MechanismType.INHIBITION),
            BiologyNode(id="B1", name="B"),
            IndicationNode(id="I1", name="I"),
            EndpointNode(
                id="E1", name="E",
                endpoint_type=EndpointType.PRIMARY,
                regulatory_status=RegulatoryStatus.ACCEPTED,
            ),
            PopulationNode(id="P1", name="P"),
        ]
        for n in nodes:
            store.add_node(n)
        trial = TrialSubgraph(
            trial_id="NCT123",
            compound_id="C1",
            target_id="T1",
            mechanism_id="M1",
            biology_id="B1",
            indication_id="I1",
            endpoint_id="E1",
            population_id="P1",
            outcome=TrialOutcome.SUCCESS,
            phase="3",
        )
        sg = store.get_trial_subgraph(trial)
        assert sg.number_of_nodes() == 7


# ── Persistence ──────────────────────────────────────────────────────────


class TestPersistence:
    def test_export_import_roundtrip(self, store, compound, target, edge, tmp_path):
        store.add_node(compound)
        store.add_node(target)
        store.add_edge(edge)
        ev = _make_evidence()
        store.update_edge_belief(compound.id, target.id, EdgeType.BINDS_TO, ev)

        filepath = str(tmp_path / "snapshot.json")
        store.export_snapshot(filepath)

        store2 = GraphStore()
        store2.import_snapshot(filepath)

        assert store2.get_node("COMPOUND_IMA")["name"] == "Imatinib"
        belief = store2.get_edge_belief(compound.id, target.id, EdgeType.BINDS_TO)
        assert belief.alpha > 1.0
        assert len(belief.evidence) == 1

    def test_export_creates_valid_json(self, store, compound, tmp_path):
        store.add_node(compound)
        filepath = str(tmp_path / "snapshot.json")
        store.export_snapshot(filepath)
        data = json.loads((tmp_path / "snapshot.json").read_text())
        assert "nodes" in data


# ── Stats ────────────────────────────────────────────────────────────────


class TestStats:
    def test_stats_counts(self, store, compound, target, edge):
        store.add_node(compound)
        store.add_node(target)
        store.add_edge(edge)
        s = store.stats()
        assert s["node_count"] == 2
        assert s["edge_count"] == 1
        assert s["node_types"]["CompoundNode"] == 1
        assert s["node_types"]["TargetNode"] == 1
        assert s["edge_types"]["binds_to"] == 1
        assert s["total_evidence"] == 0

    def test_stats_evidence_count(self, store, compound, target, edge):
        store.add_node(compound)
        store.add_node(target)
        store.add_edge(edge)
        store.update_edge_belief(
            compound.id, target.id, EdgeType.BINDS_TO, _make_evidence()
        )
        store.update_edge_belief(
            compound.id, target.id, EdgeType.BINDS_TO, _make_evidence()
        )
        s = store.stats()
        assert s["total_evidence"] == 2

    def test_stats_high_conflict(self, store, compound, target):
        store.add_node(compound)
        store.add_node(target)
        # Create an edge with high conflict: lots of evidence, p near 0.5
        belief = EdgeBeliefState(alpha=50.0, beta=50.0)
        edge = GraphEdge(
            source_id=compound.id,
            target_id=target.id,
            edge_type=EdgeType.BINDS_TO,
            belief=belief,
        )
        store.add_edge(edge)
        s = store.stats()
        assert len(s["high_conflict_edges"]) == 1
