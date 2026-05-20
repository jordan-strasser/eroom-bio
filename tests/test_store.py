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
from src.graph.store import GraphStore
from src.inference.beliefs import (
    BUCKET_TO_P_OBS,
    EVIDENCE_TYPE_N_EFF,
    SupportBucket,
)


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
        edge_type=EdgeType.AFFECTS,
    )


def _make_evidence(
    support: SupportBucket = SupportBucket.STRONG_SUPPORT,
    source_type: EvidenceType = EvidenceType.CLINICAL_PHASE3,
    quality: float = 1.0,
) -> EvidenceRecord:
    return EvidenceRecord(
        source_id="SRC_001",
        source_type=source_type,
        support=support.value,
        quality_score=quality,
        timestamp=datetime(2024, 6, 1, tzinfo=timezone.utc),
    )


# ── Node CRUD ────────────────────────────────────────────────────────────


class TestNodeCRUD:
    def test_add_and_get(self, store, compound):
        store.add_node(compound)
        data = store.get_node("COMPOUND_IMA")
        assert data["name"] == "Imatinib"
        assert data["node_type"] == "InterventionNode"

    def test_get_missing_raises(self, store):
        with pytest.raises(KeyError):
            store.get_node("NONEXISTENT")

    def test_get_nodes_by_type(self, store, compound, target):
        store.add_node(compound)
        store.add_node(target)
        compounds = store.get_nodes_by_type("InterventionNode")
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
            compound.id, target.id, EdgeType.AFFECTS
        )
        assert belief.alpha == 1.0
        assert belief.beta == 1.0

    def test_get_missing_edge_raises(self, store, compound, target):
        store.add_node(compound)
        store.add_node(target)
        with pytest.raises(KeyError):
            store.get_edge_belief(compound.id, target.id, EdgeType.AFFECTS)

    def test_get_edges_by_type(self, store, compound, target, edge):
        store.add_node(compound)
        store.add_node(target)
        store.add_edge(edge)
        edges = store.get_edges_by_type(EdgeType.AFFECTS)
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
                compound.id, edge_types=[EdgeType.AFFECTS]
            )
        ) == 1


# ── Bayesian updates ────────────────────────────────────────────────────


class TestBayesianUpdates:
    """Beta-Binomial conjugate updates: α += N_eff·p_obs, β += N_eff·(1−p_obs)."""

    def test_strong_support_lifts_alpha(self, store, compound, target, edge):
        store.add_node(compound)
        store.add_node(target)
        store.add_edge(edge)
        ev = _make_evidence(
            support=SupportBucket.STRONG_SUPPORT,
            source_type=EvidenceType.CLINICAL_PHASE3,
            quality=0.9,
        )
        belief = store.update_edge_belief(
            compound.id, target.id, EdgeType.AFFECTS, ev
        )
        n_eff = EVIDENCE_TYPE_N_EFF[EvidenceType.CLINICAL_PHASE3] * 0.9
        p_obs = BUCKET_TO_P_OBS[SupportBucket.STRONG_SUPPORT]
        assert belief.alpha == pytest.approx(1.0 + n_eff * p_obs)
        assert belief.beta == pytest.approx(1.0 + n_eff * (1.0 - p_obs))
        assert len(belief.evidence) == 1

    def test_strong_contradict_lifts_beta(self, store, compound, target, edge):
        store.add_node(compound)
        store.add_node(target)
        store.add_edge(edge)
        ev = _make_evidence(
            support=SupportBucket.STRONG_CONTRADICT,
            source_type=EvidenceType.PRECLINICAL_IN_VITRO,
            quality=0.8,
        )
        belief = store.update_edge_belief(
            compound.id, target.id, EdgeType.AFFECTS, ev
        )
        n_eff = EVIDENCE_TYPE_N_EFF[EvidenceType.PRECLINICAL_IN_VITRO] * 0.8
        p_obs = BUCKET_TO_P_OBS[SupportBucket.STRONG_CONTRADICT]
        assert belief.alpha == pytest.approx(1.0 + n_eff * p_obs)
        assert belief.beta == pytest.approx(1.0 + n_eff * (1.0 - p_obs))
        # STRONG_CONTRADICT has p_obs=0.05, so beta should pick up most of it.
        assert belief.beta > belief.alpha

    def test_ambiguous_splits_evenly(self, store, compound, target, edge):
        store.add_node(compound)
        store.add_node(target)
        store.add_edge(edge)
        ev = _make_evidence(
            support=SupportBucket.AMBIGUOUS,
            source_type=EvidenceType.LITERATURE,
            quality=1.0,
        )
        belief = store.update_edge_belief(
            compound.id, target.id, EdgeType.AFFECTS, ev
        )
        # AMBIGUOUS bucket has p_obs = 0.5, so the update splits N_eff
        # evenly between α and β—the principled successor of the old
        # 0.3-each ad-hoc split.
        half = EVIDENCE_TYPE_N_EFF[EvidenceType.LITERATURE] * 0.5
        assert belief.alpha == pytest.approx(1.0 + half)
        assert belief.beta == pytest.approx(1.0 + half)

    def test_sequential_updates_accumulate(self, store, compound, target, edge):
        store.add_node(compound)
        store.add_node(target)
        store.add_edge(edge)
        ev1 = _make_evidence(
            support=SupportBucket.STRONG_SUPPORT,
            source_type=EvidenceType.GENETIC_MR,
            quality=1.0,
        )
        ev2 = _make_evidence(
            support=SupportBucket.MODERATE_CONTRADICT,
            source_type=EvidenceType.CLINICAL_PHASE2,
            quality=0.7,
        )
        store.update_edge_belief(compound.id, target.id, EdgeType.AFFECTS, ev1)
        belief = store.update_edge_belief(
            compound.id, target.id, EdgeType.AFFECTS, ev2
        )
        n1 = EVIDENCE_TYPE_N_EFF[EvidenceType.GENETIC_MR] * 1.0
        p1 = BUCKET_TO_P_OBS[SupportBucket.STRONG_SUPPORT]
        n2 = EVIDENCE_TYPE_N_EFF[EvidenceType.CLINICAL_PHASE2] * 0.7
        p2 = BUCKET_TO_P_OBS[SupportBucket.MODERATE_CONTRADICT]
        assert belief.alpha == pytest.approx(1.0 + n1 * p1 + n2 * p2)
        assert belief.beta == pytest.approx(
            1.0 + n1 * (1.0 - p1) + n2 * (1.0 - p2)
        )
        assert len(belief.evidence) == 2

    def test_all_evidence_types_have_n_eff(self):
        for et in EvidenceType:
            assert et in EVIDENCE_TYPE_N_EFF
            assert EVIDENCE_TYPE_N_EFF[et] > 0

    def test_all_buckets_have_p_obs(self):
        for b in SupportBucket:
            assert b in BUCKET_TO_P_OBS
            assert 0.0 < BUCKET_TO_P_OBS[b] < 1.0


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
        store.add_edge(GraphEdge(source_id="A", target_id="B", edge_type=EdgeType.AFFECTS))
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
        from src.graph.models import CausalChain, TrialArm
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
            PopulationNode(id="i1__unselected", name="P"),
        ]
        for n in nodes:
            store.add_node(n)
        arm = TrialArm(
            arm_id="arm1", compound_ids=["C1"], regimen_compound_id="C1",
            is_combination=False,
        )
        chain = CausalChain(
            arm_id="arm1", compound_id="C1",
            subgroup_population_id="i1__unselected",
            target_id="T1", mechanism_id="M1", biology_id="B1",
            indication_id="I1", endpoint_id="E1",
            outcome=TrialOutcome.SUCCESS,
        )
        trial = TrialSubgraph(
            trial_id="NCT123", phase="3",
            arms=[arm], chains=[chain],
            parent_population_id="i1__unselected",
        )
        sg = store.get_trial_subgraph(trial)
        # 7 nodes: 1 compound + 1 target + 1 mechanism + 1 biology + 1 indication + 1 endpoint + 1 population
        assert sg.number_of_nodes() == 7

    def test_set_and_get_trial_subgraph_sidecar(self, store):
        from src.graph.models import CausalChain, TrialArm
        arm = TrialArm(
            arm_id="arm1", compound_ids=["C1"], regimen_compound_id="C1",
        )
        chain = CausalChain(
            arm_id="arm1", compound_id="C1",
            subgroup_population_id="i__unselected",
            target_id="T1", mechanism_id="M1", biology_id="B1",
            indication_id="I1", endpoint_id="E1",
            outcome=TrialOutcome.SUCCESS,
        )
        ts = TrialSubgraph(
            trial_id="NCT_X", phase="3", arms=[arm], chains=[chain],
            parent_population_id="i__unselected",
        )
        store.set_trial_subgraph(ts)
        roundtrip = store.get_trial_subgraph_by_id("NCT_X")
        assert roundtrip.trial_id == "NCT_X"
        assert len(roundtrip.chains) == 1


# ── Persistence ──────────────────────────────────────────────────────────


class TestPersistence:
    def test_export_import_roundtrip(self, store, compound, target, edge, tmp_path):
        store.add_node(compound)
        store.add_node(target)
        store.add_edge(edge)
        ev = _make_evidence()
        store.update_edge_belief(compound.id, target.id, EdgeType.AFFECTS, ev)

        filepath = str(tmp_path / "snapshot.json")
        store.export_snapshot(filepath)

        store2 = GraphStore()
        store2.import_snapshot(filepath)

        assert store2.get_node("COMPOUND_IMA")["name"] == "Imatinib"
        belief = store2.get_edge_belief(compound.id, target.id, EdgeType.AFFECTS)
        assert belief.alpha > 1.0
        assert len(belief.evidence) == 1

    def test_export_creates_valid_json(self, store, compound, tmp_path):
        store.add_node(compound)
        filepath = str(tmp_path / "snapshot.json")
        store.export_snapshot(filepath)
        data = json.loads((tmp_path / "snapshot.json").read_text())
        # New snapshot wrapper: {"graph": <node_link_data>, "trial_subgraphs": {}}.
        assert "graph" in data
        assert "trial_subgraphs" in data
        assert "nodes" in data["graph"]

    def test_applied_attribution_set_roundtrips(self, store, tmp_path):
        store.applied_attribution_trial_ids.add("NCT00000001")
        store.applied_attribution_trial_ids.add("NCT00000002")
        filepath = str(tmp_path / "snapshot.json")
        store.export_snapshot(filepath)

        data = json.loads((tmp_path / "snapshot.json").read_text())
        assert data["applied_attribution_trial_ids"] == [
            "NCT00000001", "NCT00000002",
        ]

        store2 = GraphStore()
        store2.import_snapshot(filepath)
        assert store2.applied_attribution_trial_ids == {
            "NCT00000001", "NCT00000002",
        }

    def test_legacy_snapshot_with_no_trials_has_empty_attribution_set(
        self, store, tmp_path,
    ):
        legacy = {
            "graph": {
                "directed": True, "multigraph": True, "graph": {},
                "nodes": [], "edges": [],
            },
            "trial_subgraphs": {},
        }
        filepath = tmp_path / "legacy.json"
        filepath.write_text(json.dumps(legacy))
        store2 = GraphStore()
        store2.import_snapshot(str(filepath))
        assert store2.applied_attribution_trial_ids == set()

    def test_legacy_snapshot_backfills_attribution_from_trial_subgraphs(
        self, store, tmp_path,
    ):
        """Round-19: a snapshot written before applied_attribution_trial_ids
        existed has its set backfilled from trial_subgraphs.keys() on
        import. This avoids the first incremental --add-trials run
        re-attributing every existing trial."""
        from src.graph.models import CausalChain, TrialArm, TrialOutcome
        store.applied_attribution_trial_ids.add("NCT_X")  # pre-state irrelevant
        ts = TrialSubgraph(
            trial_id="NCT_LEGACY",
            phase="3",
            arms=[TrialArm(arm_id="solo", compound_ids=["c1"],
                           regimen_compound_id="c1")],
            chains=[CausalChain(
                arm_id="solo", compound_id="c1",
                subgroup_population_id="pop",
                target_id="t", mechanism_id="m", biology_id="b",
                indication_id="ind", endpoint_id="e",
                outcome=TrialOutcome.UNKNOWN,
            )],
            parent_population_id="pop",
        )
        store.set_trial_subgraph(ts)

        # Build the legacy-style payload by writing the snapshot and then
        # removing the applied_attribution_trial_ids key.
        filepath = tmp_path / "legacy.json"
        store.export_snapshot(str(filepath))
        raw = json.loads(filepath.read_text())
        del raw["applied_attribution_trial_ids"]
        filepath.write_text(json.dumps(raw))

        store2 = GraphStore()
        store2.import_snapshot(str(filepath))
        assert store2.applied_attribution_trial_ids == {"NCT_LEGACY"}


# ── Stats ────────────────────────────────────────────────────────────────


class TestStats:
    def test_stats_counts(self, store, compound, target, edge):
        store.add_node(compound)
        store.add_node(target)
        store.add_edge(edge)
        s = store.stats()
        assert s["node_count"] == 2
        assert s["edge_count"] == 1
        assert s["node_types"]["InterventionNode"] == 1
        assert s["node_types"]["TargetNode"] == 1
        assert s["edge_types"]["affects"] == 1
        assert s["total_evidence"] == 0

    def test_stats_evidence_count(self, store, compound, target, edge):
        store.add_node(compound)
        store.add_node(target)
        store.add_edge(edge)
        store.update_edge_belief(
            compound.id, target.id, EdgeType.AFFECTS, _make_evidence()
        )
        store.update_edge_belief(
            compound.id, target.id, EdgeType.AFFECTS, _make_evidence()
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
            edge_type=EdgeType.AFFECTS,
            belief=belief,
        )
        store.add_edge(edge)
        s = store.stats()
        assert len(s["high_conflict_edges"]) == 1
