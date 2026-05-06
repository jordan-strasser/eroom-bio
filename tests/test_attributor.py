"""Tests for chain-aware attribution.

The attributor takes a classifier-emitted edge update with free-text
source/target entity names and routes it to the specific CausalChain
whose canonical ids match. The canonical example: in CheckMate 067 the
classifier emits both ``Nivolumab → PD-1`` and ``Ipilimumab → CTLA-4``
binds_to updates; the attributor must route each to the right chain
rather than blindly using the first compound/target tuple.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from src.annotation.attributor import (
    AppliedEdgeUpdate,
    Attributor,
    _PHASE_TO_EVIDENCE,
    _UNROUTED_LOG_PATH,
    _norm_name,
)
from src.annotation.taxonomy import (
    FailureClassification,
    FailureMode,
)
from src.graph.models import (
    BiologyNode,
    CausalChain,
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
    TrialArm,
    TrialOutcome,
    TrialSubgraph,
)
from src.graph.store import GraphStore
from src.inference.beliefs import SupportBucket


# ── Helpers ─────────────────────────────────────────────────────────────


def _seed_combo_trial_graph() -> tuple[GraphStore, TrialSubgraph]:
    """Set up a CheckMate-067-shaped graph + trial subgraph.

    Three arms (nivo mono, ipi mono, combo). One subgroup population.
    Two distinct binds_to edges: nivolumab → PD-1, ipilimumab → CTLA-4.
    """
    g = GraphStore()
    g.add_node(CompoundNode(id="nivolumab", name="Nivolumab", modality=Modality.ANTIBODY))
    g.add_node(CompoundNode(id="ipilimumab", name="Ipilimumab", modality=Modality.ANTIBODY))
    g.add_node(CompoundNode(id="ipilimumab+nivolumab", name="ipi+nivo", modality=Modality.OTHER))
    g.add_node(TargetNode(id="ENSG00000188389", name="Programmed cell death 1", gene_symbol="PD-1"))
    g.add_node(TargetNode(id="ENSG00000163599", name="Cytotoxic T-lymphocyte protein 4", gene_symbol="CTLA4"))
    g.add_node(MechanismNode(id="checkpoint_blockade", name="checkpoint blockade", mechanism_type=MechanismType.ANTAGONISM))
    g.add_node(BiologyNode(id="R-HSA-389948", name="PD-1 signaling"))
    g.add_node(IndicationNode(id="melanoma", name="Melanoma"))
    g.add_node(EndpointNode(
        id="PFS_melanoma", name="PFS (Melanoma)",
        endpoint_type=EndpointType.PRIMARY,
        regulatory_status=RegulatoryStatus.ACCEPTED,
    ))
    g.add_node(PopulationNode(id="melanoma__unselected", name="All patients (Melanoma)"))

    # Per-compound binds_to edges with their own beliefs.
    g.add_edge(GraphEdge(source_id="nivolumab", target_id="ENSG00000188389",
                        edge_type=EdgeType.BINDS_TO,
                        belief=EdgeBeliefState(alpha=4.0, beta=1.0)))
    g.add_edge(GraphEdge(source_id="ipilimumab", target_id="ENSG00000163599",
                        edge_type=EdgeType.BINDS_TO,
                        belief=EdgeBeliefState(alpha=4.0, beta=1.0)))

    arms = [
        TrialArm(arm_id="nivo_only", compound_ids=["nivolumab"],
                 regimen_compound_id="nivolumab"),
        TrialArm(arm_id="combo", compound_ids=["nivolumab", "ipilimumab"],
                 regimen_compound_id="ipilimumab+nivolumab", is_combination=True),
        TrialArm(arm_id="ipi_only", compound_ids=["ipilimumab"],
                 regimen_compound_id="ipilimumab"),
    ]
    target_for_arm = {
        "nivo_only": "ENSG00000188389",
        "combo": "ENSG00000188389",
        "ipi_only": "ENSG00000163599",
    }
    chains = [
        CausalChain(
            arm_id=a.arm_id, compound_id=a.regimen_compound_id,
            subgroup_population_id="melanoma__unselected",
            target_id=target_for_arm[a.arm_id], mechanism_id="checkpoint_blockade",
            biology_id="R-HSA-389948", indication_id="melanoma",
            endpoint_id="PFS_melanoma", outcome=TrialOutcome.UNKNOWN,
        )
        for a in arms
    ]
    ts = TrialSubgraph(
        trial_id="NCT_TEST", phase="3", arms=arms, chains=chains,
        parent_population_id="melanoma__unselected",
    )
    g.set_trial_subgraph(ts)
    return g, ts


def _make_classification(raw_edges: list[dict]) -> FailureClassification:
    clf = FailureClassification(
        trial_id="NCT_TEST",
        primary_failure_mode=FailureMode.EFFICACY_IN_SUBGROUP_ONLY,
        confidence=0.7,
        evidence_quotes=["test"],
    )
    clf._raw = {"edges_to_update": raw_edges}  # type: ignore[attr-defined]
    return clf


@pytest.fixture(autouse=True)
def _clean_unrouted_log():
    """Wipe the unrouted-attribution audit log between tests so each
    test sees only its own emissions.
    """
    if _UNROUTED_LOG_PATH.exists():
        _UNROUTED_LOG_PATH.unlink()
    yield
    if _UNROUTED_LOG_PATH.exists():
        _UNROUTED_LOG_PATH.unlink()


# ── Phase mapping (preserved from original test) ────────────────────────


class TestPhaseMapping:
    def test_phase3_maps_to_clinical_phase3(self):
        assert _PHASE_TO_EVIDENCE["3"] == EvidenceType.CLINICAL_PHASE3

    def test_phase2_maps_to_clinical_phase2(self):
        assert _PHASE_TO_EVIDENCE["2"] == EvidenceType.CLINICAL_PHASE2

    def test_phase1_maps_to_clinical_phase1(self):
        assert _PHASE_TO_EVIDENCE["1"] == EvidenceType.CLINICAL_PHASE1

    def test_phase2_3_maps_to_phase3(self):
        assert _PHASE_TO_EVIDENCE["2/3"] == EvidenceType.CLINICAL_PHASE3


# ── Name normalization (the hyphen-strip fix that unblocks CTLA-4) ──────


class TestNameNormalization:
    def test_strips_hyphens(self):
        assert _norm_name("CTLA-4") == _norm_name("CTLA4") == "ctla4"

    def test_strips_spaces_and_lowercases(self):
        assert _norm_name("PD L1") == _norm_name("PD-L1") == _norm_name("pdl1") == "pdl1"

    def test_empty_returns_empty(self):
        assert _norm_name("") == ""


# ── Chain-aware routing ─────────────────────────────────────────────────


class TestChainAwareRouting:
    def test_nivo_pd1_update_routes_to_nivo_chain(self):
        g, ts = _seed_combo_trial_graph()
        clf = _make_classification([
            {"edge_type": "binds_to", "source_entity": "Nivolumab",
             "target_entity": "PD-1", "support": "moderate_support"},
        ])
        updates = Attributor(g).attribute(clf, ts)
        assert len(updates) == 1
        assert updates[0].source_id == "nivolumab"
        assert updates[0].target_id == "ENSG00000188389"

    def test_ipi_ctla4_update_routes_to_ipi_chain_not_pd1(self):
        """The original misrouting bug: Ipilimumab→CTLA-4 must NOT land on
        the nivolumab→PD-1 binds_to edge in a multi-arm trial."""
        g, ts = _seed_combo_trial_graph()
        clf = _make_classification([
            {"edge_type": "binds_to", "source_entity": "Ipilimumab",
             "target_entity": "CTLA-4", "support": "moderate_support"},
        ])
        updates = Attributor(g).attribute(clf, ts)
        assert len(updates) == 1
        assert updates[0].source_id == "ipilimumab"
        assert updates[0].target_id == "ENSG00000163599"

    def test_both_binds_updates_route_independently(self):
        g, ts = _seed_combo_trial_graph()
        clf = _make_classification([
            {"edge_type": "binds_to", "source_entity": "Nivolumab",
             "target_entity": "PD-1", "support": "moderate_support"},
            {"edge_type": "binds_to", "source_entity": "Ipilimumab",
             "target_entity": "CTLA-4", "support": "moderate_support"},
        ])
        updates = Attributor(g).attribute(clf, ts)
        routes = {(u.source_id, u.target_id) for u in updates}
        assert ("nivolumab", "ENSG00000188389") in routes
        assert ("ipilimumab", "ENSG00000163599") in routes

    def test_off_trial_entity_is_dropped_as_hallucination(self):
        """Classifier emits an entity (Pembrolizumab) that is nowhere in
        the trial subgraph. Candidate (compound, target) pairs exist but
        none match the classifier names — graph-build trusts only
        trial-derived entities, so this is rejected as a hallucination
        rather than misrouted to whichever pair happened to be present.
        """
        g, ts = _seed_combo_trial_graph()
        clf = _make_classification([
            {"edge_type": "binds_to", "source_entity": "Pembrolizumab",
             "target_entity": "PD-1", "support": "moderate_support"},
        ])
        updates = Attributor(g).attribute(clf, ts)
        assert updates == []
        assert _UNROUTED_LOG_PATH.exists()
        records = [
            json.loads(line) for line in _UNROUTED_LOG_PATH.read_text().splitlines()
        ]
        hallucinations = [r for r in records if r["source_entity"] == "Pembrolizumab"]
        assert hallucinations, "expected the off-trial entity to be logged"
        assert all(r["reason"] == "entity_not_in_trial" for r in hallucinations)

    def test_sparse_chain_logs_no_chain_match_not_hallucination(self):
        """When the trial subgraph has UNKNOWN placeholders so no
        candidate (src, tgt) pairs can be formed for the requested edge
        type, the update is still dropped — but logged as
        ``no_chain_match`` (chain too sparse to verify) rather than
        ``entity_not_in_trial`` (classifier hallucinated). The
        distinction matters: hallucination is a model-quality signal,
        sparse chains are an ingestion-coverage signal.
        """
        g = GraphStore()
        g.add_node(CompoundNode(id="nivolumab", name="Nivolumab", modality=Modality.ANTIBODY))
        g.add_node(IndicationNode(id="melanoma", name="Melanoma"))
        g.add_node(PopulationNode(id="melanoma__unselected", name="All patients"))
        # Note: NO TargetNode and NO mechanism_affects-relevant nodes —
        # the trial chain will reference "UNKNOWN" for those.

        arms = [TrialArm(arm_id="solo", compound_ids=["nivolumab"],
                         regimen_compound_id="nivolumab")]
        chains = [
            CausalChain(
                arm_id="solo", compound_id="nivolumab",
                subgroup_population_id="melanoma__unselected",
                target_id="UNKNOWN", mechanism_id="UNKNOWN",
                biology_id="UNKNOWN", indication_id="melanoma",
                endpoint_id="UNKNOWN", outcome=TrialOutcome.UNKNOWN,
            )
        ]
        ts = TrialSubgraph(
            trial_id="NCT_SPARSE", phase="3", arms=arms, chains=chains,
            parent_population_id="melanoma__unselected",
        )
        g.set_trial_subgraph(ts)

        clf = FailureClassification(
            trial_id="NCT_SPARSE",
            primary_failure_mode=FailureMode.EFFICACY_IN_SUBGROUP_ONLY,
            confidence=0.7, evidence_quotes=["test"],
        )
        clf._raw = {"edges_to_update": [
            {"edge_type": "binds_to", "source_entity": "Nivolumab",
             "target_entity": "PD-1", "support": "moderate_support"},
        ]}  # type: ignore[attr-defined]

        updates = Attributor(g).attribute(clf, ts)
        assert updates == []
        records = [
            json.loads(line) for line in _UNROUTED_LOG_PATH.read_text().splitlines()
        ]
        assert records
        assert all(r["reason"] == "no_chain_match" for r in records)

    def test_composed_of_updates_skipped(self):
        g, ts = _seed_combo_trial_graph()
        clf = _make_classification([
            {"edge_type": "composed_of", "source_entity": "ipi+nivo",
             "target_entity": "Nivolumab", "support": "moderate_support"},
        ])
        # composed_of is a structural edge — classifier-driven updates on it
        # are dropped silently (not applied, not logged as unrouted).
        updates = Attributor(g).attribute(clf, ts)
        assert updates == []


# ── AppliedEdgeUpdate ───────────────────────────────────────────────────


class TestAppliedEdgeUpdate:
    def test_probability_change(self):
        pre = EdgeBeliefState(alpha=2.0, beta=2.0)
        post = EdgeBeliefState(alpha=2.0, beta=5.0)
        update = AppliedEdgeUpdate(
            source_id="a", target_id="b",
            edge_type=EdgeType.BINDS_TO,
            evidence=EvidenceRecord(
                source_id="trial1",
                source_type=EvidenceType.CLINICAL_PHASE3,
                support=SupportBucket.MODERATE_CONTRADICT.value,
                quality_score=0.8,
                timestamp=datetime.now(timezone.utc),
            ),
            pre_update_belief=pre,
            post_update_belief=post,
        )
        # E[p] went from 0.5 to 2/7 ≈ 0.286 → Δ ≈ -0.214
        assert update.probability_change < 0
