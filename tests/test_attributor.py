"""Tests for the failure mode attributor."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.annotation.attributor import (
    AppliedEdgeUpdate,
    Attributor,
    _EDGE_TYPE_TO_SUBGRAPH_FIELDS,
    _PHASE_TO_EVIDENCE,
)
from src.annotation.taxonomy import (
    FAILURE_MODE_RULES,
    FailureClassification,
    FailureMode,
)
from src.graph.models import (
    CompoundNode,
    TargetNode,
    MechanismNode,
    BiologyNode,
    IndicationNode,
    EndpointNode,
    PopulationNode,
    EdgeBeliefState,
    EdgeType,
    EvidenceDirection,
    EvidenceRecord,
    EvidenceType,
    GraphEdge,
    Modality,
    MechanismType,
    EndpointType,
    RegulatoryStatus,
    TrialOutcome,
    TrialSubgraph,
)
from src.graph.store import GraphStore


# ── Helpers ─────────────────────────────────────────────────────────────


def _make_graph() -> GraphStore:
    """Build a minimal graph with one full causal chain."""
    g = GraphStore()
    g.add_node(CompoundNode(id="compound_nivo", name="Nivolumab", modality=Modality.ANTIBODY))
    g.add_node(TargetNode(id="target_pd1", name="PD-1", gene_symbol="PDCD1"))
    g.add_node(MechanismNode(id="mech_checkpoint", name="Immune checkpoint blockade", mechanism_type=MechanismType.INHIBITION))
    g.add_node(BiologyNode(id="bio_immune", name="T-cell activation"))
    g.add_node(IndicationNode(id="ind_melanoma", name="Melanoma"))
    g.add_node(EndpointNode(id="ep_os", name="Overall Survival", endpoint_type=EndpointType.PRIMARY, regulatory_status=RegulatoryStatus.ACCEPTED))
    g.add_node(PopulationNode(id="pop_pdl1", name="PD-L1 positive melanoma"))

    # Add edges along the causal chain
    g.add_edge(GraphEdge(source_id="compound_nivo", target_id="target_pd1", edge_type=EdgeType.BINDS_TO))
    g.add_edge(GraphEdge(source_id="target_pd1", target_id="mech_checkpoint", edge_type=EdgeType.MODULATES_VIA))
    g.add_edge(GraphEdge(source_id="mech_checkpoint", target_id="bio_immune", edge_type=EdgeType.MECHANISM_AFFECTS))
    g.add_edge(GraphEdge(source_id="bio_immune", target_id="ind_melanoma", edge_type=EdgeType.BIOLOGY_DRIVES))
    g.add_edge(GraphEdge(source_id="bio_immune", target_id="ep_os", edge_type=EdgeType.REFLECTS_BIOLOGY))
    g.add_edge(GraphEdge(source_id="ep_os", target_id="ind_melanoma", edge_type=EdgeType.ENDPOINT_CAPTURES))
    g.add_edge(GraphEdge(source_id="pop_pdl1", target_id="ind_melanoma", edge_type=EdgeType.RESPONDS_DIFFERENTLY))

    return g


def _make_trial() -> TrialSubgraph:
    return TrialSubgraph(
        trial_id="NCT01721772",
        compound_id="compound_nivo",
        target_id="target_pd1",
        mechanism_id="mech_checkpoint",
        biology_id="bio_immune",
        indication_id="ind_melanoma",
        endpoint_id="ep_os",
        population_id="pop_pdl1",
        outcome=TrialOutcome.SUCCESS,
        phase="3",
    )


def _make_classification(
    mode: FailureMode = FailureMode.NO_TARGET_ENGAGEMENT,
    confidence: float = 0.8,
    raw_edges: list | None = None,
) -> FailureClassification:
    if raw_edges is None:
        raw_edges = [
            {
                "edge_type": "binds_to",
                "direction": "weaken",
                "magnitude": 0.7,
                "reasoning": "No evidence of target engagement",
            }
        ]
    clf = FailureClassification(
        trial_id="NCT01721772",
        primary_failure_mode=mode,
        confidence=confidence,
        reasoning="Test reasoning",
    )
    clf._raw = {"edges_to_update": raw_edges}  # type: ignore[attr-defined]
    return clf


# ── Phase mapping ───────────────────────────────────────────────────────


class TestPhaseMapping:
    def test_phase3_maps_to_clinical_phase3(self):
        assert _PHASE_TO_EVIDENCE["3"] == EvidenceType.CLINICAL_PHASE3

    def test_phase2_maps_to_clinical_phase2(self):
        assert _PHASE_TO_EVIDENCE["2"] == EvidenceType.CLINICAL_PHASE2

    def test_phase1_maps_to_clinical_phase1(self):
        assert _PHASE_TO_EVIDENCE["1"] == EvidenceType.CLINICAL_PHASE1

    def test_phase2_3_maps_to_phase3(self):
        assert _PHASE_TO_EVIDENCE["2/3"] == EvidenceType.CLINICAL_PHASE3


# ── Edge type to subgraph fields ────────────────────────────────────────


class TestEdgeTypeMapping:
    def test_all_seven_edge_types_mapped(self):
        assert len(_EDGE_TYPE_TO_SUBGRAPH_FIELDS) == 7

    def test_binds_to_uses_compound_and_target(self):
        assert _EDGE_TYPE_TO_SUBGRAPH_FIELDS["binds_to"] == ("compound_id", "target_id")

    def test_biology_drives_uses_biology_and_indication(self):
        assert _EDGE_TYPE_TO_SUBGRAPH_FIELDS["biology_drives"] == ("biology_id", "indication_id")


# ── AppliedEdgeUpdate ───────────────────────────────────────────────────


class TestAppliedEdgeUpdate:
    def test_probability_change(self):
        pre = EdgeBeliefState(alpha=2.0, beta=2.0, evidence_count=1)
        post = EdgeBeliefState(alpha=2.0, beta=5.0, evidence_count=2)
        update = AppliedEdgeUpdate(
            source_id="a",
            target_id="b",
            edge_type=EdgeType.BINDS_TO,
            evidence=EvidenceRecord(
                source_id="trial1",
                source_type=EvidenceType.CLINICAL_PHASE3,
                quality_score=0.8,
                direction=EvidenceDirection.CONTRADICTING,
                magnitude=0.7,
                timestamp=datetime.now(timezone.utc),
            ),
            pre_update_belief=pre,
            post_update_belief=post,
        )
        expected = post.expected_probability - pre.expected_probability
        assert update.probability_change == pytest.approx(expected)
        assert update.probability_change < 0  # contradicting should lower


# ── Attributor.attribute() ──────────────────────────────────────────────


class TestAttribute:
    def test_single_edge_update(self):
        graph = _make_graph()
        attributor = Attributor(graph)
        trial = _make_trial()
        clf = _make_classification(
            raw_edges=[{
                "edge_type": "binds_to",
                "direction": "weaken",
                "magnitude": 0.7,
                "reasoning": "No target engagement",
            }]
        )
        updates = attributor.attribute(clf, trial)
        assert len(updates) == 1
        assert updates[0].edge_type == EdgeType.BINDS_TO
        assert updates[0].source_id == "compound_nivo"
        assert updates[0].target_id == "target_pd1"
        assert updates[0].probability_change < 0  # weakening

    def test_strengthen_direction(self):
        graph = _make_graph()
        attributor = Attributor(graph)
        trial = _make_trial()
        clf = _make_classification(
            mode=FailureMode.TARGET_ENGAGED_BIOLOGY_NOT_MOVED,
            raw_edges=[{
                "edge_type": "binds_to",
                "direction": "strengthen",
                "magnitude": 0.8,
                "reasoning": "Target confirmed engaged",
            }]
        )
        updates = attributor.attribute(clf, trial)
        assert len(updates) == 1
        assert updates[0].probability_change > 0

    def test_neutral_direction(self):
        graph = _make_graph()
        attributor = Attributor(graph)
        trial = _make_trial()
        clf = _make_classification(
            raw_edges=[{
                "edge_type": "binds_to",
                "direction": "neutral",
                "magnitude": 0.5,
                "reasoning": "Ambiguous data",
            }]
        )
        updates = attributor.attribute(clf, trial)
        assert len(updates) == 1
        assert updates[0].evidence.direction == EvidenceDirection.AMBIGUOUS

    def test_multiple_edges(self):
        graph = _make_graph()
        attributor = Attributor(graph)
        trial = _make_trial()
        clf = _make_classification(
            raw_edges=[
                {"edge_type": "binds_to", "direction": "weaken", "magnitude": 0.7},
                {"edge_type": "modulates_via", "direction": "weaken", "magnitude": 0.6},
                {"edge_type": "mechanism_affects", "direction": "neutral", "magnitude": 0.3},
            ]
        )
        updates = attributor.attribute(clf, trial)
        assert len(updates) == 3

    def test_unknown_edge_type_skipped(self):
        graph = _make_graph()
        attributor = Attributor(graph)
        trial = _make_trial()
        clf = _make_classification(
            raw_edges=[{
                "edge_type": "nonexistent_edge",
                "direction": "weaken",
                "magnitude": 0.5,
            }]
        )
        updates = attributor.attribute(clf, trial)
        assert len(updates) == 0

    def test_unknown_node_skipped(self):
        graph = _make_graph()
        attributor = Attributor(graph)
        trial = TrialSubgraph(
            trial_id="NCT99999999",
            compound_id="UNKNOWN",
            target_id="target_pd1",
            mechanism_id="UNKNOWN",
            biology_id="UNKNOWN",
            indication_id="UNKNOWN",
            endpoint_id="UNKNOWN",
            population_id="UNKNOWN",
            outcome=TrialOutcome.UNKNOWN,
            phase="3",
        )
        clf = _make_classification(
            raw_edges=[{
                "edge_type": "binds_to",
                "direction": "weaken",
                "magnitude": 0.5,
            }]
        )
        updates = attributor.attribute(clf, trial)
        assert len(updates) == 0

    def test_edge_not_in_graph_skipped(self):
        graph = GraphStore()
        graph.add_node(CompoundNode(id="c1", name="Drug", modality=Modality.SMALL_MOLECULE))
        graph.add_node(TargetNode(id="t1", name="Target", gene_symbol="TP53"))
        # No edge added between them
        attributor = Attributor(graph)
        trial = TrialSubgraph(
            trial_id="NCT00000001",
            compound_id="c1",
            target_id="t1",
            mechanism_id="UNKNOWN",
            biology_id="UNKNOWN",
            indication_id="UNKNOWN",
            endpoint_id="UNKNOWN",
            population_id="UNKNOWN",
            outcome=TrialOutcome.FAILURE,
            phase="2",
        )
        clf = _make_classification(
            raw_edges=[{
                "edge_type": "binds_to",
                "direction": "weaken",
                "magnitude": 0.7,
            }]
        )
        updates = attributor.attribute(clf, trial)
        assert len(updates) == 0

    def test_evidence_type_from_phase(self):
        graph = _make_graph()
        attributor = Attributor(graph)
        trial = _make_trial()
        trial.phase = "2"
        clf = _make_classification(
            raw_edges=[{
                "edge_type": "binds_to",
                "direction": "weaken",
                "magnitude": 0.5,
            }]
        )
        updates = attributor.attribute(clf, trial)
        assert len(updates) == 1
        assert updates[0].evidence.source_type == EvidenceType.CLINICAL_PHASE2

    def test_magnitude_scaled_by_confidence(self):
        graph = _make_graph()
        attributor = Attributor(graph)
        trial = _make_trial()
        clf = _make_classification(
            confidence=0.6,
            raw_edges=[{
                "edge_type": "binds_to",
                "direction": "weaken",
                "magnitude": 0.8,
            }]
        )
        updates = attributor.attribute(clf, trial)
        assert updates[0].evidence.magnitude == pytest.approx(0.8 * 0.6)

    def test_empty_raw_edges(self):
        graph = _make_graph()
        attributor = Attributor(graph)
        trial = _make_trial()
        clf = _make_classification(raw_edges=[])
        updates = attributor.attribute(clf, trial)
        assert len(updates) == 0


# ── Taxonomy cross-check ────────────────────────────────────────────────


class TestTaxonomyCrossCheck:
    def test_weaken_overridden_when_taxonomy_says_strengthen(self):
        """If classifier says weaken but taxonomy says strengthen, result is ambiguous."""
        graph = _make_graph()
        attributor = Attributor(graph)
        trial = _make_trial()
        # TARGET_ENGAGED_BIOLOGY_NOT_MOVED strengthens binds_to
        clf = _make_classification(
            mode=FailureMode.TARGET_ENGAGED_BIOLOGY_NOT_MOVED,
            raw_edges=[{
                "edge_type": "binds_to",
                "direction": "weaken",  # contradicts taxonomy
                "magnitude": 0.7,
            }]
        )
        updates = attributor.attribute(clf, trial)
        assert len(updates) == 1
        assert updates[0].evidence.direction == EvidenceDirection.AMBIGUOUS

    def test_strengthen_overridden_when_taxonomy_says_weaken(self):
        """If classifier says strengthen but taxonomy says weaken, result is ambiguous."""
        graph = _make_graph()
        attributor = Attributor(graph)
        trial = _make_trial()
        # NO_TARGET_ENGAGEMENT weakens binds_to
        clf = _make_classification(
            mode=FailureMode.NO_TARGET_ENGAGEMENT,
            raw_edges=[{
                "edge_type": "binds_to",
                "direction": "strengthen",  # contradicts taxonomy
                "magnitude": 0.7,
            }]
        )
        updates = attributor.attribute(clf, trial)
        assert len(updates) == 1
        assert updates[0].evidence.direction == EvidenceDirection.AMBIGUOUS

    def test_consistent_direction_preserved(self):
        """When classifier and taxonomy agree, direction is preserved."""
        graph = _make_graph()
        attributor = Attributor(graph)
        trial = _make_trial()
        # NO_TARGET_ENGAGEMENT weakens binds_to — classifier also says weaken
        clf = _make_classification(
            mode=FailureMode.NO_TARGET_ENGAGEMENT,
            raw_edges=[{
                "edge_type": "binds_to",
                "direction": "weaken",
                "magnitude": 0.7,
            }]
        )
        updates = attributor.attribute(clf, trial)
        assert len(updates) == 1
        assert updates[0].evidence.direction == EvidenceDirection.CONTRADICTING


# ── Attributor.apply_updates() ──────────────────────────────────────────


class TestApplyUpdates:
    def test_empty_updates(self):
        graph = _make_graph()
        attributor = Attributor(graph)
        result = attributor.apply_updates([])
        assert result["edges_updated"] == 0
        assert result["largest_changes"] == []

    def test_summary_with_updates(self):
        graph = _make_graph()
        attributor = Attributor(graph)
        trial = _make_trial()
        clf = _make_classification(
            raw_edges=[
                {"edge_type": "binds_to", "direction": "weaken", "magnitude": 0.9},
                {"edge_type": "modulates_via", "direction": "weaken", "magnitude": 0.5},
            ]
        )
        updates = attributor.attribute(clf, trial)
        result = attributor.apply_updates(updates)
        assert result["edges_updated"] == 2
        assert len(result["largest_changes"]) == 2
        # Sorted by largest absolute change first
        assert abs(result["largest_changes"][0]["change"]) >= abs(result["largest_changes"][1]["change"])

    def test_summary_caps_at_five(self):
        graph = _make_graph()
        attributor = Attributor(graph)
        trial = _make_trial()
        clf = _make_classification(
            raw_edges=[
                {"edge_type": "binds_to", "direction": "weaken", "magnitude": 0.9},
                {"edge_type": "modulates_via", "direction": "weaken", "magnitude": 0.8},
                {"edge_type": "mechanism_affects", "direction": "weaken", "magnitude": 0.7},
                {"edge_type": "biology_drives", "direction": "strengthen", "magnitude": 0.6},
                {"edge_type": "reflects_biology", "direction": "neutral", "magnitude": 0.5},
                {"edge_type": "endpoint_captures", "direction": "weaken", "magnitude": 0.4},
                {"edge_type": "responds_differently", "direction": "strengthen", "magnitude": 0.3},
            ]
        )
        updates = attributor.attribute(clf, trial)
        result = attributor.apply_updates(updates)
        assert len(result["largest_changes"]) <= 5


# ── Belief state mutation ───────────────────────────────────────────────


class TestBeliefMutation:
    def test_graph_beliefs_actually_updated(self):
        graph = _make_graph()
        pre = graph.get_edge_belief("compound_nivo", "target_pd1", EdgeType.BINDS_TO)
        pre_prob = pre.expected_probability

        attributor = Attributor(graph)
        trial = _make_trial()
        clf = _make_classification(
            raw_edges=[{
                "edge_type": "binds_to",
                "direction": "weaken",
                "magnitude": 0.9,
            }]
        )
        attributor.attribute(clf, trial)

        post = graph.get_edge_belief("compound_nivo", "target_pd1", EdgeType.BINDS_TO)
        assert post.expected_probability < pre_prob
        assert len(post.evidence) > len(pre.evidence)

    def test_multiple_updates_compound(self):
        """Two sequential attributions should compound."""
        graph = _make_graph()
        attributor = Attributor(graph)
        trial = _make_trial()

        clf1 = _make_classification(
            raw_edges=[{"edge_type": "binds_to", "direction": "weaken", "magnitude": 0.7}]
        )
        attributor.attribute(clf1, trial)
        mid = graph.get_edge_belief("compound_nivo", "target_pd1", EdgeType.BINDS_TO)

        clf2 = _make_classification(
            raw_edges=[{"edge_type": "binds_to", "direction": "weaken", "magnitude": 0.7}]
        )
        attributor.attribute(clf2, trial)
        post = graph.get_edge_belief("compound_nivo", "target_pd1", EdgeType.BINDS_TO)

        assert post.expected_probability < mid.expected_probability
