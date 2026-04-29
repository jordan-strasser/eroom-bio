"""Tests for the inference layer: beliefs, updater, propagation."""

from __future__ import annotations

import math

import pytest

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
from src.annotation.taxonomy import FailureClassification, FailureMode
from src.inference.beliefs import BetaBelief
from src.inference.updater import (
    EVIDENCE_TYPE_WEIGHTS,
    QUALITY_WEIGHTS,
    EdgeUpdate,
    EvidenceUpdater,
)
from src.inference.propagation import (
    PropagatedUpdate,
    _COMPATIBLE_TYPES,
    propagate,
)


# ── Helpers ─────────────────────────────────────────────────────────────


def _make_chain_graph() -> GraphStore:
    """Build a graph: compound -> target -> mechanism -> biology -> indication
                                                         biology -> endpoint -> indication
                                                         population -> indication
    """
    g = GraphStore()
    g.add_node(CompoundNode(id="c1", name="DrugA", modality=Modality.SMALL_MOLECULE))
    g.add_node(TargetNode(id="t1", name="TargetA", gene_symbol="TGTA"))
    g.add_node(MechanismNode(id="m1", name="MechA", mechanism_type=MechanismType.INHIBITION))
    g.add_node(BiologyNode(id="b1", name="BioA"))
    g.add_node(IndicationNode(id="i1", name="DiseaseA"))
    g.add_node(EndpointNode(id="e1", name="EndpointA", endpoint_type=EndpointType.PRIMARY, regulatory_status=RegulatoryStatus.ACCEPTED))
    g.add_node(PopulationNode(id="p1", name="PopA"))

    g.add_edge(GraphEdge(source_id="c1", target_id="t1", edge_type=EdgeType.BINDS_TO))
    g.add_edge(GraphEdge(source_id="t1", target_id="m1", edge_type=EdgeType.MODULATES_VIA))
    g.add_edge(GraphEdge(source_id="m1", target_id="b1", edge_type=EdgeType.MECHANISM_AFFECTS))
    g.add_edge(GraphEdge(source_id="b1", target_id="i1", edge_type=EdgeType.BIOLOGY_DRIVES))
    g.add_edge(GraphEdge(source_id="b1", target_id="e1", edge_type=EdgeType.REFLECTS_BIOLOGY))
    g.add_edge(GraphEdge(source_id="e1", target_id="i1", edge_type=EdgeType.ENDPOINT_CAPTURES))
    g.add_edge(GraphEdge(source_id="p1", target_id="i1", edge_type=EdgeType.RESPONDS_DIFFERENTLY))

    return g


def _make_trial() -> TrialSubgraph:
    return TrialSubgraph(
        trial_id="NCT_TEST",
        compound_id="c1",
        target_id="t1",
        mechanism_id="m1",
        biology_id="b1",
        indication_id="i1",
        endpoint_id="e1",
        population_id="p1",
        outcome=TrialOutcome.FAILURE,
        phase="3",
    )


def _make_classification(
    mode: FailureMode = FailureMode.NO_TARGET_ENGAGEMENT,
    confidence: float = 0.8,
    raw_edges: list | None = None,
) -> FailureClassification:
    if raw_edges is None:
        raw_edges = [
            {"edge_type": "binds_to", "direction": "weaken", "magnitude": 0.7},
        ]
    clf = FailureClassification(
        trial_id="NCT_TEST",
        primary_failure_mode=mode,
        confidence=confidence,
        reasoning="Test",
    )
    clf._raw = {"edges_to_update": raw_edges}  # type: ignore[attr-defined]
    return clf


# ============================================================
# BetaBelief
# ============================================================


class TestBetaBeliefProperties:
    def test_uniform_prior_mean(self):
        b = BetaBelief(1.0, 1.0)
        assert b.mean == pytest.approx(0.5)

    def test_mean_formula(self):
        b = BetaBelief(3.0, 7.0)
        assert b.mean == pytest.approx(3.0 / 10.0)

    def test_variance_formula(self):
        b = BetaBelief(3.0, 7.0)
        expected = (3.0 * 7.0) / (100.0 * 11.0)
        assert b.variance == pytest.approx(expected)

    def test_evidence_strength_uniform(self):
        b = BetaBelief(1.0, 1.0)
        assert b.evidence_strength == pytest.approx(0.0)

    def test_evidence_strength_with_data(self):
        b = BetaBelief(5.0, 3.0)
        assert b.evidence_strength == pytest.approx(6.0)

    def test_conflict_score_high_for_balanced_evidence(self):
        """Beta(15,12) has lots of evidence but mean near 0.5 -> high conflict."""
        b = BetaBelief(15.0, 12.0)
        assert b.conflict_score > 5.0

    def test_conflict_score_low_for_lopsided(self):
        """Beta(15,2) has clear direction -> low conflict."""
        b = BetaBelief(15.0, 2.0)
        assert b.conflict_score < 5.0

    def test_conflict_score_zero_for_no_evidence(self):
        b = BetaBelief(1.0, 1.0)
        assert b.conflict_score == pytest.approx(0.0)

    def test_credible_interval_95(self):
        b = BetaBelief(10.0, 10.0)
        lo, hi = b.credible_interval(0.95)
        assert lo < 0.5 < hi
        assert lo > 0.0
        assert hi < 1.0

    def test_credible_interval_narrows_with_evidence(self):
        weak = BetaBelief(2.0, 2.0)
        strong = BetaBelief(20.0, 20.0)
        lo_w, hi_w = weak.credible_interval()
        lo_s, hi_s = strong.credible_interval()
        assert (hi_s - lo_s) < (hi_w - lo_w)

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            BetaBelief(0.0, 1.0)
        with pytest.raises(ValueError):
            BetaBelief(1.0, -1.0)


class TestBetaBeliefUpdate:
    def test_supporting_increases_alpha(self):
        b = BetaBelief(1.0, 1.0)
        updated = b.update("supporting", magnitude=2.0, weight=1.0)
        assert updated.alpha == pytest.approx(3.0)
        assert updated.beta == pytest.approx(1.0)

    def test_contradicting_increases_beta(self):
        b = BetaBelief(1.0, 1.0)
        updated = b.update("contradicting", magnitude=2.0, weight=1.0)
        assert updated.alpha == pytest.approx(1.0)
        assert updated.beta == pytest.approx(3.0)

    def test_ambiguous_increases_both(self):
        b = BetaBelief(1.0, 1.0)
        updated = b.update("ambiguous", magnitude=2.0, weight=1.0)
        assert updated.alpha == pytest.approx(1.0 + 2.0 * 0.3)
        assert updated.beta == pytest.approx(1.0 + 2.0 * 0.3)

    def test_weight_scales_delta(self):
        b = BetaBelief(1.0, 1.0)
        updated = b.update("supporting", magnitude=1.0, weight=5.0)
        assert updated.alpha == pytest.approx(6.0)  # 1 + 1*5

    def test_update_by_hand(self):
        """Manual verification: Phase 3 contradicting evidence.
        Start: Beta(1,1). magnitude=0.7, weight=5.0 (Phase 3).
        delta = 5.0 * 0.7 = 3.5. contradicting -> beta += 3.5.
        Result: Beta(1, 4.5). mean = 1/5.5 ≈ 0.1818.
        """
        b = BetaBelief(1.0, 1.0)
        updated = b.update("contradicting", magnitude=0.7, weight=5.0)
        assert updated.alpha == pytest.approx(1.0)
        assert updated.beta == pytest.approx(4.5)
        assert updated.mean == pytest.approx(1.0 / 5.5)

    def test_original_unchanged(self):
        b = BetaBelief(1.0, 1.0)
        _ = b.update("supporting", magnitude=2.0)
        assert b.alpha == pytest.approx(1.0)
        assert b.beta == pytest.approx(1.0)


class TestBetaBeliefPdf:
    def test_pdf_at_mode(self):
        b = BetaBelief(5.0, 2.0)
        # Mode of Beta(a,b) = (a-1)/(a+b-2) = 4/5 = 0.8
        pdf_at_mode = b.pdf(0.8)
        pdf_at_half = b.pdf(0.5)
        assert pdf_at_mode > pdf_at_half

    def test_uniform_pdf_is_one(self):
        b = BetaBelief(1.0, 1.0)
        assert b.pdf(0.3) == pytest.approx(1.0)
        assert b.pdf(0.9) == pytest.approx(1.0)


class TestBetaBeliefKL:
    def test_kl_self_is_zero(self):
        b = BetaBelief(3.0, 5.0)
        assert b.kl_divergence(b) == pytest.approx(0.0, abs=1e-10)

    def test_kl_nonnegative(self):
        a = BetaBelief(2.0, 5.0)
        b = BetaBelief(5.0, 2.0)
        assert a.kl_divergence(b) > 0

    def test_kl_asymmetric(self):
        a = BetaBelief(2.0, 5.0)
        b = BetaBelief(3.0, 8.0)
        assert a.kl_divergence(b) != pytest.approx(b.kl_divergence(a))

    def test_kl_close_distributions_small(self):
        a = BetaBelief(10.0, 10.0)
        b = BetaBelief(10.5, 10.5)
        assert a.kl_divergence(b) < 0.1

    def test_kl_distant_distributions_large(self):
        a = BetaBelief(1.0, 10.0)
        b = BetaBelief(10.0, 1.0)
        assert a.kl_divergence(b) > 1.0


# ============================================================
# EvidenceUpdater
# ============================================================


class TestEvidenceUpdaterWeights:
    def test_phase3_is_highest_clinical(self):
        assert EVIDENCE_TYPE_WEIGHTS[EvidenceType.CLINICAL_PHASE3] == 5.0

    def test_genetic_mr_higher_than_gwas(self):
        assert EVIDENCE_TYPE_WEIGHTS[EvidenceType.GENETIC_MR] > EVIDENCE_TYPE_WEIGHTS[EvidenceType.GENETIC_GWAS]

    def test_quality_weights_ordered(self):
        assert QUALITY_WEIGHTS["high"] > QUALITY_WEIGHTS["medium"] > QUALITY_WEIGHTS["low"]


class TestProcessTrialOutcome:
    def test_single_edge_update(self):
        graph = _make_chain_graph()
        updater = EvidenceUpdater(graph)
        trial = _make_trial()
        clf = _make_classification(
            raw_edges=[{"edge_type": "binds_to", "direction": "weaken", "magnitude": 0.7}]
        )
        updates = updater.process_trial_outcome(trial, clf)
        assert len(updates) == 1
        assert updates[0].edge_type == EdgeType.BINDS_TO
        assert updates[0].direction == EvidenceDirection.CONTRADICTING
        assert updates[0].probability_change < 0

    def test_multiple_edges(self):
        graph = _make_chain_graph()
        updater = EvidenceUpdater(graph)
        trial = _make_trial()
        clf = _make_classification(
            raw_edges=[
                {"edge_type": "binds_to", "direction": "weaken", "magnitude": 0.7},
                {"edge_type": "modulates_via", "direction": "weaken", "magnitude": 0.5},
            ]
        )
        updates = updater.process_trial_outcome(trial, clf)
        assert len(updates) == 2

    def test_unknown_node_skipped(self):
        graph = _make_chain_graph()
        updater = EvidenceUpdater(graph)
        trial = TrialSubgraph(
            trial_id="NCT_TEST",
            compound_id="UNKNOWN",
            target_id="t1",
            mechanism_id="UNKNOWN",
            biology_id="UNKNOWN",
            indication_id="UNKNOWN",
            endpoint_id="UNKNOWN",
            population_id="UNKNOWN",
            outcome=TrialOutcome.FAILURE,
            phase="3",
        )
        clf = _make_classification(
            raw_edges=[{"edge_type": "binds_to", "direction": "weaken", "magnitude": 0.7}]
        )
        updates = updater.process_trial_outcome(trial, clf)
        assert len(updates) == 0

    def test_taxonomy_crosscheck_overrides_direction(self):
        graph = _make_chain_graph()
        updater = EvidenceUpdater(graph)
        trial = _make_trial()
        # TARGET_ENGAGED_BIOLOGY_NOT_MOVED strengthens binds_to
        # but classifier says weaken -> should become ambiguous
        clf = _make_classification(
            mode=FailureMode.TARGET_ENGAGED_BIOLOGY_NOT_MOVED,
            raw_edges=[{"edge_type": "binds_to", "direction": "weaken", "magnitude": 0.7}]
        )
        updates = updater.process_trial_outcome(trial, clf)
        assert len(updates) == 1
        assert updates[0].direction == EvidenceDirection.AMBIGUOUS

    def test_graph_beliefs_mutated(self):
        graph = _make_chain_graph()
        pre = graph.get_edge_belief("c1", "t1", EdgeType.BINDS_TO)
        updater = EvidenceUpdater(graph)
        trial = _make_trial()
        clf = _make_classification(
            raw_edges=[{"edge_type": "binds_to", "direction": "weaken", "magnitude": 0.9}]
        )
        updater.process_trial_outcome(trial, clf)
        post = graph.get_edge_belief("c1", "t1", EdgeType.BINDS_TO)
        assert post.expected_probability < pre.expected_probability

    def test_update_math_by_hand(self):
        """Manual verification of update math.
        Phase 3 -> weight=5.0, quality=unknown -> quality_mult=0.5.
        magnitude=0.8, confidence=0.9.
        effective_magnitude = 0.8 * 0.9 = 0.72
        effective_weight = 5.0 * 0.5 = 2.5
        evidence.magnitude = 0.72 * 2.5 = 1.8
        evidence.quality_score = 0.9
        In store.update_edge_belief:
          weight_from_store = EVIDENCE_TYPE_WEIGHTS[CLINICAL_PHASE3] = 5.0
          delta = 5.0 * 0.9 * 1.8 = 8.1
          contradicting -> beta += 8.1
        Start: Beta(1,1). Result: Beta(1, 9.1). mean = 1/10.1
        """
        graph = _make_chain_graph()
        updater = EvidenceUpdater(graph)
        trial = _make_trial()
        clf = _make_classification(
            confidence=0.9,
            raw_edges=[{"edge_type": "binds_to", "direction": "weaken", "magnitude": 0.8}]
        )
        updates = updater.process_trial_outcome(trial, clf)
        assert len(updates) == 1
        u = updates[0]
        # effective_magnitude = 0.8 * 0.9 = 0.72
        assert u.magnitude == pytest.approx(0.72)
        # effective_weight = 5.0 * 0.5 = 2.5
        assert u.weight == pytest.approx(2.5)
        # evidence.magnitude = 0.72 * 2.5 = 1.8
        # store delta = 5.0 * 0.9 * 1.8 = 8.1
        assert u.post_belief.beta == pytest.approx(1.0 + 8.1)
        assert u.post_belief.expected_probability == pytest.approx(1.0 / 10.1)


# ============================================================
# Propagation
# ============================================================


class TestCompatibleTypes:
    def test_binds_to_compatible_with_modulates_via(self):
        assert EdgeType.MODULATES_VIA in _COMPATIBLE_TYPES[EdgeType.BINDS_TO]

    def test_mechanism_affects_compatible_with_biology(self):
        assert EdgeType.BIOLOGY_DRIVES in _COMPATIBLE_TYPES[EdgeType.MECHANISM_AFFECTS]

    def test_no_self_compatibility(self):
        for et, compat in _COMPATIBLE_TYPES.items():
            assert et not in compat


class TestPropagate:
    def test_propagation_reaches_hop1(self):
        graph = _make_chain_graph()
        updates = propagate(
            graph,
            source_id="c1", target_id="t1",
            edge_type=EdgeType.BINDS_TO,
            direction=EvidenceDirection.CONTRADICTING,
            magnitude=1.0,
            max_hops=1,
            damping=0.3,
        )
        assert len(updates) >= 1
        assert all(u.hop == 1 for u in updates)

    def test_propagation_reaches_hop2(self):
        graph = _make_chain_graph()
        updates = propagate(
            graph,
            source_id="c1", target_id="t1",
            edge_type=EdgeType.BINDS_TO,
            direction=EvidenceDirection.CONTRADICTING,
            magnitude=1.0,
            max_hops=2,
            damping=0.5,
        )
        hops = {u.hop for u in updates}
        assert 1 in hops
        assert 2 in hops

    def test_damping_is_multiplicative(self):
        graph = _make_chain_graph()
        damping = 0.3
        updates = propagate(
            graph,
            source_id="c1", target_id="t1",
            edge_type=EdgeType.BINDS_TO,
            direction=EvidenceDirection.CONTRADICTING,
            magnitude=1.0,
            max_hops=2,
            damping=damping,
        )
        hop1 = [u for u in updates if u.hop == 1]
        hop2 = [u for u in updates if u.hop == 2]

        # Hop 1 should have magnitude * damping
        for u in hop1:
            assert u.damped_magnitude == pytest.approx(1.0 * damping)

        # Hop 2 should have magnitude * damping^2
        for u in hop2:
            assert u.damped_magnitude == pytest.approx(1.0 * damping * damping)

    def test_stops_when_below_threshold(self):
        graph = _make_chain_graph()
        # magnitude=0.1, damping=0.3 -> hop1: 0.03, hop2: 0.009 < 0.01
        updates = propagate(
            graph,
            source_id="c1", target_id="t1",
            edge_type=EdgeType.BINDS_TO,
            direction=EvidenceDirection.CONTRADICTING,
            magnitude=0.1,
            max_hops=5,
            damping=0.3,
            min_magnitude=0.01,
        )
        for u in updates:
            assert u.damped_magnitude >= 0.01

    def test_max_hops_respected(self):
        graph = _make_chain_graph()
        updates = propagate(
            graph,
            source_id="c1", target_id="t1",
            edge_type=EdgeType.BINDS_TO,
            direction=EvidenceDirection.CONTRADICTING,
            magnitude=1.0,
            max_hops=1,
            damping=0.9,
        )
        for u in updates:
            assert u.hop <= 1

    def test_no_self_update(self):
        """The source edge itself should not be updated by propagation."""
        graph = _make_chain_graph()
        updates = propagate(
            graph,
            source_id="c1", target_id="t1",
            edge_type=EdgeType.BINDS_TO,
            direction=EvidenceDirection.CONTRADICTING,
            magnitude=1.0,
        )
        for u in updates:
            assert not (u.source_id == "c1" and u.target_id == "t1" and u.edge_type == EdgeType.BINDS_TO)

    def test_direction_preserved(self):
        graph = _make_chain_graph()
        updates = propagate(
            graph,
            source_id="c1", target_id="t1",
            edge_type=EdgeType.BINDS_TO,
            direction=EvidenceDirection.SUPPORTING,
            magnitude=1.0,
            max_hops=1,
        )
        # All propagated updates should use supporting direction
        # (they raise alpha via the store's update logic)
        for u in updates:
            assert u.post_belief.alpha >= u.pre_belief.alpha

    def test_beliefs_actually_change(self):
        graph = _make_chain_graph()
        pre = graph.get_edge_belief("t1", "m1", EdgeType.MODULATES_VIA)
        propagate(
            graph,
            source_id="c1", target_id="t1",
            edge_type=EdgeType.BINDS_TO,
            direction=EvidenceDirection.CONTRADICTING,
            magnitude=1.0,
            max_hops=1,
            damping=0.3,
        )
        post = graph.get_edge_belief("t1", "m1", EdgeType.MODULATES_VIA)
        assert post.expected_probability != pre.expected_probability

    def test_empty_when_no_compatible_neighbors(self):
        """responds_differently only propagates to biology_drives."""
        graph = GraphStore()
        graph.add_node(PopulationNode(id="p1", name="PopA"))
        graph.add_node(IndicationNode(id="i1", name="DiseaseA"))
        graph.add_edge(GraphEdge(source_id="p1", target_id="i1", edge_type=EdgeType.RESPONDS_DIFFERENTLY))
        # No biology_drives edges in this graph
        updates = propagate(
            graph,
            source_id="p1", target_id="i1",
            edge_type=EdgeType.RESPONDS_DIFFERENTLY,
            direction=EvidenceDirection.CONTRADICTING,
            magnitude=1.0,
        )
        assert len(updates) == 0

    def test_propagation_magnitude_by_hand(self):
        """Manual check: propagate from binds_to with magnitude=2.0, damping=0.5.
        Hop 1 edges get magnitude 2.0*0.5 = 1.0.
        Hop 2 edges get magnitude 1.0*0.5 = 0.5.
        """
        graph = _make_chain_graph()
        updates = propagate(
            graph,
            source_id="c1", target_id="t1",
            edge_type=EdgeType.BINDS_TO,
            direction=EvidenceDirection.CONTRADICTING,
            magnitude=2.0,
            max_hops=2,
            damping=0.5,
        )
        for u in updates:
            if u.hop == 1:
                assert u.damped_magnitude == pytest.approx(1.0)
            elif u.hop == 2:
                assert u.damped_magnitude == pytest.approx(0.5)
