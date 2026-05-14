"""Tests for the prediction layer: path queries."""

from __future__ import annotations

import numpy as np
import pytest

from src.graph.models import (
    BiologyNode,
    CompoundNode,
    EdgeBeliefState,
    EdgeType,
    EndpointNode,
    EndpointType,
    CausalChain,
    GraphEdge,
    IndicationNode,
    MechanismNode,
    MechanismType,
    Modality,
    PopulationNode,
    RegulatoryStatus,
    TargetNode,
    TrialOutcome,
)
from src.graph.store import GraphStore
from src.prediction.path_query import (
    EdgeContribution,
    PredictionEngine,
    PredictionResult,
    _aggregate_samples,
    _trust_weight,
    predict_clinical_hypothesis,
)


# ── Helpers ─────────────────────────────────────────────────────────────


def _make_graph(
    binds_to: tuple[float, float] = (1.0, 1.0),
    modulates_via: tuple[float, float] = (1.0, 1.0),
    mechanism_affects: tuple[float, float] = (1.0, 1.0),
    biology_drives: tuple[float, float] = (1.0, 1.0),
    reflects_biology: tuple[float, float] = (1.0, 1.0),
    endpoint_captures: tuple[float, float] = (1.0, 1.0),
    responds_differently: tuple[float, float] = (1.0, 1.0),
) -> GraphStore:
    """Build a full causal chain graph with configurable beliefs."""
    g = GraphStore()
    g.add_node(CompoundNode(id="c1", name="DrugA", modality=Modality.SMALL_MOLECULE))
    g.add_node(TargetNode(id="t1", name="TargetA", gene_symbol="TGTA"))
    g.add_node(MechanismNode(id="m1", name="MechA", mechanism_type=MechanismType.INHIBITION))
    g.add_node(BiologyNode(id="b1", name="BioA"))
    g.add_node(IndicationNode(id="i1", name="DiseaseA"))
    g.add_node(EndpointNode(id="e1", name="EndpointA", endpoint_type=EndpointType.PRIMARY, regulatory_status=RegulatoryStatus.ACCEPTED))
    g.add_node(PopulationNode(id="i1__unselected", name="PopA"))

    edges = [
        ("c1", "t1", EdgeType.AFFECTS, binds_to),
        ("t1", "m1", EdgeType.MODULATES_VIA, modulates_via),
        ("m1", "b1", EdgeType.MECHANISM_AFFECTS, mechanism_affects),
        ("b1", "i1", EdgeType.BIOLOGY_DRIVES, biology_drives),
        ("b1", "e1", EdgeType.REFLECTS_BIOLOGY, reflects_biology),
        ("e1", "i1", EdgeType.ENDPOINT_CAPTURES, endpoint_captures),
        ("i1__unselected", "i1", EdgeType.RESPONDS_DIFFERENTLY, responds_differently),
    ]
    for src, tgt, etype, (a, b) in edges:
        g.add_edge(GraphEdge(
            source_id=src, target_id=tgt, edge_type=etype,
            belief=EdgeBeliefState(alpha=a, beta=b),
        ))

    return g


def _make_trial() -> CausalChain:
    """Build a CausalChain matching the _make_graph topology.

    Named ``_make_trial`` for backwards compatibility with the existing
    test file; what it actually returns is a single CausalChain that the
    PredictionEngine consumes directly.
    """
    return CausalChain(
        arm_id="arm_test",
        compound_id="c1",
        subgroup_population_id="i1__unselected",
        target_id="t1",
        mechanism_id="m1",
        biology_id="b1",
        indication_id="i1",
        endpoint_id="e1",
        outcome=TrialOutcome.UNKNOWN,
    )


# ============================================================
# EdgeContribution
# ============================================================


class TestEdgeContribution:
    def test_bottleneck_score(self):
        ec = EdgeContribution(
            source_id="a", target_id="b",
            edge_type=EdgeType.AFFECTS,
            belief=EdgeBeliefState(alpha=9.0, beta=1.0),
            sampled_mean=0.9,
            bottleneck_score=0.1,
        )
        assert ec.bottleneck_score == pytest.approx(0.1)

    def test_bottleneck_is_one_minus_mean(self):
        belief = EdgeBeliefState(alpha=3.0, beta=7.0)
        ec = EdgeContribution(
            source_id="a", target_id="b",
            edge_type=EdgeType.AFFECTS,
            belief=belief,
            sampled_mean=0.3,
            bottleneck_score=1.0 - belief.expected_probability,
        )
        assert ec.bottleneck_score == pytest.approx(0.7)


# ============================================================
# PredictionEngine.predict()
# ============================================================


class TestPredict:
    def test_strong_edges_give_high_prediction(self):
        params = (20.0, 1.0)
        graph = _make_graph(
            binds_to=params, modulates_via=params, mechanism_affects=params,
            biology_drives=params, reflects_biology=params,
            endpoint_captures=params, responds_differently=params,
        )
        engine = PredictionEngine(graph)
        result = engine.predict(_make_trial(), n_samples=50_000)
        assert result.overall_probability > 0.5

    def test_weakest_link_identified(self):
        graph = _make_graph(
            binds_to=(1.0, 20.0),  # weakest: mean ≈ 0.048
            modulates_via=(20.0, 1.0),
            mechanism_affects=(20.0, 1.0),
            biology_drives=(20.0, 1.0),
        )
        engine = PredictionEngine(graph)
        result = engine.predict(_make_trial())
        assert result.weakest_link is not None
        assert result.weakest_link.edge_type == EdgeType.AFFECTS
        # Under log-scaled trust: strength=19 → trust ≈ 0.77, mean ≈ 0.048,
        # so bottleneck = (1 - 0.048) * 0.77 ≈ 0.73. Old linear trust at
        # this strength saturated to 1.0 producing ≈0.95.
        assert result.weakest_link.bottleneck_score > 0.6

    def test_edge_contributions_count(self):
        """Full chain has 7 edges."""
        graph = _make_graph()
        engine = PredictionEngine(graph)
        result = engine.predict(_make_trial())
        assert len(result.edge_contributions) == 7

    def test_n_samples_respected(self):
        graph = _make_graph()
        engine = PredictionEngine(graph)
        result = engine.predict(_make_trial(), n_samples=500)
        assert result.n_samples == 500

    def test_ci_contains_mean(self):
        graph = _make_graph()
        engine = PredictionEngine(graph)
        result = engine.predict(_make_trial(), n_samples=50_000)
        assert result.ci_lower <= result.overall_probability <= result.ci_upper

    def test_missing_edges_default_to_uniform(self):
        """Graph with only binds_to edge; others default to Beta(1,1)."""
        g = GraphStore()
        g.add_node(CompoundNode(id="c1", name="DrugA", modality=Modality.SMALL_MOLECULE))
        g.add_node(TargetNode(id="t1", name="TargetA", gene_symbol="TGTA"))
        g.add_edge(GraphEdge(
            source_id="c1", target_id="t1", edge_type=EdgeType.AFFECTS,
            belief=EdgeBeliefState(alpha=20.0, beta=1.0),
        ))
        # Need mechanism, biology, indication nodes for non-UNKNOWN fields
        g.add_node(MechanismNode(id="m1", name="MechA", mechanism_type=MechanismType.INHIBITION))
        g.add_node(BiologyNode(id="b1", name="BioA"))
        g.add_node(IndicationNode(id="i1", name="DiseaseA"))

        chain = CausalChain(
            arm_id="arm_test", compound_id="c1",
            subgroup_population_id="UNKNOWN",
            target_id="t1", mechanism_id="m1", biology_id="b1",
            indication_id="i1", endpoint_id="UNKNOWN",
            outcome=TrialOutcome.UNKNOWN,
        )
        engine = PredictionEngine(g)
        result = engine.predict(chain, n_samples=10_000)
        # binds_to has strong belief, others default to 0.5
        # Only 4 causal chain edges (no endpoint/population since UNKNOWN)
        assert len(result.edge_contributions) == 4
        # The binds_to edge should be strong
        binds = [e for e in result.edge_contributions if e.edge_type == EdgeType.AFFECTS]
        assert len(binds) == 1
        assert binds[0].belief.alpha == pytest.approx(20.0)

# ============================================================
# Trust weights + weighted_geomean
# ============================================================


class TestTrustWeight:
    def test_uniform_prior_gets_zero_trust(self):
        belief = EdgeBeliefState(alpha=1.0, beta=1.0)
        assert _trust_weight(belief) == pytest.approx(0.0)

    def test_strong_evidence_saturates_at_one(self):
        # Log-scaled: trust = log(strength+1)/log(50). Saturation at strength=49.
        # Beta(50, 1): strength=49 → trust = log(50)/log(50) = 1.0.
        belief = EdgeBeliefState(alpha=50.0, beta=1.0)
        assert _trust_weight(belief) == pytest.approx(1.0)

    def test_strong_evidence_below_saturation(self):
        # Beta(15, 5): strength=18 → trust = log(19)/log(50) ≈ 0.752
        import math
        belief = EdgeBeliefState(alpha=15.0, beta=5.0)
        assert _trust_weight(belief) == pytest.approx(
            math.log(19) / math.log(50), abs=1e-6
        )

    def test_modest_evidence(self):
        # Beta(3, 3): strength=4 → trust = log(5)/log(50) ≈ 0.411
        import math
        belief = EdgeBeliefState(alpha=3.0, beta=3.0)
        assert _trust_weight(belief) == pytest.approx(
            math.log(5) / math.log(50), abs=1e-6
        )


class TestAggregateSamples:
    def test_zero_weight_falls_back_to_unweighted(self):
        # All weights 0 → fallback to unweighted geomean of 0.5 and 0.5 = 0.5
        s1 = np.array([0.5, 0.5])
        s2 = np.array([0.5, 0.5])
        out = _aggregate_samples([s1, s2], [0.0, 0.0])
        assert np.allclose(out, [0.5, 0.5])

    def test_full_trust_recovers_geomean(self):
        s1 = np.full(1000, 0.6)
        s2 = np.full(1000, 0.4)
        out = _aggregate_samples([s1, s2], [1.0, 1.0])
        expected = np.exp(0.5 * np.log(0.6) + 0.5 * np.log(0.4))
        assert np.allclose(out, expected)

    def test_zero_weight_edges_dont_drag(self):
        strong = np.full(100, 0.9)
        weak_samples = [np.full(100, 0.5) for _ in range(6)]
        out = _aggregate_samples([strong] + weak_samples, [1.0] + [0.0] * 6)
        assert np.allclose(out, 0.9)


class TestWeightedGeomeanPredict:
    def test_default_is_weighted_geomean(self):
        graph = _make_graph()  # all Beta(1,1)
        engine = PredictionEngine(graph)
        result = engine.predict(_make_trial(), n_samples=20_000)
        # Beta(1,1) → uniform; geomean of uniforms ≈ 1/e ≈ 0.368.
        # Critically, NOT 0.5^7 (which would be the product).
        assert result.overall_probability > 0.2
        assert result.overall_probability < 0.5

    def test_one_strong_edge_dominates(self):
        # binds_to with strong evidence (alpha+beta-2=18, trust=1.0),
        # all other edges Beta(1,1) (trust=0). Result should track binds_to mean.
        graph = _make_graph(binds_to=(18.0, 2.0))
        engine = PredictionEngine(graph)
        result = engine.predict(_make_trial(), n_samples=50_000)
        # binds_to mean = 0.9; geomean dominated by it should land near 0.9
        assert result.overall_probability == pytest.approx(0.9, abs=0.05)

    def test_weakest_link_uses_trust_weighted_score(self):
        # Beta(1,1) edge has mean 0.5 but no trust, so it should NOT be flagged
        # as the weakest link. binds_to with Beta(2, 18) (mean 0.1, trust=1.0)
        # has both low mean AND evidence; that's the real bottleneck.
        graph = _make_graph(binds_to=(2.0, 18.0))  # rest Beta(1,1)
        engine = PredictionEngine(graph)
        result = engine.predict(_make_trial(), n_samples=10_000)
        assert result.weakest_link is not None
        assert result.weakest_link.edge_type == EdgeType.AFFECTS

    def test_uniform_priors_have_zero_bottleneck_score(self):
        graph = _make_graph()
        engine = PredictionEngine(graph)
        result = engine.predict(_make_trial(), n_samples=1_000)
        for ec in result.edge_contributions:
            # Beta(1,1) → trust=0 → bottleneck_score = (1 - 0.5) * 0 = 0
            assert ec.bottleneck_score == pytest.approx(0.0)


# ============================================================
# PredictionEngine.compare_hypotheses()
# ============================================================


class TestCompareHypotheses:
    def test_sorted_by_probability(self):
        # Strong hypothesis
        g_strong = _make_graph(
            binds_to=(20.0, 1.0), modulates_via=(20.0, 1.0),
            mechanism_affects=(20.0, 1.0), biology_drives=(20.0, 1.0),
            reflects_biology=(20.0, 1.0), endpoint_captures=(20.0, 1.0),
            responds_differently=(20.0, 1.0),
        )
        engine = PredictionEngine(g_strong)

        trial_strong = _make_trial()
        trial_weak = CausalChain(
            arm_id="arm_weak", compound_id="c1",
            subgroup_population_id="UNKNOWN",
            target_id="t1", mechanism_id="m1", biology_id="b1",
            indication_id="i1", endpoint_id="UNKNOWN",
            outcome=TrialOutcome.UNKNOWN,
        )

        results = engine.compare_hypotheses(
            [trial_weak, trial_strong], n_samples=5_000
        )
        # Strong trial includes more edges (all 7 vs 4), but all are strong
        # Both should be high, strong should rank first (7 strong edges still > 4)
        assert results[0].overall_probability >= results[1].overall_probability

    def test_empty_list(self):
        graph = _make_graph()
        engine = PredictionEngine(graph)
        results = engine.compare_hypotheses([])
        assert results == []


# ============================================================
# PredictionEngine.suggest_improvements()
# ============================================================


class TestSuggestImprovements:
    def test_data_gap_for_low_evidence(self):
        graph = _make_graph()  # All Beta(1,1) = no evidence
        engine = PredictionEngine(graph)
        result = engine.predict(_make_trial())
        suggestions = engine.suggest_improvements(result)
        assert any("[DATA GAP]" in s for s in suggestions)

    def test_weak_link_for_contradicted_edge(self):
        graph = _make_graph(binds_to=(2.0, 20.0))  # Strong contradicting evidence
        engine = PredictionEngine(graph)
        result = engine.predict(_make_trial())
        suggestions = engine.suggest_improvements(result)
        assert any("[WEAK LINK]" in s for s in suggestions)
        assert any("affects" in s for s in suggestions)

    def test_all_strong_no_action_needed(self):
        params = (50.0, 2.0)
        graph = _make_graph(
            binds_to=params, modulates_via=params, mechanism_affects=params,
            biology_drives=params, reflects_biology=params,
            endpoint_captures=params, responds_differently=params,
        )
        engine = PredictionEngine(graph)
        result = engine.predict(_make_trial())
        suggestions = engine.suggest_improvements(result)
        assert any("All edges strong" in s for s in suggestions)

    def test_moderate_suggestion(self):
        graph = _make_graph(
            binds_to=(5.0, 5.0),  # P=0.5, evidence_strength=8 -> moderate
            modulates_via=(50.0, 2.0),
            mechanism_affects=(50.0, 2.0),
            biology_drives=(50.0, 2.0),
            reflects_biology=(50.0, 2.0),
            endpoint_captures=(50.0, 2.0),
            responds_differently=(50.0, 2.0),
        )
        engine = PredictionEngine(graph)
        result = engine.predict(_make_trial())
        suggestions = engine.suggest_improvements(result)
        assert any("[MODERATE]" in s for s in suggestions)


# ============================================================
# Explainer
# ============================================================


class TestPredictClinicalHypothesis:
    def test_walks_full_chain(self):
        graph = _make_graph(
            binds_to=(20.0, 1.0),
            modulates_via=(20.0, 1.0),
            mechanism_affects=(20.0, 1.0),
            biology_drives=(20.0, 1.0),
        )
        # endpoint/population edges exist but we don't pass them—they're
        # auxiliary, not required for the causal-chain prediction.
        result = predict_clinical_hypothesis(
            graph, "c1", "i1", n_samples=5_000
        )
        edge_types = {ec.edge_type for ec in result.edge_contributions}
        assert EdgeType.AFFECTS in edge_types
        assert EdgeType.MODULATES_VIA in edge_types
        assert EdgeType.MECHANISM_AFFECTS in edge_types
        assert EdgeType.BIOLOGY_DRIVES in edge_types
        # Aux edges are skipped because endpoint_id / population_id default to UNKNOWN
        assert EdgeType.REFLECTS_BIOLOGY not in edge_types
        assert EdgeType.ENDPOINT_CAPTURES not in edge_types
        assert EdgeType.RESPONDS_DIFFERENTLY not in edge_types
        # All four chain edges have strong supporting beliefs → high P
        assert result.overall_probability > 0.5

    def test_includes_aux_edges_when_passed(self):
        graph = _make_graph(
            binds_to=(20.0, 1.0),
            modulates_via=(20.0, 1.0),
            mechanism_affects=(20.0, 1.0),
            biology_drives=(20.0, 1.0),
            reflects_biology=(20.0, 1.0),
            endpoint_captures=(20.0, 1.0),
            responds_differently=(20.0, 1.0),
        )
        result = predict_clinical_hypothesis(
            graph, "c1", "i1",
            endpoint_id="e1", population_id="i1__unselected",
            n_samples=2_000,
        )
        assert len(result.edge_contributions) == 7

    def test_missing_compound_raises(self):
        graph = _make_graph()
        with pytest.raises(KeyError, match="Compound"):
            predict_clinical_hypothesis(graph, "missing_compound", "i1")

    def test_missing_indication_raises(self):
        graph = _make_graph()
        with pytest.raises(KeyError, match="Indication"):
            predict_clinical_hypothesis(graph, "c1", "missing_indication")

    def test_picks_best_supported_target_when_multiple_binds_to(self):
        # Compound with two binds_to: t1 (weak prior) vs t2 (strong supporting).
        # Resolver should prefer t2.
        g = GraphStore()
        g.add_node(CompoundNode(id="c1", name="DrugA", modality=Modality.SMALL_MOLECULE))
        g.add_node(TargetNode(id="t1", name="A", gene_symbol="A"))
        g.add_node(TargetNode(id="t2", name="B", gene_symbol="B"))
        g.add_node(IndicationNode(id="i1", name="DiseaseA"))
        g.add_edge(GraphEdge(
            source_id="c1", target_id="t1", edge_type=EdgeType.AFFECTS,
            belief=EdgeBeliefState(alpha=1.0, beta=1.0),
        ))
        g.add_edge(GraphEdge(
            source_id="c1", target_id="t2", edge_type=EdgeType.AFFECTS,
            belief=EdgeBeliefState(alpha=18.0, beta=2.0),
        ))
        result = predict_clinical_hypothesis(g, "c1", "i1", n_samples=1_000)
        binds_edges = [
            ec for ec in result.edge_contributions
            if ec.edge_type == EdgeType.AFFECTS
        ]
        assert binds_edges and binds_edges[0].target_id == "t2"

    def test_unknown_target_does_not_crash(self):
        """A compound with no resolvable target (Open Targets miss — e.g.
        peptide vaccines, alternative therapies, codename compounds)
        produces target=UNKNOWN. The predictor must skip the AE neighbor
        lookup for that node rather than raising KeyError. Regression
        for NCT00003509 (antineoplaston therapy) and NCT03618641
        (cmp_001 TLR9 codename) discovered in round 5/6 audit."""
        g = GraphStore()
        g.add_node(CompoundNode(id="c_orphan", name="Orphan", modality=Modality.SMALL_MOLECULE))
        g.add_node(IndicationNode(id="i1", name="DiseaseA"))
        # No TargetNode, no affects edge — _resolve_target_for_compound
        # falls through to "UNKNOWN".
        result = predict_clinical_hypothesis(g, "c_orphan", "i1", n_samples=500)
        # No chain edges to sample → empty contributions, default 0.5.
        assert result.overall_probability == 0.5
        assert result.safety_risks == []


