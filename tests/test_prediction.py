"""Tests for the prediction layer: path queries."""

from __future__ import annotations

import numpy as np
import pytest

from src.graph.models import (
    AdverseEventNode,
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
    _collect_modulation_edges,
    _regimen_constituents,
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
        """Full chain has 7 edges; with evidence on each, all should
        appear as contributions. (Round-15: zero-evidence edges are
        dropped, so this test pins down a fully-evidenced graph.)"""
        graph = _make_graph(
            binds_to=(5.0, 2.0),
            modulates_via=(5.0, 2.0),
            mechanism_affects=(5.0, 2.0),
            biology_drives=(5.0, 2.0),
            reflects_biology=(5.0, 2.0),
            endpoint_captures=(5.0, 2.0),
            responds_differently=(5.0, 2.0),
        )
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

    def test_missing_edges_are_dropped(self):
        """Graph with only the binds_to edge populated; mechanism /
        biology / biology→indication edges don't exist at all.
        Round-15: edges that aren't in the graph (or that have zero
        evidence) are silently dropped — the prediction reflects only
        the evidenced subset of the hypothesis chain."""
        g = GraphStore()
        g.add_node(CompoundNode(id="c1", name="DrugA", modality=Modality.SMALL_MOLECULE))
        g.add_node(TargetNode(id="t1", name="TargetA", gene_symbol="TGTA"))
        g.add_edge(GraphEdge(
            source_id="c1", target_id="t1", edge_type=EdgeType.AFFECTS,
            belief=EdgeBeliefState(alpha=20.0, beta=1.0),
        ))
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
        # Only the binds_to edge has evidence; the other chain edges
        # don't exist in the graph and are dropped.
        assert len(result.edge_contributions) == 1
        ec = result.edge_contributions[0]
        assert ec.edge_type == EdgeType.AFFECTS
        assert ec.belief.alpha == pytest.approx(20.0)

# ============================================================
# Modulation-edge collection (round 8 v0.2.0)
# ============================================================


def _build_regimen_with_modulation(
    modulation_belief: tuple[float, float] = (5.0, 1.0),
) -> GraphStore:
    """Two-constituent regimen with a MODULATES_EFFICACY_OF edge."""
    g = _make_graph()
    g.add_node(CompoundNode(id="c2", name="DrugB", modality=Modality.SMALL_MOLECULE))
    g.add_node(CompoundNode(
        id="c1+c2", name="DrugA + DrugB", modality=Modality.OTHER,
    ))
    for cid in ("c1", "c2"):
        g.add_edge(GraphEdge(
            source_id="c1+c2", target_id=cid,
            edge_type=EdgeType.COMPOSED_OF,
            belief=EdgeBeliefState(alpha=1.0, beta=1.0),
        ))
    # Modulation edge stored lex-canonical (c1 < c2).
    g.add_edge(GraphEdge(
        source_id="c1", target_id="c2",
        edge_type=EdgeType.MODULATES_EFFICACY_OF,
        belief=EdgeBeliefState(
            alpha=modulation_belief[0], beta=modulation_belief[1],
        ),
    ))
    return g


class TestModulationEdgeCollection:
    def test_constituents_for_regimen(self):
        g = _build_regimen_with_modulation()
        assert sorted(_regimen_constituents(g, "c1+c2")) == ["c1", "c2"]

    def test_constituents_for_mono_compound(self):
        g = _make_graph()
        assert _regimen_constituents(g, "c1") == ["c1"]

    def test_collects_modulation_edge_for_regimen(self):
        g = _build_regimen_with_modulation()
        edges = _collect_modulation_edges(g, "c1+c2")
        assert len(edges) == 1
        src, tgt, etype, belief = edges[0]
        assert (src, tgt) == ("c1", "c2")
        assert etype == EdgeType.MODULATES_EFFICACY_OF
        assert belief.alpha == 5.0

    def test_mono_compound_collects_nothing(self):
        g = _build_regimen_with_modulation()
        assert _collect_modulation_edges(g, "c1") == []

    def test_regimen_with_no_modulation_edge_returns_empty(self):
        g = _make_graph()
        g.add_node(CompoundNode(id="c2", name="DrugB", modality=Modality.SMALL_MOLECULE))
        g.add_node(CompoundNode(
            id="c1+c2", name="DrugA + DrugB", modality=Modality.OTHER,
        ))
        for cid in ("c1", "c2"):
            g.add_edge(GraphEdge(
                source_id="c1+c2", target_id=cid,
                edge_type=EdgeType.COMPOSED_OF,
                belief=EdgeBeliefState(alpha=1.0, beta=1.0),
            ))
        assert _collect_modulation_edges(g, "c1+c2") == []


class TestRegimenPredictionWithModulation:
    def _regimen_chain(self) -> CausalChain:
        return CausalChain(
            arm_id="regimen_arm", compound_id="c1+c2",
            subgroup_population_id="i1__unselected",
            target_id="t1", mechanism_id="m1", biology_id="b1",
            indication_id="i1", endpoint_id="e1",
            outcome=TrialOutcome.UNKNOWN,
        )

    def test_modulation_edge_appears_in_edge_contributions(self):
        g = _build_regimen_with_modulation(modulation_belief=(8.0, 2.0))
        engine = PredictionEngine(g)
        result = engine.predict(self._regimen_chain(), n_samples=2_000)
        mod_contribs = [
            e for e in result.edge_contributions
            if e.edge_type == EdgeType.MODULATES_EFFICACY_OF
        ]
        assert len(mod_contribs) == 1

    def test_neutral_modulation_does_not_change_prediction(self):
        """Beta(1, 1) modulation contributes 0 trust weight and is a
        no-op on the aggregate."""
        g_with_neutral = _build_regimen_with_modulation(
            modulation_belief=(1.0, 1.0),
        )
        g_without = _make_graph()
        g_without.add_node(CompoundNode(
            id="c2", name="DrugB", modality=Modality.SMALL_MOLECULE,
        ))
        g_without.add_node(CompoundNode(
            id="c1+c2", name="DrugA + DrugB", modality=Modality.OTHER,
        ))
        for cid in ("c1", "c2"):
            g_without.add_edge(GraphEdge(
                source_id="c1+c2", target_id=cid,
                edge_type=EdgeType.COMPOSED_OF,
                belief=EdgeBeliefState(alpha=1.0, beta=1.0),
            ))
        chain = self._regimen_chain()
        engine_with = PredictionEngine(g_with_neutral)
        engine_without = PredictionEngine(g_without)
        np.random.seed(0)
        p_with = engine_with.predict(chain, n_samples=5_000).overall_probability
        np.random.seed(0)
        p_without = engine_without.predict(chain, n_samples=5_000).overall_probability
        assert abs(p_with - p_without) < 0.02

    def test_contradict_modulation_pulls_prediction_down(self):
        """A strong contradict-leaning modulation edge should reduce
        P(regimen success) relative to a strong support-leaning one."""
        g_support = _build_regimen_with_modulation(modulation_belief=(45.0, 2.0))
        g_contradict = _build_regimen_with_modulation(modulation_belief=(2.0, 45.0))
        chain = self._regimen_chain()
        np.random.seed(0)
        p_support = PredictionEngine(g_support).predict(
            chain, n_samples=5_000,
        ).overall_probability
        np.random.seed(0)
        p_contradict = PredictionEngine(g_contradict).predict(
            chain, n_samples=5_000,
        ).overall_probability
        assert p_support > p_contradict


# ============================================================
# Trust weights + weighted_geomean
# ============================================================


class TestTrustWeight:
    def test_uniform_prior_gets_zero_trust(self):
        """Beta(1,1) (no evidence) → trust 0. Round-15 reverted the
        round-14 trust floor: zero-evidence edges are now dropped at
        the engine level so they contribute nothing to the geomean,
        rather than pulling sparse chains toward 0.5 at low weight.
        Beliefs reaching `_trust_weight` are expected to already have
        evidence — this test verifies the boundary case directly."""
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
    def test_all_zero_evidence_chain_returns_default(self):
        """Round-15: chains with no evidenced edges (all Beta(1,1)) have
        no contributions to aggregate. The engine returns P=0.5 with a
        full [0, 1] CI as a "no signal" fallback instead of inventing
        an unweighted-geomean prediction from priors."""
        graph = _make_graph()  # all Beta(1,1)
        engine = PredictionEngine(graph)
        result = engine.predict(_make_trial(), n_samples=20_000)
        assert result.overall_probability == pytest.approx(0.5)
        assert result.edge_contributions == []

    def test_one_strong_edge_fully_determines_when_others_dropped(self):
        """Round-15: with zero-evidence edges dropped at collection time,
        a chain with one Beta(18, 2) edge and six Beta(1,1) edges
        produces a prediction driven entirely by the surviving edge.
        Expected value ≈ E[Beta(18,2)] = 0.9.
        """
        graph = _make_graph(binds_to=(18.0, 2.0))
        engine = PredictionEngine(graph)
        result = engine.predict(_make_trial(), n_samples=50_000)
        # Six Beta(1,1) edges dropped; only the binds_to edge remains.
        assert len(result.edge_contributions) == 1
        assert result.edge_contributions[0].edge_type == EdgeType.AFFECTS
        # Sampling Beta(18,2) and taking the mean → ~0.9 ± noise.
        assert 0.85 < result.overall_probability < 0.93

    def test_weakest_link_picks_evidenced_low_mean(self):
        # binds_to is Beta(2, 18) (mean 0.1, trust=1.0). Other edges
        # have no evidence and are dropped, so binds_to is the only
        # contribution AND the weakest link.
        graph = _make_graph(binds_to=(2.0, 18.0))
        engine = PredictionEngine(graph)
        result = engine.predict(_make_trial(), n_samples=10_000)
        assert result.weakest_link is not None
        assert result.weakest_link.edge_type == EdgeType.AFFECTS

    def test_bottleneck_score_is_trust_weighted_complement(self):
        """Bottleneck = (1 - E[p]) * trust. For a Beta(18, 2) edge:
        E[p]=0.9, trust=1.0 → bottleneck = 0.10. For Beta(5, 15):
        E[p]=0.25, trust ≈ log(19)/log(50) ≈ 0.752 → bottleneck ≈ 0.564.
        """
        import math
        graph = _make_graph(binds_to=(5.0, 15.0))  # E[p]=0.25, n_eff=18
        engine = PredictionEngine(graph)
        result = engine.predict(_make_trial(), n_samples=1_000)
        assert len(result.edge_contributions) == 1
        ec = result.edge_contributions[0]
        expected_trust = math.log(19) / math.log(50)
        expected_bottleneck = (1.0 - 0.25) * expected_trust
        assert ec.bottleneck_score == pytest.approx(expected_bottleneck, abs=1e-6)


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
    def test_no_evidence_yields_no_edges_message(self):
        """Round-15: with all Beta(1,1), the engine drops every edge,
        so `edge_contributions` is empty. `suggest_improvements` returns
        the "no edges" message — there's nothing to improve when there's
        nothing in the chain to improve from."""
        graph = _make_graph()  # All Beta(1,1) = no evidence
        engine = PredictionEngine(graph)
        result = engine.predict(_make_trial())
        suggestions = engine.suggest_improvements(result)
        assert any("No edges" in s for s in suggestions)

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

    def test_missing_compound_is_allowed_and_skips_affects(self):
        """Round-15: compound_id missing from graph (or None) no longer
        raises. The `affects` edge gets skipped, and prediction proceeds
        on whatever evidenced downstream edges exist. Supports the
        "novel compound with familiar target" use case."""
        graph = _make_graph(
            modulates_via=(5.0, 2.0),
            mechanism_affects=(5.0, 2.0),
            biology_drives=(5.0, 2.0),
        )
        result = predict_clinical_hypothesis(
            graph, "missing_compound", "i1", target_id="t1",
        )
        # affects edge skipped (compound not in graph); downstream chain
        # evidenced edges still contribute.
        edge_types = {ec.edge_type.value for ec in result.edge_contributions}
        assert "affects" not in edge_types
        assert "modulates_via" in edge_types
        # Also accepts compound_id=None.
        result2 = predict_clinical_hypothesis(
            graph, None, "i1", target_id="t1",
        )
        assert "affects" not in {
            ec.edge_type.value for ec in result2.edge_contributions
        }

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


# ============================================================
# Round-20: safety penalty
# ============================================================


def _make_chain_only_graph(efficacy_param=(8.0, 2.0)) -> tuple[GraphStore, CausalChain]:
    """Build a graph with the full mechanism chain and a moderately
    strong efficacy signal, but NO adverse-event edges. Used as a
    safety-penalty=0 baseline."""
    graph = _make_graph(
        binds_to=efficacy_param,
        modulates_via=efficacy_param,
        mechanism_affects=efficacy_param,
        biology_drives=efficacy_param,
        reflects_biology=efficacy_param,
        endpoint_captures=efficacy_param,
        responds_differently=efficacy_param,
    )
    return graph, _make_trial()


class TestSafetyPenalty:
    """Round-20: safety drag on the headline number.

    ``overall_probability = efficacy_probability * (1 - safety_penalty)``.
    Penalty is soft-or aggregation of belief × log-saturated weight over
    AE risks above ``_SAFETY_PENALTY_MIN_BELIEF`` (0.5) and
    ``_SAFETY_PENALTY_MIN_EVIDENCE`` (1.0), capped at
    ``_SAFETY_PENALTY_CAP`` (0.4).
    """

    def test_zero_ae_keeps_overall_equal_to_efficacy(self):
        graph, chain = _make_chain_only_graph()
        result = PredictionEngine(graph).predict(chain, n_samples=5_000)
        assert result.safety_penalty == 0.0
        assert result.overall_probability == pytest.approx(
            result.efficacy_probability, abs=1e-9,
        )

    def _seeded_ae(self, graph, ae_id, name, severity, *, src="c1",
                   alpha=10.0, beta=1.0,
                   edge_type=EdgeType.CAUSES_AE):
        graph.add_node(AdverseEventNode(
            id=ae_id, name=name, severity_range=severity,
        ))
        graph.add_edge(GraphEdge(
            source_id=src, target_id=ae_id,
            edge_type=edge_type,
            belief=EdgeBeliefState(alpha=alpha, beta=beta),
        ))

    def test_strong_compound_ae_drags_overall_below_efficacy(self):
        graph, chain = _make_chain_only_graph()
        # Grade 4 (life-threatening) AE, well-evidenced.
        self._seeded_ae(graph, "AE:severe", "Severe AE", "4")
        result = PredictionEngine(graph).predict(chain, n_samples=5_000)
        # Three-gate math: severity (grade 4 = 0.30) × belief_factor ×
        # trust_factor. Beta(10,1) → belief 0.909, evidence_strength 9.
        # belief_factor = (0.909-0.5)/0.5 = 0.818
        # trust_factor  = log(10)/log(50) ≈ 0.589
        # contribution ≈ 0.30 × 0.818 × 0.589 ≈ 0.145
        assert result.safety_penalty == pytest.approx(0.145, abs=0.005)
        assert result.overall_probability < result.efficacy_probability
        assert result.overall_probability == pytest.approx(
            result.efficacy_probability * (1 - result.safety_penalty),
            abs=1e-9,
        )

    def test_target_class_ae_contributes_for_novel_compound(self):
        """A novel compound has no causes_ae history, but its target
        carries a target_associated_ae signal from related compounds.
        That on-mechanism class evidence still moves the penalty —
        round-20 makes safety travel along the target node, not just
        the compound."""
        graph, chain = _make_chain_only_graph()
        # Grade 3 on the target. Beta(8,2) → belief 0.8, n_eff 10.
        self._seeded_ae(
            graph, "AE:class_tox", "Class Toxicity", "3",
            src="t1", alpha=8.0, beta=2.0,
            edge_type=EdgeType.TARGET_ASSOCIATED_AE,
        )
        result = PredictionEngine(graph).predict(chain, n_samples=5_000)
        # severity 0.15 × belief_factor 0.6 × trust_factor ≈ 0.562
        # ≈ 0.051
        assert result.safety_penalty == pytest.approx(0.051, abs=0.005)
        assert result.overall_probability < result.efficacy_probability

    def test_grade_severity_ordering(self):
        """Holding belief and evidence constant, a grade-5 AE must
        produce a strictly larger safety_penalty than a grade-3, which
        must be strictly larger than grade-1-2, which must be strictly
        larger than the unknown default. This is the core severity-
        weighting property."""
        results = {}
        for label, severity in [("g12", "1,2"), ("g3", "3"),
                                ("g4", "4"), ("g5", "5"), ("unk", "")]:
            graph, chain = _make_chain_only_graph()
            self._seeded_ae(graph, f"AE:{label}", label, severity)
            results[label] = PredictionEngine(graph).predict(
                chain, n_samples=5_000,
            ).safety_penalty
        # Same belief + evidence in every test → ordering is purely
        # severity-driven.
        assert results["g12"] < results["unk"] < results["g3"]
        assert results["g3"] < results["g4"] < results["g5"]

    def test_weak_belief_ae_barely_moves_penalty(self):
        """An AE just barely above the 0.55 belief threshold contributes
        little even at grade 5 — the (belief - 0.5)/0.5 factor scales
        the severity down. Mirrors the live-corpus case where
        AMBIGUOUS-bucket AEs accidentally drift just above 0.5 but
        carry no real safety signal."""
        graph, chain = _make_chain_only_graph()
        # Beta(5.6, 4.4) → belief ≈ 0.56, just above the 0.55 gate.
        # Grade 5, but the weak-belief factor should suppress it to
        # a small contribution.
        self._seeded_ae(
            graph, "AE:fatal_weakbel", "Fatal but weakly evidenced", "5",
            alpha=5.6, beta=4.4,
        )
        result = PredictionEngine(graph).predict(chain, n_samples=5_000)
        # belief_factor ≈ 0.12 (just above the gate),
        # trust_factor ≈ log(11)/log(50) ≈ 0.613,
        # contribution ≈ 0.50 × 0.12 × 0.613 ≈ 0.037
        assert 0.01 < result.safety_penalty < 0.08

    def test_safety_penalty_caps_at_0_6(self):
        """Pile on multiple grade-5 AEs — the penalty must not exceed
        the 0.6 cap so the chain still contributes the final 40%."""
        graph, chain = _make_chain_only_graph()
        for i in range(8):
            graph.add_node(AdverseEventNode(
                id=f"AE:fatal_{i}", name=f"Fatal {i}",
                severity_range="5",
            ))
            graph.add_edge(GraphEdge(
                source_id="c1", target_id=f"AE:fatal_{i}",
                edge_type=EdgeType.CAUSES_AE,
                belief=EdgeBeliefState(alpha=50.0, beta=1.0),
            ))
        result = PredictionEngine(graph).predict(chain, n_samples=5_000)
        assert result.safety_penalty == pytest.approx(0.60, abs=1e-9)
        assert result.overall_probability >= 0.4 * result.efficacy_probability - 1e-9

    def test_below_threshold_ae_does_not_move_penalty(self):
        """A weak Beta(1.4, 1) edge has belief ~0.58 but evidence_strength
        ~0.4 — below the min_evidence=1.0 gate. The penalty stays 0
        rather than letting a near-prior edge drag the headline number."""
        graph, chain = _make_chain_only_graph()
        graph.add_node(AdverseEventNode(
            id="AE:weak", name="Weak Tox", severity_range="5",
        ))
        graph.add_edge(GraphEdge(
            source_id="c1", target_id="AE:weak",
            edge_type=EdgeType.CAUSES_AE,
            belief=EdgeBeliefState(alpha=1.4, beta=1.0),
        ))
        result = PredictionEngine(graph).predict(chain, n_samples=5_000)
        assert result.safety_penalty == 0.0

    def test_ci_bounds_also_scale_with_safety(self):
        """CI bounds reflect the same safety_factor multiplier — otherwise
        a wide efficacy CI would visually swamp the safety drag."""
        graph, chain = _make_chain_only_graph()
        graph.add_node(AdverseEventNode(
            id="AE:severe", name="Severe AE", severity_range="4",
        ))
        graph.add_edge(GraphEdge(
            source_id="c1", target_id="AE:severe",
            edge_type=EdgeType.CAUSES_AE,
            belief=EdgeBeliefState(alpha=10.0, beta=1.0),
        ))
        result = PredictionEngine(graph).predict(chain, n_samples=5_000)
        # The CI bounds got multiplied by (1 - safety_penalty), so the
        # ratio ci_lower / ci_upper is unchanged but both shifted down.
        assert result.ci_lower < result.efficacy_probability
        assert result.ci_upper < 1.0  # squeezed
        # Lower bound should still be <= overall <= upper bound (basic
        # CI invariant preserved after the safety scale).
        assert result.ci_lower <= result.overall_probability <= result.ci_upper

    def test_max_grade_parser_handles_real_world_severity_strings(self):
        """Live AE nodes have free-text severity_range like 'any,3-4,3-5'
        from successive trial attributions. Parser must extract the
        max integer grade across all comma-separated tokens."""
        from src.prediction.path_query import _max_grade_from_severity_range
        assert _max_grade_from_severity_range("") is None
        assert _max_grade_from_severity_range(None) is None
        assert _max_grade_from_severity_range("any") is None
        assert _max_grade_from_severity_range("1") == 1
        assert _max_grade_from_severity_range("1,2,3") == 3
        assert _max_grade_from_severity_range("3-5") == 5
        assert _max_grade_from_severity_range("1,2,3-5") == 5
        assert _max_grade_from_severity_range("any,3-4,3-5") == 5
        assert _max_grade_from_severity_range("garbage,abc") is None


