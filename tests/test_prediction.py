"""Tests for the prediction layer: path queries and explainer."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest

from src.graph.models import (
    BiologyNode,
    CompoundNode,
    EdgeBeliefState,
    EdgeType,
    EndpointNode,
    EndpointType,
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
from src.prediction.path_query import (
    EdgeContribution,
    PredictionEngine,
    PredictionResult,
)
from src.prediction.explainer import explain


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
    g.add_node(PopulationNode(id="p1", name="PopA"))

    edges = [
        ("c1", "t1", EdgeType.BINDS_TO, binds_to),
        ("t1", "m1", EdgeType.MODULATES_VIA, modulates_via),
        ("m1", "b1", EdgeType.MECHANISM_AFFECTS, mechanism_affects),
        ("b1", "i1", EdgeType.BIOLOGY_DRIVES, biology_drives),
        ("b1", "e1", EdgeType.REFLECTS_BIOLOGY, reflects_biology),
        ("e1", "i1", EdgeType.ENDPOINT_CAPTURES, endpoint_captures),
        ("p1", "i1", EdgeType.RESPONDS_DIFFERENTLY, responds_differently),
    ]
    for src, tgt, etype, (a, b) in edges:
        g.add_edge(GraphEdge(
            source_id=src, target_id=tgt, edge_type=etype,
            belief=EdgeBeliefState(alpha=a, beta=b),
        ))

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
        outcome=TrialOutcome.UNKNOWN,
        phase="3",
    )


# ============================================================
# EdgeContribution
# ============================================================


class TestEdgeContribution:
    def test_bottleneck_score(self):
        ec = EdgeContribution(
            source_id="a", target_id="b",
            edge_type=EdgeType.BINDS_TO,
            belief=EdgeBeliefState(alpha=9.0, beta=1.0),
            sampled_mean=0.9,
            bottleneck_score=0.1,
        )
        assert ec.bottleneck_score == pytest.approx(0.1)

    def test_bottleneck_is_one_minus_mean(self):
        belief = EdgeBeliefState(alpha=3.0, beta=7.0)
        ec = EdgeContribution(
            source_id="a", target_id="b",
            edge_type=EdgeType.BINDS_TO,
            belief=belief,
            sampled_mean=0.3,
            bottleneck_score=1.0 - belief.expected_probability,
        )
        assert ec.bottleneck_score == pytest.approx(0.7)


# ============================================================
# PredictionEngine.predict()
# ============================================================


class TestPredict:
    def test_uniform_priors_give_moderate_prediction(self):
        """All Beta(1,1) edges -> each has mean 0.5. Product of 7 = 0.5^7 ≈ 0.0078."""
        graph = _make_graph()
        engine = PredictionEngine(graph)
        result = engine.predict(_make_trial(), n_samples=50_000)
        # Mean of product of 7 Beta(1,1) = (1/2)^7 but actually the mean of
        # product of independent Beta(1,1) is product of means = 0.5^7
        # With sampling noise, allow some tolerance
        assert result.overall_probability == pytest.approx(0.5**7, rel=0.1)
        assert result.ci_lower < result.overall_probability
        assert result.ci_upper > result.overall_probability

    def test_strong_edges_give_high_prediction(self):
        """All edges Beta(20,1) -> each mean ≈ 0.952. Product ≈ 0.952^7 ≈ 0.71."""
        params = (20.0, 1.0)
        graph = _make_graph(
            binds_to=params, modulates_via=params, mechanism_affects=params,
            biology_drives=params, reflects_biology=params,
            endpoint_captures=params, responds_differently=params,
        )
        engine = PredictionEngine(graph)
        result = engine.predict(_make_trial(), n_samples=50_000)
        assert result.overall_probability > 0.5

    def test_one_weak_edge_lowers_prediction(self):
        """One edge at Beta(1,20) (mean≈0.048) dominates the product."""
        graph = _make_graph(
            binds_to=(1.0, 20.0),  # weak
            modulates_via=(20.0, 1.0),
            mechanism_affects=(20.0, 1.0),
            biology_drives=(20.0, 1.0),
            reflects_biology=(20.0, 1.0),
            endpoint_captures=(20.0, 1.0),
            responds_differently=(20.0, 1.0),
        )
        engine = PredictionEngine(graph)
        result = engine.predict(_make_trial(), n_samples=50_000)
        assert result.overall_probability < 0.1

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
        assert result.weakest_link.edge_type == EdgeType.BINDS_TO
        assert result.weakest_link.bottleneck_score > 0.9

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
            source_id="c1", target_id="t1", edge_type=EdgeType.BINDS_TO,
            belief=EdgeBeliefState(alpha=20.0, beta=1.0),
        ))
        # Need mechanism, biology, indication nodes for non-UNKNOWN fields
        g.add_node(MechanismNode(id="m1", name="MechA", mechanism_type=MechanismType.INHIBITION))
        g.add_node(BiologyNode(id="b1", name="BioA"))
        g.add_node(IndicationNode(id="i1", name="DiseaseA"))

        trial = TrialSubgraph(
            trial_id="NCT_TEST",
            compound_id="c1", target_id="t1",
            mechanism_id="m1", biology_id="b1",
            indication_id="i1",
            endpoint_id="UNKNOWN", population_id="UNKNOWN",
            outcome=TrialOutcome.UNKNOWN, phase="3",
        )
        engine = PredictionEngine(g)
        result = engine.predict(trial, n_samples=10_000)
        # binds_to has strong belief, others default to 0.5
        # Only 4 causal chain edges (no endpoint/population since UNKNOWN)
        assert len(result.edge_contributions) == 4
        # The binds_to edge should be strong
        binds = [e for e in result.edge_contributions if e.edge_type == EdgeType.BINDS_TO]
        assert len(binds) == 1
        assert binds[0].belief.alpha == pytest.approx(20.0)

    def test_known_product_by_hand(self):
        """Hand-computed: 4 causal chain edges all Beta(10,10) (mean=0.5).
        E[product] = 0.5^4 = 0.0625. With endpoint/population UNKNOWN,
        only 4 edges contribute.
        """
        params = (10.0, 10.0)
        graph = _make_graph(
            binds_to=params, modulates_via=params,
            mechanism_affects=params, biology_drives=params,
        )
        trial = TrialSubgraph(
            trial_id="NCT_TEST",
            compound_id="c1", target_id="t1",
            mechanism_id="m1", biology_id="b1",
            indication_id="i1",
            endpoint_id="UNKNOWN", population_id="UNKNOWN",
            outcome=TrialOutcome.UNKNOWN, phase="3",
        )
        engine = PredictionEngine(graph)
        result = engine.predict(trial, n_samples=100_000)
        # E[X1*X2*X3*X4] where Xi ~ Beta(10,10)
        # For independent Betas, E[prod] = prod of E[Xi] = 0.5^4 = 0.0625
        # But E[prod(Xi)] != prod(E[Xi]) for Beta; actually they ARE equal
        # since the Xi are independent.
        assert result.overall_probability == pytest.approx(0.0625, rel=0.05)


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
        trial_weak = TrialSubgraph(
            trial_id="NCT_WEAK",
            compound_id="c1", target_id="t1",
            mechanism_id="m1", biology_id="b1",
            indication_id="i1", endpoint_id="UNKNOWN",
            population_id="UNKNOWN",
            outcome=TrialOutcome.UNKNOWN, phase="3",
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
        assert any("binds_to" in s for s in suggestions)

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


class TestExplainer:
    @pytest.mark.asyncio
    async def test_explain_returns_string(self):
        mock_client = AsyncMock()
        content_block = SimpleNamespace(text="This is an explanation.")
        mock_client.messages.create = AsyncMock(
            return_value=SimpleNamespace(content=[content_block])
        )

        graph = _make_graph(binds_to=(10.0, 2.0))
        engine = PredictionEngine(graph)
        result = engine.predict(_make_trial(), n_samples=100)

        explanation = await explain(result, mock_client)
        assert isinstance(explanation, str)
        assert len(explanation) > 0
        mock_client.messages.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_explain_sends_prediction_data(self):
        mock_client = AsyncMock()
        content_block = SimpleNamespace(text="Explanation text")
        mock_client.messages.create = AsyncMock(
            return_value=SimpleNamespace(content=[content_block])
        )

        graph = _make_graph(binds_to=(10.0, 2.0))
        engine = PredictionEngine(graph)
        result = engine.predict(_make_trial(), n_samples=100)

        await explain(result, mock_client)

        call_kwargs = mock_client.messages.create.call_args.kwargs
        user_msg = call_kwargs["messages"][0]["content"]
        assert "P(success)" in user_msg
        assert "binds_to" in user_msg
        assert result.trial_hypothesis in user_msg
