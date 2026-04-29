"""Tests for the Open Targets Platform client."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.graph.models import EdgeBeliefState, EdgeType
from src.graph.store import GraphStore
from src.ingestion.opentargets import (
    OpenTargetsClient,
    populate_target_disease_edges,
    score_to_prior,
)

# ── Mock GraphQL responses ───────────────────────────────────────────────

MOCK_SEARCH_TARGET = {
    "search": {
        "hits": [
            {
                "id": "ENSG00000146648",
                "name": "EGFR",
                "object": {
                    "id": "ENSG00000146648",
                    "approvedSymbol": "EGFR",
                    "approvedName": "epidermal growth factor receptor",
                },
            }
        ]
    }
}

MOCK_SEARCH_TARGET_EMPTY = {"search": {"hits": []}}

MOCK_TARGET_ASSOCIATIONS = {
    "target": {
        "associatedDiseases": {
            "count": 3,
            "rows": [
                {
                    "disease": {"id": "EFO_0003060", "name": "non-small cell lung carcinoma"},
                    "score": 0.85,
                    "datatypeScores": [
                        {"id": "clinical", "score": 0.99},
                        {"id": "genetic_association", "score": 0.75},
                        {"id": "literature", "score": 0.90},
                    ],
                },
                {
                    "disease": {"id": "MONDO_0008903", "name": "lung cancer"},
                    "score": 0.77,
                    "datatypeScores": [
                        {"id": "clinical", "score": 0.80},
                        {"id": "literature", "score": 0.60},
                    ],
                },
                {
                    "disease": {"id": "EFO_0000311", "name": "some rare disease"},
                    "score": 0.05,
                    "datatypeScores": [
                        {"id": "literature", "score": 0.05},
                    ],
                },
            ],
        }
    }
}

MOCK_DISEASE_ASSOCIATIONS = {
    "disease": {
        "id": "EFO_0003060",
        "name": "non-small cell lung carcinoma",
        "associatedTargets": {
            "count": 2,
            "rows": [
                {
                    "target": {"id": "ENSG00000146648", "approvedSymbol": "EGFR"},
                    "score": 0.89,
                    "datatypeScores": [
                        {"id": "clinical", "score": 0.99},
                        {"id": "genetic_association", "score": 0.93},
                        {"id": "somatic_mutation", "score": 0.83},
                    ],
                },
                {
                    "target": {"id": "ENSG00000133703", "approvedSymbol": "KRAS"},
                    "score": 0.82,
                    "datatypeScores": [
                        {"id": "clinical", "score": 0.70},
                        {"id": "somatic_mutation", "score": 0.95},
                    ],
                },
            ],
        },
    }
}


# ── Client tests ─────────────────────────────────────────────────────────


class TestSearchTarget:
    @pytest.mark.asyncio
    async def test_returns_target_info(self):
        client = OpenTargetsClient()
        client._post = AsyncMock(return_value=MOCK_SEARCH_TARGET)
        result = await client.search_target("EGFR")
        assert result["target_id"] == "ENSG00000146648"
        assert result["approved_symbol"] == "EGFR"
        assert result["name"] == "epidermal growth factor receptor"

    @pytest.mark.asyncio
    async def test_missing_target_raises(self):
        client = OpenTargetsClient()
        client._post = AsyncMock(return_value=MOCK_SEARCH_TARGET_EMPTY)
        with pytest.raises(KeyError, match="No target found"):
            await client.search_target("FAKEGENE")


class TestGetAssociations:
    @pytest.mark.asyncio
    async def test_returns_filtered_by_min_score(self):
        client = OpenTargetsClient()
        client._post = AsyncMock(return_value=MOCK_TARGET_ASSOCIATIONS)
        results = await client.get_associations("ENSG00000146648", min_score=0.1)
        # Score 0.05 should be filtered out
        assert len(results) == 2
        assert all(r["overall_score"] >= 0.1 for r in results)

    @pytest.mark.asyncio
    async def test_returns_all_above_zero_score(self):
        client = OpenTargetsClient()
        client._post = AsyncMock(return_value=MOCK_TARGET_ASSOCIATIONS)
        results = await client.get_associations("ENSG00000146648", min_score=0.0)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_filter_by_disease_id(self):
        client = OpenTargetsClient()
        client._post = AsyncMock(return_value=MOCK_TARGET_ASSOCIATIONS)
        results = await client.get_associations(
            "ENSG00000146648", disease_id="EFO_0003060"
        )
        assert len(results) == 1
        assert results[0]["disease_id"] == "EFO_0003060"

    @pytest.mark.asyncio
    async def test_association_structure(self):
        client = OpenTargetsClient()
        client._post = AsyncMock(return_value=MOCK_TARGET_ASSOCIATIONS)
        results = await client.get_associations("ENSG00000146648")
        r = results[0]
        assert r["target_id"] == "ENSG00000146648"
        assert r["disease_name"] == "non-small cell lung carcinoma"
        assert r["evidence_count"] == 3
        assert "clinical" in r["datatypes"]
        assert r["datatypes"]["clinical"] == 0.99


class TestGetDiseaseAssociations:
    @pytest.mark.asyncio
    async def test_returns_targets(self):
        client = OpenTargetsClient()
        client._post = AsyncMock(return_value=MOCK_DISEASE_ASSOCIATIONS)
        results = await client.get_disease_associations("EFO_0003060")
        assert len(results) == 2
        symbols = {r["target_symbol"] for r in results}
        assert symbols == {"EGFR", "KRAS"}

    @pytest.mark.asyncio
    async def test_association_structure(self):
        client = OpenTargetsClient()
        client._post = AsyncMock(return_value=MOCK_DISEASE_ASSOCIATIONS)
        results = await client.get_disease_associations("EFO_0003060")
        r = results[0]
        assert r["disease_id"] == "EFO_0003060"
        assert r["overall_score"] == 0.89
        assert r["evidence_count"] == 3


# ── Score conversion tests ───────────────────────────────────────────────


class TestScoreToPrior:
    def test_uninformative_at_zero(self):
        belief = score_to_prior(0.5, 0)
        # strength = min(0, 20) = 0, so alpha=1, beta=1
        assert belief.alpha == 1.0
        assert belief.beta == 1.0

    def test_strong_positive(self):
        belief = score_to_prior(0.9, 10)
        # strength=10, alpha=1+0.9*10=10, beta=1+0.1*10=2
        assert belief.alpha == pytest.approx(10.0)
        assert belief.beta == pytest.approx(2.0)
        assert belief.expected_probability > 0.8

    def test_strong_negative(self):
        belief = score_to_prior(0.1, 10)
        # strength=10, alpha=1+0.1*10=2, beta=1+0.9*10=10
        assert belief.alpha == pytest.approx(2.0)
        assert belief.beta == pytest.approx(10.0)
        assert belief.expected_probability < 0.2

    def test_evidence_capped_at_20(self):
        belief = score_to_prior(0.8, 100)
        # strength = min(100, 20) = 20
        assert belief.alpha == pytest.approx(1.0 + 0.8 * 20)
        assert belief.beta == pytest.approx(1.0 + 0.2 * 20)

    def test_balanced_score(self):
        belief = score_to_prior(0.5, 10)
        # alpha = 1+5 = 6, beta = 1+5 = 6
        assert belief.alpha == pytest.approx(6.0)
        assert belief.beta == pytest.approx(6.0)
        assert belief.expected_probability == pytest.approx(0.5)


# ── Graph population tests ───────────────────────────────────────────────


class TestPopulateGraph:
    @pytest.mark.asyncio
    async def test_adds_edges_and_nodes(self):
        client = OpenTargetsClient()
        client.get_disease_associations = AsyncMock(
            return_value=[
                {
                    "target_id": "ENSG00000146648",
                    "target_symbol": "EGFR",
                    "disease_id": "EFO_0003060",
                    "overall_score": 0.89,
                    "evidence_count": 3,
                    "datatypes": {"clinical": 0.99},
                },
                {
                    "target_id": "ENSG00000133703",
                    "target_symbol": "KRAS",
                    "disease_id": "EFO_0003060",
                    "overall_score": 0.82,
                    "evidence_count": 2,
                    "datatypes": {"somatic_mutation": 0.95},
                },
            ]
        )
        graph = GraphStore()
        added = await populate_target_disease_edges(client, graph, "EFO_0003060")
        assert added == 2

        # Target nodes created
        egfr = graph.get_node("ENSG00000146648")
        assert egfr["gene_symbol"] == "EGFR"
        kras = graph.get_node("ENSG00000133703")
        assert kras["gene_symbol"] == "KRAS"

        # Indication node created
        ind = graph.get_node("EFO_0003060")
        assert ind["node_type"] == "IndicationNode"

        # Edges with beliefs
        belief = graph.get_edge_belief(
            "ENSG00000146648", "EFO_0003060", EdgeType.BIOLOGY_DRIVES
        )
        expected = score_to_prior(0.89, 3)
        assert belief.alpha == pytest.approx(expected.alpha)
        assert belief.beta == pytest.approx(expected.beta)

    @pytest.mark.asyncio
    async def test_does_not_duplicate_existing_nodes(self):
        from src.graph.models import IndicationNode, TargetNode

        client = OpenTargetsClient()
        client.get_disease_associations = AsyncMock(
            return_value=[
                {
                    "target_id": "ENSG00000146648",
                    "target_symbol": "EGFR",
                    "disease_id": "EFO_0003060",
                    "overall_score": 0.89,
                    "evidence_count": 3,
                    "datatypes": {},
                },
            ]
        )
        graph = GraphStore()
        # Pre-add nodes with richer data
        graph.add_node(
            TargetNode(
                id="ENSG00000146648",
                name="epidermal growth factor receptor",
                gene_symbol="EGFR",
                druggability_score=0.85,
            )
        )
        graph.add_node(
            IndicationNode(id="EFO_0003060", name="non-small cell lung carcinoma")
        )

        await populate_target_disease_edges(client, graph, "EFO_0003060")

        # Original richer data preserved (not overwritten)
        node = graph.get_node("ENSG00000146648")
        assert node["druggability_score"] == 0.85

    @pytest.mark.asyncio
    async def test_empty_associations(self):
        client = OpenTargetsClient()
        client.get_disease_associations = AsyncMock(return_value=[])
        graph = GraphStore()
        added = await populate_target_disease_edges(client, graph, "EFO_9999999")
        assert added == 0


# ── Integration test ─────────────────────────────────────────────────────


@pytest.mark.integration
class TestIntegration:
    @pytest.mark.asyncio
    async def test_search_target_real_api(self):
        client = OpenTargetsClient()
        result = await client.search_target("BRAF")
        assert result["approved_symbol"] == "BRAF"
        assert result["target_id"].startswith("ENSG")
