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


_MOCK_DRUG_NIVO = {
    "search": {
        "hits": [{
            "id": "CHEMBL1742982",
            "name": "Nivolumab",
            "object": {
                "id": "CHEMBL1742982",
                "name": "Nivolumab",
                "synonyms": ["BMS-936558", "MDX-1106", "ONO-4538", "Nivolumab"],
                "tradeNames": ["Opdivo"],
            },
        }],
    }
}

_MOCK_DRUG_EMPTY = {"search": {"hits": []}}


class TestSearchDrug:
    @pytest.mark.asyncio
    async def test_returns_chembl_id_and_aliases(self):
        client = OpenTargetsClient()
        client._post = AsyncMock(return_value=_MOCK_DRUG_NIVO)
        result = await client.search_drug("Nivolumab")
        assert result["chembl_id"] == "CHEMBL1742982"
        assert result["name"] == "Nivolumab"
        # Cap at 3, trade name first, no canonical
        assert len(result["aliases"]) == 3
        assert result["aliases"][0] == "Opdivo"
        assert "BMS-936558" in result["aliases"]
        assert "Nivolumab" not in result["aliases"]

    @pytest.mark.asyncio
    async def test_missing_drug_raises(self):
        client = OpenTargetsClient()
        client._post = AsyncMock(return_value=_MOCK_DRUG_EMPTY)
        with pytest.raises(KeyError, match="No drug found"):
            await client.search_drug("notarealdrug")

    @pytest.mark.asyncio
    async def test_capped_at_three_with_trade_name_first(self):
        mock = {
            "search": {"hits": [{
                "id": "CHEMBL1", "name": "TestDrug",
                "object": {
                    "id": "CHEMBL1", "name": "TestDrug",
                    "synonyms": ["AliasA", "AliasB", "AliasC", "AliasD", "AliasE"],
                    "tradeNames": ["BrandX"],
                },
            }]}
        }
        client = OpenTargetsClient()
        client._post = AsyncMock(return_value=mock)
        result = await client.search_drug("TestDrug")
        # Trade name takes the first slot, then 2 synonyms in order.
        assert result["aliases"] == ["BrandX", "AliasA", "AliasB"]

    @pytest.mark.asyncio
    async def test_filters_relationship_descriptors_and_canonical_decorations(self):
        # Real-world ChEMBL noise: "X component of Y", "X bms", etc.
        mock = {
            "search": {"hits": [{
                "id": "CHEMBL1", "name": "Nivolumab",
                "object": {
                    "id": "CHEMBL1", "name": "Nivolumab",
                    "synonyms": [
                        "BMS-936558",
                        "Nivolumab bms",                          # canonical decoration
                        "Nivolumab component of bms-986213",      # relationship
                        "MDX-1106",
                        "Nivolumab Solution",                     # canonical decoration
                        "ONO-4538",
                    ],
                    "tradeNames": ["Opdivo"],
                },
            }]}
        }
        client = OpenTargetsClient()
        client._post = AsyncMock(return_value=mock)
        result = await client.search_drug("Nivolumab")
        # Junk dropped → Opdivo + BMS-936558 + MDX-1106 (3 total)
        assert result["aliases"] == ["Opdivo", "BMS-936558", "MDX-1106"]
        assert all("of " not in a for a in result["aliases"])
        assert all("nivolumab" not in a.lower() for a in result["aliases"])

    @pytest.mark.asyncio
    async def test_punctuation_insensitive_dedup(self):
        # MDX-1106 and MDX1106 should collapse to one entry.
        mock = {
            "search": {"hits": [{
                "id": "CHEMBL1", "name": "TestDrug",
                "object": {
                    "id": "CHEMBL1", "name": "TestDrug",
                    "synonyms": ["MDX-1106", "MDX1106", "BMS-936558", "BMS936558"],
                    "tradeNames": [],
                },
            }]}
        }
        client = OpenTargetsClient()
        client._post = AsyncMock(return_value=mock)
        result = await client.search_drug("TestDrug")
        assert result["aliases"] == ["MDX-1106", "BMS-936558"]


class TestGetDrugWithTargetsDisambiguation:
    """Top-1 OT search picks ranking, not name match. We widened to 5 +
    prefer hits whose canonical/synonym/tradeName matches the queried
    name. These tests pin that behavior."""

    def _make_hit(self, chembl_id: str, name: str, synonyms=None, trade_names=None):
        return {
            "id": chembl_id,
            "name": name,
            "object": {
                "id": chembl_id,
                "name": name,
                "synonyms": synonyms or [],
                "tradeNames": trade_names or [],
                "mechanismsOfAction": {"rows": [{
                    "actionType": "INHIBITOR",
                    "mechanismOfAction": "stub",
                    "targets": [{
                        "id": f"ENS_{chembl_id}",
                        "approvedSymbol": f"SYM_{chembl_id}",
                        "approvedName": f"Approved {chembl_id}",
                    }],
                }]},
            },
        }

    @pytest.mark.asyncio
    async def test_prefers_exact_canonical_name_over_first_hit(self):
        # OT ranks "AvastinBio" first but the user asked for "Avastin".
        mock = {"search": {"hits": [
            self._make_hit("CHEMBLBIO", "AvastinBio"),
            self._make_hit("CHEMBLAVA", "Avastin"),
        ]}}
        client = OpenTargetsClient()
        client._post = AsyncMock(return_value=mock)
        result = await client.get_drug_with_targets("Avastin")
        assert result["chembl_id"] == "CHEMBLAVA"
        assert result["name"] == "Avastin"
        assert result["targets"][0]["target_id"] == "ENS_CHEMBLAVA"

    @pytest.mark.asyncio
    async def test_prefers_synonym_match_over_first_hit(self):
        mock = {"search": {"hits": [
            self._make_hit("CHEMBL_OTHER", "OtherDrug"),
            self._make_hit("CHEMBL_NIVO", "Nivolumab", synonyms=["BMS-936558"]),
        ]}}
        client = OpenTargetsClient()
        client._post = AsyncMock(return_value=mock)
        result = await client.get_drug_with_targets("BMS-936558")
        assert result["chembl_id"] == "CHEMBL_NIVO"

    @pytest.mark.asyncio
    async def test_prefers_trade_name_match_over_first_hit(self):
        mock = {"search": {"hits": [
            self._make_hit("CHEMBL_OTHER", "OtherDrug"),
            self._make_hit(
                "CHEMBL_NIVO", "Nivolumab", trade_names=["Opdivo"],
            ),
        ]}}
        client = OpenTargetsClient()
        client._post = AsyncMock(return_value=mock)
        result = await client.get_drug_with_targets("Opdivo")
        assert result["chembl_id"] == "CHEMBL_NIVO"

    @pytest.mark.asyncio
    async def test_punctuation_insensitive_match(self):
        # "BMS936558" should match "BMS-936558" once punctuation is stripped.
        mock = {"search": {"hits": [
            self._make_hit("CHEMBL_OTHER", "OtherDrug"),
            self._make_hit(
                "CHEMBL_NIVO", "Nivolumab", synonyms=["BMS-936558"],
            ),
        ]}}
        client = OpenTargetsClient()
        client._post = AsyncMock(return_value=mock)
        result = await client.get_drug_with_targets("BMS936558")
        assert result["chembl_id"] == "CHEMBL_NIVO"

    @pytest.mark.asyncio
    async def test_falls_back_to_first_when_no_exact_match(self):
        # No hit matches "Mystery"—keep OT's ranking.
        mock = {"search": {"hits": [
            self._make_hit("CHEMBL_FIRST", "FirstDrug"),
            self._make_hit("CHEMBL_SECOND", "SecondDrug"),
        ]}}
        client = OpenTargetsClient()
        client._post = AsyncMock(return_value=mock)
        result = await client.get_drug_with_targets("Mystery")
        assert result["chembl_id"] == "CHEMBL_FIRST"


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
    """Round-25: score_to_prior now routes through DATABASE_CURATED
    EvidenceRecord. Posterior reflects one curated record applied to
    Beta(1, 1) — direction is preserved (high score → posterior > 0.5,
    low score → posterior < 0.5) but the magnitude is smaller because
    n_eff is fixed at 3.0 rather than scaling with evidence_count.
    """

    def test_uninformative_at_zero_count(self):
        # No supporting evidence_count → uninformative Beta(1, 1).
        belief = score_to_prior(0.5, 0)
        assert belief.alpha == 1.0
        assert belief.beta == 1.0
        # And critically: no evidence records, so the prediction engine
        # will drop this edge.
        assert belief.evidence == []

    def test_strong_positive_emits_strong_support(self):
        belief = score_to_prior(0.9, 10)
        # bucket=strong_support (0.9 >= 0.75), quality=1.0 (count >= 5),
        # source=DATABASE_OT_ASSOCIATION (n_eff=2.0).
        # posterior: alpha = 1 + 2*0.95 = 2.9, beta = 1 + 2*0.05 = 1.1
        assert belief.alpha == pytest.approx(2.9)
        assert belief.beta == pytest.approx(1.1)
        assert belief.expected_probability > 0.7
        # Provenance preserved via evidence record.
        assert len(belief.evidence) == 1
        assert belief.evidence[0].support == "strong_support"
        assert belief.evidence[0].source_type.value == "database_ot_association"

    def test_strong_negative_emits_contradict(self):
        belief = score_to_prior(0.04, 10)
        # 0.04 < 0.05 → weak_contradict (p_obs=0.35), quality=1.0, n_eff=2
        # alpha = 1 + 2*0.35 = 1.7, beta = 1 + 2*0.65 = 2.3
        assert belief.alpha == pytest.approx(1.7)
        assert belief.beta == pytest.approx(2.3)
        assert belief.expected_probability < 0.5
        assert belief.evidence[0].support == "weak_contradict"

    def test_evidence_count_caps_quality_at_five(self):
        # Beyond evidence_count=5, quality_score is already 1.0 and
        # additional sources don't shift the posterior further.
        b5 = score_to_prior(0.8, 5)
        b100 = score_to_prior(0.8, 100)
        assert b5.alpha == pytest.approx(b100.alpha)
        assert b5.beta == pytest.approx(b100.beta)

    def test_balanced_score_emits_moderate_support(self):
        # 0.5 is at the moderate_support boundary (>= 0.5) per
        # ot_association_score_to_bucket. n_eff=2, p_obs=0.80.
        belief = score_to_prior(0.5, 10)
        assert belief.alpha == pytest.approx(1.0 + 2 * 0.80)
        assert belief.beta == pytest.approx(1.0 + 2 * 0.20)
        # Posterior leans positive at score=0.5 — under DATABASE_OT_
        # ASSOCIATION n_eff=2 with moderate_support (p_obs=0.80),
        # alpha=2.6 / (alpha+beta=4) = 0.65.
        assert belief.expected_probability == pytest.approx(0.65, abs=0.01)


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
        # Pre-create the canonical indication node—populate_target_disease_edges
        # writes against the canonical id, not the EFO id.
        from src.graph.models import IndicationNode

        graph.add_node(IndicationNode(id="non_small_cell_lung_carcinoma", name="NSCLC"))
        added = await populate_target_disease_edges(
            client, graph, "EFO_0003060", "non_small_cell_lung_carcinoma"
        )
        assert added == 2

        # Target nodes created
        egfr = graph.get_node("ENSG00000146648")
        assert egfr["gene_symbol"] == "EGFR"
        kras = graph.get_node("ENSG00000133703")
        assert kras["gene_symbol"] == "KRAS"

        # Edges land on the canonical indication node
        belief = graph.get_edge_belief(
            "ENSG00000146648",
            "non_small_cell_lung_carcinoma",
            EdgeType.BIOLOGY_DRIVES,
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
            IndicationNode(id="non_small_cell_lung_carcinoma", name="NSCLC")
        )

        await populate_target_disease_edges(
            client, graph, "EFO_0003060", "non_small_cell_lung_carcinoma"
        )

        # Original richer data preserved (not overwritten)
        node = graph.get_node("ENSG00000146648")
        assert node["druggability_score"] == 0.85

    @pytest.mark.asyncio
    async def test_empty_associations(self):
        client = OpenTargetsClient()
        client.get_disease_associations = AsyncMock(return_value=[])
        graph = GraphStore()
        added = await populate_target_disease_edges(
            client, graph, "EFO_9999999", "no_disease"
        )
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
