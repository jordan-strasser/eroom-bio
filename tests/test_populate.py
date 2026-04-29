"""Tests for the graph population orchestrator."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.graph.models import (
    CompoundNode,
    EdgeType,
    EndpointNode,
    EndpointType,
    IndicationNode,
    Modality,
    RegulatoryStatus,
    TargetNode,
    TrialOutcome,
)
from src.graph.populate import PopulationPipeline, _normalize
from src.graph.store import GraphStore
from src.ingestion.clinicaltrials import Intervention, OutcomeMeasure, TrialRecord


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def graph():
    return GraphStore()


@pytest.fixture
def pipeline(graph):
    return PopulationPipeline(graph)


def _make_trial(
    nct_id: str = "NCT00000001",
    conditions: list[str] | None = None,
    drug_name: str = "Imatinib",
    drug_desc: str = "oral kinase inhibitor targeting BCR-ABL",
    outcome_measure: str = "Overall Survival",
    phase: str = "3",
    title: str = "A Phase 3 Study of Imatinib in CML",
    status: str = "COMPLETED",
) -> TrialRecord:
    return TrialRecord(
        nct_id=nct_id,
        title=title,
        phase=phase,
        status=status,
        conditions=conditions or ["Chronic Myeloid Leukemia"],
        interventions=[
            Intervention(name=drug_name, type="DRUG", description=drug_desc),
            Intervention(name="Placebo", type="OTHER", description=""),
        ],
        primary_outcomes=[
            OutcomeMeasure(measure=outcome_measure, timeframe="36 months"),
        ],
        enrollment=400,
        has_results=True,
    )


# ── Normalize ────────────────────────────────────────────────────────────


class TestNormalize:
    def test_lowercases(self):
        assert _normalize("EGFR") == "egfr"

    def test_strips_whitespace(self):
        assert _normalize("  lung cancer  ") == "lung cancer"

    def test_collapses_spaces(self):
        assert _normalize("non  small   cell") == "non small cell"


# ── Entity resolution ────────────────────────────────────────────────────


class TestResolveEntity:
    def test_resolve_indexed_entity(self, pipeline):
        pipeline._index_node("IND_001", "Lung Cancer", "indication")
        assert pipeline.resolve_entity("Lung Cancer", "indication") == "IND_001"

    def test_case_insensitive(self, pipeline):
        pipeline._index_node("IND_001", "Lung Cancer", "indication")
        assert pipeline.resolve_entity("lung cancer", "indication") == "IND_001"

    def test_returns_none_for_unknown(self, pipeline):
        assert pipeline.resolve_entity("Unknown Disease", "indication") is None

    def test_different_types_independent(self, pipeline):
        pipeline._index_node("IND_001", "ABC", "indication")
        pipeline._index_node("COMP_001", "ABC", "compound")
        assert pipeline.resolve_entity("ABC", "indication") == "IND_001"
        assert pipeline.resolve_entity("ABC", "compound") == "COMP_001"


# ── Trial subgraph building ──────────────────────────────────────────────


class TestBuildTrialSubgraphs:
    def test_builds_subgraph_when_all_resolve(self, pipeline, graph):
        trial = _make_trial()
        # Simulate what populate_oncology does: add nodes and index them
        graph.add_node(CompoundNode(id="COMP_001", name="Imatinib", modality=Modality.SMALL_MOLECULE))
        pipeline._index_node("COMP_001", "Imatinib", "compound")
        graph.add_node(IndicationNode(id="IND_001", name="Chronic Myeloid Leukemia"))
        pipeline._index_node("IND_001", "Chronic Myeloid Leukemia", "indication")
        graph.add_node(EndpointNode(
            id="EP_001", name="Overall Survival",
            endpoint_type=EndpointType.PRIMARY,
            regulatory_status=RegulatoryStatus.EXPLORATORY,
        ))
        pipeline._index_node("EP_001", "Overall Survival", "endpoint")

        subgraphs = pipeline.build_trial_subgraphs([trial])
        assert len(subgraphs) == 1
        sg = subgraphs[0]
        assert sg.trial_id == "NCT00000001"
        assert sg.compound_id == "COMP_001"
        assert sg.indication_id == "IND_001"
        assert sg.endpoint_id == "EP_001"
        assert sg.outcome == TrialOutcome.UNKNOWN
        assert sg.phase == "3"

    def test_skips_unresolvable_trial(self, pipeline):
        trial = _make_trial()
        # Nothing indexed → nothing resolves
        subgraphs = pipeline.build_trial_subgraphs([trial])
        assert len(subgraphs) == 0

    def test_skips_if_only_compound_resolves(self, pipeline):
        trial = _make_trial()
        pipeline._index_node("COMP_001", "Imatinib", "compound")
        subgraphs = pipeline.build_trial_subgraphs([trial])
        assert len(subgraphs) == 0

    def test_placeholder_ids_for_unresolved_fields(self, pipeline, graph):
        trial = _make_trial()
        graph.add_node(CompoundNode(id="C1", name="Imatinib", modality=Modality.SMALL_MOLECULE))
        pipeline._index_node("C1", "Imatinib", "compound")
        graph.add_node(IndicationNode(id="I1", name="Chronic Myeloid Leukemia"))
        pipeline._index_node("I1", "Chronic Myeloid Leukemia", "indication")
        graph.add_node(EndpointNode(
            id="E1", name="Overall Survival",
            endpoint_type=EndpointType.PRIMARY,
            regulatory_status=RegulatoryStatus.EXPLORATORY,
        ))
        pipeline._index_node("E1", "Overall Survival", "endpoint")

        subgraphs = pipeline.build_trial_subgraphs([trial])
        sg = subgraphs[0]
        assert sg.target_id == "UNKNOWN"
        assert sg.mechanism_id == "UNKNOWN"
        assert sg.biology_id == "UNKNOWN"
        assert sg.population_id == "UNKNOWN"


# ── Compound-target cross-reference ──────────────────────────────────────


class TestCompoundTargetEdges:
    def test_adds_binds_to_when_symbol_in_text(self, pipeline, graph):
        graph.add_node(CompoundNode(id="C1", name="Imatinib", modality=Modality.SMALL_MOLECULE))
        pipeline._index_node("C1", "Imatinib", "compound")
        graph.add_node(TargetNode(id="T_ABL", name="ABL1 kinase", gene_symbol="ABL1"))

        trial = _make_trial(
            drug_name="Imatinib",
            drug_desc="targets ABL1 kinase",
            title="Phase 3 Imatinib in CML",
        )
        added = pipeline._add_compound_target_edges([trial])
        assert added >= 1
        belief = graph.get_edge_belief("C1", "T_ABL", EdgeType.BINDS_TO)
        assert belief.alpha == 2.0

    def test_no_edge_when_no_match(self, pipeline, graph):
        graph.add_node(CompoundNode(id="C1", name="Imatinib", modality=Modality.SMALL_MOLECULE))
        pipeline._index_node("C1", "Imatinib", "compound")
        graph.add_node(TargetNode(id="T_HER2", name="HER2 receptor", gene_symbol="ERBB2"))

        trial = _make_trial(
            drug_name="Imatinib",
            drug_desc="oral kinase inhibitor",
            title="Phase 3 Imatinib in CML",
        )
        added = pipeline._add_compound_target_edges([trial])
        assert added == 0


# ── Full pipeline (mocked I/O) ──────────────────────────────────────────


class TestPopulateOncology:
    @pytest.mark.asyncio
    async def test_end_to_end_mocked(self, pipeline, graph):
        trial = _make_trial()

        # Mock ClinicalTrials.gov
        pipeline._ct_client.fetch_oncology_with_results = AsyncMock(
            return_value=[trial]
        )

        # Mock Open Targets — disease search returns nothing (simplify)
        pipeline._ot_client._post = AsyncMock(
            return_value={"search": {"hits": []}}
        )

        summary = await pipeline.populate_oncology(max_trials=10)

        assert summary["trials_fetched"] == 1
        assert summary["compounds"] >= 1
        assert summary["indications"] >= 1
        assert summary["endpoints"] >= 1
        assert summary["trial_subgraphs"] >= 1

    @pytest.mark.asyncio
    async def test_with_ot_associations(self, pipeline, graph):
        trial = _make_trial(
            conditions=["non-small cell lung carcinoma"],
        )

        pipeline._ct_client.fetch_oncology_with_results = AsyncMock(
            return_value=[trial]
        )

        # Mock OT: disease search returns EFO ID, then disease associations
        call_count = 0

        async def mock_post(query, variables):
            nonlocal call_count
            call_count += 1
            if "SearchDisease" in query or "search" in query.lower() and "disease" in query.lower():
                return {"search": {"hits": [{"id": "EFO_0003060"}]}}
            # DiseaseAssociations query
            return {
                "disease": {
                    "id": "EFO_0003060",
                    "name": "non-small cell lung carcinoma",
                    "associatedTargets": {
                        "count": 1,
                        "rows": [
                            {
                                "target": {"id": "ENSG00000146648", "approvedSymbol": "EGFR"},
                                "score": 0.89,
                                "datatypeScores": [{"id": "clinical", "score": 0.99}],
                            }
                        ],
                    },
                }
            }

        pipeline._ot_client._post = mock_post

        summary = await pipeline.populate_oncology(max_trials=10)
        assert summary["targets"] >= 1
        assert summary["edges"] >= 1

    @pytest.mark.asyncio
    async def test_handles_ot_failure_gracefully(self, pipeline, graph):
        trial = _make_trial()

        pipeline._ct_client.fetch_oncology_with_results = AsyncMock(
            return_value=[trial]
        )

        # OT raises on every call
        pipeline._ot_client._post = AsyncMock(side_effect=RuntimeError("API down"))

        # Should not raise — just logs and continues
        summary = await pipeline.populate_oncology(max_trials=10)
        assert summary["trials_fetched"] == 1
