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
from src.graph.populate import (
    PopulationPipeline,
    _normalize,
    build_trial_subgraph_from_extraction,
    classify_endpoint_deterministic,
)
from src.graph.store import GraphStore
from src.ingestion.clinicaltrials import (
    ArmGroup,
    Intervention,
    OutcomeMeasure,
    TrialRecord,
)


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def graph():
    return GraphStore()


@pytest.fixture
def pipeline(graph, tmp_path):
    """Pipeline with a mocked anthropic client so endpoint classification runs."""
    client = AsyncMock()

    async def fake_create(**kwargs):
        # Default to a valid EndpointClass value; tests that need different
        # behavior can override pipeline._anthropic.
        from types import SimpleNamespace
        return SimpleNamespace(content=[SimpleNamespace(text="OS")])

    client.messages.create = fake_create
    return PopulationPipeline(graph, anthropic_client=client, cache_dir=tmp_path)


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
    # Single-arm trial: one arm group with one drug.
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
        arm_groups=[
            ArmGroup(group_id=drug_name.lower(), title=drug_name, intervention_names=[drug_name]),
        ],
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
        graph.add_node(CompoundNode(id="imatinib", name="Imatinib", modality=Modality.SMALL_MOLECULE))
        pipeline._index_node("imatinib", "Imatinib", "compound")
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
        assert sg.phase == "3"
        # One arm with the drug; one chain at the parent population.
        assert len(sg.arms) == 1
        assert sg.arms[0].compound_ids == ["imatinib"]
        assert not sg.arms[0].is_combination
        assert len(sg.chains) == 1
        chain = sg.chains[0]
        assert chain.compound_id == "imatinib"
        assert chain.indication_id == "IND_001"
        assert chain.endpoint_id == "EP_001"
        assert sg.parent_population_id == "IND_001__unselected"

    def test_skips_unresolvable_trial(self, pipeline):
        trial = _make_trial()
        # Nothing indexed → nothing resolves
        subgraphs = pipeline.build_trial_subgraphs([trial])
        assert len(subgraphs) == 0

    def test_skips_if_only_compound_resolves(self, pipeline):
        trial = _make_trial()
        pipeline._index_node("imatinib", "Imatinib", "compound")
        subgraphs = pipeline.build_trial_subgraphs([trial])
        assert len(subgraphs) == 0

    def test_placeholder_ids_for_unresolved_backbone(self, pipeline, graph):
        # target/mechanism/biology start as UNKNOWN; subgroup populations
        # are added later by extraction. Parent population is created.
        trial = _make_trial()
        graph.add_node(CompoundNode(id="imatinib", name="Imatinib", modality=Modality.SMALL_MOLECULE))
        pipeline._index_node("imatinib", "Imatinib", "compound")
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
        chain = sg.chains[0]
        assert chain.target_id == "UNKNOWN"
        assert chain.mechanism_id == "UNKNOWN"
        assert chain.biology_id == "UNKNOWN"
        assert chain.subgroup_population_id == "I1__unselected"

    def test_combo_arm_synthesizes_combo_compound(self, pipeline, graph):
        # A combo arm (two intervention names) gets a synthesized
        # CompoundNode + composed_of edges and a chain rooted on it.
        trial = _make_trial()
        trial.arm_groups.append(ArmGroup(
            group_id="imatinib_dasatinib",
            title="Imatinib + Dasatinib",
            intervention_names=["Imatinib", "Dasatinib"],
        ))
        graph.add_node(CompoundNode(id="imatinib", name="Imatinib", modality=Modality.SMALL_MOLECULE))
        graph.add_node(CompoundNode(id="dasatinib", name="Dasatinib", modality=Modality.SMALL_MOLECULE))
        pipeline._index_node("imatinib", "Imatinib", "compound")
        pipeline._index_node("dasatinib", "Dasatinib", "compound")
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
        combo_arms = [a for a in sg.arms if a.is_combination]
        assert len(combo_arms) == 1
        assert combo_arms[0].regimen_compound_id == "dasatinib+imatinib"
        # Combo CompoundNode synthesized
        combo_node = graph.get_node("dasatinib+imatinib")
        assert combo_node["node_type"] == "InterventionNode"
        # composed_of edges to both constituents
        composed = graph.get_edges_by_type(EdgeType.COMPOSED_OF)
        targets = sorted(e["target_id"] for e in composed if e["source_id"] == "dasatinib+imatinib")
        assert targets == ["dasatinib", "imatinib"]
        # Chain count: 2 arms (mono Imatinib + combo) × 1 parent population
        assert len(sg.chains) == 2

    def test_trial_node_persisted_in_sidecar(self, pipeline, graph):
        trial = _make_trial()
        graph.add_node(CompoundNode(id="imatinib", name="Imatinib", modality=Modality.SMALL_MOLECULE))
        pipeline._index_node("imatinib", "Imatinib", "compound")
        graph.add_node(IndicationNode(id="I1", name="Chronic Myeloid Leukemia"))
        pipeline._index_node("I1", "Chronic Myeloid Leukemia", "indication")
        graph.add_node(EndpointNode(
            id="E1", name="Overall Survival",
            endpoint_type=EndpointType.PRIMARY,
            regulatory_status=RegulatoryStatus.EXPLORATORY,
        ))
        pipeline._index_node("E1", "Overall Survival", "endpoint")

        pipeline.build_trial_subgraphs([trial])
        # TrialNode was added to the graph as a marker
        node = graph.get_node("NCT00000001")
        assert node["node_type"] == "TrialNode"
        # And the full TrialSubgraph is in the sidecar
        ts = graph.get_trial_subgraph_by_id("NCT00000001")
        assert ts.trial_id == "NCT00000001"
        assert len(ts.chains) >= 1

    def test_non_drug_intervention_names_are_filtered_from_arms(self, pipeline, graph):
        """CT.gov lists procedures, diagnostics, radiation and devices in
        arm_groups[].intervention_names alongside actual drugs. Those
        non-drug names previously became orphan untyped compound nodes
        ("biopsy", "quality_of_life_assessment", "radiation_therapy", ...)
        polluting the graph. The arm-building step must drop them when
        they're explicitly typed non-drug in trial.interventions.
        """
        from src.ingestion.clinicaltrials import Intervention

        trial = _make_trial()
        # Procedures, radiation, biopsy — all non-drug per CT.gov type.
        trial.interventions.extend([
            Intervention(name="Biopsy", type="PROCEDURE", description=""),
            Intervention(name="Radiation Therapy", type="RADIATION", description=""),
            Intervention(name="Quality-of-Life Assessment", type="OTHER", description=""),
        ])
        trial.arm_groups.append(ArmGroup(
            group_id="combined_treatment",
            title="Imatinib + Radiation Therapy",
            intervention_names=[
                "Imatinib", "Radiation Therapy", "Biopsy",
                "Quality-of-Life Assessment",
            ],
        ))
        graph.add_node(CompoundNode(id="imatinib", name="Imatinib", modality=Modality.SMALL_MOLECULE))
        pipeline._index_node("imatinib", "Imatinib", "compound")
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
        combined_arm = next(a for a in sg.arms if a.arm_id == "combined_treatment")
        # Only the drug ("imatinib") survived the filter.
        assert combined_arm.compound_ids == ["imatinib"]
        # No orphan radiation/biopsy/qol nodes leaked into the graph as
        # untyped CompoundNodes.
        for orphan in ("biopsy", "radiation_therapy", "quality_of_life_assessment"):
            try:
                node = graph.get_node(orphan)
            except KeyError:
                continue  # not in graph — good
            assert "node_type" in node, (
                f"non-drug intervention {orphan!r} leaked into the graph "
                "as an untyped node"
            )


# ── Primary intervention identification ──────────────────────────────────


class TestIdentifyPrimaryInterventions:
    """The round-3.3 primary-vs-supportive split. Per-constituent chains
    only fan out for compounds whose name appears in the trial's
    therapeutic_hypothesis.compound text. Supportive infrastructure
    (preconditioning chemo, growth factors, premeds) is tracked as an
    arm member but doesn't multiply chains."""

    def _make_extraction(self, compound_name: str):
        from src.annotation.taxonomy import TrialExtraction
        return TrialExtraction(trial_id="NCT_TEST", compound_name=compound_name)

    def _make_arms(self, compound_ids_per_arm):
        from src.graph.models import TrialArm
        return [
            TrialArm(
                arm_id=f"arm_{i}",
                compound_ids=cids,
                regimen_compound_id="+".join(sorted(cids)),
                is_combination=len(cids) > 1,
            )
            for i, cids in enumerate(compound_ids_per_arm)
        ]

    def test_single_drug_hypothesis_picks_one_primary(self):
        from src.graph.populate import _identify_primary_intervention_ids
        arms = self._make_arms([["nivolumab"]])
        ext = self._make_extraction("Nivolumab for melanoma")
        assert _identify_primary_intervention_ids(arms, ext) == ["nivolumab"]

    def test_combo_hypothesis_picks_all_named_primaries(self):
        """CheckMate-067-shaped case: hypothesis names both
        investigational drugs."""
        from src.graph.populate import _identify_primary_intervention_ids
        arms = self._make_arms([
            ["nivolumab"],
            ["nivolumab", "ipilimumab"],
            ["ipilimumab"],
        ])
        ext = self._make_extraction("Nivolumab and Ipilimumab combination")
        primaries = _identify_primary_intervention_ids(arms, ext)
        assert set(primaries) == {"nivolumab", "ipilimumab"}

    def test_car_t_with_supportive_drugs_picks_only_car_t(self):
        """NCT01218867-shaped case: CAR-T is primary, lymphodepletion
        chemo and IL-2 are supportive infrastructure."""
        from src.graph.populate import _identify_primary_intervention_ids
        arms = self._make_arms([
            ["anti_vegfr2_car_cd8_t_cells", "cyclophosphamide", "fludarabine", "aldesleukin"],
        ])
        ext = self._make_extraction(
            "Anti-VEGFR2 CAR CD8+ T cells with lymphodepletion"
        )
        primaries = _identify_primary_intervention_ids(arms, ext)
        # Only the CAR-T should be marked primary; cyclophosphamide /
        # fludarabine / aldesleukin aren't named in the hypothesis text.
        assert primaries == ["anti_vegfr2_car_cd8_t_cells"]

    def test_empty_hypothesis_returns_empty_list_for_back_compat(self):
        """No hypothesis text → back-compat: treat all as primary (return
        empty list, which downstream interprets as 'no filter')."""
        from src.graph.populate import _identify_primary_intervention_ids
        arms = self._make_arms([["nivolumab", "cyclophosphamide"]])
        ext = self._make_extraction("")
        assert _identify_primary_intervention_ids(arms, ext) == []

    def test_no_match_falls_back_to_empty(self):
        """If no arm compound matches the hypothesis (extraction text too
        freeform), fall back to empty list rather than killing every
        chain. The trial isn't silent — chains stay built per-constituent."""
        from src.graph.populate import _identify_primary_intervention_ids
        arms = self._make_arms([["compound_x", "compound_y"]])
        ext = self._make_extraction("Something completely unrelated")
        assert _identify_primary_intervention_ids(arms, ext) == []


# ── Indication slug normalization ────────────────────────────────────────


class TestSlugifyDiseaseName:
    """The canonicalizer-side normalization that collapses near-duplicate
    IndicationNode ids (singular/plural, word-order variants)."""

    @pytest.mark.parametrize("raw,expected", [
        ("Solid Tumors",          "solid_tumor"),
        ("solid tumor",           "solid_tumor"),       # already singular
        ("Brain Metastases",      "brain_metastasis"),
        ("Brain Metastasis",      "brain_metastasis"),
        ("Pancreatic Cancers",    "pancreatic_cancer"),
        ("Soft Tissue Sarcomas",  "soft_tissue_sarcoma"),
        ("Non-Hodgkin Lymphomas", "non_hodgkin_lymphoma"),
        ("Acute Leukemias",       "acute_leukemia"),
        ("Squamous Cell Carcinomas", "squamous_cell_carcinoma"),
        ("Benign Neoplasms",      "benign_neoplasm"),
    ])
    def test_plural_disease_nouns_collapse_to_singular(self, raw, expected):
        from src.graph.indication_taxonomy import slugify_disease_name
        assert slugify_disease_name(raw) == expected

    def test_known_aliases_normalize_to_canonical_form(self):
        """Word-order variants of the same disease — CT.gov phrases the
        same disease two different ways across trials; the alias table
        picks one canonical form so evidence accumulates together."""
        from src.graph.indication_taxonomy import slugify_disease_name
        canonical = "head_and_neck_squamous_cell_carcinoma"
        assert slugify_disease_name(
            "Head and Neck Squamous Cell Carcinoma"
        ) == canonical
        # The CT.gov variant "Carcinoma, Squamous Cell of Head and Neck"
        # slugifies to a different word order; the alias dict maps it back.
        assert slugify_disease_name(
            "squamous cell carcinoma head and neck"
        ) == canonical


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
        belief = graph.get_edge_belief("C1", "T_ABL", EdgeType.AFFECTS)
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

        # Mock ClinicalTrials.gov (both fetch paths the orchestrator hits)
        pipeline._ct_client.fetch_oncology_with_results = AsyncMock(
            return_value=[trial]
        )
        pipeline._ct_client.fetch_oncology_terminated_with_reason = AsyncMock(
            return_value=[]
        )

        # Mock Open Targets—disease search returns nothing (simplify)
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
        pipeline._ct_client.fetch_oncology_terminated_with_reason = AsyncMock(
            return_value=[]
        )

        # Mock OT for the trial-driven flow:
        #   DrugWithTargets → returns linkedTargets containing EGFR
        #   SearchDisease   → returns the EFO id
        #   TargetAssociations → returns the (target, disease) score
        async def mock_post(query, variables):
            if "DrugWithTargets" in query or "mechanismsOfAction" in query:
                return {
                    "search": {
                        "hits": [{
                            "id": "CHEMBL941",
                            "name": "Imatinib",
                            "object": {
                                "id": "CHEMBL941",
                                "name": "Imatinib",
                                "synonyms": [],
                                "tradeNames": ["Gleevec"],
                                "mechanismsOfAction": {
                                    "rows": [{
                                        "actionType": "INHIBITOR",
                                        "mechanismOfAction": "EGFR inhibitor",
                                        "targets": [{
                                            "id": "ENSG00000146648",
                                            "approvedSymbol": "EGFR",
                                            "approvedName": "epidermal growth factor receptor",
                                        }],
                                    }],
                                },
                            },
                        }],
                    },
                }
            if "SearchDisease" in query:
                return {"search": {"hits": [{"id": "EFO_0003060"}]}}
            if "TargetAssociations" in query:
                return {
                    "target": {
                        "associatedDiseases": {
                            "count": 1,
                            "rows": [{
                                "disease": {
                                    "id": "EFO_0003060",
                                    "name": "non-small cell lung carcinoma",
                                },
                                "score": 0.89,
                                "datatypeScores": [{"id": "clinical", "score": 0.99}],
                            }],
                        },
                    },
                }
            # SearchDrug (without targets)—fall through with a generic hit
            return {"search": {"hits": []}}

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
        pipeline._ct_client.fetch_oncology_terminated_with_reason = AsyncMock(
            return_value=[]
        )

        # OT raises on every call
        pipeline._ot_client._post = AsyncMock(side_effect=RuntimeError("API down"))

        # Should not raise—just logs and continues
        summary = await pipeline.populate_oncology(max_trials=10)
        assert summary["trials_fetched"] == 1


# ── Endpoint classifier (deterministic, no LLM) ─────────────────────────


class TestClassifyEndpointDeterministic:
    @pytest.mark.parametrize("text,expected", [
        ("Progression Free Survival (PFS)", "PFS"),
        ("Progression-Free Survival", "PFS"),
        ("Median PFS", "PFS"),
        ("Overall Survival (OS)", "OS"),
        ("OS at 24 months", "OS"),
        ("Overall Survival", "OS"),
        ("Disease-Free Survival", "DFS"),
        ("Time to Progression", "TTP"),
        ("Objective Response Rate", "ORR"),
        ("Overall Response Rate", "ORR"),
        ("Complete Response Rate", "CR"),
    ])
    def test_known_endpoints(self, text, expected):
        assert classify_endpoint_deterministic(text) == expected

    def test_unknown_text_returns_other(self):
        assert classify_endpoint_deterministic("Quality of Life Score") == "other"

    def test_empty_text_returns_other(self):
        assert classify_endpoint_deterministic("") == "other"

    @pytest.mark.parametrize("text,expected", [
        # Cases pulled from real n=50 skips: stripped trailing reviewer
        # suffix and parenthetical clarifiers should not block matching.
        ("Progression-Free Survival (PFS) by investigator", "PFS"),
        ("Progression-Free Survival by investigator assessment", "PFS"),
        ("PFS by BICR", "PFS"),
        ("OS by Independent Review Committee", "OS"),
        ("Disease-Free Survival (DFS), as Assessed by Investigator", "DFS"),
        ("ORR by independent central review", "ORR"),
        ("overall response rate (CR + PR) in BRAF V600E", "ORR"),
        ("Best Overall Response (BOR) [Phase 2 cohort]", "ORR"),
    ])
    def test_qualifier_stripping(self, text, expected):
        assert classify_endpoint_deterministic(text) == expected


class TestEndpointReindexing:
    """Pin the n=50 root-cause: a second trial whose primary outcome maps to
    the same EndpointClass as an earlier trial's outcome—but with
    different wording—used to be silently skipped because resolve_entity
    couldn't find the new measure string in the index."""

    @pytest.mark.asyncio
    async def test_second_measure_string_indexes_existing_endpoint(
        self, pipeline, graph,
    ):
        graph.add_node(IndicationNode(id="melanoma", name="Melanoma"))
        pipeline._index_node("melanoma", "Melanoma", "indication")

        # Two trials, two different PFS phrasings—both should resolve to
        # the same EndpointNode after _create_canonical_endpoints runs.
        # Deterministic regex catches "PFS" so no LLM call is made.
        t1 = TrialRecord(
            nct_id="NCT_A", title="A", phase="3", status="COMPLETED",
            conditions=["Melanoma"],
            primary_outcomes=[OutcomeMeasure(measure="Progression-Free Survival (PFS)")],
        )
        t2 = TrialRecord(
            nct_id="NCT_B", title="B", phase="3", status="COMPLETED",
            conditions=["Melanoma"],
            primary_outcomes=[OutcomeMeasure(measure="Progression-Free Survival (PFS) by investigator")],
        )
        await pipeline._populate_canonical_endpoints([t1, t2])

        ep_id_a = pipeline.resolve_entity(t1.primary_outcomes[0].measure, "endpoint")
        ep_id_b = pipeline.resolve_entity(t2.primary_outcomes[0].measure, "endpoint")
        assert ep_id_a is not None
        assert ep_id_b is not None
        # Both measure strings must resolve to the same canonical PFS node.
        assert ep_id_a == ep_id_b


# ── build_trial_subgraph_from_extraction (multi-arm × multi-subgroup × multi-endpoint) ─


class TestBuildTrialSubgraphFromExtraction:
    def test_fans_chains_across_arms_subgroups_and_endpoints(self, graph):
        """3 arms × 2 subgroups × 2 endpoints = 12 chains, each routed to the
        correct endpoint_id and carrying the matching effect_size."""
        from src.annotation.taxonomy import (
            ExtractedArm, ExtractedSubgroup, ChainResult, TrialExtraction,
        )
        from src.graph.models import (
            BiologyNode, MechanismNode, MechanismType,
            EndpointNode, EndpointType, RegulatoryStatus,
            IndicationNode, CompoundNode, Modality, TargetNode,
        )
        from src.ingestion.clinicaltrials import ArmGroup

        # Trial with 3 arms (mono A, mono B, combo) and 2 primary outcomes
        trial = TrialRecord(
            nct_id="NCT_FAN", title="Fan test", phase="3", status="COMPLETED",
            conditions=["Melanoma"],
            interventions=[
                Intervention(name="Nivolumab", type="BIOLOGICAL"),
                Intervention(name="Ipilimumab", type="BIOLOGICAL"),
            ],
            primary_outcomes=[
                OutcomeMeasure(measure="Progression Free Survival (PFS)"),
                OutcomeMeasure(measure="Overall Survival"),
            ],
            arm_groups=[
                ArmGroup(group_id="nivo", title="Nivolumab", intervention_names=["Nivolumab"]),
                ArmGroup(group_id="combo", title="Nivo+Ipi", intervention_names=["Nivolumab", "Ipilimumab"]),
                ArmGroup(group_id="ipi", title="Ipilimumab", intervention_names=["Ipilimumab"]),
            ],
            has_results=True,
        )

        # Seed nodes
        graph.add_node(CompoundNode(id="nivolumab", name="Nivolumab", modality=Modality.ANTIBODY))
        graph.add_node(CompoundNode(id="ipilimumab", name="Ipilimumab", modality=Modality.ANTIBODY))
        graph.add_node(IndicationNode(id="melanoma", name="Melanoma"))
        graph.add_node(TargetNode(id="ENSG_PD1", name="PD-1", gene_symbol="PD-1"))
        graph.add_node(TargetNode(id="ENSG_CTLA4", name="CTLA-4", gene_symbol="CTLA4"))
        graph.add_node(MechanismNode(id="checkpoint_blockade", name="cb", mechanism_type=MechanismType.ANTAGONISM))
        graph.add_node(BiologyNode(id="R-HSA-389948", name="PD-1 sig"))
        graph.add_node(EndpointNode(
            id="PFS_melanoma", name="PFS",
            endpoint_type=EndpointType.PRIMARY, regulatory_status=RegulatoryStatus.ACCEPTED,
        ))
        graph.add_node(EndpointNode(
            id="OS_melanoma", name="OS",
            endpoint_type=EndpointType.PRIMARY, regulatory_status=RegulatoryStatus.ACCEPTED,
        ))

        # Extraction with 2 reported subgroups, results filled for one cell
        extraction = TrialExtraction(
            trial_id="NCT_FAN",
            arms=[
                ExtractedArm(arm_id="nivo", compounds=["Nivolumab"]),
                ExtractedArm(arm_id="combo", compounds=["Nivolumab", "Ipilimumab"]),
                ExtractedArm(arm_id="ipi", compounds=["Ipilimumab"]),
            ],
            subgroups=[
                ExtractedSubgroup(raw_descriptor="PD-L1 ≥1%",
                                 features=[{"axis": "gene", "key": "CD274", "level": "high"}]),
                ExtractedSubgroup(raw_descriptor="PD-L1 <1%",
                                 features=[{"axis": "gene", "key": "CD274", "level": "low"}]),
            ],
            results_by_chain=[
                ChainResult(arm_id="nivo", subgroup_descriptor="PD-L1 ≥1%",
                           endpoint="PFS", effect_size=0.42, outcome="success"),
                ChainResult(arm_id="combo", subgroup_descriptor="PD-L1 ≥1%",
                           endpoint="OS", effect_size=0.55, outcome="success"),
            ],
        )

        ts = build_trial_subgraph_from_extraction(
            graph, trial, extraction,
            target_by_arm={
                "nivo": "ENSG_PD1",
                "combo": "ENSG_PD1",
                "ipi": "ENSG_CTLA4",
            },
            mechanism_id="checkpoint_blockade",
            biology_id="R-HSA-389948",
            indication_id="melanoma",
            endpoint_ids={"PFS": "PFS_melanoma", "OS": "OS_melanoma"},
        )

        # 3 arms × 2 subgroups × 2 endpoints = 12 chains
        assert len(ts.chains) == 12
        # Each chain carries its endpoint_class in metadata
        ep_classes = {c.metadata.get("endpoint_class") for c in ts.chains}
        assert ep_classes == {"PFS", "OS"}
        # The two filled results landed on the correct chains
        nivo_pfs_high = [
            c for c in ts.chains
            if c.arm_id == "nivo" and c.endpoint_id == "PFS_melanoma"
            and c.subgroup_population_id == "melanoma__cd274_positive"
        ]
        assert len(nivo_pfs_high) == 1 and nivo_pfs_high[0].effect_size == 0.42
        combo_os_high = [
            c for c in ts.chains
            if c.arm_id == "combo" and c.endpoint_id == "OS_melanoma"
            and c.subgroup_population_id == "melanoma__cd274_positive"
        ]
        assert len(combo_os_high) == 1 and combo_os_high[0].effect_size == 0.55
        # Cells with no reported result default to UNKNOWN outcome
        unknowns = [c for c in ts.chains if c.outcome == TrialOutcome.UNKNOWN]
        assert len(unknowns) == 10  # 12 total - 2 filled

    def test_skips_subgroup_when_all_features_canonicalize_to_other(self, graph):
        """Continuous PD readouts and analysis-timepoint markers aren't
        real subgroups. When canonicalization can't place any feature on
        a known axis, the subgroup is dropped from the chain fan-out so
        we don't pollute the graph with one-off
        ``melanoma__other_baseline_cd8_tumor`` populations."""
        from src.annotation.taxonomy import (
            ExtractedArm, ExtractedSubgroup, TrialExtraction,
        )
        from src.graph.models import (
            BiologyNode, MechanismNode, MechanismType,
            EndpointNode, EndpointType, RegulatoryStatus,
            IndicationNode, CompoundNode, Modality, TargetNode,
        )

        trial = TrialRecord(
            nct_id="NCT_OTHER", title="OtherOnly", phase="2", status="COMPLETED",
            conditions=["Melanoma"],
            interventions=[Intervention(name="Nivolumab", type="BIOLOGICAL")],
            primary_outcomes=[OutcomeMeasure(measure="Overall Survival")],
            arm_groups=[
                ArmGroup(group_id="A1", title="Mono", intervention_names=["Nivolumab"]),
            ],
        )
        graph.add_node(CompoundNode(id="nivolumab", name="Nivolumab", modality=Modality.ANTIBODY))
        graph.add_node(IndicationNode(id="melanoma", name="Melanoma"))
        graph.add_node(TargetNode(id="ENSG_PD1", name="PD-1", gene_symbol="PD-1"))
        graph.add_node(MechanismNode(id="cb", name="cb", mechanism_type=MechanismType.ANTAGONISM))
        graph.add_node(BiologyNode(id="bio", name="bio"))
        graph.add_node(EndpointNode(
            id="OS_mel", name="OS",
            endpoint_type=EndpointType.PRIMARY, regulatory_status=RegulatoryStatus.ACCEPTED,
        ))

        extraction = TrialExtraction(
            trial_id="NCT_OTHER",
            arms=[ExtractedArm(arm_id="A1", compounds=["Nivolumab"])],
            subgroups=[
                # PD readout—not a real subgroup, no canonical axis fits.
                ExtractedSubgroup(
                    raw_descriptor="CD8 T cells per mm² day 22",
                    features=[{"axis": "biomarker", "key": "CD8", "level": "day22"}],
                ),
                # Analysis timepoint—also not a subgroup.
                ExtractedSubgroup(
                    raw_descriptor="Final analysis",
                    features=[{"axis": "timepoint", "key": "", "level": "final"}],
                ),
            ],
            results_by_chain=[],
        )

        ts = build_trial_subgraph_from_extraction(
            graph, trial, extraction,
            target_by_arm={"A1": "ENSG_PD1"},
            mechanism_id="cb",
            biology_id="bio",
            indication_id="melanoma",
            endpoint_ids={"OS": "OS_mel"},
        )

        # No subgroup PopulationNodes created.
        pop_ids = [c.subgroup_population_id for c in ts.chains]
        assert all(pid == "melanoma__unselected" for pid in pop_ids)
        # No "other_*" PopulationNode leaked into the graph.
        other_pops = [
            n for n in graph._graph.nodes
            if isinstance(n, str) and "__other_" in n
        ]
        assert other_pops == []
        # Only the parent-population fan: 1 arm × 1 endpoint = 1 chain.
        assert len(ts.chains) == 1

    def test_skips_subgroup_when_only_feature_is_response_axis(self, graph):
        """RECIST response strata (CR / PR / SD / PD) are outcome
        stratifiers, not patient strata — they describe how subgroups
        responded post-treatment, not who was enrolled. Forking the
        arm × population matrix across them creates one chain per
        response category × per arm with no a-priori meaning, and
        none of them get classifier updates. Drop those subgroups
        from the chain fan-out; the trial-level chain still carries
        the real result.
        """
        from src.annotation.taxonomy import (
            ExtractedArm, ExtractedSubgroup, TrialExtraction,
        )
        from src.graph.models import (
            BiologyNode, MechanismNode, MechanismType,
            EndpointNode, EndpointType, RegulatoryStatus,
            IndicationNode, CompoundNode, Modality, TargetNode,
        )

        trial = TrialRecord(
            nct_id="NCT_RESP", title="ResponseOnly", phase="2", status="COMPLETED",
            conditions=["Melanoma"],
            interventions=[Intervention(name="Nivolumab", type="BIOLOGICAL")],
            primary_outcomes=[OutcomeMeasure(measure="Overall Survival")],
            arm_groups=[
                ArmGroup(group_id="A1", title="Mono", intervention_names=["Nivolumab"]),
            ],
        )
        graph.add_node(CompoundNode(id="nivolumab", name="Nivolumab", modality=Modality.ANTIBODY))
        graph.add_node(IndicationNode(id="melanoma", name="Melanoma"))
        graph.add_node(TargetNode(id="ENSG_PD1", name="PD-1", gene_symbol="PD-1"))
        graph.add_node(MechanismNode(id="cb", name="cb", mechanism_type=MechanismType.ANTAGONISM))
        graph.add_node(BiologyNode(id="bio", name="bio"))
        graph.add_node(EndpointNode(
            id="OS_mel", name="OS",
            endpoint_type=EndpointType.PRIMARY, regulatory_status=RegulatoryStatus.ACCEPTED,
        ))

        extraction = TrialExtraction(
            trial_id="NCT_RESP",
            arms=[ExtractedArm(arm_id="A1", compounds=["Nivolumab"])],
            subgroups=[
                ExtractedSubgroup(
                    raw_descriptor="Complete Response",
                    features=[{"axis": "response", "key": "", "level": "complete_response"}],
                ),
                # Stale-cache variant: axis="other" but RECIST level — should
                # auto-promote in canonicalize_feature and be skipped here.
                ExtractedSubgroup(
                    raw_descriptor="Progressive Disease",
                    features=[{"axis": "other", "key": "", "level": "progressive_disease"}],
                ),
            ],
            results_by_chain=[],
        )

        ts = build_trial_subgraph_from_extraction(
            graph, trial, extraction,
            target_by_arm={"A1": "ENSG_PD1"},
            mechanism_id="cb",
            biology_id="bio",
            indication_id="melanoma",
            endpoint_ids={"OS": "OS_mel"},
        )

        # Only the parent-population chain — no response-strata forks.
        assert len(ts.chains) == 1
        assert ts.chains[0].subgroup_population_id == "melanoma__unselected"
        # And no melanoma__response_* PopulationNode created.
        response_pops = [
            n for n in graph._graph.nodes
            if isinstance(n, str) and "__response_" in n
        ]
        assert response_pops == []
