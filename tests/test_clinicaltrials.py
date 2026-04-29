"""Tests for the ClinicalTrials.gov v2 API client."""

import pytest

from src.graph.models import EndpointType, Modality
from src.ingestion.clinicaltrials import (
    ClinicalTrialsClient,
    Intervention,
    OutcomeMeasure,
    TrialRecord,
    _guess_modality,
    _parse_study,
    map_trial_to_graph_nodes,
)

# ── Fixtures ─────────────────────────────────────────────────────────────

RAW_STUDY = {
    "protocolSection": {
        "identificationModule": {
            "nctId": "NCT00000001",
            "briefTitle": "Test Trial of Imatinib in CML",
        },
        "statusModule": {
            "overallStatus": "COMPLETED",
            "startDateStruct": {"date": "2020-01-15", "type": "ACTUAL"},
            "completionDateStruct": {"date": "2023-06-30", "type": "ACTUAL"},
        },
        "designModule": {
            "phases": ["PHASE3"],
            "enrollmentInfo": {"count": 450},
        },
        "conditionsModule": {
            "conditions": ["Chronic Myeloid Leukemia", "Philadelphia Positive CML"],
        },
        "armsInterventionsModule": {
            "interventions": [
                {
                    "name": "Imatinib",
                    "type": "DRUG",
                    "description": "400mg daily oral tyrosine kinase inhibitor",
                },
                {
                    "name": "Placebo",
                    "type": "OTHER",
                    "description": "Matching placebo",
                },
            ],
        },
        "outcomesModule": {
            "primaryOutcomes": [
                {
                    "measure": "Major Molecular Response (MMR)",
                    "timeFrame": "12 months",
                    "description": "BCR-ABL1 transcript level <=0.1%",
                },
            ],
            "secondaryOutcomes": [
                {
                    "measure": "Progression-Free Survival",
                    "timeFrame": "36 months",
                    "description": "Time from randomization to progression or death",
                },
            ],
        },
        "sponsorCollaboratorsModule": {
            "leadSponsor": {"name": "Novartis", "class": "INDUSTRY"},
        },
    },
    "hasResults": True,
    "resultsSection": {"some": "data"},
}


@pytest.fixture
def trial_record():
    return _parse_study(RAW_STUDY)


@pytest.fixture
def antibody_trial():
    return TrialRecord(
        nct_id="NCT99999999",
        title="Test Antibody Trial",
        phase="3",
        status="COMPLETED",
        conditions=["Breast Cancer"],
        interventions=[
            Intervention(
                name="Trastuzumab",
                type="DRUG",
                description="Anti-HER2 monoclonal antibody",
            ),
        ],
        primary_outcomes=[
            OutcomeMeasure(measure="Overall Survival", timeframe="60 months"),
        ],
        has_results=True,
    )


# ── Parsing ──────────────────────────────────────────────────────────────


class TestParseStudy:
    def test_nct_id(self, trial_record):
        assert trial_record.nct_id == "NCT00000001"

    def test_title(self, trial_record):
        assert "Imatinib" in trial_record.title

    def test_phase(self, trial_record):
        assert trial_record.phase == "3"

    def test_status(self, trial_record):
        assert trial_record.status == "COMPLETED"

    def test_conditions(self, trial_record):
        assert len(trial_record.conditions) == 2
        assert "Chronic Myeloid Leukemia" in trial_record.conditions

    def test_interventions(self, trial_record):
        assert len(trial_record.interventions) == 2
        assert trial_record.interventions[0].name == "Imatinib"
        assert trial_record.interventions[0].type == "DRUG"

    def test_primary_outcomes(self, trial_record):
        assert len(trial_record.primary_outcomes) == 1
        assert "Molecular Response" in trial_record.primary_outcomes[0].measure

    def test_secondary_outcomes(self, trial_record):
        assert len(trial_record.secondary_outcomes) == 1

    def test_enrollment(self, trial_record):
        assert trial_record.enrollment == 450

    def test_dates(self, trial_record):
        assert trial_record.start_date == "2020-01-15"
        assert trial_record.completion_date == "2023-06-30"

    def test_sponsor(self, trial_record):
        assert trial_record.sponsor == "Novartis"

    def test_has_results(self, trial_record):
        assert trial_record.has_results is True
        assert trial_record.results_summary is not None

    def test_multi_phase(self):
        raw = {
            "protocolSection": {
                "identificationModule": {"nctId": "NCT00000002", "briefTitle": "X"},
                "statusModule": {"overallStatus": "ACTIVE"},
                "designModule": {"phases": ["PHASE2", "PHASE3"]},
                "conditionsModule": {},
                "armsInterventionsModule": {},
                "outcomesModule": {},
                "sponsorCollaboratorsModule": {},
            },
            "hasResults": False,
        }
        record = _parse_study(raw)
        assert record.phase == "2/3"

    def test_missing_fields_use_defaults(self):
        raw = {
            "protocolSection": {
                "identificationModule": {"nctId": "NCT00000003", "briefTitle": "Minimal"},
                "statusModule": {},
                "designModule": {},
                "conditionsModule": {},
                "armsInterventionsModule": {},
                "outcomesModule": {},
                "sponsorCollaboratorsModule": {},
            },
            "hasResults": False,
        }
        record = _parse_study(raw)
        assert record.nct_id == "NCT00000003"
        assert record.phase == ""
        assert record.conditions == []
        assert record.interventions == []
        assert record.enrollment is None


# ── Modality guessing ────────────────────────────────────────────────────


class TestModalityGuess:
    def test_antibody(self):
        iv = Intervention(name="Trastuzumab", type="DRUG", description="monoclonal antibody")
        assert _guess_modality(iv) == Modality.ANTIBODY

    def test_adc(self):
        iv = Intervention(name="T-DXd", type="DRUG", description="antibody drug conjugate")
        assert _guess_modality(iv) == Modality.ADC

    def test_car_t(self):
        iv = Intervention(name="Axicabtagene", type="DRUG", description="CAR-T cell therapy")
        assert _guess_modality(iv) == Modality.CELL_THERAPY

    def test_gene_therapy(self):
        iv = Intervention(name="Zolgensma", type="DRUG", description="AAV gene therapy vector")
        assert _guess_modality(iv) == Modality.GENE_THERAPY

    def test_small_molecule_default(self):
        iv = Intervention(name="Imatinib", type="DRUG", description="oral kinase inhibitor")
        assert _guess_modality(iv) == Modality.SMALL_MOLECULE


# ── Graph node mapping ───────────────────────────────────────────────────


class TestMapTrialToNodes:
    def test_indications(self, trial_record):
        nodes = map_trial_to_graph_nodes(trial_record)
        assert len(nodes["indications"]) == 2
        assert nodes["indications"][0].name == "Chronic Myeloid Leukemia"

    def test_compounds_only_drugs(self, trial_record):
        nodes = map_trial_to_graph_nodes(trial_record)
        # Only Imatinib (DRUG), not Placebo (OTHER)
        assert len(nodes["compounds"]) == 1
        assert nodes["compounds"][0].name == "Imatinib"
        assert nodes["compounds"][0].modality == Modality.SMALL_MOLECULE

    def test_antibody_modality(self, antibody_trial):
        nodes = map_trial_to_graph_nodes(antibody_trial)
        assert nodes["compounds"][0].modality == Modality.ANTIBODY

    def test_endpoints(self, trial_record):
        nodes = map_trial_to_graph_nodes(trial_record)
        assert len(nodes["endpoints"]) == 1
        ep = nodes["endpoints"][0]
        assert ep.endpoint_type == EndpointType.PRIMARY
        assert ep.measurement_properties["timeframe"] == "12 months"

    def test_node_ids_include_nct(self, trial_record):
        nodes = map_trial_to_graph_nodes(trial_record)
        for group in nodes.values():
            for node in group:
                assert "NCT00000001" in node.id


# ── Integration test ─────────────────────────────────────────────────────


@pytest.mark.integration
class TestIntegration:
    @pytest.mark.asyncio
    async def test_search_real_api(self):
        client = ClinicalTrialsClient()
        records = await client.search(
            condition="lung cancer",
            status="COMPLETED",
            max_results=5,
        )
        assert len(records) > 0
        assert len(records) <= 5
        for r in records:
            assert r.nct_id.startswith("NCT")
            assert r.status == "COMPLETED"

    @pytest.mark.asyncio
    async def test_get_study_real_api(self):
        client = ClinicalTrialsClient()
        record = await client.get_study("NCT04880863")
        assert record.nct_id == "NCT04880863"
        assert record.has_results is True
