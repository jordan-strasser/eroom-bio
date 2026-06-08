"""Tests for the ClinicalTrials.gov v2 API client."""

import pytest

from src.graph.models import Modality
from src.ingestion.clinicaltrials import (
    ClinicalTrialsClient,
    Intervention,
    OutcomeMeasure,
    TrialRecord,
    _guess_modality,
    _parse_study,
    is_processable_drug_trial,
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


class TestProcessableDrugTrial:
    """Corpus-quality gate for multi-indication: keep drug/biologic trials with a
    primary outcome; drop the structural mismatches the drug-centric pipeline
    can't use (the non-onco n=10 smoke surfaced all four)."""

    def _rec(self, itypes, n_primary):
        return TrialRecord(
            nct_id="NCT", title="t",
            interventions=[Intervention(name=f"x{i}", type=t)
                           for i, t in enumerate(itypes)],
            primary_outcomes=[OutcomeMeasure(measure=f"m{i}", timeframe="")
                              for i in range(n_primary)],
        )

    def test_keeps_drug_with_primary(self):
        assert is_processable_drug_trial(self._rec(["DRUG"], 1))

    def test_keeps_biologic_and_drug_plus_procedure_combo(self):
        # a cancer drug+surgery combo still passes — it has ≥1 drug
        assert is_processable_drug_trial(self._rec(["BIOLOGICAL", "DRUG", "PROCEDURE"], 2))

    def test_drops_behavioral_only(self):
        assert not is_processable_drug_trial(self._rec(["BEHAVIORAL"], 1))

    def test_drops_procedure_only(self):
        assert not is_processable_drug_trial(self._rec(["PROCEDURE"], 2))

    def test_drops_observational_no_interventions(self):
        assert not is_processable_drug_trial(self._rec([], 1))

    def test_drops_drug_without_primary_outcome(self):
        # sparse record (e.g. the AN-1792 Alzheimer's trial: BIOLOGICAL, 0 primary)
        assert not is_processable_drug_trial(self._rec(["BIOLOGICAL"], 0))


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

    def test_no_endpoint_nodes_emitted(self, trial_record):
        # Endpoint nodes are intentionally not produced here; their canonical
        # id requires LLM classification + indication pairing.
        nodes = map_trial_to_graph_nodes(trial_record)
        assert "endpoints" not in nodes

    def test_canonical_compound_id(self, trial_record):
        nodes = map_trial_to_graph_nodes(trial_record)
        assert nodes["compounds"][0].id == "imatinib"

    def test_canonical_indication_id(self, trial_record):
        nodes = map_trial_to_graph_nodes(trial_record)
        assert nodes["indications"][0].id == "chronic_myeloid_leukemia"


# ── Round-27 dose-suffix stripping ───────────────────────────────────────


class TestCodenameINNParenthetical:
    """Biologic-resolution fix: 'BMS 188667 (Abatacept)' must resolve to the INN
    (abatacept, ChEMBL/OT-resolvable), not the codename bms_188667 (unresolvable
    → diagnostic-filtered → empty arm → trial dropped). Surfaced by the n=10
    multi-indication smoke (autoimmune trials are biologic-heavy)."""

    def test_codename_paren_inn_prefers_inn(self):
        from src.ingestion.clinicaltrials import strip_parenthetical_brand as s
        assert s("BMS 188667 (Abatacept)") == "Abatacept"
        assert s("AMG 510 (Sotorasib)") == "Sotorasib"
        assert s("MK-3475 (Pembrolizumab)") == "Pembrolizumab"

    def test_inn_main_with_brand_paren_unchanged(self):
        from src.ingestion.clinicaltrials import strip_parenthetical_brand as s
        # main token is already the INN → strip the brand paren as before
        assert s("Sorafenib (Nexavar; BAY43-9006)") == "Sorafenib"
        assert s("Imatinib (Gleevec)") == "Imatinib"

    def test_combo_paren_still_none(self):
        from src.ingestion.clinicaltrials import strip_parenthetical_brand as s
        assert s("Drug X (with Y)") is None


class TestStripDoseSuffix:
    """Round 27 fix for the round-26 bevacizumab AVANT regression: CT.gov
    intervention names often embed dose info. Without stripping, OT /
    ChEMBL lookup fails, SapBERT keeps the dose-laden slug canonical,
    and the cross-reference name-match heuristic creates phantom edges.
    """

    def test_strips_mass_per_kg_dose(self):
        from src.ingestion.clinicaltrials import strip_dose_suffix
        assert strip_dose_suffix("Bevacizumab 7.5 mg/kg") == "Bevacizumab"
        assert strip_dose_suffix("Bevacizumab 7.5mg/kg") == "Bevacizumab"

    def test_strips_mass_per_m2_dose(self):
        from src.ingestion.clinicaltrials import strip_dose_suffix
        assert strip_dose_suffix("Capecitabine 1000 mg/m^2") == "Capecitabine"
        assert strip_dose_suffix("Capecitabine 1000 mg/m2") == "Capecitabine"

    def test_strips_simple_mg_dose(self):
        from src.ingestion.clinicaltrials import strip_dose_suffix
        assert strip_dose_suffix("Aspirin 81 mg") == "Aspirin"
        assert strip_dose_suffix("Donepezil 10 mg") == "Donepezil"

    def test_strips_biologic_iu(self):
        from src.ingestion.clinicaltrials import strip_dose_suffix
        assert strip_dose_suffix("Insulin 100 IU") == "Insulin"
        assert strip_dose_suffix("Interferon 5 MIU") == "Interferon"

    def test_strips_route_frequency(self):
        from src.ingestion.clinicaltrials import strip_dose_suffix
        assert strip_dose_suffix("Atorvastatin 10 mg PO") == "Atorvastatin"
        assert strip_dose_suffix("Furosemide 40 mg BID") == "Furosemide"
        assert strip_dose_suffix("Omeprazole 20 mg q12h") == "Omeprazole"

    def test_preserves_leading_chemical_prefix(self):
        """Critical: '5-Fluorouracil' must NOT lose its leading 5 (part
        of the chemical name, not a dose). Dose pattern requires
        whitespace BEFORE the number."""
        from src.ingestion.clinicaltrials import strip_dose_suffix
        assert strip_dose_suffix("5-Fluorouracil") == "5-Fluorouracil"
        assert strip_dose_suffix("5_fluorouracil") == "5_fluorouracil"

    def test_idempotent_on_clean_name(self):
        from src.ingestion.clinicaltrials import strip_dose_suffix
        assert strip_dose_suffix("Bevacizumab") == "Bevacizumab"
        assert strip_dose_suffix("imatinib") == "imatinib"

    def test_empty_after_strip_returns_original(self):
        from src.ingestion.clinicaltrials import strip_dose_suffix
        # Pathological: name is JUST a dose. Return original so the
        # downstream populator at least keeps something to attempt.
        assert strip_dose_suffix("10 mg/kg") == "10 mg/kg"

    def test_canonical_compound_names_strips_dose(self):
        """The integration: canonical_compound_names is what the
        populator actually calls. It must produce dose-free names."""
        from src.ingestion.clinicaltrials import canonical_compound_names
        assert canonical_compound_names("Bevacizumab 7.5 mg/kg") == ["Bevacizumab"]
        assert canonical_compound_names("Capecitabine 1000 mg/m^2") == ["Capecitabine"]
        # Combo with dose-laden constituents
        result = canonical_compound_names(
            "Oxaliplatin + Capecitabine 1000 mg/m^2 + Bevacizumab 7.5 mg/kg"
        )
        assert set(result) == {"Oxaliplatin", "Capecitabine", "Bevacizumab"}


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
