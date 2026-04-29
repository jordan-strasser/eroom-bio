"""Tests for the AI extraction pipeline."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.annotation.extractor import (
    Extractor,
    _format_trial_for_prompt,
    _parse_extraction_response,
)
from src.ingestion.clinicaltrials import Intervention, OutcomeMeasure, TrialRecord


# ── Fixtures ─────────────────────────────────────────────────────────────

VALID_RESPONSE_JSON = {
    "nct_id": "NCT00000001",
    "therapeutic_hypothesis": {
        "compound": "Imatinib",
        "claimed_target": "BCR-ABL",
        "proposed_mechanism": "Tyrosine kinase inhibition",
        "intended_biology": "Inhibit leukemic cell proliferation",
        "target_population": "Newly diagnosed CML patients",
        "primary_endpoint": "Major Molecular Response at 12 months",
        "rationale_strength": "strong",
    },
    "results": {
        "primary_endpoint_met": True,
        "effect_size": "HR 0.73, 95% CI 0.59-0.90",
        "p_value": 0.003,
        "target_engagement_evidence": "BCR-ABL transcript levels decreased >3 log",
        "biomarker_changes": ["BCR-ABL IS reduced to <0.1%"],
        "subgroup_findings": ["Better response in patients with e13a2 transcript"],
        "safety_signals": ["Grade 3 neutropenia in 12% of patients"],
        "dose_information": "400mg daily oral",
    },
    "context": {
        "comparator": "Interferon-alpha + cytarabine",
        "sample_size": 1106,
        "duration": "12 months primary, 60 months follow-up",
        "investigator_commentary": "Imatinib showed unprecedented efficacy",
    },
}


@pytest.fixture
def trial():
    return TrialRecord(
        nct_id="NCT00000001",
        title="Phase 3 Imatinib vs IFN-alpha in CML",
        phase="3",
        status="COMPLETED",
        conditions=["Chronic Myeloid Leukemia"],
        interventions=[
            Intervention(
                name="Imatinib",
                type="DRUG",
                description="400mg daily oral TKI",
            ),
        ],
        primary_outcomes=[
            OutcomeMeasure(
                measure="Major Molecular Response",
                timeframe="12 months",
            ),
        ],
        enrollment=1106,
        sponsor="Novartis",
        has_results=True,
    )


def _mock_api_response(text: str):
    """Create a mock Anthropic API response."""
    content_block = SimpleNamespace(text=text)
    return SimpleNamespace(content=[content_block])


@pytest.fixture
def mock_client():
    client = AsyncMock()
    client.messages.create = AsyncMock(
        return_value=_mock_api_response(json.dumps(VALID_RESPONSE_JSON))
    )
    return client


# ── Prompt formatting ────────────────────────────────────────────────────


class TestFormatPrompt:
    def test_includes_nct_id(self, trial):
        prompt = _format_trial_for_prompt(trial, None)
        assert "NCT00000001" in prompt

    def test_includes_conditions(self, trial):
        prompt = _format_trial_for_prompt(trial, None)
        assert "Chronic Myeloid Leukemia" in prompt

    def test_includes_interventions(self, trial):
        prompt = _format_trial_for_prompt(trial, None)
        assert "Imatinib" in prompt
        assert "400mg" in prompt

    def test_includes_abstract_when_provided(self, trial):
        prompt = _format_trial_for_prompt(trial, "This is the abstract text.")
        assert "This is the abstract text." in prompt

    def test_no_abstract_section_when_none(self, trial):
        prompt = _format_trial_for_prompt(trial, None)
        assert "## Abstract" not in prompt


# ── Response parsing ─────────────────────────────────────────────────────


class TestParseResponse:
    def test_parses_valid_response(self):
        extraction = _parse_extraction_response(VALID_RESPONSE_JSON, "NCT00000001")
        assert extraction.trial_id == "NCT00000001"
        assert extraction.compound_name == "Imatinib"
        assert extraction.target_name == "BCR-ABL"
        assert extraction.primary_endpoint_met is True
        assert extraction.p_value == 0.003

    def test_extracts_effect_size_float(self):
        extraction = _parse_extraction_response(VALID_RESPONSE_JSON, "NCT00000001")
        assert extraction.effect_size == pytest.approx(0.73)

    def test_extracts_safety_signals(self):
        extraction = _parse_extraction_response(VALID_RESPONSE_JSON, "NCT00000001")
        assert len(extraction.safety_signals) == 1
        assert "neutropenia" in extraction.safety_signals[0]

    def test_extracts_subgroup_findings(self):
        extraction = _parse_extraction_response(VALID_RESPONSE_JSON, "NCT00000001")
        assert len(extraction.subgroup_findings) == 1

    def test_extracts_biomarker_data(self):
        extraction = _parse_extraction_response(VALID_RESPONSE_JSON, "NCT00000001")
        assert extraction.biomarker_data["target_engagement"] is not None
        assert len(extraction.biomarker_data["biomarker_changes"]) == 1

    def test_handles_missing_fields(self):
        minimal = {
            "nct_id": "NCT999",
            "therapeutic_hypothesis": {},
            "results": {},
            "context": {},
        }
        extraction = _parse_extraction_response(minimal, "NCT999")
        assert extraction.trial_id == "NCT999"
        assert extraction.compound_name == ""
        assert extraction.primary_endpoint_met is None
        assert extraction.effect_size is None

    def test_handles_null_p_value(self):
        data = {**VALID_RESPONSE_JSON, "results": {**VALID_RESPONSE_JSON["results"], "p_value": None}}
        extraction = _parse_extraction_response(data, "NCT00000001")
        assert extraction.p_value is None

    def test_summary_contains_full_json(self):
        extraction = _parse_extraction_response(VALID_RESPONSE_JSON, "NCT00000001")
        summary_data = json.loads(extraction.summary)
        assert summary_data["nct_id"] == "NCT00000001"


# ── Extractor class ──────────────────────────────────────────────────────


class TestExtractor:
    @pytest.mark.asyncio
    async def test_extract_success(self, mock_client, trial, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "src.annotation.extractor._ANNOTATIONS_DIR", tmp_path
        )
        extractor = Extractor(mock_client)
        extraction = await extractor.extract(trial)

        assert extraction.trial_id == "NCT00000001"
        assert extraction.compound_name == "Imatinib"
        mock_client.messages.create.assert_called_once()

        # Check annotation file saved
        saved = tmp_path / "NCT00000001_extraction.json"
        assert saved.exists()
        saved_data = json.loads(saved.read_text())
        assert saved_data["nct_id"] == "NCT00000001"

    @pytest.mark.asyncio
    async def test_extract_retries_on_bad_json(self, trial, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "src.annotation.extractor._ANNOTATIONS_DIR", tmp_path
        )
        client = AsyncMock()
        # First call returns invalid JSON, second returns valid
        client.messages.create = AsyncMock(
            side_effect=[
                _mock_api_response("not valid json {{{"),
                _mock_api_response(json.dumps(VALID_RESPONSE_JSON)),
            ]
        )
        extractor = Extractor(client)
        extraction = await extractor.extract(trial)
        assert extraction.trial_id == "NCT00000001"
        assert client.messages.create.call_count == 2

    @pytest.mark.asyncio
    async def test_extract_fails_after_max_retries(self, trial, monkeypatch):
        monkeypatch.setattr("src.annotation.extractor.MAX_RETRIES", 2)
        client = AsyncMock()
        client.messages.create = AsyncMock(
            return_value=_mock_api_response("not json at all")
        )
        extractor = Extractor(client)
        with pytest.raises(ValueError, match="Failed to get valid JSON"):
            await extractor.extract(trial)

    @pytest.mark.asyncio
    async def test_extract_strips_markdown_fences(self, mock_client, trial, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "src.annotation.extractor._ANNOTATIONS_DIR", tmp_path
        )
        # Wrap valid JSON in markdown code fences
        fenced = f"```json\n{json.dumps(VALID_RESPONSE_JSON)}\n```"
        mock_client.messages.create = AsyncMock(
            return_value=_mock_api_response(fenced)
        )
        extractor = Extractor(mock_client)
        extraction = await extractor.extract(trial)
        assert extraction.trial_id == "NCT00000001"


class TestExtractBatch:
    @pytest.mark.asyncio
    async def test_batch_extraction(self, mock_client, trial, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "src.annotation.extractor._ANNOTATIONS_DIR", tmp_path
        )
        extractor = Extractor(mock_client)
        trials = [trial, trial]
        results = await extractor.extract_batch(trials, concurrency=2)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_batch_skips_failures(self, trial, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "src.annotation.extractor._ANNOTATIONS_DIR", tmp_path
        )
        monkeypatch.setattr("src.annotation.extractor.MAX_RETRIES", 1)
        client = AsyncMock()
        # All calls return bad JSON
        client.messages.create = AsyncMock(
            return_value=_mock_api_response("bad")
        )
        extractor = Extractor(client)
        results = await extractor.extract_batch([trial], concurrency=1)
        assert len(results) == 0
