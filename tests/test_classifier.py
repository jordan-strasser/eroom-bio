"""Tests for the failure mode classifier."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.annotation.classifier import (
    Classifier,
    _format_classification_prompt,
    _needs_expert_review,
    _parse_classification,
    annotate_trial,
)
from src.annotation.extractor import Extractor
from src.annotation.taxonomy import FailureMode, TrialExtraction
from src.ingestion.clinicaltrials import Intervention, OutcomeMeasure, TrialRecord


# ── Fixtures ─────────────────────────────────────────────────────────────


VALID_CLASSIFICATION_JSON = {
    "nct_id": "NCT00000001",
    "trial_outcome": "success",
    "reasoning": "Step 1: Target engagement confirmed via BCR-ABL transcript reduction. Step 2: Pathway modulation confirmed. Step 3: Primary endpoint met with HR 0.42. Step 4: PD-L1 subgroup showed greater benefit. Step 5: No confounders.",
    "failure_modes": [
        {
            "mode": "no_target_engagement",
            "confidence": 0.95,
            "evidence": "BCR-ABL transcript levels decreased >3 log, confirming target engagement",
        }
    ],
    "edges_to_update": [
        {
            "edge_type": "binds_to",
            "source_entity": "Nivolumab",
            "target_entity": "PD-1",
            "direction": "strengthen",
            "magnitude": 0.9,
            "reasoning": "Strong target engagement evidence",
        },
        {
            "edge_type": "mechanism_affects",
            "source_entity": "PD-1 blockade",
            "target_entity": "T-cell activation",
            "direction": "strengthen",
            "magnitude": 0.8,
            "reasoning": "Downstream immune activation confirmed",
        },
    ],
    "confidence_overall": 0.92,
    "needs_expert_review": False,
    "review_reason": None,
}


_DEFAULT_BIOMARKER = {
    "target_engagement": "BCR-ABL reduced",
    "biomarker_changes": ["BCR-ABL IS <0.1%"],
}
_DEFAULT_SAFETY = ["Grade 3 neutropenia"]
_DEFAULT_SUBGROUPS = ["Better in e13a2 subgroup"]


def _make_extraction(
    trial_id: str = "NCT00000001",
    endpoint_met: bool | None = True,
    biomarker_data: dict | None = None,
    subgroup_findings: list | None = None,
    safety_signals: list | None = None,
) -> TrialExtraction:
    return TrialExtraction(
        trial_id=trial_id,
        compound_name="Imatinib",
        target_name="BCR-ABL",
        primary_endpoint="Overall Survival",
        primary_endpoint_met=endpoint_met,
        effect_size=0.42,
        p_value=0.0001,
        biomarker_data=_DEFAULT_BIOMARKER if biomarker_data is None else biomarker_data,
        safety_signals=_DEFAULT_SAFETY if safety_signals is None else safety_signals,
        subgroup_findings=_DEFAULT_SUBGROUPS if subgroup_findings is None else subgroup_findings,
        summary=json.dumps(VALID_CLASSIFICATION_JSON),
    )


def _mock_api_response(text: str):
    content_block = SimpleNamespace(text=text)
    return SimpleNamespace(content=[content_block])


@pytest.fixture
def mock_client():
    client = AsyncMock()
    client.messages.create = AsyncMock(
        return_value=_mock_api_response(json.dumps(VALID_CLASSIFICATION_JSON))
    )
    return client


# ── Prompt formatting ────────────────────────────────────────────────────


class TestFormatPrompt:
    def test_includes_trial_id(self):
        ext = _make_extraction()
        prompt = _format_classification_prompt(ext)
        assert "NCT00000001" in prompt

    def test_includes_compound(self):
        ext = _make_extraction()
        prompt = _format_classification_prompt(ext)
        assert "Imatinib" in prompt

    def test_includes_endpoint_result(self):
        ext = _make_extraction()
        prompt = _format_classification_prompt(ext)
        assert "True" in prompt  # primary_endpoint_met
        assert "0.42" in prompt  # effect_size

    def test_includes_biomarker_data(self):
        ext = _make_extraction()
        prompt = _format_classification_prompt(ext)
        assert "BCR-ABL" in prompt

    def test_includes_safety(self):
        ext = _make_extraction()
        prompt = _format_classification_prompt(ext)
        assert "neutropenia" in prompt

    def test_handles_empty_data(self):
        ext = _make_extraction(
            biomarker_data={}, subgroup_findings=[], safety_signals=[]
        )
        prompt = _format_classification_prompt(ext)
        assert "No biomarker data" in prompt
        assert "No safety signals" in prompt


# ── Expert review logic ──────────────────────────────────────────────────


class TestNeedsExpertReview:
    def test_low_confidence_triggers_review(self):
        raw = {"confidence_overall": 0.4, "failure_modes": []}
        ext = _make_extraction()
        needs, reason = _needs_expert_review(raw, ext)
        assert needs is True
        assert "Low classification confidence" in reason

    def test_high_confidence_no_review(self):
        raw = {
            "confidence_overall": 0.9,
            "failure_modes": [{"mode": "no_target_engagement", "confidence": 0.9}],
        }
        ext = _make_extraction()
        needs, _ = _needs_expert_review(raw, ext)
        assert needs is False

    def test_ambiguous_top_modes_trigger_review(self):
        raw = {
            "confidence_overall": 0.8,
            "failure_modes": [
                {"mode": "no_target_engagement", "confidence": 0.55},
                {"mode": "dose_limiting_toxicity", "confidence": 0.50},
            ],
        }
        ext = _make_extraction()
        needs, reason = _needs_expert_review(raw, ext)
        assert needs is True
        assert "Ambiguous" in reason

    def test_well_separated_modes_no_review(self):
        raw = {
            "confidence_overall": 0.85,
            "failure_modes": [
                {"mode": "no_target_engagement", "confidence": 0.85},
                {"mode": "dose_limiting_toxicity", "confidence": 0.10},
            ],
        }
        ext = _make_extraction()
        needs, _ = _needs_expert_review(raw, ext)
        assert needs is False

    def test_no_biomarker_no_subgroup_triggers_review(self):
        raw = {
            "confidence_overall": 0.8,
            "failure_modes": [{"mode": "no_target_engagement", "confidence": 0.8}],
        }
        ext = _make_extraction(biomarker_data={}, subgroup_findings=[])
        needs, reason = _needs_expert_review(raw, ext)
        assert needs is True
        assert "biomarker" in reason.lower()

    def test_insufficient_info_triggers_review(self):
        raw = {
            "confidence_overall": 0.7,
            "failure_modes": [
                {"mode": "insufficient_information", "confidence": 0.7},
            ],
        }
        ext = _make_extraction()
        needs, reason = _needs_expert_review(raw, ext)
        assert needs is True
        assert "insufficient_information" in reason


# ── Response parsing ─────────────────────────────────────────────────────


class TestParseClassification:
    def test_parses_valid_response(self):
        ext = _make_extraction()
        clf = _parse_classification(VALID_CLASSIFICATION_JSON, "NCT00000001", ext)
        assert clf.trial_id == "NCT00000001"
        assert clf.primary_failure_mode == FailureMode.NO_TARGET_ENGAGEMENT
        assert clf.confidence == pytest.approx(0.92)
        assert len(clf.evidence_quotes) == 1

    def test_picks_highest_confidence_as_primary(self):
        raw = {
            **VALID_CLASSIFICATION_JSON,
            "failure_modes": [
                {"mode": "dose_limiting_toxicity", "confidence": 0.3, "evidence": "low"},
                {"mode": "no_target_engagement", "confidence": 0.8, "evidence": "high"},
            ],
        }
        ext = _make_extraction()
        clf = _parse_classification(raw, "NCT00000001", ext)
        assert clf.primary_failure_mode == FailureMode.NO_TARGET_ENGAGEMENT
        assert clf.secondary_failure_modes == [FailureMode.DOSE_LIMITING_TOXICITY]

    def test_empty_modes_defaults_to_insufficient(self):
        raw = {**VALID_CLASSIFICATION_JSON, "failure_modes": []}
        ext = _make_extraction()
        clf = _parse_classification(raw, "NCT00000001", ext)
        assert clf.primary_failure_mode == FailureMode.INSUFFICIENT_INFORMATION
        assert clf.confidence == 0.0

    def test_stashes_raw_and_review_fields(self):
        ext = _make_extraction()
        clf = _parse_classification(VALID_CLASSIFICATION_JSON, "NCT00000001", ext)
        assert hasattr(clf, "_raw")
        assert clf._raw["trial_outcome"] == "success"


# ── Classifier class ─────────────────────────────────────────────────────


class TestClassifier:
    @pytest.mark.asyncio
    async def test_classify_success(self, mock_client, tmp_path, monkeypatch):
        monkeypatch.setattr("src.annotation.classifier._ANNOTATIONS_DIR", tmp_path)
        classifier = Classifier(mock_client)
        ext = _make_extraction()
        clf = await classifier.classify(ext)
        assert clf.trial_id == "NCT00000001"
        assert clf.primary_failure_mode == FailureMode.NO_TARGET_ENGAGEMENT
        mock_client.messages.create.assert_called_once()

        saved = tmp_path / "NCT00000001_classification.json"
        assert saved.exists()

    @pytest.mark.asyncio
    async def test_classify_retries_on_bad_json(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.annotation.classifier._ANNOTATIONS_DIR", tmp_path)
        client = AsyncMock()
        client.messages.create = AsyncMock(
            side_effect=[
                _mock_api_response("{bad json"),
                _mock_api_response(json.dumps(VALID_CLASSIFICATION_JSON)),
            ]
        )
        classifier = Classifier(client)
        clf = await classifier.classify(_make_extraction())
        assert clf.trial_id == "NCT00000001"
        assert client.messages.create.call_count == 2

    @pytest.mark.asyncio
    async def test_classify_strips_markdown(self, mock_client, tmp_path, monkeypatch):
        monkeypatch.setattr("src.annotation.classifier._ANNOTATIONS_DIR", tmp_path)
        fenced = f"```json\n{json.dumps(VALID_CLASSIFICATION_JSON)}\n```"
        mock_client.messages.create = AsyncMock(
            return_value=_mock_api_response(fenced)
        )
        classifier = Classifier(mock_client)
        clf = await classifier.classify(_make_extraction())
        assert clf.trial_id == "NCT00000001"


# ── End-to-end annotation ────────────────────────────────────────────────


class TestAnnotateTrial:
    @pytest.mark.asyncio
    async def test_annotate_trial_end_to_end(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.annotation.extractor._ANNOTATIONS_DIR", tmp_path)
        monkeypatch.setattr("src.annotation.classifier._ANNOTATIONS_DIR", tmp_path)

        extraction_json = {
            "nct_id": "NCT00000001",
            "therapeutic_hypothesis": {"compound": "TestDrug", "claimed_target": "TP53"},
            "results": {"primary_endpoint_met": False, "effect_size": "", "p_value": 0.3},
            "context": {"sample_size": 200},
        }

        client = AsyncMock()
        # First call: extraction, second call: classification
        client.messages.create = AsyncMock(
            side_effect=[
                _mock_api_response(json.dumps(extraction_json)),
                _mock_api_response(json.dumps(VALID_CLASSIFICATION_JSON)),
            ]
        )

        extractor = Extractor(client)
        classifier = Classifier(client)

        trial = TrialRecord(
            nct_id="NCT00000001",
            title="Test Trial",
            phase="3",
            status="COMPLETED",
            conditions=["Cancer"],
            interventions=[Intervention(name="TestDrug", type="DRUG", description="")],
            primary_outcomes=[OutcomeMeasure(measure="OS", timeframe="12mo")],
            has_results=True,
        )

        extraction, classification = await annotate_trial(trial, extractor, classifier)
        assert extraction.trial_id == "NCT00000001"
        assert classification.trial_id == "NCT00000001"
        assert client.messages.create.call_count == 2
