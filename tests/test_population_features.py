"""Round-17 tests for the LLM population-feature extractor.

The actual LLM call is mocked. The interesting behavior is the
conversion from raw LLM JSON output to SubgroupFeature list and how
the cache interacts with repeated calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from src.graph.models import PopulationNode, SubgroupFeature
from src.graph.population_features import (
    _features_from_llm_response,
    _parse_llm_json,
    extract_population_features_with_llm,
)


# ── Static parser logic ─────────────────────────────────────────────────


class TestFeaturesFromLLMResponse:
    def test_full_response_produces_all_axes(self):
        raw = {
            "line_of_therapy": "second",
            "prior_treatments": ["anti_pd1", "chemotherapy"],
            "required_mutations": [{"gene": "BRAF", "variant": "v600e"}],
            "biomarker_selection": [{"gene": "CD274", "level": "high"}],
            "disease_stage": "metastatic",
        }
        feats = _features_from_llm_response(raw)
        # 1 line + 1 extent + 2 prior_tx + 1 mutation + 1 biomarker = 6
        assert len(feats) == 6
        axes = {f.axis for f in feats}
        assert axes == {"line", "extent", "prior_tx", "gene"}

    def test_list_valued_scalar_fields_are_tolerated(self):
        """The LLM sometimes returns disease_stage / line_of_therapy as a LIST
        (surfaced at n=500). Each valid value adds a feature; no crash."""
        raw = {
            "disease_stage": ["iii", "metastatic"],  # was: AttributeError on .strip()
            "line_of_therapy": ["first"],
        }
        feats = _features_from_llm_response(raw)
        axes = {(f.axis, f.level) for f in feats}
        assert ("stage", "iii") in axes
        assert ("extent", "metastatic") in axes
        assert ("line", "first") in axes

    def test_non_dict_items_in_object_lists_are_skipped(self):
        """required_mutations / biomarker_selection items that come back as bare
        strings (not objects) are skipped rather than crashing on .get()."""
        raw = {
            "required_mutations": ["BRAF V600E", {"gene": "KRAS", "variant": "G12C"}],
            "biomarker_selection": ["nonsense"],
        }
        feats = _features_from_llm_response(raw)
        assert [(f.key, f.level) for f in feats] == [("KRAS", "g12c")]

    def test_empty_response_produces_nothing(self):
        raw = {
            "line_of_therapy": None,
            "prior_treatments": [],
            "required_mutations": [],
            "biomarker_selection": [],
            "disease_stage": None,
        }
        feats = _features_from_llm_response(raw)
        assert feats == []

    def test_unknown_line_value_is_dropped(self):
        raw = {"line_of_therapy": "fourth"}  # not in _LINE_LEVELS
        feats = _features_from_llm_response(raw)
        assert feats == []

    def test_unknown_prior_tx_is_dropped(self):
        raw = {"prior_treatments": ["fictional_therapy"]}
        feats = _features_from_llm_response(raw)
        assert feats == []

    def test_mutation_compose_axis_gene(self):
        raw = {
            "required_mutations": [{"gene": "kras", "variant": "G12C"}],
        }
        feats = _features_from_llm_response(raw)
        assert len(feats) == 1
        assert feats[0].axis == "gene"
        assert feats[0].key == "KRAS"
        # Variant normalized to lowercase alnum
        assert feats[0].level == "g12c"

    def test_biomarker_uses_gene_axis(self):
        raw = {
            "biomarker_selection": [{"gene": "MSI", "level": "High"}],
        }
        feats = _features_from_llm_response(raw)
        assert len(feats) == 1
        assert feats[0].axis == "gene"
        assert feats[0].key == "MSI"
        assert feats[0].level == "high"

    def test_extent_vs_stage_axis_routing(self):
        """metastatic/resectable/etc. → extent axis; roman numerals → stage."""
        raw_extent = {"disease_stage": "metastatic"}
        raw_stage = {"disease_stage": "iv"}
        assert _features_from_llm_response(raw_extent)[0].axis == "extent"
        assert _features_from_llm_response(raw_stage)[0].axis == "stage"

    def test_demographics_axes(self):
        """Round-31: age/sex/race extracted as subgroup stratifiers."""
        raw = {"age_group": "elderly", "sex": "female", "race_ethnicity": "asian"}
        got = {(f.axis, f.level) for f in _features_from_llm_response(raw)}
        assert got == {("age", "elderly"), ("sex", "female"), ("race", "asian")}

    def test_invalid_demographics_dropped(self):
        """Out-of-vocabulary demographic values don't create junk features."""
        raw = {"age_group": "middle_aged", "sex": "other", "race_ethnicity": "martian"}
        assert _features_from_llm_response(raw) == []


# ── JSON parser ─────────────────────────────────────────────────────────


class TestParseLLMJSON:
    def test_clean_json(self):
        text = '{"line_of_therapy": "first"}'
        assert _parse_llm_json(text) == {"line_of_therapy": "first"}

    def test_markdown_fence_stripped(self):
        text = '```json\n{"line_of_therapy": "first"}\n```'
        assert _parse_llm_json(text) == {"line_of_therapy": "first"}

    def test_embedded_prose_extracted(self):
        text = (
            'Here is the JSON:\n'
            '{"line_of_therapy": "first", "prior_treatments": []}\n'
            'Hope this helps.'
        )
        parsed = _parse_llm_json(text)
        assert parsed["line_of_therapy"] == "first"

    def test_malformed_returns_none(self):
        assert _parse_llm_json("not json at all") is None


# ── End-to-end with mocked LLM + cache ──────────────────────────────────


def _make_llm_mock(response_obj: dict):
    """Build an AsyncMock that returns a response shaped like
    anthropic.AsyncAnthropic.messages.create's return."""
    from types import SimpleNamespace
    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(return_value=SimpleNamespace(
        content=[SimpleNamespace(text=json.dumps(response_obj))],
    ))
    return mock_client


class TestExtractPopulationFeaturesWithLLM:
    @pytest.mark.asyncio
    async def test_extracts_and_caches(self, tmp_path):
        client = _make_llm_mock({
            "line_of_therapy": "second",
            "prior_treatments": ["anti_pd1"],
            "required_mutations": [],
            "biomarker_selection": [],
            "disease_stage": "metastatic",
        })
        cache_path = tmp_path / "pop_features.json"

        feats = await extract_population_features_with_llm(
            nct_id="NCT_TEST",
            title="Phase 2 X in metastatic melanoma post anti-PD-1",
            conditions=["Melanoma"],
            eligibility_criteria="Patients must have progressed on prior anti-PD-1.",
            client=client,
            cache_path=cache_path,
        )
        assert len(feats) == 3  # line + extent + prior_tx
        assert cache_path.exists()

        # Re-call should hit cache (no LLM call)
        client.messages.create.reset_mock()
        feats_again = await extract_population_features_with_llm(
            nct_id="NCT_TEST",
            title="(ignored)", conditions=[], eligibility_criteria="(ignored)",
            client=client, cache_path=cache_path,
        )
        client.messages.create.assert_not_called()
        assert len(feats_again) == 3

    @pytest.mark.asyncio
    async def test_offline_returns_empty(self, tmp_path):
        feats = await extract_population_features_with_llm(
            nct_id="NCT_TEST",
            title="x", conditions=["Melanoma"],
            eligibility_criteria="(any)",
            client=None,  # offline mode
            cache_path=tmp_path / "pf.json",
        )
        assert feats == []

    @pytest.mark.asyncio
    async def test_empty_eligibility_and_conditions_returns_empty(self, tmp_path):
        client = _make_llm_mock({"line_of_therapy": "first"})
        feats = await extract_population_features_with_llm(
            nct_id="NCT_TEST",
            title="x", conditions=[],
            eligibility_criteria="",
            client=client,
            cache_path=tmp_path / "pf.json",
        )
        # No input → no LLM call → no features
        client.messages.create.assert_not_called()
        assert feats == []

    @pytest.mark.asyncio
    async def test_malformed_llm_response_returns_empty_and_caches(self, tmp_path):
        from types import SimpleNamespace
        client = AsyncMock()
        client.messages.create = AsyncMock(return_value=SimpleNamespace(
            content=[SimpleNamespace(text="oh hai not json")],
        ))
        cache_path = tmp_path / "pf.json"
        feats = await extract_population_features_with_llm(
            nct_id="NCT_TEST",
            title="x", conditions=["Melanoma"],
            eligibility_criteria="Some text here.",
            client=client, cache_path=cache_path,
        )
        assert feats == []
        # Cache stores the {features, eligibility_labs} entry so we don't retry the
        # same broken response. Key is version-namespaced (prompt-version prefix) so
        # a prompt bump regenerates instead of returning stale features.
        from src.graph.population_features import _PROMPT_VERSION
        cached = json.loads(cache_path.read_text())
        assert cached[f"{_PROMPT_VERSION}:NCT_TEST"] == {"features": [], "eligibility_labs": []}


# ── Integration: composing into PopulationNode.compose_id ───────────────


class TestComposeWithExistingQualifiers:
    """Verify the LLM features merge cleanly with deterministic
    qualifier extractor output and produce a sensible population_id."""

    def test_compose_with_regex_qualifiers(self):
        from src.graph.indication_taxonomy import extract_indication_qualifiers

        regex_feats = extract_indication_qualifiers("Stage IV Metastatic Melanoma")
        llm_feats = _features_from_llm_response({
            "line_of_therapy": "second",
            "prior_treatments": ["anti_pd1"],
        })

        # Merge with dedup by slug.
        seen = {f.slug() for f in regex_feats}
        merged = list(regex_feats)
        for f in llm_feats:
            if f.slug() not in seen:
                merged.append(f)
                seen.add(f.slug())

        pop_id = PopulationNode.compose_id(merged)
        # The slug should include both regex- and LLM-derived axes.
        assert "extent_metastatic" in pop_id
        assert "stage_iv" in pop_id
        assert "line_second" in pop_id
        assert "prior_tx_anti_pd1_treated" in pop_id


# ── Static prompt regression ────────────────────────────────────────────


class TestRound17ClassifierPromptDirective:
    """Round-17 added an explicit "prefer the most-specific
    population_id" rule to the classifier prompt. Static test catches
    accidental revert."""

    def test_round17_directive_present(self):
        prompt = Path("src/annotation/prompts/classification_system.txt").read_text()
        # Anchor on round-17 marker.
        assert "Round-17" in prompt, (
            "Round-17 most-specific-population directive section missing"
        )
        lower = prompt.lower()
        # Key concepts.
        assert "most-specific" in lower or "most specific" in lower
        assert "__unselected" in prompt
        # Must mention the fallback-defeats-discrimination rationale.
        assert "discrimination" in lower or "fallback" in lower
