"""Tests for the subgroup feature canonicalization vocabulary."""

from __future__ import annotations

import json

import pytest

from src.graph.models import SubgroupFeature
from src.graph.subgroup_taxonomy import (
    GENE_LEVELS,
    NON_GENE_AXES,
    UNMAPPED_LOG_PATH,
    VARIANT_PATTERN,
    canonicalize_feature,
    is_canonical,
    log_unmapped,
    vocabulary_for_prompt,
)


@pytest.fixture(autouse=True)
def _clean_unmapped_log(tmp_path, monkeypatch):
    """Redirect the unmapped-features log to a tmp path so tests don't
    pollute the project's data/dev directory.
    """
    test_log = tmp_path / "unmapped.jsonl"
    monkeypatch.setattr(
        "src.graph.subgroup_taxonomy.UNMAPPED_LOG_PATH", test_log
    )
    return test_log


# ── Gene axis (open vocab) ──────────────────────────────────────────────


class TestGeneAxis:
    def test_general_level_canonicalizes(self):
        f = canonicalize_feature("gene", "CD274", "high", "PD-L1 ≥1%")
        assert f.axis == "gene"
        assert f.key == "CD274"
        assert f.level == "high"
        assert f.raw_descriptor == "PD-L1 ≥1%"

    def test_specific_variant_passes_through(self):
        f = canonicalize_feature("gene", "KRAS", "G12C")
        assert f.axis == "gene"
        assert f.level == "G12C"

    def test_variant_pattern_matches_canonical_forms(self):
        for v in ("G12C", "V600E", "T790M", "L858R", "R248Q"):
            assert VARIANT_PATTERN.match(v), v

    def test_invalid_gene_level_falls_to_other(self):
        f = canonicalize_feature("gene", "EGFR", "somewhat_mutated")
        assert f.axis == "other"
        assert is_canonical(f) is False

    def test_invalid_hugo_symbol_falls_to_other(self):
        f = canonicalize_feature("gene", "???", "high")
        assert f.axis == "other"


# ── Non-gene closed vocab ────────────────────────────────────────────────


class TestNonGeneAxes:
    @pytest.mark.parametrize("axis,level", [
        ("line", "first"), ("line", "second"),
        ("performance", "good"), ("performance", "poor"),
        ("age", "elderly"),
        ("prior_tx", "naive"),
        ("signature", "msi_high"),
    ])
    def test_known_axis_level_canonicalizes(self, axis, level):
        f = canonicalize_feature(axis, "", level)
        assert f.axis == axis
        assert f.level == level
        assert is_canonical(f)

    def test_unknown_level_falls_to_other(self):
        f = canonicalize_feature("line", "", "twentieth")
        assert f.axis == "other"

    def test_unknown_axis_falls_to_other(self):
        f = canonicalize_feature("weight", "", "overweight", "BMI > 30")
        assert f.axis == "other"
        assert f.raw_descriptor == "BMI > 30"


# ── Response axis (RECIST strata) ───────────────────────────────────────


class TestResponseAxis:
    @pytest.mark.parametrize("level_in,expected", [
        ("complete_response", "complete_response"),
        ("partial_response", "partial_response"),
        ("stable_disease", "stable_disease"),
        ("progressive_disease", "progressive_disease"),
        ("Complete Response", "complete_response"),
        ("CR", "complete_response"),
        ("PR", "partial_response"),
        ("SD", "stable_disease"),
        ("PD", "progressive_disease"),
        ("responder", "responder"),
        ("non_responder", "non_responder"),
    ])
    def test_response_levels_canonicalize(self, level_in, expected):
        f = canonicalize_feature("response", "", level_in, "Best response")
        assert f.axis == "response"
        assert f.level == expected
        assert is_canonical(f)

    def test_unknown_response_level_falls_to_other(self):
        f = canonicalize_feature("response", "", "mixed_response")
        assert f.axis == "other"

    def test_response_slug_round_trip(self):
        f = canonicalize_feature("response", "", "CR", "Complete Response")
        assert f.slug() == "response_complete_response"


# ── Slug rendering (used to compose PopulationNode ids) ─────────────────


class TestFeatureSlug:
    def test_gene_slug_uses_key_and_level(self):
        f = SubgroupFeature(axis="gene", key="CD274", level="high")
        assert f.slug() == "cd274_high"

    def test_gene_variant_slug(self):
        f = SubgroupFeature(axis="gene", key="KRAS", level="G12C")
        assert f.slug() == "kras_g12c"

    def test_non_gene_slug_uses_axis_and_level(self):
        f = SubgroupFeature(axis="line", level="first")
        assert f.slug() == "line_first"


# ── Logging unmapped features ───────────────────────────────────────────


class TestLogUnmapped:
    def test_canonical_feature_is_no_op(self, _clean_unmapped_log):
        canonical = canonicalize_feature("line", "", "first")
        log_unmapped(canonical, "NCT123", log_path=_clean_unmapped_log)
        assert not _clean_unmapped_log.exists()

    def test_other_feature_appends(self, _clean_unmapped_log):
        other = canonicalize_feature("weight", "", "overweight", "BMI > 30")
        log_unmapped(other, "NCT456", log_path=_clean_unmapped_log)
        records = [
            json.loads(line)
            for line in _clean_unmapped_log.read_text().splitlines()
        ]
        assert len(records) == 1
        assert records[0]["trial_id"] == "NCT456"
        assert records[0]["raw_descriptor"] == "BMI > 30"


# ── Vocabulary export for prompt ────────────────────────────────────────


class TestVocabularyForPrompt:
    def test_includes_gene_axis(self):
        text = vocabulary_for_prompt()
        assert "axis='gene'" in text
        assert "HUGO" in text

    def test_includes_each_non_gene_axis(self):
        text = vocabulary_for_prompt()
        for axis in NON_GENE_AXES:
            assert f"axis='{axis}'" in text

    def test_mentions_other_fallback(self):
        assert "other" in vocabulary_for_prompt()
