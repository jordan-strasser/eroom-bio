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
        # "high" collapses to "positive" so cd274_high and cd274_positive
        # don't fragment for the same expression biology (fixes.md #5).
        f = canonicalize_feature("gene", "CD274", "high", "PD-L1 ≥1%")
        assert f.axis == "gene"
        assert f.key == "CD274"
        assert f.level == "positive"
        assert f.raw_descriptor == "PD-L1 ≥1%"

    def test_low_collapses_to_negative(self):
        f = canonicalize_feature("gene", "CD274", "low", "PD-L1 < 1%")
        assert f.axis == "gene"
        assert f.level == "negative"

    def test_positive_passes_through(self):
        f = canonicalize_feature("gene", "CD274", "positive", "PD-L1 ≥1%")
        assert f.level == "positive"

    def test_threshold_variants_collapse_onto_one_pop_id(self):
        # PD-L1 ≥ 1%, ≥ 5%, ≥ 10% all map to (cd274, positive). Different
        # thresholds for the same direction should not fork the population.
        from src.graph.models import PopulationNode
        pop_ids = {
            PopulationNode.compose_id(
                "melanoma",
                [canonicalize_feature("gene", "CD274", level, desc)],
            )
            for level, desc in [
                ("high", "PD-L1 ≥ 1%"),
                ("high", "PD-L1 ≥ 5%"),
                ("high", "PD-L1 ≥ 10%"),
                ("positive", "PD-L1 positive"),
            ]
        }
        assert pop_ids == {"melanoma__cd274_positive"}

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

    @pytest.mark.parametrize("stale_axis", ["other", "", "  ", "OTHER"])
    @pytest.mark.parametrize("level_in", [
        "complete_response", "partial_response", "stable_disease",
        "progressive_disease", "CR", "PR", "SD", "PD",
    ])
    def test_stale_axis_other_promotes_to_response(self, stale_axis, level_in):
        """Cached extractions emitted RECIST states with axis='other' before
        the response-axis prompt rule was added. Auto-promote so the dev log
        doesn't fill with re-mappable noise and the populator can treat them
        as response strata (and skip chain-forking on them per the patient-
        strata rule)."""
        f = canonicalize_feature(stale_axis, "", level_in, "Best response")
        assert f.axis == "response"
        assert is_canonical(f)


# ── Round 3.2 dev-log cleanup ──────────────────────────────────────────


class TestAntibodyStatusAxis:
    """The antibody_status axis is a real patient stratifier — HAHA-positive
    patients can neutralize biologic drugs. Round 3.2 added it after the
    dev-log audit surfaced HAHA-positive / HAHA-negative entries falling to
    axis='other'."""

    @pytest.mark.parametrize("level_in,expected", [
        ("haha_positive_baseline", "positive"),
        ("haha_positive", "positive"),
        ("haha_negative", "negative"),
        ("ada_positive", "positive"),
        ("ada_negative", "negative"),
        ("anti_drug_antibody_positive", "positive"),
        ("antidrug_negative", "negative"),
    ])
    def test_haha_promotes_to_antibody_status(self, level_in, expected):
        f = canonicalize_feature("other", "", level_in, "HAHA status")
        assert f.axis == "antibody_status"
        assert f.level == expected
        assert is_canonical(f)

    def test_canonical_axis_passthrough(self):
        f = canonicalize_feature("antibody_status", "", "positive", "ADA positive")
        assert f.axis == "antibody_status"
        assert f.level == "positive"
        assert is_canonical(f)


class TestKnownNonStratifierDescriptors:
    """Some descriptors come through extraction looking like subgroups but
    are actually analysis time points, individual patient labels, or
    outcome-change Likert labels — not patient strata. Mark them with a
    sentinel level so log_unmapped suppresses them silently."""

    @pytest.mark.parametrize("descriptor", [
        "Final analysis",
        "Primary completion",
        "Interim analysis",
        "Month 3",
        "Month 21",
        "Week 12",
        "Day 22",
        "Cycle 4",
        "12-week landmark",
        "CD8 T cells per mm² day 22",
        "CD4 T cells per mm2 of tumor prevaccine (day 0)",
        "Patient #1",
        "Patient 12",
        "Subject #5",
        "better",
        "worse",
        "no change",
        "missing",
        "improved",
        "worsened",
        "Yes",
        "No",
        "Not Evaluable",
        "Not Evaluable (NE)",
        "Not Assessed (NA)",
        "Not Evaluated",
    ])
    def test_descriptor_returns_non_stratifier_sentinel(self, descriptor):
        from src.graph.subgroup_taxonomy import _NON_STRATIFIER_LEVEL
        f = canonicalize_feature("other", "", "whatever", descriptor)
        assert f.axis == "other"
        assert f.level == _NON_STRATIFIER_LEVEL

    def test_log_unmapped_suppresses_known_non_stratifiers(self, tmp_path):
        from src.graph.subgroup_taxonomy import log_unmapped
        log_path = tmp_path / "unmapped.jsonl"
        # First: a known-non-stratifier descriptor.
        f_noise = canonicalize_feature(
            "other", "", "month_3", raw_descriptor="Month 3",
        )
        log_unmapped(f_noise, "NCT_NOISE", log_path=log_path)
        # Second: a real unmapped descriptor (not in the pattern set).
        f_real = canonicalize_feature(
            "other", "", "mystery_term", raw_descriptor="Something genuinely new",
        )
        log_unmapped(f_real, "NCT_REAL", log_path=log_path)
        # Only the real one should land in the log.
        records = [
            json.loads(line) for line in log_path.read_text().splitlines()
        ] if log_path.exists() else []
        assert len(records) == 1
        assert records[0]["trial_id"] == "NCT_REAL"

    def test_real_unmapped_descriptors_still_log(self, tmp_path):
        """Regression guard: actual unknown patterns must keep flowing
        to the log so vocab expansion has signal."""
        from src.graph.subgroup_taxonomy import log_unmapped
        log_path = tmp_path / "unmapped.jsonl"
        f = canonicalize_feature(
            "other", "", "novel_axis", raw_descriptor="Some novel descriptor",
        )
        log_unmapped(f, "NCT_X", log_path=log_path)
        assert log_path.exists()


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
