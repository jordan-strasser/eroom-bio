"""Round-28 MedDRA hierarchy loader tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.annotation.meddra_hierarchy import (
    HIERARCHY_PATH,
    MedDRAHierarchy,
)


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def minimal_hierarchy_file(tmp_path: Path) -> Path:
    """A small hand-crafted hierarchy file for round-tripping tests."""
    p = tmp_path / "meddra.json"
    p.write_text(json.dumps({
        "version": "test",
        "soc_names": {
            "cardiac_disorders": "Cardiac disorders",
            "endocrine_disorders": "Endocrine disorders",
        },
        "soc_name_aliases": {
            "ear, nose and throat disorders": "ear_and_labyrinth_disorders",
        },
        "pt_to_soc": {
            "atrial_fibrillation": "cardiac_disorders",
            "myocardial_infarction": "cardiac_disorders",
            "bradycardia": "cardiac_disorders",
            "hypophysitis": "endocrine_disorders",
        },
        "pt_to_hlt": {
            "atrial_fibrillation": "cardiac_arrhythmias_supraventricular",
        },
        "hlt_to_hlgt": {
            "cardiac_arrhythmias_supraventricular": "cardiac_arrhythmias",
        },
        "hlgt_to_soc": {
            "cardiac_arrhythmias": "cardiac_disorders",
        },
    }))
    return p


# ── Loading ──────────────────────────────────────────────────────────────


class TestLoading:
    def test_load_explicit_path(self, minimal_hierarchy_file):
        h = MedDRAHierarchy.load(minimal_hierarchy_file)
        assert h.n_socs == 2
        assert h.n_pts == 4
        assert h.version == "test"

    def test_load_missing_path_returns_empty(self, tmp_path):
        h = MedDRAHierarchy.load(tmp_path / "nope.json")
        assert h.n_socs == 0
        assert h.n_pts == 0

    def test_load_default_exists(self):
        # The bootstrap file ships with the repo — sanity-check it loads.
        assert HIERARCHY_PATH.exists()
        h = MedDRAHierarchy.load_default()
        assert h.n_socs >= 25
        assert h.n_pts >= 100


# ── parents_for_pt ───────────────────────────────────────────────────────


class TestParentsForPT:
    def test_known_pt_returns_soc(self, minimal_hierarchy_file):
        h = MedDRAHierarchy.load(minimal_hierarchy_file)
        parents = h.parents_for_pt("atrial_fibrillation")
        assert parents["soc_id"] == "cardiac_disorders"
        assert parents["soc_name"] == "Cardiac disorders"

    def test_known_pt_returns_hlt_chain(self, minimal_hierarchy_file):
        h = MedDRAHierarchy.load(minimal_hierarchy_file)
        parents = h.parents_for_pt("atrial_fibrillation")
        assert parents["hlt_id"] == "cardiac_arrhythmias_supraventricular"
        assert parents["hlgt_id"] == "cardiac_arrhythmias"

    def test_pt_with_ae_prefix_normalized(self, minimal_hierarchy_file):
        h = MedDRAHierarchy.load(minimal_hierarchy_file)
        a = h.parents_for_pt("AE:atrial_fibrillation")
        b = h.parents_for_pt("atrial_fibrillation")
        assert a == b

    def test_raw_pt_text_slugified(self, minimal_hierarchy_file):
        h = MedDRAHierarchy.load(minimal_hierarchy_file)
        parents = h.parents_for_pt("Atrial fibrillation")
        assert parents["soc_id"] == "cardiac_disorders"

    def test_unknown_pt_returns_empty_parents(self, minimal_hierarchy_file):
        h = MedDRAHierarchy.load(minimal_hierarchy_file)
        parents = h.parents_for_pt("nonexistent_pt")
        assert parents["soc_id"] == ""
        assert parents["soc_name"] == ""
        assert parents["hlt_id"] == ""
        assert parents["hlgt_id"] == ""

    def test_unknown_pt_with_fallback_soc(self, minimal_hierarchy_file):
        """Unknown PTs resolve to SOC via the fallback canonical name."""
        h = MedDRAHierarchy.load(minimal_hierarchy_file)
        parents = h.parents_for_pt(
            "nonexistent_pt", fallback_soc_name="Cardiac disorders",
        )
        assert parents["soc_id"] == "cardiac_disorders"
        assert parents["soc_name"] == "Cardiac disorders"

    def test_unknown_pt_unknown_fallback_stays_empty(self, minimal_hierarchy_file):
        h = MedDRAHierarchy.load(minimal_hierarchy_file)
        parents = h.parents_for_pt(
            "nonexistent_pt", fallback_soc_name="Not a real SOC",
        )
        assert parents["soc_id"] == ""

    def test_keys_always_present(self, minimal_hierarchy_file):
        h = MedDRAHierarchy.load(minimal_hierarchy_file)
        parents = h.parents_for_pt("nonexistent_pt")
        for k in ("hlt_id", "hlgt_id", "soc_id", "soc_name"):
            assert k in parents


# ── aggregate_tier ───────────────────────────────────────────────────────


class TestAggregateTier:
    def test_soc_aggregation_groups_cardiac_pts(self, minimal_hierarchy_file):
        h = MedDRAHierarchy.load(minimal_hierarchy_file)
        groups = h.aggregate_tier(
            ["atrial_fibrillation", "myocardial_infarction", "bradycardia",
             "hypophysitis"],
            tier="soc",
        )
        assert sorted(groups.keys()) == [
            "cardiac_disorders", "endocrine_disorders",
        ]
        assert sorted(groups["cardiac_disorders"]) == [
            "atrial_fibrillation", "bradycardia", "myocardial_infarction",
        ]
        assert groups["endocrine_disorders"] == ["hypophysitis"]

    def test_unknown_pts_skipped(self, minimal_hierarchy_file):
        h = MedDRAHierarchy.load(minimal_hierarchy_file)
        groups = h.aggregate_tier(
            ["atrial_fibrillation", "totally_unknown"],
            tier="soc",
        )
        assert groups == {"cardiac_disorders": ["atrial_fibrillation"]}

    def test_fallback_socs_per_pt(self, minimal_hierarchy_file):
        h = MedDRAHierarchy.load(minimal_hierarchy_file)
        groups = h.aggregate_tier(
            ["AE:atrial_fibrillation", "AE:weird_unknown_ae"],
            tier="soc",
            fallback_soc_by_pt={
                "AE:weird_unknown_ae": "Cardiac disorders",
            },
        )
        assert sorted(groups["cardiac_disorders"]) == [
            "atrial_fibrillation", "weird_unknown_ae",
        ]

    def test_hlt_aggregation(self, minimal_hierarchy_file):
        h = MedDRAHierarchy.load(minimal_hierarchy_file)
        groups = h.aggregate_tier(
            ["atrial_fibrillation", "myocardial_infarction"],
            tier="hlt",
        )
        # Only atrial_fibrillation has an HLT in the minimal fixture;
        # myocardial_infarction is skipped.
        assert groups == {
            "cardiac_arrhythmias_supraventricular": ["atrial_fibrillation"],
        }

    def test_invalid_tier_raises(self, minimal_hierarchy_file):
        h = MedDRAHierarchy.load(minimal_hierarchy_file)
        with pytest.raises(ValueError):
            h.aggregate_tier(["x"], tier="bogus")


# ── SOC name canonicalization ────────────────────────────────────────────


class TestSocForName:
    def test_canonical_name_returns_slug(self, minimal_hierarchy_file):
        h = MedDRAHierarchy.load(minimal_hierarchy_file)
        assert h.soc_for_name("Cardiac disorders") == "cardiac_disorders"

    def test_case_insensitive(self, minimal_hierarchy_file):
        h = MedDRAHierarchy.load(minimal_hierarchy_file)
        assert h.soc_for_name("cardiac DISORDERS") == "cardiac_disorders"

    def test_alias_resolves(self, minimal_hierarchy_file):
        h = MedDRAHierarchy.load(minimal_hierarchy_file)
        # "Ear, nose and throat disorders" is an alias in the fixture.
        assert (
            h.soc_for_name("Ear, nose and throat disorders")
            == "ear_and_labyrinth_disorders"
        )

    def test_unknown_name_returns_empty(self, minimal_hierarchy_file):
        h = MedDRAHierarchy.load(minimal_hierarchy_file)
        assert h.soc_for_name("Not a SOC") == ""

    def test_empty_input_returns_empty(self, minimal_hierarchy_file):
        h = MedDRAHierarchy.load(minimal_hierarchy_file)
        assert h.soc_for_name("") == ""


# ── Bootstrap file content ───────────────────────────────────────────────


class TestBootstrapFile:
    """Sanity-check the shipped bootstrap file covers round-28 needs."""

    def test_cardiac_irae_pts_present(self):
        h = MedDRAHierarchy.load_default()
        # CETP holdout case study — these PTs need to roll up to
        # cardiac_disorders so torcetrapib's safety penalty rises.
        for pt in (
            "atrial_fibrillation", "myocardial_infarction", "bradycardia",
        ):
            soc = h.parents_for_pt(pt)["soc_id"]
            assert soc == "cardiac_disorders", (
                f"{pt!r} should roll up to cardiac_disorders, got {soc!r}"
            )

    def test_immune_related_aes_endocrine(self):
        h = MedDRAHierarchy.load_default()
        # Round-27 found solanezumab irAE leak (hypophysitis from
        # phantom checkpoint→APP propagation). The SOC for hypophysitis
        # is endocrine_disorders — sibling endocrine signals should
        # aggregate there.
        for pt in ("hypophysitis", "adrenal_insufficiency", "thyroiditis"):
            soc = h.parents_for_pt(pt)["soc_id"]
            assert soc == "endocrine_disorders", (
                f"{pt!r} should roll up to endocrine_disorders, got {soc!r}"
            )

    def test_all_27_canonical_socs_present(self):
        h = MedDRAHierarchy.load_default()
        # The 27 standard MedDRA SOCs.
        assert h.n_socs == 27
