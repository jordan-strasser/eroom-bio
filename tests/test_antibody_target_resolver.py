"""Unit tests for the round-24 antibody target resolver."""

from __future__ import annotations

import pytest

from src.graph.antibody_target_resolver import (
    MAB_COMPOUND_TO_TARGET_GENE,
    NL_TARGET_TO_GENE,
    is_monoclonal_antibody,
    mab_target_gene,
    nl_target_to_gene,
)


class TestIsMonoclonalAntibody:
    @pytest.mark.parametrize(
        "name",
        [
            "pembrolizumab", "nivolumab", "ipilimumab", "rituximab",
            "solanezumab", "bevacizumab", "trastuzumab", "cetuximab",
            "PEMBROLIZUMAB",  # case-insensitive
            "  pembrolizumab  ",  # whitespace
        ],
    )
    def test_recognizes_mab_suffixes(self, name: str) -> None:
        assert is_monoclonal_antibody(name) is True

    @pytest.mark.parametrize(
        "name",
        [
            "atorvastatin", "torcetrapib", "selumetinib", "vemurafenib",
            "doxorubicin", "5_fluorouracil", "imatinib",  # -nib, kinase inhibitor
            "everolimus",  # -limus
            "",  # empty
        ],
    )
    def test_rejects_non_mabs(self, name: str) -> None:
        assert is_monoclonal_antibody(name) is False


class TestMabTargetGene:
    @pytest.mark.parametrize(
        "compound, expected",
        [
            ("solanezumab", "APP"),
            ("aducanumab", "APP"),
            ("lecanemab", "APP"),
            ("donanemab", "APP"),
            ("pembrolizumab", "PDCD1"),
            ("nivolumab", "PDCD1"),
            ("ipilimumab", "CTLA4"),
            ("bevacizumab", "VEGFA"),
            ("trastuzumab", "ERBB2"),
            ("cetuximab", "EGFR"),
            ("rituximab", "MS4A1"),
            ("tocilizumab", "IL6R"),
            ("SOLANEZUMAB", "APP"),  # case-insensitive
        ],
    )
    def test_known_mabs(self, compound: str, expected: str) -> None:
        assert mab_target_gene(compound) == expected

    @pytest.mark.parametrize(
        "compound",
        ["torcetrapib", "vemurafenib", "atorvastatin", "", "unknown_mab"],
    )
    def test_returns_none_for_unknown(self, compound: str) -> None:
        assert mab_target_gene(compound) is None


class TestNlTargetToGene:
    @pytest.mark.parametrize(
        "nl, expected",
        [
            ("amyloid beta", "APP"),
            ("amyloid-beta", "APP"),
            ("AMYLOID BETA", "APP"),  # case-insensitive
            ("  amyloid beta  ", "APP"),  # whitespace-tolerant
            ("amyloid β", "APP"),
            ("Aβ", "APP"),  # case-folded
            ("PD-1", "PDCD1"),
            ("PD-L1", "CD274"),
            ("CTLA-4", "CTLA4"),
            ("HER2", "ERBB2"),
            ("VEGF", "VEGFA"),
            ("CD20", "MS4A1"),
            ("MEK", "MAP2K1"),
            ("MEK1/2", "MAP2K1"),
            ("CETP", "CETP"),
            ("cholesteryl ester transfer protein", "CETP"),
            ("BRAF", "BRAF"),
        ],
    )
    def test_known_natural_language_targets(self, nl: str, expected: str) -> None:
        assert nl_target_to_gene(nl) == expected

    @pytest.mark.parametrize(
        "nl",
        ["random_target_xyz", "", None],
    )
    def test_returns_none_for_unknown(self, nl: str | None) -> None:
        assert nl_target_to_gene(nl) is None  # type: ignore[arg-type]


class TestTableConsistency:
    """Confidence checks that the two tables are internally consistent —
    every mAb in the curated table has its target also accessible from
    the natural-language side."""

    def test_all_mab_targets_are_valid_gene_symbols(self) -> None:
        # Heuristic: gene symbols are uppercase alphanumeric (no spaces).
        for compound, gene in MAB_COMPOUND_TO_TARGET_GENE.items():
            assert gene.isupper(), f"{compound} → {gene} is not an uppercase gene symbol"
            assert gene.replace("_", "").isalnum(), (
                f"{compound} → {gene} contains unexpected characters"
            )

    def test_nl_keys_are_lowercased(self) -> None:
        for key in NL_TARGET_TO_GENE:
            assert key == key.lower(), f"NL key {key!r} is not lowercased"
