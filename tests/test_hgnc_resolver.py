"""Tests for the HGNC alias resolver."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.graph import hgnc_resolver
from src.graph.hgnc_resolver import (
    _normalize_alias,
    _parse_hgnc_tsv,
    aliases_for,
    canonical_symbol,
    is_loaded,
    load,
    reset_for_test,
    uniprot_for_symbol,
)


# ── Fixtures ────────────────────────────────────────────────────────────


_HGNC_FIXTURE = """\
hgnc_id\tsymbol\tname\talias_symbol\tprev_symbol\tuniprot_ids
HGNC:17635\tCD274\tCD274 molecule\tB7-H1|PD-L1|PDCD1L1|PDCD1LG1|PDL1\t\tQ9NZQ7
HGNC:8760\tPDCD1\tprogrammed cell death 1\tPD-1|hPD-1|hSLE1\tPD1\tQ15116
HGNC:2376\tCTLA4\tcytotoxic T-lymphocyte associated protein 4\tCD152|CTLA-4\t\tP16410
HGNC:3236\tEGFR\tepidermal growth factor receptor\tERBB|HER1|ERBB1\tERBB\tP00533
HGNC:6407\tKRAS\tKRAS proto-oncogene\tNS3|RASK2|KRAS1|KRAS2\tKRAS2\tP01116
HGNC:30185\tCRBN\tcereblon\tAD-006\t\tQ96SW2
"""


@pytest.fixture
def hgnc_tsv_path(tmp_path: Path) -> Path:
    p = tmp_path / "hgnc.tsv"
    p.write_text(_HGNC_FIXTURE)
    return p


@pytest.fixture(autouse=True)
def _reset_resolver():
    """Each test starts with an unloaded resolver."""
    reset_for_test()
    yield
    reset_for_test()


# ── Normalization ───────────────────────────────────────────────────────


class TestNormalize:
    def test_lowercases(self):
        assert _normalize_alias("PDCD1") == "pdcd1"

    def test_strips_punctuation(self):
        assert _normalize_alias("PD-L1") == "pdl1"
        assert _normalize_alias("PD_L1") == "pdl1"
        assert _normalize_alias("PD L1") == "pdl1"

    def test_empty_returns_empty(self):
        assert _normalize_alias("") == ""


# ── TSV parsing ─────────────────────────────────────────────────────────


class TestParseTsv:
    def test_canonical_maps_to_itself(self, hgnc_tsv_path: Path):
        lookup, _, _ = _parse_hgnc_tsv(hgnc_tsv_path)
        assert lookup["cd274"] == "CD274"
        assert lookup["pdcd1"] == "PDCD1"
        assert lookup["egfr"] == "EGFR"

    def test_aliases_map_to_canonical(self, hgnc_tsv_path: Path):
        lookup, originals, _ = _parse_hgnc_tsv(hgnc_tsv_path)
        # PD-L1 family
        assert lookup["pdl1"] == "CD274"
        assert lookup["b7h1"] == "CD274"
        assert lookup["pdcd1l1"] == "CD274"
        # PD-1 family (PD1 → PDCD1 via prev_symbol; PD-1 normalizes to pd1)
        assert lookup["pd1"] == "PDCD1"
        # CTLA-4 alias with hyphen normalized
        assert lookup["ctla4"] == "CTLA4"
        # Previous symbols
        assert lookup["erbb"] == "EGFR"
        # Original-case aliases preserved (with hyphens etc.)
        assert "PD-L1" in originals["CD274"]
        assert "B7-H1" in originals["CD274"]
        assert "PD-1" in originals["PDCD1"]


# ── Public lookup ───────────────────────────────────────────────────────


class TestCanonicalSymbol:
    def test_resolves_pd_l1_alias(self, hgnc_tsv_path: Path):
        load(cache_path=hgnc_tsv_path, download_if_missing=False)
        assert canonical_symbol("PD-L1") == "CD274"
        assert canonical_symbol("PDL1") == "CD274"
        assert canonical_symbol("B7-H1") == "CD274"

    def test_resolves_canonical_to_itself(self, hgnc_tsv_path: Path):
        load(cache_path=hgnc_tsv_path, download_if_missing=False)
        assert canonical_symbol("CD274") == "CD274"
        assert canonical_symbol("cd274") == "CD274"

    def test_unknown_symbol_returns_none(self, hgnc_tsv_path: Path):
        load(cache_path=hgnc_tsv_path, download_if_missing=False)
        assert canonical_symbol("FOOBAR") is None

    def test_empty_input_returns_none(self):
        assert canonical_symbol("") is None
        assert canonical_symbol("   ") is None

    def test_aliases_for_returns_originals(self, hgnc_tsv_path: Path):
        load(cache_path=hgnc_tsv_path, download_if_missing=False)
        # Resolves through canonical, returns hyphenated originals.
        aliases = aliases_for("CD274")
        assert "PD-L1" in aliases
        assert "B7-H1" in aliases
        # Lookup via an alias also works.
        aliases_via_alias = aliases_for("PD-L1")
        assert sorted(aliases) == sorted(aliases_via_alias)

    def test_aliases_for_unknown_returns_empty(self, hgnc_tsv_path: Path):
        load(cache_path=hgnc_tsv_path, download_if_missing=False)
        assert aliases_for("FOOBAR_NOT_A_GENE") == []


class TestUniProtLookup:
    def test_canonical_symbol_returns_uniprot(self, hgnc_tsv_path: Path):
        load(cache_path=hgnc_tsv_path, download_if_missing=False)
        assert uniprot_for_symbol("CRBN") == "Q96SW2"
        assert uniprot_for_symbol("CD274") == "Q9NZQ7"

    def test_alias_resolves_through_canonical(self, hgnc_tsv_path: Path):
        """uniprot_for_symbol should resolve aliases the same way
        canonical_symbol does — PD-L1 → CD274 → Q9NZQ7."""
        load(cache_path=hgnc_tsv_path, download_if_missing=False)
        assert uniprot_for_symbol("PD-L1") == "Q9NZQ7"
        assert uniprot_for_symbol("PD-1") == "Q15116"

    def test_unknown_symbol_returns_none(self, hgnc_tsv_path: Path):
        load(cache_path=hgnc_tsv_path, download_if_missing=False)
        assert uniprot_for_symbol("FOOBAR_NOT_A_GENE") is None

    def test_empty_input_returns_none(self):
        assert uniprot_for_symbol("") is None

    def test_returns_none_when_resolver_disabled(self, tmp_path: Path):
        # Cache absent + download skipped → resolver becomes a no-op,
        # returns None for everything but is_loaded() is True (with
        # an empty dict, signaling we tried).
        missing = tmp_path / "no_cache.tsv"
        load(cache_path=missing, download_if_missing=False)
        assert is_loaded()
        assert canonical_symbol("CD274") is None


# ── Integration with subgroup_taxonomy ──────────────────────────────────


class TestSubgroupTaxonomyIntegration:
    def test_pdl1_canonicalizes_to_cd274(self, hgnc_tsv_path: Path):
        from src.graph.subgroup_taxonomy import canonicalize_feature
        load(cache_path=hgnc_tsv_path, download_if_missing=False)
        # "high" collapses to "positive" per fixes.md #5.
        f = canonicalize_feature("gene", "PD-L1", "high", "PD-L1 ≥1%")
        assert f.axis == "gene"
        assert f.key == "CD274"
        assert f.level == "positive"

    def test_alias_collapse_yields_same_population_id(self, hgnc_tsv_path: Path):
        """Different trials reporting PD-L1 positive under different names
        ('PD-L1', 'PDL1', 'CD274') all yield the same PopulationNode id —
        which is the whole point of canonicalization."""
        from src.graph.models import PopulationNode
        from src.graph.subgroup_taxonomy import canonicalize_feature
        load(cache_path=hgnc_tsv_path, download_if_missing=False)
        ids = {
            PopulationNode.compose_id(
                [canonicalize_feature("gene", name, "high")]
            )
            for name in ("PD-L1", "PDL1", "CD274", "B7-H1")
        }
        assert ids == {"cd274_positive"}

    def test_unknown_gene_falls_to_other_when_resolver_loaded(self, hgnc_tsv_path: Path):
        from src.graph.subgroup_taxonomy import canonicalize_feature
        load(cache_path=hgnc_tsv_path, download_if_missing=False)
        f = canonicalize_feature("gene", "FOOBAR_NOT_A_GENE", "high")
        # Resolver loaded but doesn't know this—strict reject.
        assert f.axis == "other"
