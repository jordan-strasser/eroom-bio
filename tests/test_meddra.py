"""Tests for MedDRA AE-term normalization."""

from __future__ import annotations

from src.annotation.meddra import (
    _safe_parse,
    _strip_code_fence,
    ae_node_id,
    is_meta_ae_term,
)


class TestStripCodeFence:
    def test_strips_json_fence(self):
        text = '```json\n{"preferred_term": "Myocarditis"}\n```'
        assert _strip_code_fence(text) == '{"preferred_term": "Myocarditis"}'

    def test_strips_bare_fence(self):
        text = '```\n{"a": 1}\n```'
        assert _strip_code_fence(text) == '{"a": 1}'

    def test_no_fence_passes_through(self):
        text = '{"preferred_term": "Myocarditis"}'
        assert _strip_code_fence(text) == text

    def test_preserves_inner_whitespace(self):
        text = '```json\n  {"a": 1}  \n```'
        assert _strip_code_fence(text) == '{"a": 1}'


class TestSafeParseFenced:
    """Pin the bug fix: Haiku wraps responses in ```json fences and the
    parser was rejecting them silently, falling back to capitalized raw
    term + empty SOC (stale data populated 161/161 cached AE entries
    before the fix)."""

    def test_parses_fenced_response(self):
        text = (
            '```json\n'
            '{"preferred_term": "Myocarditis", '
            '"system_organ_class": "Cardiac disorders"}\n'
            '```'
        )
        result = _safe_parse(text)
        assert result is not None
        assert result["preferred_term"] == "Myocarditis"
        assert result["system_organ_class"] == "Cardiac disorders"

    def test_parses_unfenced_response(self):
        text = '{"preferred_term": "Hepatotoxicity", "system_organ_class": "Hepatobiliary disorders"}'
        result = _safe_parse(text)
        assert result is not None
        assert result["preferred_term"] == "Hepatotoxicity"
        assert result["system_organ_class"] == "Hepatobiliary disorders"

    def test_rejects_truly_invalid_json(self):
        assert _safe_parse("not json") is None

    def test_rejects_missing_preferred_term(self):
        assert _safe_parse('{"system_organ_class": "x"}') is None


class TestAeNodeId:
    def test_simple_term(self):
        assert ae_node_id("Hepatotoxicity") == "AE:hepatotoxicity"

    def test_multi_word(self):
        assert (
            ae_node_id("Aspartate aminotransferase increased")
            == "AE:aspartate_aminotransferase_increased"
        )

    def test_empty_falls_back(self):
        assert ae_node_id("") == "AE:unspecified"

    def test_british_to_american_collapse(self):
        # haemo/anaem variants collapse onto the American slug so the
        # same AE doesn't fragment across nodes (fixes.md #8).
        assert ae_node_id("Haemolytic anaemia") == "AE:hemolytic_anemia"
        assert ae_node_id("Hemolytic anemia") == "AE:hemolytic_anemia"
        assert ae_node_id("Anaemia") == "AE:anemia"
        assert ae_node_id("Oedema") == "AE:edema"
        assert ae_node_id("Diarrhoea") == "AE:diarrhea"
        assert ae_node_id("Leucopenia") == "AE:leukopenia"


class TestIsMetaAeTerm:
    """Pre-filter for trial-level summary rows (fixes.md #7). These have
    no single clinical concept and shouldn't produce causes_ae edges."""

    def test_grade_range_summary(self):
        assert is_meta_ae_term("Grade 3-5 adverse events")
        assert is_meta_ae_term("grade 3 AEs")
        assert is_meta_ae_term("≥Grade 3 adverse events")

    def test_serious_aes_summary(self):
        assert is_meta_ae_term("Serious adverse events")
        assert is_meta_ae_term("Serious AEs")

    def test_treatment_emergent_summary(self):
        assert is_meta_ae_term("Treatment-emergent adverse events")
        assert is_meta_ae_term("Treatment related adverse events")

    def test_bare_meta_terms(self):
        assert is_meta_ae_term("Adverse events")
        assert is_meta_ae_term("Toxicity")
        assert is_meta_ae_term("All AEs")
        assert is_meta_ae_term("")

    def test_real_aes_not_meta(self):
        # Real clinical concepts must NOT be rejected, even when the AE
        # term carries a grade modifier — the LLM strips the grade.
        assert not is_meta_ae_term("Hepatotoxicity")
        assert not is_meta_ae_term("Grade 3 hepatotoxicity")
        assert not is_meta_ae_term("Anaemia")
        assert not is_meta_ae_term("Hemolytic anemia")
        assert not is_meta_ae_term("Aspartate aminotransferase increased")
