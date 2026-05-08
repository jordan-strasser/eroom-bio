"""Tests for MedDRA AE-term normalization."""

from __future__ import annotations

from src.annotation.meddra import _safe_parse, _strip_code_fence, ae_node_id


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
