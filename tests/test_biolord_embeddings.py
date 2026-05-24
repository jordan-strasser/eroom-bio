"""Tests for the BioLORD embedding pipeline (A.1).

The model is mocked everywhere — CI shouldn't download weights to verify the
caching + cosine + cache-roundtrip logic. Mirrors test_sapbert_embeddings.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.graph import biolord_embeddings as biolord_mod
from src.graph.biolord_embeddings import (
    BioLordUnavailable,
    _normalize_key,
    cosine_similarity,
    embed_text,
    embed_texts,
)


@pytest.fixture(autouse=True)
def _reset_model(monkeypatch):
    monkeypatch.setattr(biolord_mod, "_model", None)
    yield


@pytest.fixture
def fake_model():
    model = MagicMock()

    def encode(texts, convert_to_numpy=True):
        out = []
        for t in texts:
            h = abs(hash(t.lower())) % 100
            out.append([h / 100.0, 1.0 - h / 100.0])
        return np.array(out)

    model.encode = encode
    return model


class TestNormalizeKey:
    def test_lowercases_and_collapses_whitespace(self):
        assert _normalize_key("  VEGF  Signaling ") == "vegf signaling"

    def test_distinct_descriptions_distinct(self):
        assert _normalize_key("kinase inhibition") != _normalize_key("angiogenesis")


class TestEmbeddingCache:
    def test_first_call_encodes_and_caches(self, fake_model, tmp_path, monkeypatch):
        monkeypatch.setattr(biolord_mod, "_model", fake_model)
        cache = tmp_path / "biolord.json"
        vec = embed_text("VEGFR2 inhibition", cache_path=cache)
        assert len(vec) == 2
        assert cache.exists()

    def test_second_call_hits_cache(self, fake_model, tmp_path, monkeypatch):
        monkeypatch.setattr(biolord_mod, "_model", fake_model)
        cache = tmp_path / "biolord.json"
        calls = {"n": 0}
        original = fake_model.encode

        def counting(texts, convert_to_numpy=True):
            calls["n"] += 1
            return original(texts, convert_to_numpy=convert_to_numpy)

        fake_model.encode = counting
        embed_text("angiogenesis", cache_path=cache)
        embed_text("angiogenesis", cache_path=cache)
        assert calls["n"] == 1  # second call served from disk cache

    def test_empty_text_raises(self, fake_model, monkeypatch):
        monkeypatch.setattr(biolord_mod, "_model", fake_model)
        with pytest.raises(ValueError):
            embed_text("   ")


class TestBatch:
    def test_partial_cache_only_encodes_misses(self, fake_model, tmp_path, monkeypatch):
        monkeypatch.setattr(biolord_mod, "_model", fake_model)
        cache = tmp_path / "biolord.json"
        embed_text("apoptosis", cache_path=cache)  # pre-warm one
        calls = {"n": 0}
        original = fake_model.encode

        def counting(texts, convert_to_numpy=True):
            calls["n"] += 1
            return original(texts, convert_to_numpy=convert_to_numpy)

        fake_model.encode = counting
        out = embed_texts(["apoptosis", "necrosis"], cache_path=cache)
        assert len(out) == 2
        assert calls["n"] == 1  # one batched encode for the single miss


class TestCosine:
    def test_identical_vectors(self):
        assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            cosine_similarity([1.0], [1.0, 2.0])


class TestUnavailable:
    def test_missing_extra_raises_biolord_unavailable(self, monkeypatch):
        # Simulate the [biolord] extra not being installed.
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "sentence_transformers":
                raise ImportError("not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        monkeypatch.setattr(biolord_mod, "_model", None)
        with pytest.raises(BioLordUnavailable):
            embed_text("anything", cache_path=Path("/tmp/unused_biolord.json"), use_cache=False)
