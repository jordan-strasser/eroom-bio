"""Tests for the SapBERT embedding pipeline (round-21 Phase A).

The actual SapBERT model is mocked everywhere — CI shouldn't have to
download ~440 MB of weights to verify the caching + cosine + cache
roundtrip logic. The model interaction surface is small (one
`encode(list[str], convert_to_numpy=True) -> np.ndarray` call), so a
plain mock object covers it.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.graph import sapbert_embeddings as sapbert_mod
from src.graph.sapbert_embeddings import (
    SapBertUnavailable,
    _normalize_key,
    cosine_similarity,
    embed_compound_name,
    embed_compound_names,
)


@pytest.fixture(autouse=True)
def _reset_model(monkeypatch):
    """Reset the module-level lazy-load cache between tests so each test
    sees a clean state."""
    monkeypatch.setattr(sapbert_mod, "_model", None)
    yield


@pytest.fixture
def fake_model():
    """A drop-in for SentenceTransformer that returns predictable
    embeddings: a tiny per-name fingerprint that lets us assert "same
    name → same vector" and "different names → different vectors"."""
    model = MagicMock()

    def encode(names, convert_to_numpy=True):
        out = []
        for n in names:
            # Two-dimensional fingerprint: hash → in-range floats.
            h = abs(hash(n.lower())) % 100
            out.append([h / 100.0, 1.0 - h / 100.0])
        return np.array(out)

    model.encode = encode
    return model


# ── Key normalization ───────────────────────────────────────────────────


class TestNormalizeKey:
    def test_lowercases(self):
        assert _normalize_key("Pembrolizumab") == "pembrolizumab"

    def test_collapses_whitespace(self):
        assert _normalize_key("  5-Fluorouracil   plus  ") == "5-fluorouracil plus"

    def test_distinct_drugs_still_distinct(self):
        assert (
            _normalize_key("Fluorouracil") != _normalize_key("5-Fluorouracil")
        ), (
            "Case + whitespace normalization should NOT collapse "
            "spelling differences — that's the canonicalization layer's "
            "job in Phase B, not the cache key"
        )


# ── Cache behavior ──────────────────────────────────────────────────────


class TestEmbeddingCache:
    def test_first_call_invokes_model_and_writes_cache(
        self, fake_model, tmp_path, monkeypatch,
    ):
        cache_path = tmp_path / "cache.json"
        monkeypatch.setattr(sapbert_mod, "_model", fake_model)
        vec = embed_compound_name("Fluorouracil", cache_path=cache_path)
        assert len(vec) == 2  # fake model returns 2-dim
        # Cache written
        assert cache_path.exists()
        import json
        cached = json.loads(cache_path.read_text())
        assert "fluorouracil" in cached
        assert cached["fluorouracil"] == vec

    def test_second_call_hits_cache_not_model(
        self, fake_model, tmp_path, monkeypatch,
    ):
        cache_path = tmp_path / "cache.json"
        monkeypatch.setattr(sapbert_mod, "_model", fake_model)

        # Wrap the encode method so we can count calls.
        encode_calls = {"n": 0}
        original = fake_model.encode
        def counting(names, convert_to_numpy=True):
            encode_calls["n"] += 1
            return original(names, convert_to_numpy=convert_to_numpy)
        fake_model.encode = counting

        # First call: model fires.
        v1 = embed_compound_name("Fluorouracil", cache_path=cache_path)
        assert encode_calls["n"] == 1
        # Second call with same input: cache hit, no encode.
        v2 = embed_compound_name("Fluorouracil", cache_path=cache_path)
        assert encode_calls["n"] == 1
        assert v1 == v2

    def test_use_cache_false_bypasses_cache(
        self, fake_model, tmp_path, monkeypatch,
    ):
        cache_path = tmp_path / "cache.json"
        monkeypatch.setattr(sapbert_mod, "_model", fake_model)
        embed_compound_name(
            "Pembrolizumab", cache_path=cache_path, use_cache=False,
        )
        # No cache written.
        assert not cache_path.exists()


# ── Batched encoding ────────────────────────────────────────────────────


class TestBatchEmbedding:
    def test_all_missing_encodes_once(self, fake_model, tmp_path, monkeypatch):
        cache_path = tmp_path / "cache.json"
        monkeypatch.setattr(sapbert_mod, "_model", fake_model)
        calls = {"n": 0}
        original = fake_model.encode
        def counting(names, convert_to_numpy=True):
            calls["n"] += 1
            return original(names, convert_to_numpy=convert_to_numpy)
        fake_model.encode = counting

        out = embed_compound_names(
            ["Fluorouracil", "Pembrolizumab", "Nivolumab"],
            cache_path=cache_path,
        )
        assert len(out) == 3
        assert calls["n"] == 1  # one batched call

    def test_partial_cache_only_encodes_misses(
        self, fake_model, tmp_path, monkeypatch,
    ):
        cache_path = tmp_path / "cache.json"
        monkeypatch.setattr(sapbert_mod, "_model", fake_model)
        # Pre-populate cache with one name.
        embed_compound_name("Fluorouracil", cache_path=cache_path)

        # Reset model + counter.
        calls = {"n": 0, "args": []}
        original = fake_model.encode
        def counting(names, convert_to_numpy=True):
            calls["n"] += 1
            calls["args"].append(list(names))
            return original(names, convert_to_numpy=convert_to_numpy)
        fake_model.encode = counting

        embed_compound_names(
            ["Fluorouracil", "Pembrolizumab"],
            cache_path=cache_path,
        )
        # Only Pembrolizumab gets re-encoded — Fluorouracil came from cache.
        assert calls["n"] == 1
        assert calls["args"] == [["Pembrolizumab"]]


# ── Cosine similarity ───────────────────────────────────────────────────


class TestCosineSimilarity:
    def test_identical_vectors_score_1(self):
        assert cosine_similarity([0.6, 0.8], [0.6, 0.8]) == pytest.approx(1.0)

    def test_orthogonal_vectors_score_0(self):
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0

    def test_opposite_vectors_score_minus_1(self):
        assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_zero_vector_returns_zero(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0

    def test_dimension_mismatch_raises(self):
        with pytest.raises(ValueError, match="length mismatch"):
            cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])


# ── Unavailability + edge cases ─────────────────────────────────────────


class TestAvailability:
    def test_missing_dependency_raises_sapbert_unavailable(
        self, tmp_path, monkeypatch,
    ):
        """When sentence-transformers isn't installed, ``_load_model``
        raises ``SapBertUnavailable``. Phase B callers should catch this
        and fall back to curated-alias behavior."""
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "sentence_transformers":
                raise ImportError("simulated missing dep")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        monkeypatch.setattr(sapbert_mod, "_model", None)
        with pytest.raises(SapBertUnavailable, match="sentence-transformers"):
            embed_compound_name("anything", cache_path=tmp_path / "c.json")

    def test_empty_name_raises_value_error(self, tmp_path):
        with pytest.raises(ValueError, match="empty"):
            embed_compound_name("", cache_path=tmp_path / "c.json")
        with pytest.raises(ValueError, match="empty"):
            embed_compound_name("   ", cache_path=tmp_path / "c.json")

    def test_corrupt_cache_treated_as_empty(
        self, fake_model, tmp_path, monkeypatch,
    ):
        cache_path = tmp_path / "cache.json"
        cache_path.write_text("not json")
        monkeypatch.setattr(sapbert_mod, "_model", fake_model)
        # Should fall back to encoding, not raise.
        vec = embed_compound_name("Fluorouracil", cache_path=cache_path)
        assert len(vec) == 2


# ── Schema roundtrip (round-21 Phase A schema additions) ────────────────


class TestInterventionNodeSchema:
    """The Phase A schema additions to InterventionNode (`stable_id`,
    `embedding`) must roundtrip through Pydantic + the GraphStore JSON
    snapshot without losing precision or shape."""

    def test_stable_id_and_embedding_persist(self, tmp_path):
        from src.graph.models import (
            CompoundNode, EdgeBeliefState, EdgeType, GraphEdge, Modality,
        )
        from src.graph.store import GraphStore

        store = GraphStore()
        store.add_node(CompoundNode(
            id="fluorouracil", name="Fluorouracil",
            modality=Modality.SMALL_MOLECULE,
            stable_id="CHEMBL185",
            embedding=[0.123, -0.456, 0.789],
            aliases=["5-FU", "Adrucil"],
        ))

        snapshot = tmp_path / "snap.json"
        store.export_snapshot(str(snapshot))

        roundtripped = GraphStore()
        roundtripped.import_snapshot(str(snapshot))
        node = roundtripped.get_node("fluorouracil")
        assert node["stable_id"] == "CHEMBL185"
        assert node["embedding"] == [0.123, -0.456, 0.789]
        # Existing fields stay intact.
        assert node["aliases"] == ["5-FU", "Adrucil"]

    def test_defaults_to_none_when_not_set(self):
        from src.graph.models import CompoundNode, Modality
        node = CompoundNode(
            id="x", name="x", modality=Modality.SMALL_MOLECULE,
        )
        assert node.stable_id is None
        assert node.embedding is None
