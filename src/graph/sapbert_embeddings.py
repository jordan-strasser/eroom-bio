"""SapBERT embedding pipeline for compound canonicalization.

Round-21 Phase A. Provides ``embed_compound_name(name) -> list[float]``
backed by SapBERT (``cambridgeltl/SapBERT-from-PubMedBERT-fulltext``).
Results cached to ``data/cache/sapbert_embeddings.json`` so repeats
across builds hit disk instead of re-running inference.

Phase A only exposes the embedding function + cache; Phase B will
wire it into the populator's compound-add loop for vector-similarity
canonicalization (`audit/round_21_sapbert_canonicalization.md`).

Lazy-loaded by design: the model isn't imported until the first
``embed_compound_name`` call so the rest of the codebase doesn't pay
the ~1.3 GB ML-dep cost just to import this module. Tests mock the
model so CI doesn't need the dependency.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)

# SapBERT model identifier on HuggingFace. The self-aligned biomedical
# entity embeddings are reported to produce >0.85 cosine for same-drug-
# different-name pairs (e.g. "5-Fluorouracil" vs "Fluorouracil").
SAPBERT_MODEL_ID = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"

# Where compound name → embedding gets cached on disk so subsequent
# builds skip inference. Mirrors the pattern of
# ``data/cache/ot_drug_targets.json`` used by `_populate_compound_targets`.
DEFAULT_CACHE_PATH = Path("data/cache/sapbert_embeddings.json")


class SapBertUnavailable(RuntimeError):
    """Raised when sentence-transformers isn't installed or the model
    can't be loaded. Callers decide whether to skip canonicalization or
    propagate. The populator's Phase B integration treats this as
    'fall back to curated alias-only behavior', not an error."""


# Module-level cache of the loaded model, guarded by a lock so concurrent
# embed() calls don't race on first load.
_model: Any = None
_model_lock = Lock()


def _load_model() -> Any:
    """Lazy-load the SapBERT SentenceTransformer. Raises
    ``SapBertUnavailable`` if the `sapbert` extra isn't installed."""
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:  # double-checked
            return _model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise SapBertUnavailable(
                "sentence-transformers not installed. Install the "
                "round-21 dependency with `pip install -e \".[sapbert]\"` "
                "to enable SapBERT-based compound canonicalization."
            ) from exc
        try:
            _model = SentenceTransformer(SAPBERT_MODEL_ID)
        except Exception as exc:
            raise SapBertUnavailable(
                f"Failed to load SapBERT model {SAPBERT_MODEL_ID!r}. "
                f"On first use the ~440 MB weights download to "
                f"~/.cache/huggingface; check network access. "
                f"Underlying error: {exc}"
            ) from exc
        return _model


def _load_cache(cache_path: Path) -> dict[str, list[float]]:
    """Read the on-disk embedding cache; empty dict if missing."""
    if not cache_path.exists():
        return {}
    try:
        raw = json.loads(cache_path.read_text())
    except json.JSONDecodeError:
        logger.warning(
            "SapBERT cache at %s is corrupt; treating as empty", cache_path,
        )
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, list[float]] = {}
    for k, v in raw.items():
        if isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v):
            out[k] = [float(x) for x in v]
    return out


def _save_cache(cache_path: Path, cache: dict[str, list[float]]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, sort_keys=True))


def _normalize_key(name: str) -> str:
    """Cache key normalization: lowercase + collapse whitespace. The
    SapBERT model itself handles tokenization, but we want
    'Fluorouracil' and 'fluorouracil  ' to share a cache entry."""
    return " ".join((name or "").lower().split())


def embed_compound_name(
    name: str,
    *,
    cache_path: Path | None = None,
    use_cache: bool = True,
) -> list[float]:
    """Return the SapBERT embedding for a compound name.

    Cache hit: returns the cached vector immediately.
    Cache miss: loads the model (one-time cost), encodes, persists.

    Raises ``SapBertUnavailable`` when the optional `sapbert` extra
    isn't installed — Phase B integration should catch this and fall
    back to curated-alias behavior rather than crashing the populate
    pipeline.
    """
    if not name or not name.strip():
        raise ValueError("Cannot embed an empty compound name")
    cache_path = cache_path or DEFAULT_CACHE_PATH
    key = _normalize_key(name)
    cache = _load_cache(cache_path) if use_cache else {}
    if key in cache:
        return cache[key]

    model = _load_model()
    vec = model.encode([name], convert_to_numpy=True)[0]
    embedding = [float(x) for x in vec.tolist()]
    if use_cache:
        cache[key] = embedding
        _save_cache(cache_path, cache)
    return embedding


def embed_compound_names(
    names: list[str],
    *,
    cache_path: Path | None = None,
    use_cache: bool = True,
) -> list[list[float]]:
    """Batched version of ``embed_compound_name``. Cache-hits avoid
    inference entirely; cache-misses get encoded in one model.encode()
    call. Useful for Phase A backfill of an existing snapshot."""
    cache_path = cache_path or DEFAULT_CACHE_PATH
    cache = _load_cache(cache_path) if use_cache else {}

    missing_keys: list[str] = []
    missing_names: list[str] = []
    for name in names:
        if not name or not name.strip():
            raise ValueError(f"Cannot embed empty name in batch (got {name!r})")
        key = _normalize_key(name)
        if key not in cache:
            missing_keys.append(key)
            missing_names.append(name)

    if missing_names:
        model = _load_model()
        vecs = model.encode(missing_names, convert_to_numpy=True)
        for key, vec in zip(missing_keys, vecs):
            cache[key] = [float(x) for x in vec.tolist()]
        if use_cache:
            _save_cache(cache_path, cache)

    return [cache[_normalize_key(name)] for name in names]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Plain Python cosine similarity for two embedding vectors.

    Used by Phase B canonicalization to compare a candidate compound's
    embedding against existing nodes' embeddings without pulling in a
    larger vector-search library. For corpora with <10k compounds this
    is fast enough; revisit with faiss/HNSW when that scales out.
    """
    if len(a) != len(b):
        raise ValueError(
            f"Embedding length mismatch: {len(a)} vs {len(b)}"
        )
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / ((norm_a ** 0.5) * (norm_b ** 0.5))
