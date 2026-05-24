"""BioLORD embedding pipeline for the semantic node substrate (manifold 1).

A.1 of the node-graph kickoff. Provides ``embed_text(text) -> list[float]``
backed by BioLORD (``FremyCompany/BioLORD-2023``). Where SapBERT embeds
compound *names* for canonicalization, BioLORD embeds the rich free-text
*descriptions* that A.0 now preserves on Biology / Mechanism / Population
nodes — the embedding becomes the semantic substrate, the canonical id stays a
routing tag. Results cache to ``data/cache/biolord_embeddings.json`` (gitignored
— per ``BOUNDARY.md``, embeddings are a recomputable cache artifact, never
snapshot state).

Lazy-loaded by design: the model isn't imported until the first ``embed_text``
call, so importing this module is free. Tests mock the model so CI doesn't need
the ~440 MB weights or the ``[biolord]`` extra.

Licensing caveat (see the project memory / BOUNDARY.md): BioLORD-2023 is
MIT-code but trained on UMLS + SNOMED CT, and the model card requires proper
UMLS/SNOMED licensing to use it. That matters specifically because the
*fine-tuned* derivative becomes the distributed enterprise artifact (manifold 1
weights). The *base* embeddings produced here are recomputable by anyone and
carry no eroom moat; clear SNOMED/UMLS before shipping fine-tuned weights.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)

# BioLORD model identifier on HuggingFace. Biomedical sentence embeddings
# aligned to ontology definitions — strong at placing rich pathway/mechanism
# descriptions near their canonical concepts. ~768-dim.
BIOLORD_MODEL_ID = "FremyCompany/BioLORD-2023"

# Where description → embedding gets cached on disk so subsequent builds skip
# inference. Mirrors ``data/cache/sapbert_embeddings.json``. Gitignored.
DEFAULT_CACHE_PATH = Path("data/cache/biolord_embeddings.json")


class BioLordUnavailable(RuntimeError):
    """Raised when sentence-transformers isn't installed or the model can't be
    loaded. Callers decide whether to skip embedding (fall back to id-only
    behavior) or propagate — the substrate work treats this as 'no embeddings
    this run', not a crash."""


# Module-level cache of the loaded model, guarded by a lock so concurrent
# embed() calls don't race on first load.
_model: Any = None
_model_lock = Lock()


def _load_model() -> Any:
    """Lazy-load the BioLORD SentenceTransformer. Raises ``BioLordUnavailable``
    if the ``[biolord]`` extra isn't installed or the weights can't load."""
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:  # double-checked
            return _model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise BioLordUnavailable(
                "sentence-transformers not installed. Install the A.1 "
                "dependency with `pip install -e \".[biolord]\"` to enable "
                "BioLORD-based node embeddings."
            ) from exc
        try:
            _model = SentenceTransformer(BIOLORD_MODEL_ID)
        except Exception as exc:
            raise BioLordUnavailable(
                f"Failed to load BioLORD model {BIOLORD_MODEL_ID!r}. On first "
                f"use the weights download to ~/.cache/huggingface; check "
                f"network access. Underlying error: {exc}"
            ) from exc
        return _model


def _load_cache(cache_path: Path) -> dict[str, list[float]]:
    """Read the on-disk embedding cache; empty dict if missing/corrupt."""
    if not cache_path.exists():
        return {}
    try:
        raw = json.loads(cache_path.read_text())
    except json.JSONDecodeError:
        logger.warning(
            "BioLORD cache at %s is corrupt; treating as empty", cache_path,
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


def _normalize_key(text: str) -> str:
    """Cache key normalization: lowercase + collapse whitespace, so trivially
    different phrasings of the same description share a cache entry. The model
    handles tokenization; this only dedups exact-modulo-whitespace strings."""
    return " ".join((text or "").lower().split())


def embed_text(
    text: str,
    *,
    cache_path: Path | None = None,
    use_cache: bool = True,
) -> list[float]:
    """Return the BioLORD embedding for a free-text description.

    Cache hit returns immediately; cache miss loads the model (one-time cost),
    encodes, and persists. Raises ``BioLordUnavailable`` when the ``[biolord]``
    extra isn't installed — callers should catch and fall back to id-only.
    """
    if not text or not text.strip():
        raise ValueError("Cannot embed an empty description")
    cache_path = cache_path or DEFAULT_CACHE_PATH
    key = _normalize_key(text)
    cache = _load_cache(cache_path) if use_cache else {}
    if key in cache:
        return cache[key]

    model = _load_model()
    vec = model.encode([text], convert_to_numpy=True)[0]
    embedding = [float(x) for x in vec.tolist()]
    if use_cache:
        cache[key] = embedding
        _save_cache(cache_path, cache)
    return embedding


def embed_texts(
    texts: list[str],
    *,
    cache_path: Path | None = None,
    use_cache: bool = True,
) -> list[list[float]]:
    """Batched ``embed_text``. Cache-hits avoid inference; cache-misses are
    encoded in one ``model.encode`` call. Used to backfill an existing
    snapshot's node descriptions in one pass."""
    cache_path = cache_path or DEFAULT_CACHE_PATH
    cache = _load_cache(cache_path) if use_cache else {}

    missing_keys: list[str] = []
    missing_texts: list[str] = []
    for text in texts:
        if not text or not text.strip():
            raise ValueError(f"Cannot embed empty text in batch (got {text!r})")
        key = _normalize_key(text)
        if key not in cache and key not in missing_keys:
            missing_keys.append(key)
            missing_texts.append(text)

    if missing_texts:
        model = _load_model()
        vecs = model.encode(missing_texts, convert_to_numpy=True)
        for key, vec in zip(missing_keys, vecs):
            cache[key] = [float(x) for x in vec.tolist()]
        if use_cache:
            _save_cache(cache_path, cache)

    return [cache[_normalize_key(text)] for text in texts]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Plain Python cosine similarity for two embedding vectors.

    Kept local (mirroring ``sapbert_embeddings``) so this module is
    self-contained. For corpora well under 10k nodes this is fast enough;
    revisit with faiss/HNSW if it scales out.
    """
    if len(a) != len(b):
        raise ValueError(f"Embedding length mismatch: {len(a)} vs {len(b)}")
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
