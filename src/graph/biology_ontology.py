"""B1 — map a biology description to a GO-biological-process controlled-vocabulary id.

Applied by the explicit rekey step (``scripts/rekey_biology_ontology.py``) using
``CONFIG.bio_ontology_gate`` (src/config.py); formerly the ``EROOM_BIO_ONTOLOGY``
on/off flag, which is gone — only the cosine gate remains. A BiologyNode is keyed by
the nearest GO-BP term instead of a content hash of its free-text description, so the
same biology across trials lands on ONE node — mirroring the Reactome id-merge that
makes the mechanism layer the densest, highest-reuse unit in the graph. Unmappable
biology (below the cosine gate) falls back to the ``bio:<sha1>`` content hash, so no
node is ever lost.

Rationale + measured collapse/coverage: ``B1_DECISION.md`` (GO; ~40% of singletons
collapse, %≥8 reuse 5.2%→~10%) and ``B1_DESCRIPTOR_AUDIT.md`` (the descriptors are a
consistent-but-impoverished functional phrase, so an ontology id — not descriptor
re-normalization — is the lever).

The desc→GO map is precomputed at ``data/cache/biology_ontology_map.json`` (built by
``scratch/diagnostics/b1_build_ontology_map.py``) so a build is deterministic and
offline. A cache miss falls back to a live BioLORD nearest-GO lookup (invoked by the
rekey step whenever it requests a mapping).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from src.config import CONFIG

_MAP_PATH = Path("data/cache/biology_ontology_map.json")


def gate() -> float:
    """BioLORD cosine floor for accepting a GO-BP mapping (CONFIG.bio_ontology_gate).
    The biology→GO grouping is no longer flag-gated — the rekey step
    (scripts/rekey_biology_ontology.py) applies it unconditionally; this gate is
    its only tuneable parameter."""
    return CONFIG.bio_ontology_gate


def _normkey(s: str) -> str:
    return " ".join((s or "").lower().split())


@lru_cache(maxsize=1)
def _load_map() -> dict:
    if _MAP_PATH.exists():
        return json.loads(_MAP_PATH.read_text())
    return {}


# ── live fallback (only on cache miss, only when enabled) ────────────────────
_live_vocab: tuple | None = None


def _ensure_vocab():
    global _live_vocab
    if _live_vocab is not None:
        return _live_vocab
    import glob

    import numpy as np

    vocab = {}
    for f in glob.glob("data/cache/quickgo/*.json"):
        try:
            d = json.loads(Path(f).read_text())
        except Exception:
            continue
        for x in d if isinstance(d, list) else []:
            if (x.get("aspect") == "biological_process"
                    and x.get("stable_id") and x.get("display_name")):
                vocab[x["stable_id"]] = x["display_name"]
    from src.graph.biolord_embeddings import embed_texts

    ids = list(vocab)
    labels = [vocab[g] for g in ids]
    GE = np.asarray(embed_texts(labels), dtype=np.float32)
    GE /= np.linalg.norm(GE, axis=1, keepdims=True) + 1e-9
    _live_vocab = (ids, labels, GE, vocab)
    return _live_vocab


def _live_nearest(description: str) -> dict | None:
    try:
        import numpy as np

        ids, labels, GE, vocab = _ensure_vocab()
        from src.graph.biolord_embeddings import embed_texts

        v = np.asarray(embed_texts([description])[0], dtype=np.float32)
        v /= np.linalg.norm(v) + 1e-9
        sims = GE @ v
        i = int(sims.argmax())
        return {"go_id": ids[i], "go_label": labels[i], "cos": float(sims[i])}
    except Exception:
        return None


def ontology_id_for(description: str) -> str | None:
    """``'bio:GO:xxxxxxx'`` if the description maps to a GO-BP term at/above the gate,
    else ``None`` (caller falls back to the content-hash id)."""
    if not description:
        return None
    rec = _load_map().get(_normkey(description))
    if rec is None:
        rec = _live_nearest(description)
    if rec and rec.get("cos", 0.0) >= gate():
        return "bio:" + rec["go_id"]
    return None
