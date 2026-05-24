"""Reactome pathway hierarchy (ancestors) for biology is-a supervision (A.2).

Free HTTP to the Reactome ContentService ``ancestors`` endpoint, cached on disk
(``data/cache/`` — gitignored). Best-effort: any network/parse failure returns
``[]`` so box fitting degrades to leaf cubes rather than crashing. No LLM.

Used by ``box_embeddings.biology_parent_child_pairs`` to give BiologyNodes real
is-a structure (a parent pathway's box contains its sub-pathways' boxes).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_CACHE_PATH = Path("data/cache/reactome_ancestors.json")
_ANCESTORS_URL = "https://reactome.org/ContentService/data/event/{}/ancestors"


def _load_cache(cache_path: Path) -> dict[str, list[str]]:
    if not cache_path.exists():
        return {}
    try:
        raw = json.loads(cache_path.read_text())
    except json.JSONDecodeError:
        return {}
    return raw if isinstance(raw, dict) else {}


def _save_cache(cache_path: Path, cache: dict[str, list[str]]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, sort_keys=True))


def fetch_reactome_ancestors(
    stable_id: str,
    *,
    cache_path: Path | None = None,
    timeout: float = 15.0,
) -> list[str]:
    """Return the set of ancestor pathway stIds for a Reactome pathway.

    The endpoint returns one or more ancestor *paths* (a pathway can sit under
    multiple parents); we flatten them to the union of ancestor stIds,
    excluding the pathway itself. Non-Reactome ids and failures return ``[]``.
    Cached on disk so a re-fit is offline.
    """
    if not stable_id.startswith("R-"):
        return []
    cache_path = cache_path or DEFAULT_CACHE_PATH
    cache = _load_cache(cache_path)
    if stable_id in cache:
        return cache[stable_id]

    import httpx

    try:
        resp = httpx.get(_ANCESTORS_URL.format(stable_id), timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        logger.debug("Reactome ancestors lookup failed for %s", stable_id, exc_info=True)
        return []

    ids: set[str] = set()
    paths = data if isinstance(data, list) else []
    for path in paths:
        if not isinstance(path, list):
            continue
        for node in path:
            if isinstance(node, dict):
                sid = node.get("stId")
                if sid and sid != stable_id:
                    ids.add(sid)
    out = sorted(ids)
    cache[stable_id] = out
    _save_cache(cache_path, cache)
    return out


def fetch_many_ancestors(
    stable_ids: list[str], *, cache_path: Path | None = None
) -> dict[str, list[str]]:
    """Batched (sequential, cached) ancestor lookup for many pathways."""
    return {
        sid: fetch_reactome_ancestors(sid, cache_path=cache_path)
        for sid in stable_ids
    }
