"""PubChem REST client — codename → synonym list fallback.

Round-23 codename canonicalization. SapBERT (round 21) resolves spelling
variants and salt forms but treats arbitrary alphanumeric codenames as
random tokens ("MK-3475" → "Pembrolizumab" only scored 0.332 cosine).
PubChem's `compound/name/<name>/synonyms/JSON` endpoint returns the
comprehensive synonym list NCBI curators have already gathered — codes,
INN, brand names, salt forms — which we use to bridge a codename onto
an already-known compound's canonical name.

The match step lives in `src/graph/codename_resolver.py`; this module is
just the HTTP + on-disk cache.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

PUBCHEM_BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"


class PubChemClient:
    """Minimal PubChem REST client for name → synonyms lookup.

    Only the `/compound/name/{name}/synonyms/JSON` endpoint is used.
    Responses are cached on disk so repeated builds against the same
    corpus don't re-hit NCBI.
    """

    def __init__(
        self,
        cache_path: Path | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._timeout = timeout
        self._cache_path = (
            cache_path or Path("data/cache/pubchem_synonyms.json")
        )
        self._cache: dict[str, dict[str, Any]] | None = None

    def _load_cache(self) -> dict[str, dict[str, Any]]:
        if self._cache is not None:
            return self._cache
        if self._cache_path.exists():
            try:
                self._cache = json.loads(self._cache_path.read_text())
            except json.JSONDecodeError:
                logger.warning(
                    "PubChem cache %s unreadable; starting empty",
                    self._cache_path,
                )
                self._cache = {}
        else:
            self._cache = {}
        return self._cache

    def _save_cache(self) -> None:
        if self._cache is None:
            return
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_text(
            json.dumps(self._cache, indent=2, sort_keys=True)
        )

    async def get_synonyms(self, name: str) -> dict[str, Any] | None:
        """Return ``{"cid": int, "synonyms": [str, ...]}`` or None.

        ``None`` is returned (and cached) when PubChem has no record for
        the name — the typical fault response is HTTP 404 with body
        ``{"Fault": {"Code": "PUGREST.NotFound", ...}}``.

        Cached on disk per normalized name. Cache misses pay one HTTP
        call; subsequent lookups are local.
        """
        if not name or not name.strip():
            return None
        cache = self._load_cache()
        normalized = name.strip().lower()
        if normalized in cache:
            return cache[normalized].get("result")

        url = (
            f"{PUBCHEM_BASE_URL}/compound/name/{name.strip()}/synonyms/JSON"
        )
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(url)
                if resp.status_code == 404:
                    cache[normalized] = {"result": None}
                    self._save_cache()
                    return None
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            logger.debug(
                "PubChem synonyms lookup failed for %r: %s", name, exc,
            )
            cache[normalized] = {"result": None}
            self._save_cache()
            return None

        info_list = (data.get("InformationList") or {}).get("Information") or []
        if not info_list:
            cache[normalized] = {"result": None}
            self._save_cache()
            return None

        first = info_list[0]
        cid = first.get("CID")
        synonyms = first.get("Synonym") or []
        result = {"cid": cid, "synonyms": synonyms}
        cache[normalized] = {"result": result}
        self._save_cache()
        return result
