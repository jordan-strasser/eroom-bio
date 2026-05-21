"""RxNorm RxNav REST client — codename → ingredient INN fallback.

Round-23 codename canonicalization. NLM's RxNav exposes RxNorm — the
US drug-vocabulary standard that links brand names, generic names,
investigational codenames, and ingredient (IN) concepts. For a codename
like "MK-3475", RxNorm walks to ingredient "pembrolizumab" with a single
related-concept hop. Free, no auth required.

The match step lives in `src/graph/codename_resolver.py`; this module is
just the HTTP + on-disk cache.

API reference: https://rxnav.nlm.nih.gov/RxNormAPIs.html
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

RXNAV_BASE_URL = "https://rxnav.nlm.nih.gov/REST"


class RxNormClient:
    """Minimal RxNav client for name → ingredient resolution.

    Two endpoints used:
      - ``/rxcui.json?name=<name>`` → top-level RxCUI for a name
      - ``/rxcui/<rxcui>/related.json?tty=IN`` → ingredient concepts

    Responses are cached on disk per name so repeated builds don't
    re-hit NLM.
    """

    def __init__(
        self,
        cache_path: Path | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._timeout = timeout
        self._cache_path = (
            cache_path or Path("data/cache/rxnorm_lookups.json")
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
                    "RxNorm cache %s unreadable; starting empty",
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

    async def find_rxcui(self, name: str) -> str | None:
        """Look up the top-level RxCUI for a drug name. None if missing."""
        if not name or not name.strip():
            return None
        url = f"{RXNAV_BASE_URL}/rxcui.json"
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(url, params={"name": name.strip()})
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            logger.debug(
                "RxNorm rxcui lookup failed for %r: %s", name, exc,
            )
            return None
        ids = ((data.get("idGroup") or {}).get("rxnormId") or [])
        if not ids:
            return None
        return ids[0]

    async def get_ingredient(self, rxcui: str) -> dict[str, Any] | None:
        """Return the first IN-tty (ingredient) concept for an RxCUI.

        Shape: ``{"rxcui": str, "name": str}``. Returns None when the
        RxCUI has no IN-typed relatives.
        """
        if not rxcui:
            return None
        url = f"{RXNAV_BASE_URL}/rxcui/{rxcui}/related.json"
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(url, params={"tty": "IN"})
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            logger.debug(
                "RxNorm related lookup failed for %s: %s", rxcui, exc,
            )
            return None
        groups = (data.get("relatedGroup") or {}).get("conceptGroup") or []
        for grp in groups:
            if grp.get("tty") != "IN":
                continue
            props = grp.get("conceptProperties") or []
            if not props:
                continue
            first = props[0]
            return {
                "rxcui": first.get("rxcui"),
                "name": first.get("name"),
            }
        return None

    async def resolve_to_ingredient(self, name: str) -> dict[str, Any] | None:
        """Resolve a codename / brand / generic name to its IN concept.

        Returns ``{"rxcui": str, "ingredient_name": str, "ingredient_rxcui": str}``
        or None. The top-level rxcui can equal the ingredient rxcui if
        ``name`` is already the ingredient name; both are returned so the
        caller can distinguish brand-to-generic from same-as.

        Cached on disk per normalized name (both hits and misses).
        """
        if not name or not name.strip():
            return None
        cache = self._load_cache()
        normalized = name.strip().lower()
        if normalized in cache:
            return cache[normalized].get("result")

        rxcui = await self.find_rxcui(name)
        if not rxcui:
            cache[normalized] = {"result": None}
            self._save_cache()
            return None

        ingredient = await self.get_ingredient(rxcui)
        if not ingredient or not ingredient.get("name"):
            cache[normalized] = {"result": None}
            self._save_cache()
            return None

        result = {
            "rxcui": rxcui,
            "ingredient_name": ingredient["name"],
            "ingredient_rxcui": ingredient.get("rxcui"),
        }
        cache[normalized] = {"result": result}
        self._save_cache()
        return result
