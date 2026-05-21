"""Codename → canonical-compound resolver (round 23).

Orchestrates a fallback chain for compound names that haven't been
resolved to a known canonical compound by the earlier layers:

  1. ``CODENAME_TO_INN`` curated dict (handled in populate.py before
     this resolver runs — kept here as a reference)
  2. Open Targets chembl_id (handled in populate.py Step 1.5)
  3. RxNorm RxNav → ingredient (INN) — this module
  4. PubChem REST synonyms → match against known compound names — this
     module
  5. Unresolved (logged to ``data/dev/unresolved_codenames.jsonl``)

The resolver does NOT mutate compound nodes itself — it returns a
``ResolvedCodename`` describing what it found, and populate.py applies
the result (rename id/name, append original to aliases, persist rxcui /
pubchem_cid into metadata).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from src.ingestion.pubchem import PubChemClient
from src.ingestion.rxnorm import RxNormClient

logger = logging.getLogger(__name__)

UNRESOLVED_LOG_PATH = Path("data/dev/unresolved_codenames.jsonl")

# Loose normalization for synonym-matching: lowercase + strip every
# non-alphanumeric character. Matches the dedup-normalizer used by
# Open Targets alias cleanup so "MDX-1106" and "MDX1106" collapse.
_NORM_RE = re.compile(r"[^a-z0-9]+")


def _norm(s: str) -> str:
    return _NORM_RE.sub("", s.strip().lower())


@dataclass
class ResolvedCodename:
    """The outcome of a codename-resolution attempt.

    ``canonical_name`` is the INN-equivalent name to rename the compound
    to. ``source`` records which fallback layer found it. ``rxcui`` and
    ``pubchem_cid`` are persisted as compound metadata so cross-trial
    canonicalization can match on them even when the name differs.

    ``matched_existing_id`` is non-None when PubChem's synonym list let
    us map the codename onto an *already-existing* compound node — the
    caller should merge rather than create a new node.
    """

    canonical_name: str
    source: Literal["rxnorm", "pubchem"]
    rxcui: str | None = None
    pubchem_cid: int | None = None
    synonyms: list[str] = field(default_factory=list)
    matched_existing_id: str | None = None


class CodenameResolver:
    """Wraps RxNorm + PubChem with a single resolve-or-None entry point."""

    def __init__(
        self,
        rxnorm: RxNormClient | None = None,
        pubchem: PubChemClient | None = None,
        unresolved_log_path: Path | None = None,
    ) -> None:
        self._rxnorm = rxnorm or RxNormClient()
        self._pubchem = pubchem or PubChemClient()
        self._unresolved_log_path = (
            unresolved_log_path or UNRESOLVED_LOG_PATH
        )

    async def resolve(
        self,
        name: str,
        *,
        known_compound_index: dict[str, str] | None = None,
    ) -> ResolvedCodename | None:
        """Try to resolve ``name`` to a canonical compound name.

        ``known_compound_index`` maps normalized name/alias → existing
        compound id. When provided, PubChem synonyms that match an entry
        in this index set ``matched_existing_id`` so the caller can
        merge instead of creating a new node.

        Returns None when both RxNorm and PubChem fail; the unresolved
        name is appended to ``data/dev/unresolved_codenames.jsonl`` for
        future curation of ``CODENAME_TO_INN``.
        """
        if not name or not name.strip():
            return None

        # 1) RxNorm: codename → top-level RxCUI → ingredient (INN).
        rxnorm_result = await self._rxnorm.resolve_to_ingredient(name)
        if rxnorm_result and rxnorm_result.get("ingredient_name"):
            ingredient = rxnorm_result["ingredient_name"]
            # Only treat as a real resolution if the ingredient name
            # differs from the input — otherwise RxNorm just echoed
            # the input back (the input was already the ingredient).
            if _norm(ingredient) != _norm(name):
                matched_existing = None
                if known_compound_index:
                    matched_existing = known_compound_index.get(
                        _norm(ingredient)
                    )
                return ResolvedCodename(
                    canonical_name=ingredient,
                    source="rxnorm",
                    rxcui=rxnorm_result.get("ingredient_rxcui")
                    or rxnorm_result.get("rxcui"),
                    matched_existing_id=matched_existing,
                )

        # 2) PubChem: codename → synonym list → match against known
        # compound names. Synonyms are ordered roughly by NCBI's
        # confidence — the first synonym whose normalized form matches
        # an existing compound wins. If nothing matches, fall back to
        # the most "INN-shaped" synonym (lowercased word, no digits)
        # as the canonical name.
        pubchem_result = await self._pubchem.get_synonyms(name)
        if pubchem_result:
            synonyms = pubchem_result.get("synonyms") or []
            cid = pubchem_result.get("cid")
            if known_compound_index:
                for syn in synonyms:
                    matched = known_compound_index.get(_norm(syn))
                    if matched:
                        return ResolvedCodename(
                            canonical_name=syn,
                            source="pubchem",
                            pubchem_cid=cid,
                            synonyms=list(synonyms[:10]),
                            matched_existing_id=matched,
                        )
            inn_like = _pick_inn_like_synonym(synonyms, exclude=name)
            if inn_like:
                return ResolvedCodename(
                    canonical_name=inn_like,
                    source="pubchem",
                    pubchem_cid=cid,
                    synonyms=list(synonyms[:10]),
                )

        # 3) Unresolved — log + return None.
        self._log_unresolved(name)
        return None

    def _log_unresolved(self, name: str) -> None:
        try:
            self._unresolved_log_path.parent.mkdir(
                parents=True, exist_ok=True
            )
            with self._unresolved_log_path.open("a") as fh:
                fh.write(json.dumps({"name": name}) + "\n")
        except OSError as exc:
            logger.debug(
                "Could not append to unresolved-codenames log %s: %s",
                self._unresolved_log_path, exc,
            )


_INN_SHAPED_RE = re.compile(r"^[a-z][a-z\-]{3,}$")


def _pick_inn_like_synonym(synonyms: list[str], *, exclude: str) -> str | None:
    """Return the first INN-shaped synonym from a PubChem synonym list.

    Heuristic: an INN (international nonproprietary name) is a single
    lowercased English-ish token — letters and hyphens only, at least 4
    characters, no digits. PubChem orders synonyms with curated names
    first, codes/CAS numbers/SMILES strings later — so the first match
    is usually the INN.

    Returns None when nothing matches the shape; the caller treats the
    name as unresolved and logs it.
    """
    exclude_norm = _norm(exclude)
    for raw in synonyms:
        if not raw:
            continue
        candidate = raw.strip().lower()
        if _norm(candidate) == exclude_norm:
            continue
        if _INN_SHAPED_RE.match(candidate):
            return candidate
    return None
