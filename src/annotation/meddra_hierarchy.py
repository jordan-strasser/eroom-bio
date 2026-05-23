"""MedDRA hierarchy loader for round-28 SOC-tier AE roll-up.

The round-28 architecture wants sibling-compound `target_associated_ae`
propagation to aggregate at MedDRA System Organ Class (SOC) tier so that
e.g. three CETP siblings each reporting different per-event cardiac AEs
(atrial_fibrillation, myocardial_infarction, bradycardia) still
contribute a single `target_associated_ae → AE:cardiac_disorders` edge
on the CETP target node. Without SOC roll-up, the three siblings'
PT-level extractions sit on disjoint AE nodes and the propagation
heuristic ("≥ 2 siblings share an AE") never fires.

This module loads a curated PT → HLT → HLGT → SOC hierarchy from
``data/external/meddra_hierarchy.json``. The file is bootstrapped from
the existing MedDRA normalizer cache plus a curated supplement covering
cardiovascular / endocrine / hepatobiliary AEs common in the corpus —
because MedDRA itself is proprietary, we use the existing LLM-emitted
SOC strings (already canonicalized to the 27 standard SOCs by the
normalizer prompt) as the source of truth.

HLT and HLGT intermediate tiers are stubbed in the file but not
populated in round 28; the round's deliverable is SOC roll-up. When a
future round wires up an OHDSI ATHENA-derived export, drop in the
populated `pt_to_hlt` / `hlt_to_hlgt` maps and the loader picks them up
without further code changes.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default location of the bootstrapped hierarchy file.
HIERARCHY_PATH = Path("data/external/meddra_hierarchy.json")

_AE_PT_RE = re.compile(r"[^a-z0-9]+")


def _ae_prefix_strip(node_or_pt: str) -> str:
    """Strip the ``AE:`` prefix used in AdverseEventNode ids."""
    return node_or_pt[3:] if node_or_pt.startswith("AE:") else node_or_pt


def _slugify_pt(raw: str) -> str:
    """Match the slug rules used by ``src.annotation.meddra.ae_node_id``."""
    return _AE_PT_RE.sub("_", raw.lower()).strip("_")


class MedDRAHierarchy:
    """In-memory lookup over the bootstrapped MedDRA PT → SOC mapping.

    Singleton per process — call ``MedDRAHierarchy.load_default()`` to
    get the shared instance, or construct directly with a path for
    tests. The class is stateless after load; safe to share across
    asyncio coroutines.
    """

    def __init__(
        self,
        *,
        soc_names: dict[str, str],
        pt_to_soc: dict[str, str],
        soc_name_aliases: dict[str, str] | None = None,
        pt_to_hlt: dict[str, str] | None = None,
        hlt_to_hlgt: dict[str, str] | None = None,
        hlgt_to_soc: dict[str, str] | None = None,
        version: str = "",
    ) -> None:
        self.soc_names = dict(soc_names)
        self.pt_to_soc = dict(pt_to_soc)
        self.soc_name_aliases = dict(soc_name_aliases or {})
        self.pt_to_hlt = dict(pt_to_hlt or {})
        self.hlt_to_hlgt = dict(hlt_to_hlgt or {})
        self.hlgt_to_soc = dict(hlgt_to_soc or {})
        self.version = version

    # ── Loading ────────────────────────────────────────────────────────

    _SHARED: "MedDRAHierarchy | None" = None

    @classmethod
    def load_default(cls, path: Path | None = None) -> "MedDRAHierarchy":
        """Return the shared MedDRAHierarchy, loading on first call.

        ``path`` overrides the default file location for tests. The
        per-class cache is invalidated if a different path is passed.
        """
        target = path or HIERARCHY_PATH
        if cls._SHARED is None or getattr(cls._SHARED, "_loaded_from", None) != target:
            cls._SHARED = cls.load(target)
            cls._SHARED._loaded_from = target  # type: ignore[attr-defined]
        return cls._SHARED

    @classmethod
    def load(cls, path: Path) -> "MedDRAHierarchy":
        if not path.exists():
            logger.warning(
                "MedDRA hierarchy file not found at %s; returning empty "
                "hierarchy (SOC roll-up disabled)",
                path,
            )
            return cls(soc_names={}, pt_to_soc={})
        raw: dict[str, Any] = json.loads(path.read_text())
        return cls(
            soc_names=raw.get("soc_names", {}),
            pt_to_soc=raw.get("pt_to_soc", {}),
            soc_name_aliases=raw.get("soc_name_aliases", {}),
            pt_to_hlt=raw.get("pt_to_hlt", {}),
            hlt_to_hlgt=raw.get("hlt_to_hlgt", {}),
            hlgt_to_soc=raw.get("hlgt_to_soc", {}),
            version=raw.get("version", ""),
        )

    # ── Public lookups ─────────────────────────────────────────────────

    def soc_for_name(self, raw_soc_name: str) -> str:
        """Canonicalize a free-text MedDRA SOC string to its slug.

        Returns "" if the name doesn't match the 27 canonical SOCs or
        any alias. Tolerant to case and trailing whitespace; doesn't
        otherwise normalize punctuation (the canonical set keeps the
        exact MedDRA-published comma + paren wording).
        """
        if not raw_soc_name:
            return ""
        lookup = raw_soc_name.strip().lower()
        # Build a reverse map of canonical-name → slug on demand. The
        # canonical set is small (27 entries) so rebuilding per call is
        # cheap and avoids a stale cache.
        for slug, name in self.soc_names.items():
            if name.lower() == lookup:
                return slug
        return self.soc_name_aliases.get(lookup, "")

    def parents_for_pt(
        self,
        pt: str,
        *,
        fallback_soc_name: str = "",
    ) -> dict[str, str]:
        """Return the curated parents for a PT, or empty strings if unknown.

        ``pt`` accepts either a raw MedDRA Preferred Term ("Atrial
        fibrillation") or an AdverseEventNode-style slug ("atrial_fibrillation"
        or "AE:atrial_fibrillation"). The PT is normalized to the slug
        form before lookup.

        ``fallback_soc_name`` lets callers pass the AE node's stored
        ``system_organ_class`` so unknown PTs still resolve to a SOC via
        the canonical-name lookup. Without that fallback, PTs absent
        from the curated table return all-empty parents.

        Output keys: ``hlt_id``, ``hlgt_id``, ``soc_id``, ``soc_name``.
        All four are always present (possibly empty).
        """
        slug = _slugify_pt(_ae_prefix_strip(pt))
        out = {
            "hlt_id": self.pt_to_hlt.get(slug, ""),
            "hlgt_id": "",
            "soc_id": "",
            "soc_name": "",
        }
        if out["hlt_id"]:
            out["hlgt_id"] = self.hlt_to_hlgt.get(out["hlt_id"], "")
        if out["hlgt_id"]:
            out["soc_id"] = self.hlgt_to_soc.get(out["hlgt_id"], "")
        if not out["soc_id"]:
            out["soc_id"] = self.pt_to_soc.get(slug, "")
        if not out["soc_id"] and fallback_soc_name:
            out["soc_id"] = self.soc_for_name(fallback_soc_name)
        if out["soc_id"]:
            out["soc_name"] = self.soc_names.get(out["soc_id"], "")
        return out

    def aggregate_tier(
        self,
        pts: list[str],
        tier: str = "soc",
        *,
        fallback_soc_by_pt: dict[str, str] | None = None,
    ) -> dict[str, list[str]]:
        """Group ``pts`` by their parent id at the requested ``tier``.

        ``tier`` ∈ {"hlt", "hlgt", "soc"}. PTs whose parent at that tier
        is unknown are skipped (NOT bucketed under "").

        ``fallback_soc_by_pt`` lets the caller pass per-PT SOC strings
        (typically the AE node's stored ``system_organ_class``) so PTs
        absent from the curated table still resolve via canonical-name
        lookup. Only consulted for the SOC tier.

        Returns ``{parent_id: [pt_slug, …]}``.
        """
        if tier not in ("hlt", "hlgt", "soc"):
            raise ValueError(
                f"tier must be one of hlt/hlgt/soc, got {tier!r}"
            )
        fallback = fallback_soc_by_pt or {}
        out: dict[str, list[str]] = {}
        for raw_pt in pts:
            slug = _slugify_pt(_ae_prefix_strip(raw_pt))
            parents = self.parents_for_pt(
                slug, fallback_soc_name=fallback.get(raw_pt, ""),
            )
            parent_id = parents.get(f"{tier}_id", "")
            if not parent_id:
                continue
            out.setdefault(parent_id, []).append(slug)
        return out

    # ── Introspection helpers ──────────────────────────────────────────

    @property
    def n_pts(self) -> int:
        return len(self.pt_to_soc)

    @property
    def n_socs(self) -> int:
        return len(self.soc_names)
