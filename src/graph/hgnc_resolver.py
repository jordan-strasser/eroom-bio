"""Resolve free-text gene names to canonical HUGO (HGNC) symbols.

Canonical lookups are essential for the subgroup vocabulary: the LLM may
emit "PD-L1", "PDL1", "B7-H1" — all aliases for the same gene — and we
need them to collapse onto the same node id (``cd274_high``) so two
trials reporting the same biomarker actually link to the same
PopulationNode.

Source of truth: HGNC's complete_set TSV (~15 MB), downloaded once and
cached at ``data/cache/hgnc_complete_set.tsv``. The in-memory dict is
~10 MB after parsing — every approved symbol plus every alias_symbol /
prev_symbol entry maps to the canonical ``symbol`` column.

If the cache file is absent and ``download_if_missing=True`` is passed
to ``load()``, the resolver fetches it on first use. Otherwise
``canonical_symbol`` returns ``None`` for everything until the cache
is populated, and callers can fall back to format-only validation.
"""

from __future__ import annotations

import logging
import re
import threading
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)


HGNC_URL = (
    "https://storage.googleapis.com/public-download-files/hgnc/"
    "tsv/tsv/hgnc_complete_set.txt"
)
DEFAULT_CACHE_PATH = Path("data/cache/hgnc_complete_set.tsv")


# ── Module state ────────────────────────────────────────────────────────

_LOCK = threading.Lock()
# alias (lowercased, punctuation stripped) → canonical HUGO symbol
_ALIAS_TO_CANONICAL: dict[str, str] | None = None
# canonical HUGO symbol → list of original-case aliases (preserved
# punctuation, e.g. "PD-1" with the hyphen rather than "pd1") so that
# downstream graph nodes can carry the names a human or LLM would write.
_CANONICAL_TO_ALIASES: dict[str, list[str]] = {}


_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


def _normalize_alias(name: str) -> str:
    """Lowercase + strip non-alphanumerics. ``PD-L1`` / ``PDL1`` / ``pd_l1``
    all collapse to ``pdl1``."""
    return _NORMALIZE_RE.sub("", name.lower())


# ── Loading ─────────────────────────────────────────────────────────────


def is_loaded() -> bool:
    """True iff the alias dict has been populated."""
    return _ALIAS_TO_CANONICAL is not None


def load(
    cache_path: Path = DEFAULT_CACHE_PATH,
    download_if_missing: bool = True,
    timeout: float = 60.0,
) -> int:
    """Populate the alias dicts from the HGNC TSV. Returns alias count.

    Idempotent: subsequent calls with a populated dict are no-ops unless
    ``cache_path`` is missing and re-download succeeds.
    """
    global _ALIAS_TO_CANONICAL, _CANONICAL_TO_ALIASES
    with _LOCK:
        if _ALIAS_TO_CANONICAL is not None:
            return len(_ALIAS_TO_CANONICAL)

        if not cache_path.exists() and download_if_missing:
            _download_hgnc_tsv(cache_path, timeout=timeout)

        if not cache_path.exists():
            logger.warning(
                "HGNC cache %s missing and download skipped; resolver disabled",
                cache_path,
            )
            _ALIAS_TO_CANONICAL = {}
            _CANONICAL_TO_ALIASES = {}
            return 0

        _ALIAS_TO_CANONICAL, _CANONICAL_TO_ALIASES = _parse_hgnc_tsv(cache_path)
        logger.info(
            "Loaded %d HGNC alias mappings from %s",
            len(_ALIAS_TO_CANONICAL), cache_path,
        )
        return len(_ALIAS_TO_CANONICAL)


def _download_hgnc_tsv(cache_path: Path, timeout: float) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading HGNC complete_set TSV from %s", HGNC_URL)
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        resp = client.get(HGNC_URL)
        resp.raise_for_status()
        cache_path.write_bytes(resp.content)
    logger.info(
        "Cached HGNC TSV to %s (%d bytes)",
        cache_path, cache_path.stat().st_size,
    )


def _parse_hgnc_tsv(
    cache_path: Path,
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Parse the HGNC TSV into two indexes:

    - ``alias_lookup`` (normalized alias → canonical symbol) for fast lookups
    - ``canonical_to_aliases`` (canonical → list of original-case aliases)

    Aliases pulled from three columns: ``symbol`` (the canonical itself,
    so symbol→symbol roundtrips), ``alias_symbol`` (pipe-separated), and
    ``prev_symbol`` (pipe-separated).
    """
    alias_lookup: dict[str, str] = {}
    canonical_to_aliases: dict[str, list[str]] = {}

    with cache_path.open("r", encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        try:
            symbol_col = header.index("symbol")
        except ValueError:
            raise ValueError(
                f"HGNC TSV at {cache_path} missing 'symbol' column"
            )
        alias_col = header.index("alias_symbol") if "alias_symbol" in header else None
        prev_col = header.index("prev_symbol") if "prev_symbol" in header else None

        for line in fh:
            cols = line.rstrip("\n").split("\t")
            if len(cols) <= symbol_col:
                continue
            canonical = cols[symbol_col].strip()
            if not canonical:
                continue
            # The canonical maps to itself in the lookup, but doesn't go into
            # its own aliases list (callers know the canonical separately).
            _add_alias(alias_lookup, canonical, canonical)
            originals: list[str] = []
            if alias_col is not None and len(cols) > alias_col:
                for alias in _split_pipe(cols[alias_col]):
                    _add_alias(alias_lookup, alias, canonical)
                    originals.append(alias)
            if prev_col is not None and len(cols) > prev_col:
                for prev in _split_pipe(cols[prev_col]):
                    _add_alias(alias_lookup, prev, canonical)
                    originals.append(prev)
            if originals:
                # Dedupe while preserving first-seen order.
                seen: set[str] = set()
                deduped = [a for a in originals if not (a in seen or seen.add(a))]
                canonical_to_aliases[canonical] = deduped

    return alias_lookup, canonical_to_aliases


def _split_pipe(text: str) -> list[str]:
    """HGNC pipe-separated lists are also surrounded by quotes in some
    rows. Handle both ``A|B|C`` and ``"A|B|C"`` shapes."""
    cleaned = text.strip().strip('"')
    if not cleaned:
        return []
    return [t.strip() for t in cleaned.split("|") if t.strip()]


def _add_alias(d: dict[str, str], alias: str, canonical: str) -> None:
    key = _normalize_alias(alias)
    if not key:
        return
    # On collision, prefer the lexicographically smaller canonical so
    # behavior is deterministic. This matters in rare cases where an
    # ambiguous alias has been used for two different genes.
    existing = d.get(key)
    if existing is None or canonical < existing:
        d[key] = canonical


# ── Lookup ──────────────────────────────────────────────────────────────


def canonical_symbol(name: str) -> str | None:
    """Return the canonical HGNC symbol for ``name``, or ``None`` if unknown.

    Lazy-loads the cache on first call. If the cache is missing AND
    download fails, returns ``None`` for all inputs (caller should
    handle the resolver-disabled case explicitly).
    """
    if not name or not name.strip():
        return None
    if _ALIAS_TO_CANONICAL is None:
        load(download_if_missing=True)
    if not _ALIAS_TO_CANONICAL:
        return None
    return _ALIAS_TO_CANONICAL.get(_normalize_alias(name))


def aliases_for(symbol: str) -> list[str]:
    """Return original-case aliases for ``symbol`` (a HUGO canonical or alias).

    First resolves to the canonical, then returns its alias list. Empty
    list when the symbol is unknown OR has no recorded aliases. The
    canonical itself is NOT included in the returned list (callers
    already know the canonical).
    """
    canonical = canonical_symbol(symbol)
    if canonical is None:
        return []
    return list(_CANONICAL_TO_ALIASES.get(canonical, []))


def reset_for_test() -> None:
    """Reset module state. Use only from tests."""
    global _ALIAS_TO_CANONICAL, _CANONICAL_TO_ALIASES
    with _LOCK:
        _ALIAS_TO_CANONICAL = None
        _CANONICAL_TO_ALIASES = {}
