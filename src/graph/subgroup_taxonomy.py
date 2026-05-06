"""Controlled vocabulary for canonicalizing patient subgroup descriptions.

Trials report subgroups in free text ("PD-L1 ≥1% by IHC 28-8", "previously
treated with anti-PD-1"). The graph wants stable, abstract identifiers so
that two trials reporting *equivalent* subgroups land on the same
PopulationNode and can share evidence.

Two axis flavors:
  - **Open-vocab gene axis** (axis="gene"): key is a HUGO symbol — the
    vocabulary grows automatically with new genes. Levels are limited to a
    small canonical set plus pass-through specific variants (G12C, V600E).
  - **Closed-vocab non-gene axes** (line, performance, age, prior_tx,
    signature): small, slow-growing — extending requires editing this file.

Anything that doesn't canonicalize falls into ``axis="other"`` with the
raw descriptor preserved AND appended to
``data/dev/unmapped_subgroup_features.jsonl`` for vocab-extension review.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from src.graph.models import SubgroupFeature

logger = logging.getLogger(__name__)


# ── Closed vocabulary: non-gene axes ────────────────────────────────────

NON_GENE_AXES: dict[str, list[str]] = {
    "line":        ["first", "second", "later", "unknown"],
    "prior_tx":    ["naive", "treated", "unknown"],
    # ECOG good = 0–1; poor = ≥2
    "performance": ["good", "poor", "unknown"],
    "age":         ["pediatric", "adult", "elderly", "unknown"],
    # Non-gene molecular signatures: MSI, TMB, HRD-style aggregate scores
    "signature":   ["msi_high", "mss", "tmb_high", "tmb_low", "hrd", "unknown"],
}


# ── Open vocabulary: gene axis ──────────────────────────────────────────

# General levels common across genes
GENE_LEVELS: set[str] = {
    "high", "low", "positive", "negative",
    "mutant", "wildtype", "unselected", "unknown",
}

# Specific point mutations like G12C, V600E, T790M, R248Q* — single letter,
# one to four digits, single letter (or '*' for nonsense). Pass-through.
VARIANT_PATTERN = re.compile(r"^[A-Z]\d{1,4}[A-Z*]$")

# HUGO symbol shape: starts with a letter, alphanumerics + dashes, 1–20 chars.
# Permissive on purpose — we don't ship the full HUGO table here, this is a
# format gate not a membership check.
HUGO_PATTERN = re.compile(r"^[A-Z][A-Z0-9-]{0,19}$")


# ── Where unmapped features get logged ──────────────────────────────────

UNMAPPED_LOG_PATH = Path("data/dev/unmapped_subgroup_features.jsonl")


def _normalize_level_token(level_raw: str) -> str:
    """Lowercase, strip whitespace, collapse separators to underscores."""
    return re.sub(r"[\s\-]+", "_", level_raw.strip().lower())


def _normalize_axis_token(axis_raw: str) -> str:
    return re.sub(r"[\s\-]+", "_", axis_raw.strip().lower())


def canonicalize_feature(
    axis_raw: str,
    key_raw: str,
    level_raw: str,
    raw_descriptor: str = "",
) -> SubgroupFeature:
    """Map free-text axis/key/level into a canonical ``SubgroupFeature``.

    Returns a ``SubgroupFeature(axis="other", ...)`` carrying the raw text
    when no canonical mapping applies. The caller is responsible for
    invoking ``log_unmapped`` so the dev review file accumulates examples.
    """
    descriptor = raw_descriptor or _compose_descriptor(axis_raw, key_raw, level_raw)

    axis = _normalize_axis_token(axis_raw)
    level = _normalize_level_token(level_raw)

    if axis == "gene":
        # Variant tokens are case-sensitive; normalize the input but accept
        # the canonical "G12C"/"V600E" capitalization.
        variant_candidate = level_raw.strip().upper()
        if VARIANT_PATTERN.match(variant_candidate):
            level_out = variant_candidate
        elif level in GENE_LEVELS:
            level_out = level
        else:
            return _other(axis_raw, level_raw, descriptor)

        # Resolve aliases to the canonical HUGO symbol so PD-L1 / PDL1 /
        # B7-H1 / CD274 all collapse onto the same key. Falls back to
        # format-only validation when the resolver isn't loaded (no cache,
        # no network) so the system still works offline.
        from src.graph.hgnc_resolver import canonical_symbol, is_loaded
        canonical = canonical_symbol(key_raw)
        if canonical is not None:
            key = canonical
        else:
            key = key_raw.strip().upper()
            if not HUGO_PATTERN.match(key):
                return _other(axis_raw, level_raw, descriptor)
            if is_loaded():
                # Resolver is loaded but doesn't know this symbol — likely
                # not a real HUGO gene. Reject rather than silently accept
                # an unrecognized name as if it were canonical.
                return _other(axis_raw, level_raw, descriptor)

        return SubgroupFeature(
            axis="gene", key=key, level=level_out, raw_descriptor=descriptor,
        )

    if axis in NON_GENE_AXES:
        if level in NON_GENE_AXES[axis]:
            return SubgroupFeature(
                axis=axis, key="", level=level, raw_descriptor=descriptor,
            )
        return _other(axis_raw, level_raw, descriptor)

    return _other(axis_raw, level_raw, descriptor)


def _other(axis_raw: str, level_raw: str, descriptor: str) -> SubgroupFeature:
    fallback_level = _normalize_level_token(level_raw) or "unmapped"
    return SubgroupFeature(
        axis="other", key="", level=fallback_level, raw_descriptor=descriptor,
    )


def _compose_descriptor(axis_raw: str, key_raw: str, level_raw: str) -> str:
    parts = [p for p in (axis_raw, key_raw, level_raw) if p]
    return " ".join(parts)


def is_canonical(feature: SubgroupFeature) -> bool:
    """True if the feature came out of a successful canonicalization."""
    return feature.axis != "other"


def log_unmapped(
    feature: SubgroupFeature,
    trial_id: str,
    *,
    log_path: Path = UNMAPPED_LOG_PATH,
) -> None:
    """Append an unmapped feature to the dev log (jsonl, one record per line).

    No-op for canonical features. Caller decides when to invoke — typically
    only when ``feature.axis == "other"``. The log is the input for vocab
    expansion: terms that show up often get promoted into ``NON_GENE_AXES``
    or ``GENE_LEVELS``.
    """
    if feature.axis != "other":
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "trial_id": trial_id,
        "raw_descriptor": feature.raw_descriptor,
        "fallback_level": feature.level,
        "logged_at": datetime.now(timezone.utc).isoformat(),
    }
    with log_path.open("a") as fh:
        fh.write(json.dumps(record) + "\n")


def vocabulary_for_prompt() -> str:
    """Render the controlled vocab as a string for inclusion in LLM prompts.

    Used by the extractor's system prompt so the model knows which axes
    and levels are canonical (and what to map free-text descriptors to).
    """
    lines = ["Canonical subgroup feature vocabulary:"]
    lines.append("")
    lines.append("axis='gene' — key=HUGO gene symbol; level one of:")
    lines.append(
        "  " + ", ".join(sorted(GENE_LEVELS))
        + ", or a specific variant like G12C / V600E / T790M"
    )
    lines.append("")
    for axis, levels in NON_GENE_AXES.items():
        lines.append(f"axis='{axis}' — key=''; level one of: {', '.join(levels)}")
    lines.append("")
    lines.append(
        "If a subgroup descriptor doesn't fit any of the above, "
        "emit axis='other' with the raw descriptor preserved."
    )
    return "\n".join(lines)
