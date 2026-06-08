"""Controlled vocabulary for canonicalizing patient subgroup descriptions.

Trials report subgroups in free text ("PD-L1 ≥1% by IHC 28-8", "previously
treated with anti-PD-1"). The graph wants stable, abstract identifiers so
that two trials reporting *equivalent* subgroups land on the same
PopulationNode and can share evidence.

Two axis flavors:
  - **Open-vocab gene axis** (axis="gene"): key is a HUGO symbol—the
    vocabulary grows automatically with new genes. Levels are limited to a
    small canonical set plus pass-through specific variants (G12C, V600E).
  - **Closed-vocab non-gene axes** (line, performance, age, prior_tx,
    signature): small, slow-growing—extending requires editing this file.

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
    # RECIST response strata. Trials report subgroup outcomes by best
    # response category (CR/PR/SD/PD); collapsing them onto a canonical
    # axis lets evidence aggregate across trials instead of producing
    # one-off "other_complete_response" populations per study.
    "response":    [
        "complete_response", "partial_response",
        "stable_disease", "progressive_disease",
        "responder", "non_responder", "unknown",
    ],
    # Anti-drug antibody / immunogenicity status. Real stratifier for
    # biologics trials — HAHA-positive patients can neutralize biologic
    # drugs and respond differently. Round 3.2 added this axis after
    # the dev-log audit surfaced HAHA-positive / HAHA-negative / ADA
    # entries that were dropping to axis="other".
    "antibody_status": ["positive", "negative", "unknown"],
    # Multi-indication (non-onco) cohort definers. disease activity/severity
    # (RA active, UC moderate-severe, "in remission") and functional class
    # (ACR / NYHA I–IV) are the primary patient-selection axes outside
    # oncology, where line/stage rarely apply.
    "severity": ["mild", "moderate", "severe", "active", "remission", "unknown"],
    "functional_class": ["i", "ii", "iii", "iv", "unknown"],
}

# Canonical levels for the open-vocab non-gene ``biomarker`` axis (RF,
# anti-CCP, LVEF, HbA1c, CRP, …). Key is the free marker slug; level is a
# direction, mirroring the gene axis's positive/negative collapse.
BIOMARKER_LEVELS: set[str] = {"positive", "negative", "high", "low", "unknown"}
_BIOMARKER_LEVEL_SYNONYMS: dict[str, str] = {
    "elevated": "high", "raised": "high", "reduced": "low", "decreased": "low",
    "present": "positive", "absent": "negative", "seropositive": "positive",
    "seronegative": "negative",
}


# Hypothetical level → canonical level for the antibody_status axis.
# LLM emissions like "haha_positive_baseline", "ada_positive", "haha_pos"
# all collapse to the bare positive/negative level — the timing context
# ("at baseline") lives in raw_descriptor metadata, not the level.
_ANTIBODY_STATUS_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^(haha|ada|anti[-_]drug[-_]antibody|antidrug)[-_]?(positive|pos)(?:[-_].*)?$", re.I), "positive"),
    (re.compile(r"^(haha|ada|anti[-_]drug[-_]antibody|antidrug)[-_]?(negative|neg)(?:[-_].*)?$", re.I), "negative"),
    (re.compile(r"^(positive|pos)[-_]?(haha|ada)(?:[-_].*)?$", re.I), "positive"),
    (re.compile(r"^(negative|neg)[-_]?(haha|ada)(?:[-_].*)?$", re.I), "negative"),
]


# Raw descriptor patterns we silently drop (axis="other" feature with a
# special sentinel level). These aren't patient subgroups — they're
# analysis time points, individual patient identifiers, ECOG-change
# Likert labels, generic yes/no markers without context. They reach
# canonicalize_feature only because the extractor prompt isn't perfect;
# rather than re-extract every trial, recognize them at canonicalization
# time and skip the dev-log emission.
_KNOWN_NON_STRATIFIER_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Analysis time points: "Final analysis", "Primary completion",
    # "Month N", "Week N", "Day N", "Cycle N", "Interim analysis"
    re.compile(r"^(final|primary|interim)\s+(analysis|completion)$", re.I),
    re.compile(r"^(month|week|day|cycle)\s+\d+$", re.I),
    re.compile(r"^\d+[-\s]?(week|month|day)s?\s+(landmark|follow[-\s]?up)$", re.I),
    # Continuous biomarker measurements at time points (PD readouts)
    re.compile(r"\bper\s+mm[²2]\b.*\bday\s*\d+", re.I),
    re.compile(r"\b(pre|post)[-\s]?(vaccine|baseline|treatment)\b.*\bday\s*\d+", re.I),
    re.compile(r"^cd\d+\s+t\s+cells?.*day\s*\d+", re.I),
    # Individual patient identifiers
    re.compile(r"^patient\s+#?\s*\d+$", re.I),
    re.compile(r"^subject\s+#?\s*\d+$", re.I),
    # Likert change-from-baseline outcome labels (ECOG / QoL change)
    re.compile(r"^(better|worse|no\s+change|missing|improved|worsened|stable)$", re.I),
    # Generic yes / no without context
    re.compile(r"^(yes|no)$", re.I),
    # Non-evaluable outcome states (RECIST partial — companion to the
    # response-axis auto-promotion in round 3.1)
    re.compile(r"^not\s+(evaluable|evaluated|assessed)(\s*\([a-z]+\))?$", re.I),
)


# Sentinel level used on returned SubgroupFeature when the raw_descriptor
# matched _KNOWN_NON_STRATIFIER_PATTERNS. Picked up by log_unmapped to
# skip logging — the populator drops these via the existing axis="other"
# filter, so the only effect is suppressing dev-log noise.
_NON_STRATIFIER_LEVEL = "_known_non_stratifier"


# Short-form aliases for the response axis (LLM emits "CR" / "PR" / etc.).
_RESPONSE_ALIASES: dict[str, str] = {
    "cr": "complete_response",
    "pr": "partial_response",
    "sd": "stable_disease",
    "pd": "progressive_disease",
}


# ── Open vocabulary: gene axis ──────────────────────────────────────────

# Canonical gene-axis levels. The expression-level axis uses a single
# direction cut: positive (above threshold) / negative (below threshold).
# Older inputs of "high" / "low" are accepted at the canonicalize step
# and collapsed to positive / negative — the percentage threshold itself
# (≥ 1% vs ≥ 5% vs ≥ 10% for PD-L1, for instance) lives in the raw
# descriptor metadata, not the level. Forking populations on threshold
# created cd274_low + cd274_positive + cd274_high nodes for the same
# trial; positive/negative is the one consistent axis.
GENE_LEVELS: set[str] = {
    "positive", "negative",
    "mutant", "wildtype", "unselected", "unknown",
}


# Expression-level synonyms — collapsed to the positive/negative axis
# during canonicalization so cd274_high and cd274_positive don't
# fragment into two nodes for the same biology.
_GENE_LEVEL_SYNONYMS: dict[str, str] = {
    "high": "positive",
    "low": "negative",
    "expressed": "positive",
    "overexpressed": "positive",
    "not_expressed": "negative",
    "amplified": "positive",
    "deleted": "negative",
}

# Specific point mutations like G12C, V600E, T790M, R248Q*—single letter,
# one to four digits, single letter (or '*' for nonsense). Pass-through.
VARIANT_PATTERN = re.compile(r"^[A-Z]\d{1,4}[A-Z*]$")

# HUGO symbol shape: starts with a letter, alphanumerics + dashes, 1–20 chars.
# Permissive on purpose—we don't ship the full HUGO table here, this is a
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

    # Drop known non-stratifier descriptors silently. These come through
    # as axis="other" from the extractor (analysis time points, ECOG-
    # change Likert labels, "Patient #N" labels, etc.) but the populator
    # will discard them via the standard axis="other" filter anyway —
    # so the only thing left to do is mark them with a sentinel level
    # that suppresses the dev-log emission. Cheap stopgap; the proper
    # fix would re-extract every trial under a stricter prompt.
    if descriptor and any(
        p.search(descriptor) for p in _KNOWN_NON_STRATIFIER_PATTERNS
    ):
        return SubgroupFeature(
            axis="other", key="", level=_NON_STRATIFIER_LEVEL,
            raw_descriptor=descriptor,
        )

    # Self-heal stale axis labels. Older cached extractions (and Sonnet's
    # occasional miscategorization) emit RECIST states with axis="other" or
    # blank. Promote those to axis="response" when the level matches a
    # known response category — same effect as fresh extraction under the
    # current prompt, no re-extraction cost.
    if axis in ("", "other"):
        response_aliases = NON_GENE_AXES["response"]
        if level in response_aliases or level in _RESPONSE_ALIASES:
            axis = "response"
        # Same self-heal for anti-drug antibody status. The extractor
        # often emits HAHA / ADA descriptors as axis="other" with a
        # level like "haha_positive_baseline". Promote to the dedicated
        # antibody_status axis.
        for pattern, canonical_level in _ANTIBODY_STATUS_PATTERNS:
            if pattern.match(level) or pattern.match(descriptor):
                axis = "antibody_status"
                level = canonical_level
                break

    if axis == "gene":
        # Variant tokens are case-sensitive; normalize the input but accept
        # the canonical "G12C"/"V600E" capitalization.
        variant_candidate = level_raw.strip().upper()
        if VARIANT_PATTERN.match(variant_candidate):
            level_out = variant_candidate
        elif level in _GENE_LEVEL_SYNONYMS:
            # Expression-level synonyms collapse to positive/negative —
            # PD-L1 "high" / "low" / "≥1%" / "≥5%" all share one biology
            # so cross-trial evidence should accumulate on one node.
            level_out = _GENE_LEVEL_SYNONYMS[level]
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
                # Resolver is loaded but doesn't know this symbol—likely
                # not a real HUGO gene. Reject rather than silently accept
                # an unrecognized name as if it were canonical.
                return _other(axis_raw, level_raw, descriptor)

        return SubgroupFeature(
            axis="gene", key=key, level=level_out, raw_descriptor=descriptor,
        )

    if axis == "biomarker":
        # Open-vocab NON-gene marker (RF, anti-CCP, LVEF, HbA1c, CRP, …). Free
        # key slug + a direction level (positive/negative/high/low). Unlike the
        # gene axis there's no HUGO gate — these aren't genes.
        key = re.sub(r"[^a-z0-9]+", "_", key_raw.lower()).strip("_")
        lvl = _BIOMARKER_LEVEL_SYNONYMS.get(level, level)
        if key and lvl in BIOMARKER_LEVELS:
            return SubgroupFeature(
                axis="biomarker", key=key, level=lvl, raw_descriptor=descriptor,
            )
        return _other(axis_raw, level_raw, descriptor)

    if axis in NON_GENE_AXES:
        # Resolve common short-forms before membership check so "CR" /
        # "PR" / "SD" / "PD" land on the canonical level.
        if axis == "response":
            level = _RESPONSE_ALIASES.get(level, level)
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

    No-op for canonical features and for features whose descriptor was
    recognized as a known non-stratifier pattern (analysis time points,
    individual patient labels, etc.) — those don't carry vocab-extension
    signal. Caller decides when to invoke; typically only when
    ``feature.axis == "other"``. The log is the input for vocab
    expansion: terms that show up often get promoted into ``NON_GENE_AXES``
    or ``GENE_LEVELS``.
    """
    if feature.axis != "other":
        return
    if feature.level == _NON_STRATIFIER_LEVEL:
        return  # known non-stratifier pattern, suppressed
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
    lines.append("axis='gene'—key=HUGO gene symbol; level one of:")
    lines.append(
        "  " + ", ".join(sorted(GENE_LEVELS))
        + ", or a specific variant like G12C / V600E / T790M"
    )
    lines.append(
        "axis='biomarker'—key=NON-gene marker name (RF, anti_ccp, lvef, hba1c, "
        "crp, esr); level one of: " + ", ".join(sorted(BIOMARKER_LEVELS))
    )
    lines.append("")
    for axis, levels in NON_GENE_AXES.items():
        lines.append(f"axis='{axis}'—key=''; level one of: {', '.join(levels)}")
    lines.append("")
    lines.append(
        "If a subgroup descriptor doesn't fit any of the above, "
        "emit axis='other' with the raw descriptor preserved."
    )
    return "\n".join(lines)
