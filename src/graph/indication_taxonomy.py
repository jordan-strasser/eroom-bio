"""Canonicalize indication strings and extract default population qualifiers.

CT.gov condition strings are free text. Two trials studying the same
disease can phrase the indication very differently ("Stage IIIC Cutaneous
Melanoma AJCC v7" vs "Unresectable or Metastatic Melanoma"), which
fragments evidence across what should be one IndicationNode. This module
provides:

A) ``extract_indication_qualifiers`` — a deterministic regex parser that
   pulls staging, subtype, severity, treatment-setting, and similar
   qualifiers out of a condition string and returns them as
   ``SubgroupFeature`` objects. Compose those features with a canonical
   indication id via ``PopulationNode.compose_id`` to produce the
   trial's default PopulationNode (e.g.
   ``melanoma__histology_cutaneous__stage_iii``).

B) Disease canonicalization itself (raw text → ``melanoma``) is an LLM
   call that lives in ``PopulationPipeline._canonicalize_indication``
   because it needs the Anthropic client; the prompts there are kept
   intentionally generic so the same routine works across oncology,
   autoimmune disease, infectious disease, etc.

The pattern library below is intentionally cross-domain. Generic axes
(``stage``, ``severity``, ``extent``, ``setting``, ``line``, ``age``)
apply to many disease areas. Disease-specific histology subtypes are
listed in tiers and earlier patterns win when the same axis would match
twice — most-specific first within each axis.
"""

from __future__ import annotations

import re

from src.graph.models import SubgroupFeature


# Each entry: (compiled_regex, axis, level). Earlier rows beat later rows
# for the same axis. Patterns are case-insensitive.
_QUALIFIER_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    # ── Stage tier (cancer + other staged diseases) ──
    # Most-specific first so "stage iii" doesn't shadow "stage iv".
    (re.compile(r"\bstage\s+iv\b|\bstage\s+4\b", re.I),                       "stage", "iv"),
    (re.compile(r"\bstage\s+iii[abc]?\b|\bstage\s+3\b", re.I),                "stage", "iii"),
    (re.compile(r"\bstage\s+ii[abc]?\b(?!i)|\bstage\s+2\b", re.I),            "stage", "ii"),
    (re.compile(r"\bstage\s+i[abc]?\b(?!v|i)|\bstage\s+1\b", re.I),           "stage", "i"),
    (re.compile(r"\bstage\s+0\b|\bcarcinoma\s+in\s+situ\b", re.I),            "stage", "0"),

    # ── Severity tier (chronic / autoimmune / inflammatory) ──
    # "moderate to severe" matched before plain "moderate" / "severe".
    (re.compile(r"\bmoderate\s*(?:to|-)\s*severe\b", re.I),                   "severity", "moderate_severe"),
    (re.compile(r"\bmild\s*(?:to|-)\s*moderate\b", re.I),                     "severity", "mild_moderate"),
    (re.compile(r"\bsevere\b", re.I),                                         "severity", "severe"),
    (re.compile(r"\bmoderate\b", re.I),                                       "severity", "moderate"),
    (re.compile(r"\bmild\b", re.I),                                           "severity", "mild"),

    # ── Extent / spread ──
    (re.compile(r"\bmetastatic\b", re.I),                                     "extent", "metastatic"),
    (re.compile(r"\bunresectable\b", re.I),                                   "extent", "unresectable"),
    (re.compile(r"\bresectable\b", re.I),                                     "extent", "resectable"),
    (re.compile(r"\blocally\s+advanced\b", re.I),                             "extent", "locally_advanced"),
    (re.compile(r"\bdisseminated\b", re.I),                                   "extent", "disseminated"),
    (re.compile(r"\blocalized\b|\bnon[\s-]?metastatic\b", re.I),              "extent", "localized"),
    (re.compile(r"\badvanced\b", re.I),                                       "extent", "advanced"),

    # ── Treatment setting / context ──
    (re.compile(r"\bneo[\s-]?adjuvant\b", re.I),                              "setting", "neoadjuvant"),
    (re.compile(r"\badjuvant\b", re.I),                                       "setting", "adjuvant"),
    (re.compile(r"\bmaintenance\b", re.I),                                    "setting", "maintenance"),
    (re.compile(r"\bperioperative\b", re.I),                                  "setting", "perioperative"),

    # ── Line of therapy ──
    (re.compile(
        r"\b(?:previously\s+untreated|treatment[\s-]na(?:ï|i)ve|"
        r"newly\s+diagnosed|first[\s-]line|1l\b)",
        re.I,
    ),                                                                        "line", "first"),
    (re.compile(r"\bsecond[\s-]line\b|\b2l\b", re.I),                         "line", "second"),
    (re.compile(
        r"\bthird[\s-]line\b|\b3rd[\s-]line\b|\blater[\s-]line\b|"
        r"\brefractory\b|\brelapsed\b|\bpreviously\s+treated\b",
        re.I,
    ),                                                                        "line", "later"),

    # ── Demographic segment ──
    (re.compile(r"\bpediatric\b|\bchild(?:ren|hood)?\b", re.I),               "age", "pediatric"),
    (re.compile(r"\belderly\b|\bgeriatric\b", re.I),                          "age", "elderly"),

    # ── Histology: melanoma ──
    (re.compile(r"\bcutaneous\b", re.I),                                      "histology", "cutaneous"),
    (re.compile(r"\buveal\b|\bocular\b", re.I),                               "histology", "uveal"),
    (re.compile(r"\bmucosal\b", re.I),                                        "histology", "mucosal"),
    (re.compile(r"\bacral\b", re.I),                                          "histology", "acral"),

    # ── Histology: lung ──
    (re.compile(r"\bnon[\s-]?small[\s-]?cell\b|\bnsclc\b", re.I),             "histology", "non_small_cell"),
    (re.compile(r"\bsmall[\s-]?cell\b|\bsclc\b", re.I),                       "histology", "small_cell"),
    (re.compile(r"\b(?:lung\s+)?adenocarcinoma\b", re.I),                     "histology", "adenocarcinoma"),
    (re.compile(r"\bsquamous(?:\s+cell)?\b", re.I),                           "histology", "squamous"),
    (re.compile(r"\blarge[\s-]?cell\b", re.I),                                "histology", "large_cell"),

    # ── Histology: breast ──
    (re.compile(r"\btriple[\s-]?negative\b|\btnbc\b", re.I),                  "histology", "triple_negative"),
    (re.compile(r"\bhormone[\s-]?receptor[\s-]?positive\b|\bhr\s*\+\b", re.I),"histology", "hr_positive"),
    (re.compile(r"\bher2[\s-]?positive\b|\bher2\+\b", re.I),                  "histology", "her2_positive"),
    (re.compile(r"\b(?:invasive\s+)?ductal\b", re.I),                         "histology", "ductal"),
    (re.compile(r"\blobular\b", re.I),                                        "histology", "lobular"),
    (re.compile(r"\bin[\s-]?situ\b|\bdcis\b", re.I),                          "histology", "in_situ"),

    # ── Histology: hematologic malignancies ──
    (re.compile(r"\bdiffuse\s+large\s+b[\s-]?cell\b|\bdlbcl\b", re.I),        "histology", "dlbcl"),
    (re.compile(r"\bfollicular\b", re.I),                                     "histology", "follicular"),
    (re.compile(r"\bnon[\s-]?hodgkin\b|\bnhl\b", re.I),                       "histology", "non_hodgkin"),
    (re.compile(r"\bhodgkin\b", re.I),                                        "histology", "hodgkin"),
    (re.compile(r"\bmantle[\s-]?cell\b", re.I),                               "histology", "mantle_cell"),
    (re.compile(r"\bmyeloma\b", re.I),                                        "histology", "myeloma"),

    # ── Histology: autoimmune / neurology ──
    (re.compile(r"\brelapsing[\s-]?remitting\b|\brrms\b", re.I),              "histology", "relapsing_remitting"),
    (re.compile(r"\bprimary[\s-]?progressive\b", re.I),                       "histology", "primary_progressive"),
    (re.compile(r"\bsecondary[\s-]?progressive\b", re.I),                     "histology", "secondary_progressive"),
]


def extract_indication_qualifiers(condition_text: str) -> list[SubgroupFeature]:
    """Parse a CT.gov condition string for population qualifiers.

    Returns a list of ``SubgroupFeature``s — one per axis matched. Each
    axis fires AT MOST ONCE; within an axis, the pattern list is ordered
    most-specific first, so e.g. "Stage IIIC" maps to ``stage=iii`` even
    though "Stage IV" patterns would also match the leading "Stage".

    Used to compose the trial's default PopulationNode id alongside a
    canonical IndicationNode. Biomarker-derived subgroups discovered
    later (BRAF V600E, PD-L1 high) add ADDITIONAL population nodes that
    fork off this default population.
    """
    features: list[SubgroupFeature] = []
    seen_axes: set[str] = set()
    for pattern, axis, level in _QUALIFIER_PATTERNS:
        if axis in seen_axes:
            continue
        if pattern.search(condition_text):
            features.append(SubgroupFeature(
                axis=axis,
                level=level,
                raw_descriptor=condition_text,
            ))
            seen_axes.add(axis)
    return features


# Known same-disease aliases the LLM canonicalizer emits inconsistently.
# Each entry maps a non-preferred slug to its canonical equivalent. Surfaced
# round-over-round by the duplicate-sweep diagnostic; add new entries here
# as they appear in audit/checklist.md #5.
_DISEASE_SLUG_ALIASES: dict[str, str] = {
    # Word-order variants: "Carcinoma, Squamous Cell of Head and Neck" vs
    # "Head and Neck Squamous Cell Carcinoma" — same disease, different
    # CT.gov condition phrasings.
    "squamous_cell_carcinoma_head_and_neck": "head_and_neck_squamous_cell_carcinoma",
}


# Disease hierarchy: subtype IndicationNode id → parent IndicationNode id.
# Used by the populator to add SUBTYPE_OF edges after IndicationNodes are
# created, so cross-indication queries can roll up specific subtypes to
# their parent disease ("show me all melanoma trials" picks up uveal /
# mucosal / cutaneous / etc. as well, not just the bare melanoma node).
#
# Added in round 3.2 (#11) as the scaling-readiness foundation. The map
# is hand-curated for the current oncology corpus; multi-indication
# scaling (#10) extends it as new diseases come in. A MeSH / EFO-backed
# hierarchy is the eventual replacement, but a hand-curated dict is the
# right scope while the corpus is small enough to enumerate.
_INDICATION_HIERARCHY: dict[str, str] = {
    # Melanoma subtypes — all biologically distinct (different driver
    # mutations, different populations, different responses to checkpoint
    # blockade) but all roll up under "melanoma" for cross-subtype
    # queries.
    "uveal_melanoma":                "melanoma",
    "mucosal_melanoma":              "melanoma",
    "intraocular_melanoma":          "melanoma",
    "choroidal_melanoma":            "melanoma",
    "ocular_melanoma":               "melanoma",
    "iris_melanoma":                 "melanoma",
    "cutaneous_melanoma":            "melanoma",
    "acral_melanoma":                "melanoma",
    "acral_lentiginous_melanoma":    "melanoma",
    "mucosal_lentiginous_melanoma":  "melanoma",
    # Future entries (multi-indication scaling):
    # "triple_negative_breast_cancer":   "breast_cancer",
    # "her2_positive_breast_cancer":     "breast_cancer",
    # "non_small_cell_lung_cancer":      "lung_cancer",
    # "small_cell_lung_cancer":          "lung_cancer",
    # "esophageal_squamous_cell_carcinoma":  "esophageal_cancer",
    # ...
}


def parent_indication_for(slug: str) -> str | None:
    """Return the parent IndicationNode id for a subtype slug, or None.

    First checks the hand-curated ``_INDICATION_HIERARCHY``. If that
    misses, falls back to a simple suffix-pattern heuristic for cancers
    that follow ``<qualifier>_<canonical>`` naming
    (``triple_negative_breast_cancer`` → ``breast_cancer``) — only
    fires when the suffix is a known parent disease in the hierarchy
    map's values, so we don't invent novel parents.
    """
    parent = _INDICATION_HIERARCHY.get(slug)
    if parent:
        return parent
    # Heuristic: suffix-match on known parents. E.g.
    # "advanced_melanoma" → "melanoma" if "melanoma" is a known parent.
    known_parents = set(_INDICATION_HIERARCHY.values())
    for candidate in known_parents:
        if slug != candidate and slug.endswith("_" + candidate):
            return candidate
    return None


def slugify_disease_name(raw: str) -> str:
    """Coerce an LLM-emitted disease label into a stable snake_case slug.

    Lowercases, collapses non-alphanumeric runs to single underscores,
    strips leading / trailing underscores, then applies two normalizations:
      1. Strip the trailing 's' on uncountable disease nouns
         (``solid_tumors`` → ``solid_tumor``, ``brain_metastases`` →
         ``brain_metastasis``). Disease names should be singular so
         evidence accumulates on one node per disease.
      2. Resolve known reorderings via ``_DISEASE_SLUG_ALIASES`` —
         CT.gov condition strings sometimes invert word order
         (``squamous_cell_carcinoma_head_and_neck`` vs
         ``head_and_neck_squamous_cell_carcinoma``). The alias table
         picks one canonical form.

    The result is used as the canonical IndicationNode id.
    """
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", raw.strip().lower()).strip("_")
    # Plural → singular for uncountable disease nouns. Order matters:
    # 'metastases' must be caught before the generic '_s' strip.
    plural_rewrites = [
        ("_metastases", "_metastasis"),
        ("_tumors", "_tumor"),
        ("_cancers", "_cancer"),
        ("_carcinomas", "_carcinoma"),
        ("_neoplasms", "_neoplasm"),
        ("_sarcomas", "_sarcoma"),
        ("_lymphomas", "_lymphoma"),
        ("_leukemias", "_leukemia"),
        # -y/-ies rule: covers neuropathy/neuropathies,
        # arthropathy/arthropathies, retinopathy/retinopathies, etc.
        # Disease terms ending in `-pathy` describe the disorder
        # category; the plural is grammatical only.
        ("_neuropathies", "_neuropathy"),
        ("_arthropathies", "_arthropathy"),
        ("_retinopathies", "_retinopathy"),
        ("_myopathies", "_myopathy"),
        ("_nephropathies", "_nephropathy"),
        ("_encephalopathies", "_encephalopathy"),
    ]
    for suffix, replacement in plural_rewrites:
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)] + replacement
            break
    return _DISEASE_SLUG_ALIASES.get(cleaned, cleaned)
