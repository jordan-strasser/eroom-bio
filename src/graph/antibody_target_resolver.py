"""Monoclonal antibody target resolver.

Two related functions:

1. **mAb compound → gene symbol** (`mab_target_gene`).
   Used by ``populate._populate_compound_targets`` as a fallback when
   Open Targets returns no protein targets for a recognizable mAb.
   Without this, OT misses (or sparse OT entries) yield UNKNOWN target
   chains for well-characterized antibodies.

2. **Natural-language target → gene symbol** (``nl_target_to_gene``).
   Used by ``eval_holdout_v2._lookup_target`` so the audit and holdout
   eval can map extraction strings like "amyloid beta", "IL-6 receptor",
   or "CD20" to the gene symbol the graph stores. Without this, the
   chain target_id resolves to UNKNOWN even when the AFFECTS edge
   already exists in the graph (solanezumab's symptom — OT had APP,
   but the chain resolver couldn't bridge "amyloid beta" → APP).

The two tables are kept distinct because their failure modes are
different. (1) needs to know "this mAb binds this antigen" — a
biological fact. (2) needs to know "trial extractions colloquially
call the antigen X" — a linguistics fact.
"""

from __future__ import annotations

import re

# ── (1) mAb compound → target gene symbol ──────────────────────────────

# Curated table of well-characterized monoclonal antibodies and the
# gene symbol of their primary target. Used when Open Targets fails to
# resolve a compound the way it does for, e.g., solanezumab in some
# corpora. New entries should be backed by an authoritative source
# (DrugBank, ChEMBL, PubMed). Kept small and explicit rather than
# scraped — biology cost-of-error here is high.
MAB_COMPOUND_TO_TARGET_GENE: dict[str, str] = {
    # PD-(L)1 axis
    "pembrolizumab": "PDCD1",
    "nivolumab": "PDCD1",
    "cemiplimab": "PDCD1",
    "dostarlimab": "PDCD1",
    "atezolizumab": "CD274",
    "durvalumab": "CD274",
    "avelumab": "CD274",
    # CTLA-4
    "ipilimumab": "CTLA4",
    "tremelimumab": "CTLA4",
    # LAG-3, TIGIT
    "relatlimab": "LAG3",
    "tiragolumab": "TIGIT",
    # VEGF / VEGFR
    "bevacizumab": "VEGFA",
    "ramucirumab": "KDR",
    # HER2
    "trastuzumab": "ERBB2",
    "pertuzumab": "ERBB2",
    # EGFR
    "cetuximab": "EGFR",
    "panitumumab": "EGFR",
    "necitumumab": "EGFR",
    # CD20
    "rituximab": "MS4A1",
    "obinutuzumab": "MS4A1",
    "ofatumumab": "MS4A1",
    # Amyloid-β (anti-amyloid mAbs in Alzheimer's)
    "solanezumab": "APP",
    "aducanumab": "APP",
    "lecanemab": "APP",
    "donanemab": "APP",
    "gantenerumab": "APP",
    "bapineuzumab": "APP",
    "crenezumab": "APP",
    # IL-6R, IL-17
    "tocilizumab": "IL6R",
    "sarilumab": "IL6R",
    "secukinumab": "IL17A",
    "ixekizumab": "IL17A",
    # TNF-α
    "adalimumab": "TNF",
    "infliximab": "TNF",
    "golimumab": "TNF",
    # CD38 (myeloma)
    "daratumumab": "CD38",
    "isatuximab": "CD38",
}

# Antibody name suffixes (WHO INN nomenclature). When the compound
# name ends in one of these, it's almost certainly a monoclonal antibody.
_MAB_SUFFIXES: tuple[str, ...] = (
    "mab", "umab", "zumab", "ximab", "omab", "imab",
)

_MAB_RE = re.compile(r"^[a-z0-9_-]+(?:" + "|".join(_MAB_SUFFIXES) + r")$")


def is_monoclonal_antibody(compound_name: str) -> bool:
    """True if the compound name matches the WHO mAb suffix convention.

    Not a perfect classifier — some non-mAb biologics (e.g. fusion
    proteins, BiTEs) also use ``-mab``-like suffixes, and some early
    mAbs predate the convention. But high-precision enough to gate a
    target-lookup fallback.
    """
    if not compound_name:
        return False
    return bool(_MAB_RE.match(compound_name.strip().lower()))


def mab_target_gene(compound_name: str) -> str | None:
    """Curated-table lookup. Returns None for unknown mAbs.

    Caller normalizes the compound name (lowercase + slugify) before
    asking; this function doesn't try to be clever about variant
    spellings beyond the basic lower/strip.
    """
    if not compound_name:
        return None
    return MAB_COMPOUND_TO_TARGET_GENE.get(compound_name.strip().lower())


# ── (2) Natural-language target → gene symbol ──────────────────────────

# Trial extraction `claimed_target` strings often use the colloquial
# antigen name (the thing patients and clinicians say) rather than the
# HGNC gene symbol. Maps the colloquial form to the symbol the graph
# stores on TargetNodes.
#
# Keys MUST be lowercased. Order doesn't matter (no overrides).
NL_TARGET_TO_GENE: dict[str, str] = {
    # Amyloid (Alzheimer's case study + mAb class)
    "amyloid beta": "APP",
    "amyloid-beta": "APP",
    "amyloid-β": "APP",
    "amyloid β": "APP",
    "aβ": "APP",
    "abeta": "APP",
    "amyloid precursor protein": "APP",
    # Tau (next-gen Alzheimer's)
    "tau": "MAPT",
    "tau protein": "MAPT",
    # PD-(L)1 axis (colloquial)
    "pd-1": "PDCD1", "pd1": "PDCD1",
    "pd-l1": "CD274", "pdl1": "CD274", "pd-l2": "PDCD1LG2",
    # CTLA-4
    "ctla-4": "CTLA4", "ctla4": "CTLA4",
    # CD20, CD38, CD19, CD22
    "cd20": "MS4A1",
    "cd38": "CD38",
    "cd19": "CD19",
    "cd22": "CD22",
    # IL receptors
    "il-6r": "IL6R", "il6r": "IL6R", "il-6 receptor": "IL6R",
    "il-17a": "IL17A", "il17a": "IL17A",
    "il-23": "IL23A", "il23": "IL23A",
    # TNF
    "tnf": "TNF", "tnf-alpha": "TNF", "tnf-α": "TNF", "tnfα": "TNF",
    # VEGF / VEGFR
    "vegf": "VEGFA", "vegfa": "VEGFA", "vegf-a": "VEGFA",
    "vegfr2": "KDR", "vegfr-2": "KDR", "vegfr": "KDR",
    # HER2, EGFR
    "her2": "ERBB2", "her-2": "ERBB2", "erbb2": "ERBB2",
    "egfr": "EGFR", "egf receptor": "EGFR",
    # CETP (torcetrapib + siblings)
    "cetp": "CETP", "cholesteryl ester transfer protein": "CETP",
    # MEK / ERK / KIT (selumetinib + siblings)
    "mek": "MAP2K1", "mek1": "MAP2K1", "mek2": "MAP2K2",
    "mek1/2": "MAP2K1", "mek 1/2": "MAP2K1",
    "erk": "MAPK1", "erk1": "MAPK3", "erk2": "MAPK1",
    "kit": "KIT", "c-kit": "KIT",
    # Other targets that recur in melanoma corpus
    "csf1r": "CSF1R", "fms": "CSF1R",
    "liv-1": "SLC39A6", "liv1": "SLC39A6",
    "ny-eso-1": "CTAG1B",
    "gp100": "PMEL",
    "ido": "IDO1", "ido1": "IDO1",
    "lag-3": "LAG3", "lag3": "LAG3",
    "tim-3": "HAVCR2", "tim3": "HAVCR2",
    "tigit": "TIGIT",
    "braf": "BRAF",
    "hmg-coa reductase": "HMGCR", "hmgcr": "HMGCR",
}


def nl_target_to_gene(target_str: str) -> str | None:
    """Map a natural-language target string to the canonical gene symbol.

    Case-insensitive. Whitespace-normalized. Returns None when the
    string is empty or unrecognized — caller should fall back to its
    usual gene-symbol lookup.
    """
    if not target_str:
        return None
    key = " ".join(target_str.strip().lower().split())
    return NL_TARGET_TO_GENE.get(key)
