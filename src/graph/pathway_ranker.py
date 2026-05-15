"""Context-aware re-ranking of Reactome pathways for a chain's biology node.

Reactome's default ranking puts the gene's most-cited pathway first, which is
often disease-context-inappropriate: CRBN → "Potential therapeutics for SARS",
VEGFA → "Platelet degranulation", PDPK1 → "GPVI-mediated activation cascade".
These pathways are real for the gene but the wrong biology for the trial's
mechanism + indication, so they false-positive learning when chosen as the
chain's primary biology node.

This module re-ranks the Reactome list by token overlap between each pathway's
display name and the chain context (mechanism, indication, gene symbol),
preserving Reactome's order on ties (stable sort). The full pathway list is
still kept on BiologyNode.pathway_ids — only the *order* changes, so
cross-indication queries against the alternates remain answerable.
"""

from __future__ import annotations

import re
from typing import Iterable

from src.ingestion.lincs import ReactomePathway


# Generic English connectives + a handful of Reactome boilerplate tokens that
# carry no disease/mechanism signal. Kept small on purpose — over-aggressive
# stopwording silently swallows useful matches.
_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the",
    "and", "or",
    "of", "in", "on", "to", "for", "by", "via", "with", "from",
    "is", "are", "be",
    "as",
})

# Minimum token length for substring-aware overlap. Below this the matches are
# almost all coincidental (e.g. "il" in "kinase_inhibition" ↔ "iliac"). Four is
# long enough to keep real gene-name overlaps like vegf↔vegfa working.
_MIN_TOKEN_LEN = 4


# Mechanism-of-action → vocabulary that actually appears in Reactome pathway
# names. The mechanism slug itself ("kinase_inhibition", "protein_degradation")
# rarely matches anything because Reactome describes pathways in terms of the
# specific signaling proteins involved, not the drug-mechanism category. This
# map bridges that gap.
#
# Each entry is a tight set of tokens that are *biologically characteristic*
# of the mechanism — kept small on purpose. Over-broad vocabularies create
# spurious matches (e.g. adding "signaling" everywhere makes every
# receptor-related pathway tie with every kinase pathway). Each token must be
# at least ``_MIN_TOKEN_LEN`` chars or the overlap rule ignores it.
MECHANISM_PATHWAY_TOKENS: dict[str, set[str]] = {
    "checkpoint_blockade": {"checkpoint", "inhibition", "stimulation", "pdcd1", "ctla4", "lag3"},
    "kinase_inhibition": {"kinase", "phosphorylation", "phosphorylates", "akt", "pi3k", "mapk"},
    "receptor_antagonism": {"receptor", "antagonist"},
    "receptor_agonism": {"receptor", "agonist", "interleukin", "cytokine"},
    "enzyme_inhibition": {"enzyme", "catalysis", "metabolism", "hydrolysis"},
    "protein_degradation": {"ubiquitin", "ubiquitination", "proteasome", "cullin", "ligase"},
    "gene_editing": {"homologous", "recombination", "repair"},
    "antibody_dependent_cytotoxicity": {"antibody", "complement", "natural", "killer"},
    "hormone_modulation": {"hormone", "estrogen", "androgen", "steroid"},
    "antimetabolite": {"nucleotide", "nucleoside", "purine", "pyrimidine"},
    "dna_damage": {"damage", "repair", "double-strand", "break"},
    "angiogenesis_inhibition": {"angiogenesis", "vascular", "vasculature"},
    "immune_costimulation": {"costimulation", "interleukin", "tcr", "lymphocyte"},
    "other": set(),
}


def _tokenize(text: str) -> set[str]:
    if not text:
        return set()
    raw = re.split(r"[^a-zA-Z0-9]+", text.lower())
    return {t for t in raw if t and t not in _STOPWORDS}


def _tokens_overlap(a: str, b: str) -> bool:
    """True if a and b are equal or one is a prefix/suffix of the other.

    Boundary-aligned matching is intentional. Pure substring containment
    would false-match short tokens embedded in longer compound words
    ("gene" ⊂ "angiogenesis" → spurious overlap). Prefix/suffix matching
    still catches the gene-name family we care about (vegf prefix of
    vegfa, akt prefix of akt1, jak prefix of jak1) without the noise.
    """
    if a == b:
        return True
    if len(a) > len(b):
        a, b = b, a
    return b.startswith(a) or b.endswith(a)


def _overlap_score(path_tokens: set[str], context_tokens: set[str]) -> int:
    """Count context tokens that overlap any pathway token.

    Each context token contributes at most +1 (we break on first match), so
    a pathway with many overlapping mentions of the same context token
    doesn't dominate one that overlaps with more distinct context tokens.
    """
    score = 0
    for ct in context_tokens:
        if len(ct) < _MIN_TOKEN_LEN:
            continue
        for pt in path_tokens:
            if len(pt) < _MIN_TOKEN_LEN:
                continue
            if _tokens_overlap(ct, pt):
                score += 1
                break
    return score


def _mechanism_expansion(mechanism_name: str) -> set[str]:
    """Look up the canonical Reactome-vocabulary tokens for a mechanism slug.

    Reactome pathway names rarely contain mechanism slugs verbatim (no pathway
    name says "kinase_inhibition"), so we expand the slug into the actual
    protein-level terms Reactome uses (akt, pi3k, mapk, …). Returns an empty
    set for unknown mechanisms — the caller falls back to slug tokenization.
    """
    if not mechanism_name:
        return set()
    return MECHANISM_PATHWAY_TOKENS.get(mechanism_name, set())


def rerank_pathways(
    pathways: Iterable[ReactomePathway],
    *,
    mechanism_name: str = "",
    indication_name: str = "",
    gene_symbol: str = "",
) -> list[ReactomePathway]:
    """Re-rank a list of Reactome pathways by relevance to the chain context.

    The score is the number of context tokens that overlap any pathway token
    via the boundary-aligned overlap rule. Context tokens are drawn from:
      * the mechanism slug itself, tokenized (e.g. ``kinase_inhibition`` →
        {kinase, inhibition})
      * the mechanism's expansion vocabulary from ``MECHANISM_PATHWAY_TOKENS``
        (e.g. ``kinase_inhibition`` → {kinase, phosphorylation, akt, pi3k, mapk})
      * the indication slug
      * the gene symbol

    Higher score → higher rank. Ties preserve input order (stable sort), so
    a context with no signal leaves Reactome's order unchanged.
    """
    items = list(pathways)
    context_tokens = (
        _tokenize(mechanism_name)
        | _mechanism_expansion(mechanism_name)
        | _tokenize(indication_name)
        | _tokenize(gene_symbol)
    )
    if not context_tokens:
        return items

    def key(p: ReactomePathway) -> int:
        return -_overlap_score(_tokenize(p.display_name), context_tokens)

    return sorted(items, key=key)
