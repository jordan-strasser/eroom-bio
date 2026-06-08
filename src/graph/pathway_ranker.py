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

import math
import re
from typing import Callable, Iterable, Protocol, TypeVar


class BiologyCandidate(Protocol):
    """Structural type for anything the ranker can score.

    Both ``src.ingestion.lincs.ReactomePathway`` and
    ``src.ingestion.quickgo.GOTerm`` satisfy this protocol — they each
    carry a stable id and a human-readable display name. The ranker
    works on any iterable of objects with these two fields, so the same
    scoring rule applies whether we're picking among Reactome pathways
    or GO biological-process terms.
    """
    stable_id: str
    display_name: str


_T = TypeVar("_T", bound=BiologyCandidate)


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
    # Taxanes / vinca alkaloids. The target is the microtubule/tubulin, whose
    # gene (TUBB*) annotates in GO to many off-mechanism processes (sperm
    # motility, NK cytotoxicity). These tokens let the on-mechanism terms
    # ("mitotic cell cycle", "microtubule-based process") score > 0 so the
    # relevance floor keeps them even when the extracted description is sparse.
    "microtubule_binding": {"microtubule", "mitotic", "spindle", "tubulin", "kinetochore"},
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


def _raw_overlap_count(path_tokens: set[str], context_tokens: set[str]) -> int:
    """Count context tokens that overlap any pathway token (no length norm).

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


def _overlap_score(path_tokens: set[str], context_tokens: set[str]) -> float:
    """Pathway-name-length-normalized overlap score.

    Returns ``matches / len(path_tokens)`` — the fraction of the pathway's
    own name that is on-topic for the chain context. Verbose specific names
    (e.g. GO:0038033 "positive regulation of endothelial cell chemotaxis by
    VEGF-activated vascular endothelial growth factor receptor signaling
    pathway", 14 tokens) get penalized relative to short canonical names
    (e.g. R-HSA-194138 "Signaling by VEGF", 2 tokens). Without this, longer
    names win simply by giving more tokens chances to overlap, which biases
    selection toward over-specific biology and fragments evidence pooling.

    Returns 0.0 when the pathway has no scoreable tokens after filtering.
    """
    if not path_tokens:
        return 0.0
    scoreable = [t for t in path_tokens if len(t) >= _MIN_TOKEN_LEN]
    if not scoreable:
        return 0.0
    matches = _raw_overlap_count(path_tokens, context_tokens)
    return matches / len(scoreable)


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


def _context_tokens(
    mechanism_name: str, indication_name: str, gene_symbol: str
) -> set[str]:
    return (
        _tokenize(mechanism_name)
        | _mechanism_expansion(mechanism_name)
        | _tokenize(indication_name)
        | _tokenize(gene_symbol)
    )


def score_candidate(
    candidate: BiologyCandidate,
    *,
    mechanism_name: str = "",
    indication_name: str = "",
    gene_symbol: str = "",
) -> float:
    """Score a single biology candidate against the chain context.

    Returns the pathway-name-length-normalized overlap score (matches /
    name_tokens, a fraction in [0, 1]). 0.0 means no context signal at all;
    higher values mean a larger fraction of the pathway's own name is
    on-topic for the chain context. Used by the populator to decide whether
    a GO term outscores Reactome's best for the same gene.
    """
    context = _context_tokens(mechanism_name, indication_name, gene_symbol)
    if not context:
        return 0.0
    return _overlap_score(_tokenize(candidate.display_name), context)


def rerank_pathways(
    pathways: Iterable[_T],
    *,
    mechanism_name: str = "",
    indication_name: str = "",
    gene_symbol: str = "",
) -> list[_T]:
    """Re-rank a list of biology candidates by relevance to the chain context.

    Accepts any iterable of objects satisfying ``BiologyCandidate`` (a stable
    id + a display name). The score is pathway-name-length-normalized: the
    fraction of the pathway's own name tokens that overlap any context token
    via the boundary-aligned overlap rule. Normalization keeps short canonical
    names (e.g. "Signaling by VEGF") competitive against verbose specific
    names (e.g. "positive regulation of endothelial cell chemotaxis by ...")
    that would otherwise win simply by having more tokens to roll the dice on.
    The downstream consequence is that biology nodes tend toward general,
    re-usable terms — better evidence pooling across trials.

    Context tokens are drawn from:
      * the mechanism slug itself, tokenized (e.g. ``kinase_inhibition`` →
        {kinase, inhibition})
      * the mechanism's expansion vocabulary from ``MECHANISM_PATHWAY_TOKENS``
        (e.g. ``kinase_inhibition`` → {kinase, phosphorylation, akt, pi3k, mapk})
      * the indication slug
      * the gene symbol

    Higher score → higher rank. Ties preserve input order (stable sort), so
    a context with no signal leaves the original order unchanged.
    """
    items = list(pathways)
    context = _context_tokens(mechanism_name, indication_name, gene_symbol)
    if not context:
        return items

    def key(p: _T) -> float:
        return -_overlap_score(_tokenize(p.display_name), context)

    return sorted(items, key=key)


def relevance_floor(
    candidates: Iterable[_T],
    *,
    mechanism_name: str = "",
    indication_name: str = "",
    gene_symbol: str = "",
) -> list[_T]:
    """Re-rank, then keep only candidates whose context-overlap score is
    strictly positive; if none clear the floor, keep the single top-ranked
    candidate (never returns empty unless given nothing).

    The mechanism fan-out keeps *every* term a gene maps to when there are
    fewer than the cap — so a tubulin gene's full GO biological-process set
    (TUBB4B → "mitotic cell cycle" AND "flagellated sperm motility") all
    become MechanismNodes, the off-context leaves included. Flooring at
    score > 0 drops the leaves that share no token with the chain context
    while keeping the on-mechanism ones; the top-1 fallback guarantees the
    chain still gets a mechanism. Intended for the GO-augmentation path,
    where the candidate set is a gene's raw annotation list rather than
    Reactome's curated signaling-pathway names.
    """
    ranked = rerank_pathways(
        candidates,
        mechanism_name=mechanism_name,
        indication_name=indication_name,
        gene_symbol=gene_symbol,
    )
    if not ranked:
        return []
    kept = [
        c
        for c in ranked
        if score_candidate(
            c,
            mechanism_name=mechanism_name,
            indication_name=indication_name,
            gene_symbol=gene_symbol,
        )
        > 0.0
    ]
    return kept or ranked[:1]


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


# Keep GO candidates scoring within this fraction of the top candidate's
# semantic similarity. Tuned against the BioLORD validation (taxane/CRBN/SMO):
# the on-mechanism terms score ~2-3x the off-context leaves, so a 0.55 margin
# keeps the gene's relevant pathway footprint while dropping the developmental/
# reproductive leaves. A knob; see /tuning-log.
_SEMANTIC_FLOOR_REL = 0.55


def semantic_relevance_floor(
    candidates: Iterable[_T],
    context_text: str,
    embed_fn: Callable[[list[str]], list[list[float]]],
    *,
    min_relative: float = _SEMANTIC_FLOOR_REL,
) -> list[_T]:
    """Rank candidates by BioLORD semantic similarity to ``context_text`` and
    keep those scoring within ``min_relative`` of the top; top-1 fallback.

    This is the scalable, root-cause replacement for the token-overlap
    ``relevance_floor`` on the GO-augmentation path: a gene's GO biological-
    process set is full of real-but-off-mechanism leaves (TUBB4B →
    "flagellated sperm motility", CRBN → "limb development") that lexical
    overlap can't reliably reject (e.g. "natural killer CELL …" sneaks past a
    "cell cycle" context). Semantic embeddings separate them cleanly and need
    no hand-maintained mechanism vocabulary, so the prune generalizes to any
    gene/mechanism as the corpus scales.

    ``embed_fn`` maps a list of strings to their embeddings (the context text
    is prepended); injected so this stays pure and unit-testable, and so the
    caller controls the on-disk cache. Falls back to keeping all candidates
    when ``context_text`` is empty (no signal to floor against).
    """
    items = list(candidates)
    if len(items) <= 1 or not context_text.strip():
        return items
    names = [c.display_name for c in items]
    vecs = embed_fn([context_text] + names)
    ctx_vec, name_vecs = vecs[0], vecs[1:]
    scored = sorted(
        ((_cosine(nv, ctx_vec), c) for nv, c in zip(name_vecs, items)),
        key=lambda sc: sc[0],
        reverse=True,
    )
    top = scored[0][0]
    if top <= 0.0:
        return [scored[0][1]]
    kept = [c for s, c in scored if s >= min_relative * top]
    return kept or [scored[0][1]]
