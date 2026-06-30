"""Principled Beta-Binomial belief updates from evidence records.

The Beta(α, β) belief on each graph edge is updated by *virtual evidence*
in the conjugate sense: each evidence record contributes ``N_eff``
virtual Bernoulli draws with empirical success rate ``p_obs``. The
posterior parameters are::

    α_post = α_prior + N_eff · p_obs
    β_post = β_prior + N_eff · (1 - p_obs)

Two tables drive the update—and they are the only two knobs in the
system, which makes the update both **principled** (real conjugate
update with a clear generative interpretation: α-1 and β-1 count
effective virtual successes and failures) and **repeatable** (deterministic
function of the evidence type and the categorical bucket).

`EVIDENCE_TYPE_N_EFF`—how many virtual trials a single piece of
evidence of each class is worth. Calibrated against per-class historical
replication / predictive value; defaults below mirror the prior weight
ordering (clinical > genetic > preclinical > in vitro) but now carry
explicit "effective sample size" semantics rather than ad-hoc multipliers.

`BUCKET_TO_P_OBS`—the LLM picks one of seven discrete buckets per
evidence record; the bucket maps to a fixed probability. Free-floating
0–1 confidence floats from an LLM are notoriously miscalibrated; bucketed
emissions are far more repeatable, and the seven values can be empirically
recalibrated with a small held-out set (calibration is a pending follow-up).

Both tables are deliberately exported as plain dicts so callers can swap
them in tests or after calibration without subclassing.
"""

from __future__ import annotations

from enum import Enum

from src.config import CONFIG
from src.graph.models import (
    EdgeBeliefState,
    EvidenceDirection,
    EvidenceType,
)


class SupportBucket(str, Enum):
    """Categorical strength-and-direction of one piece of evidence.

    Symmetric around AMBIGUOUS. The classifier picks exactly one bucket
    per (edge, evidence) pair using the rubric in the classification
    prompt—never a free-form float.
    """
    STRONG_SUPPORT = "strong_support"
    MODERATE_SUPPORT = "moderate_support"
    WEAK_SUPPORT = "weak_support"
    AMBIGUOUS = "ambiguous"
    WEAK_CONTRADICT = "weak_contradict"
    MODERATE_CONTRADICT = "moderate_contradict"
    STRONG_CONTRADICT = "strong_contradict"


# Categorical → probability of "the edge is true" implied by one virtual
# observation in this bucket. Symmetric around AMBIGUOUS=0.5. The 0.05/0.95
# floors at the extremes prevent any single piece of evidence from driving
# the posterior to logical certainty, even with large N_eff.
BUCKET_TO_P_OBS: dict[SupportBucket, float] = {
    SupportBucket.STRONG_SUPPORT:      0.95,
    SupportBucket.MODERATE_SUPPORT:    0.80,
    SupportBucket.WEAK_SUPPORT:        0.65,
    SupportBucket.AMBIGUOUS:           0.50,
    SupportBucket.WEAK_CONTRADICT:     0.35,
    SupportBucket.MODERATE_CONTRADICT: 0.20,
    SupportBucket.STRONG_CONTRADICT:   0.05,
}


# Effective sample size per evidence class. These are pseudocount
# contributions to (α + β) per record—i.e. how many virtual Bernoulli
# trials this evidence is worth. Larger = stronger shrinkage of the
# posterior mean and faster CI tightening.
#
# Defensible relative scaling pre-calibration:
#   - Phase 3 ≈ 15× in vitro: registrational RCTs (N≥500, randomized,
#     blinded) carry orders of magnitude more information than a single
#     LINCS knockout in one cell line. With three Phase 3s ≈ 45 virtual
#     trials, clinical evidence appropriately dominates a typical
#     LINCS bundle (~10 cell-line hits per mechanism).
#   - Genetic MR ≈ 10×: well-powered MR with strong instruments
#     approaches Phase-2-trial-equivalent causal evidence (uses
#     Mendelian randomization to identify causal effects, replicates
#     better than GWAS).
#   - GWAS ≈ 4×: ~50% replication rate in independent cohorts, no
#     direct causal claim.
#   - Phase 1 ≈ 2×: dosed primarily for safety/PK, not efficacy —
#     drops below Phase 2 despite being clinical.
#   - Literature/computational at <1×: nominal pseudocounts; should
#     not move beliefs much absent corroborating evidence.
#
# These are pre-calibration defaults—once a labeled holdout of
# ≥50 trials exists, refit against the labeled set to minimize Brier.
EVIDENCE_TYPE_N_EFF: dict[EvidenceType, float] = {
    EvidenceType.CLINICAL_PHASE3:    15.0,
    EvidenceType.CLINICAL_PHASE2:     6.0,
    EvidenceType.CLINICAL_PHASE1:     2.0,
    EvidenceType.GENETIC_MR:         10.0,
    EvidenceType.GENETIC_GWAS:        4.0,
    EvidenceType.PRECLINICAL_IN_VIVO: 2.0,
    EvidenceType.PRECLINICAL_IN_VITRO: 1.0,
    # ── Per-source curated database n_eff values ───────────────────────
    #
    # Picked from source character (curation depth, replication, primary
    # vs aggregate), NOT tuned against the 5-trial holdout audit. Doing
    # the latter on a 5-trial set would be hyperparameter-overfitting —
    # the holdout result would then be a function of our tuning, not a
    # measurement of cross-trial learning.
    #
    # Defensible reasoning per tier:
    #
    # OT-direct → 12.0  (round-28 bump from 3.0)
    # ChEMBL-direct, mAb-table → 10.0  (round-28 bump from 3.0)
    #   Multi-source curated assertions about a SPECIFIC compound-target
    #   binding pair. These are molecular facts — "this antibody binds
    #   this antigen", "this small molecule occupies this kinase's ATP
    #   pocket" — not probabilistic claims about a clinical outcome.
    #   The round-27 forensic audit found that a single OT-direct record
    #   at n_eff=3 was being overwhelmed by ~3 Phase-3 trials at
    #   n_eff=15 each, all classified AMBIGUOUS because trials assume
    #   binding rather than demonstrate it. The AMBIGUOUS pseudocounts
    #   dragged the AFFECTS posterior from molecular near-certainty
    #   toward 0.5. Promoting curated binding records to a tier that
    #   rivals one Phase-3 trial reflects what they actually represent
    #   epistemically: cross-checked molecular biology, not noisy
    #   clinical signal. OT-direct edges multi-source: ChEMBL + IUPHAR +
    #   DGIdb + drug-label curation, so gets a small edge over ChEMBL
    #   or hand-curated mAb tables alone.
    EvidenceType.DATABASE_OT_DIRECT:          12.0,
    EvidenceType.DATABASE_CHEMBL:             10.0,
    EvidenceType.DATABASE_MAB_TABLE:          10.0,
    #
    # OT-association score, endpoint-class prior → 2.0
    #   Aggregate score COMBINING multiple evidence types via a heuristic
    #   weighting. Each underlying source is real evidence but the score
    #   itself is interpretive. Endpoint-class priors are similarly an
    #   FDA / ICH consensus that an endpoint captures disease but applied
    #   broadly. Comparable to preclinical in vivo.
    EvidenceType.DATABASE_OT_ASSOCIATION:      2.0,
    EvidenceType.DATABASE_ENDPOINT_PRIOR:      2.0,
    #
    # Reactome / GO pathway annotation → 1.5
    #   Curated by pathway-database experts, but a single curator's
    #   interpretive call per entry. Between in-vitro (1.0) and in-vivo
    #   (2.0) in evidential weight.
    EvidenceType.DATABASE_REACTOME_GO:         1.5,
    #
    # LINCS L1000 perturbation signature → 1.0
    #   Real in-vitro experiment; same tier as PRECLINICAL_IN_VITRO,
    #   which is exactly what it is.
    EvidenceType.DATABASE_LINCS:               1.0,
    #
    # Indication-taxonomy structural → 1.0
    #   Structural roll-up ("metastatic melanoma" → "melanoma"), not
    #   measurement. One observation's worth of weight.
    EvidenceType.DATABASE_INDICATION_TAXONOMY: 1.0,
    #
    # Synthesized fallback (trial_biology_fallback, combo_inherit) → 0.5
    #   Derived from existing curated facts by an inferential step
    #   ("compound A inherits target T from constituent X"). Halve the
    #   evidence since we're double-counting partially.
    EvidenceType.DATABASE_FALLBACK:            0.5,
    #
    # Name-match cross-reference → 0.3
    #   Gene symbol found in intervention free text. Not curation, just
    #   string overlap. Same as COMPUTATIONAL.
    EvidenceType.DATABASE_CROSS_REFERENCE:     0.3,
    #
    # LLM-inferred drug→target gene → 3.0
    #   For compounds OT/ChEMBL/mAb couldn't resolve. The gene is validated
    #   to a real Ensembl id (OT search_target) before use, and for known
    #   drugs the target is usually correct, so this earns more than a raw
    #   name-match — but the binding is INFERRED, not curated, so it sits an
    #   order of magnitude below OT-direct (12) / ChEMBL (10). Calibrated by
    #   source character, not the holdout. See /tuning-log.
    EvidenceType.DATABASE_LLM_INFERENCE:       3.0,
    # ────────────────────────────────────────────────────────────────────
    EvidenceType.COMPUTATIONAL:       0.3,
    EvidenceType.LITERATURE:          0.2,
}


def p_obs_for_bucket(bucket: SupportBucket) -> float:
    return BUCKET_TO_P_OBS[bucket]


# ── Precision-aware N_eff (formerly the EROOM_NEFF_PRECISION flag) ─────────
#
# The flag is GONE and the path that folded precision into
# ``effective_n_for_evidence`` was REMOVED — that function now always returns
# the legacy ``base × quality_score`` (its ``n_obs`` / ``edge_type`` params are
# vestigial), so existing builds, snapshots, and the holdout stay byte-for-byte
# unchanged. What SURVIVES and is baked in is ``_precision_multiplier`` below,
# which the attributor's outcome path calls DIRECTLY and unconditionally
# (belief formation is always √N-weighted): a clinical record scales by the
# *precision* of the reported result, anchored so a median-N trial reproduces
# its legacy type-constant.
#
# All values are pre-calibration defaults; a future calibration pass is
# meant to refit them against held-out outcomes (Brier / ECE).
_N_REF_ANCHOR = 350.0        # reference patient N (≈ corpus median enrollment 353); mult == 1 here
_PRECISION_EXPONENT = 0.5    # concave in N (sqrt): 4x patients -> 2x weight
_PRECISION_MULT_FLOOR = 0.5  # a small trial is still worth >= half the anchor
_PRECISION_MULT_CEIL = 2.5   # a huge trial caps at 2.5x (trust saturates anyway)

def _precision_multiplier(n_obs: int | None) -> float:
    """Anchored, concave precision weight from a patient/observation count.

    Returns 1.0 when N is unknown, so a record with no count reproduces its
    anchor (the legacy type-constant). A trial at ``_N_REF_ANCHOR`` returns
    1.0; larger N scales up concavely, smaller N down, clamped to keep any
    single record from dominating or vanishing.
    """
    if not n_obs or n_obs <= 0:
        return 1.0
    raw = (n_obs / _N_REF_ANCHOR) ** _PRECISION_EXPONENT
    return max(_PRECISION_MULT_FLOOR, min(_PRECISION_MULT_CEIL, raw))


def effective_n_for_evidence(
    source_type: EvidenceType,
    quality_score: float = 1.0,
    *,
    n_obs: int | None = None,
    edge_type: str | None = None,
) -> float:
    """Effective virtual sample size for one evidence record:
    ``EVIDENCE_TYPE_N_EFF[source_type] * quality_score`` — the type-constant
    weighting, where ``quality_score`` ∈ [0, 1] folds in the LLM classifier's
    self-reported confidence (1.0 for non-LLM streams like LINCS / GWAS that
    have no classification step).

    ``n_obs`` / ``edge_type`` were inputs to the removed precision-aware n_eff
    path (the EROOM_NEFF_PRECISION flag); they are kept on the signature for
    call-site stability but no longer affect the result.
    """
    if not 0.0 <= quality_score <= 1.0:
        raise ValueError(
            f"quality_score must be in [0, 1], got {quality_score!r}"
        )
    return EVIDENCE_TYPE_N_EFF[source_type] * quality_score


def redundancy_factor(prior_same_cluster: int) -> float:
    """No-op (returns 1.0). The clustered-evidence redundancy discount was part
    of the removed precision-aware n_eff path (EROOM_NEFF_PRECISION); the baked
    behavior applies no discount. Retained as a 1.0-returning stub because the
    store's belief update calls it on the hot path — kept behavior-identical."""
    return 1.0


def applied_weights(record) -> tuple[float, float]:
    """The EXACT ``(n_eff, p_obs)`` this record contributed to its edge's Beta — the
    faithful replay weight for ANY re-derivation: the (s,t) field materializer, LOO
    self-exclusion, AND the node-MERGE belief replay.

    Prefers the persisted ``applied_n_eff`` / ``applied_p_obs`` (which capture the
    attributor's explaining-away split and any redundancy discount applied at
    attribution time); falls back to the nominal ``effective_n_for_evidence`` /
    bucket ``p_obs`` for legacy records written before persistence. Recomputing
    nominal weights in a replay is the Bug-B family of defects — it re-applies the
    full contradiction the explaining-away split away (over-penalizing high-belief
    edges) and, in the merge, re-counts replicated database facts at full weight.
    """
    p_obs = (
        record.applied_p_obs if getattr(record, "applied_p_obs", None) is not None
        else p_obs_for_bucket(SupportBucket(record.support))
    )
    n_eff = (
        record.applied_n_eff if getattr(record, "applied_n_eff", None) is not None
        else effective_n_for_evidence(
            record.source_type, record.quality_score,
            n_obs=getattr(record, "n_obs", None),
        )
    )
    return n_eff, p_obs


def bucket_to_direction(bucket: SupportBucket) -> EvidenceDirection:
    """Coarse projection of a bucket onto the legacy 3-way direction.

    Used for filtering, taxonomy cross-checks, and display. The bucket
    remains the source of truth for the conjugate update.
    """
    if bucket in (
        SupportBucket.STRONG_SUPPORT,
        SupportBucket.MODERATE_SUPPORT,
        SupportBucket.WEAK_SUPPORT,
    ):
        return EvidenceDirection.SUPPORTING
    if bucket in (
        SupportBucket.WEAK_CONTRADICT,
        SupportBucket.MODERATE_CONTRADICT,
        SupportBucket.STRONG_CONTRADICT,
    ):
        return EvidenceDirection.CONTRADICTING
    return EvidenceDirection.AMBIGUOUS


def flip_bucket(bucket: SupportBucket) -> SupportBucket:
    """Return the bucket symmetric across AMBIGUOUS.

    Used by the attributor when the taxonomy rule disagrees with the
    classifier's direction: instead of silently dropping to AMBIGUOUS
    (which discards strength information), we can opt to flip—though
    the current attributor downgrades to AMBIGUOUS to be conservative.
    Provided here for completeness.
    """
    return {
        SupportBucket.STRONG_SUPPORT:      SupportBucket.STRONG_CONTRADICT,
        SupportBucket.MODERATE_SUPPORT:    SupportBucket.MODERATE_CONTRADICT,
        SupportBucket.WEAK_SUPPORT:        SupportBucket.WEAK_CONTRADICT,
        SupportBucket.AMBIGUOUS:           SupportBucket.AMBIGUOUS,
        SupportBucket.WEAK_CONTRADICT:     SupportBucket.WEAK_SUPPORT,
        SupportBucket.MODERATE_CONTRADICT: SupportBucket.MODERATE_SUPPORT,
        SupportBucket.STRONG_CONTRADICT:   SupportBucket.STRONG_SUPPORT,
    }[bucket]


def modulation_bucket(
    direction: str,
    confidence: float,
) -> SupportBucket:
    """Map LLM-emitted (direction, confidence) to a SupportBucket.

    The v0.3.0 modulation classifier emits each modulation as a
    ``direction`` (``amplifies`` | ``suppresses`` | ``neutral``) plus a
    numeric ``confidence`` in [0, 1]. Direction picks the side
    (support vs contradict); confidence picks the magnitude. Below the
    ``AMBIGUOUS`` floor any modulation collapses to AMBIGUOUS — at that
    point the LLM is unsure enough that we don't want it driving an
    edge belief either way.

    ``neutral`` ALWAYS maps to AMBIGUOUS — at any confidence. A trial
    that didn't demonstrate amplification has many possible explanations
    beyond "the modulator does nothing" (wrong dose, wrong population,
    underpowered, AE-driven discontinuation, endpoint didn't capture the
    modulation's effect, …). So "neutral" is genuinely "no information
    about the modulation edge," not "evidence the modulation is false."

    A *confident* neutral still does work, though — at AMBIGUOUS p_obs=0.5
    the Beta-Binomial update shrinks the posterior toward 0.5 with weight
    proportional to ``n_eff`` (Phase 3 = 15, etc.). So confident neutral
    on a Phase 3 failed trial encodes "strong evidence we don't know"
    rather than "we have no signal at all" — exactly the right epistemic
    behavior given the failure-mode confounders above.

    Thresholds:
      - ``< 0.55`` confidence → AMBIGUOUS regardless of direction
      - ``≥ 0.85`` + amplifies → STRONG_SUPPORT
      - ``≥ 0.70`` + amplifies → MODERATE_SUPPORT
      - ``≥ 0.55`` + amplifies → WEAK_SUPPORT
      - ``≥ 0.85`` + suppresses → STRONG_CONTRADICT
      - ``≥ 0.70`` + suppresses → MODERATE_CONTRADICT
      - ``≥ 0.55`` + suppresses → WEAK_CONTRADICT
      - neutral (any confidence ≥ 0.55) → AMBIGUOUS

    Effective sample size for modulation edges is set by the trial's
    evidence type (Phase 3 = 15, Phase 2 = 6, etc.) just like every
    other LLM-attributed edge — the LLM doesn't estimate N_eff.
    """
    if direction == "neutral" or confidence < 0.55:
        return SupportBucket.AMBIGUOUS
    if direction == "amplifies":
        if confidence >= 0.85:
            return SupportBucket.STRONG_SUPPORT
        if confidence >= 0.70:
            return SupportBucket.MODERATE_SUPPORT
        return SupportBucket.WEAK_SUPPORT
    if direction == "suppresses":
        if confidence >= 0.85:
            return SupportBucket.STRONG_CONTRADICT
        if confidence >= 0.70:
            return SupportBucket.MODERATE_CONTRADICT
        return SupportBucket.WEAK_CONTRADICT
    # Unknown direction string → conservative AMBIGUOUS rather than crash.
    return SupportBucket.AMBIGUOUS


# ── Hierarchical partial pooling (cross-indication / cross-population backoff) ──
#
# A specific (leaf) edge belief borrows strength from its coarser ancestor: the
# ancestor acts as a PRIOR whose concentration is capped at ``prior_strength``
# virtual observations, preserving its mean. So a sparse leaf is dominated by a
# rich-but-capped parent (the backoff borrows cross-trial / cross-INDICATION
# evidence — the literal north-star: a mechanism learned in melanoma informs
# uveal_melanoma), while a leaf that amasses ≳ prior_strength of its OWN evidence
# overrides the parent (specificity is reclaimed once earned). The cap encodes
# that a broader population is only a PROXY for the specific edge — its transfer
# credit saturates regardless of how many trials it pooled, so one HER2+ trial
# can't be steamrolled forever by a 94-strength breast_cancer parent, and a
# 16-strength rheumatoid_arthritis leaf isn't dragged back to a weak arthritis
# parent.
#
# Hierarchy levels are DISJOINT in evidence (each trial is attributed to its leaf
# indication / own population slug; SUBTYPE_OF / axis-subset parents exist for
# structure, with roll-up deferred to here), so summing evidence across levels
# does not double-count. This is fixed-concentration hierarchical Beta-Binomial
# partial pooling.
#
# ``prior_strength`` (τ) defaults to 20 — ≈ the p90 of observed n=500 parent-edge
# strengths, so the cap bites only the richest proxies, not the typical one. It
# is calibrated by the evidence-strength SCALE, NOT tuned against any holdout
# AUROC. BAKED via ``CONFIG.pool_prior_strength`` (src/config.py); formerly the
# env-overridable ``EROOM_POOL_PRIOR_STRENGTH``. See /tuning-log.
_POOL_PRIOR_STRENGTH = CONFIG.pool_prior_strength


def _pool_prior_strength() -> float:
    return _POOL_PRIOR_STRENGTH


def _cap_concentration(
    alpha: float, beta: float, cap: float
) -> tuple[float, float]:
    """Scale ``(alpha, beta)`` down to total concentration ``cap`` (preserving the
    mean) when it exceeds ``cap``; otherwise return it unchanged. Bounds a parent
    belief's influence when it stands in as a prior for a finer level."""
    conc = alpha + beta
    if conc <= cap or conc <= 0.0:
        return alpha, beta
    scale = cap / conc
    return alpha * scale, beta * scale


def pool_hierarchical(
    beliefs_specific_to_general: list[EdgeBeliefState],
    *,
    prior_strength: float | None = None,
) -> EdgeBeliefState | None:
    """Partial-pool an edge belief with its coarser-ancestor beliefs.

    ``beliefs_specific_to_general`` lists the EVIDENCED hierarchy levels
    most-specific first (the order ``_indication_ancestors`` /
    ``_population_ancestors`` yield), filtered to those carrying evidence. The
    coarsest level seeds the running pool; each finer level then updates a
    concentration-capped copy of that pool with its OWN evidence mass (``α−1``,
    ``β−1`` beyond the Beta(1,1) base).

    The returned belief carries the MOST-SPECIFIC (leaf) level's ``evidence``
    records, not the union. This is exactly what makes self-exclusion faithful:
    the leaf enters the pool UNCAPPED, so the held-out trial's leaf records are
    linearly removable by ``provenance._belief_excluding``'s delta-adjust, while
    the (capped) ancestor mass is a constant prior offset. When the held-out
    trial is the leaf's only evidence, self-excluding it leaves that capped
    ancestor prior — so the honest holdout MEASURES cross-indication transfer
    (does the parent's evidence predict the leaf trial?) rather than dropping the
    edge. Single-level pooling is therefore an exact no-op (same α, β, evidence,
    field as the input). Returns ``None`` for an empty input.
    """
    if not beliefs_specific_to_general:
        return None
    leaf = beliefs_specific_to_general[0]
    cap = _pool_prior_strength() if prior_strength is None else prior_strength
    levels = list(reversed(beliefs_specific_to_general))  # general → specific
    alpha, beta = levels[0].alpha, levels[0].beta
    for belief in levels[1:]:
        prior_a, prior_b = _cap_concentration(alpha, beta, cap)
        own_a = max(0.0, belief.alpha - 1.0)
        own_b = max(0.0, belief.beta - 1.0)
        alpha, beta = prior_a + own_a, prior_b + own_b
    return EdgeBeliefState(
        alpha=alpha, beta=beta,
        evidence=leaf.evidence, belief_field=leaf.belief_field,
    )


def apply_virtual_evidence(
    belief: EdgeBeliefState,
    *,
    n_eff: float,
    p_obs: float,
) -> EdgeBeliefState:
    """Beta-Binomial conjugate update with one virtual-evidence record.

    Returns a new EdgeBeliefState; does not mutate the input. The
    evidence list on the input is preserved (callers append the
    triggering EvidenceRecord separately so the update is replayable
    from raw evidence).
    """
    if n_eff < 0:
        raise ValueError(f"n_eff must be non-negative, got {n_eff!r}")
    if not 0.0 <= p_obs <= 1.0:
        raise ValueError(f"p_obs must be in [0, 1], got {p_obs!r}")
    return belief.model_copy(update={
        "alpha": belief.alpha + n_eff * p_obs,
        "beta":  belief.beta  + n_eff * (1.0 - p_obs),
    })
