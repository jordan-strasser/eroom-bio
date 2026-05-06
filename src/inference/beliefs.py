"""Principled Beta-Binomial belief updates from evidence records.

The Beta(α, β) belief on each graph edge is updated by *virtual evidence*
in the conjugate sense: each evidence record contributes ``N_eff``
virtual Bernoulli draws with empirical success rate ``p_obs``. The
posterior parameters are::

    α_post = α_prior + N_eff · p_obs
    β_post = β_prior + N_eff · (1 - p_obs)

Two tables drive the update — and they are the only two knobs in the
system, which makes the update both **principled** (real conjugate
update with a clear generative interpretation: α-1 and β-1 count
effective virtual successes and failures) and **repeatable** (deterministic
function of the evidence type and the categorical bucket).

`EVIDENCE_TYPE_N_EFF` — how many virtual trials a single piece of
evidence of each class is worth. Calibrated against per-class historical
replication / predictive value; defaults below mirror the prior weight
ordering (clinical > genetic > preclinical > in vitro) but now carry
explicit "effective sample size" semantics rather than ad-hoc multipliers.

`BUCKET_TO_P_OBS` — the LLM picks one of seven discrete buckets per
evidence record; the bucket maps to a fixed probability. Free-floating
0–1 confidence floats from an LLM are notoriously miscalibrated; bucketed
emissions are far more repeatable, and the seven values can be empirically
recalibrated with a small held-out set (see ``calibration.py``).

Both tables are deliberately exported as plain dicts so callers can swap
them in tests or after calibration without subclassing.
"""

from __future__ import annotations

from enum import Enum

from src.graph.models import (
    EdgeBeliefState,
    EvidenceDirection,
    EvidenceType,
)


class SupportBucket(str, Enum):
    """Categorical strength-and-direction of one piece of evidence.

    Symmetric around AMBIGUOUS. The classifier picks exactly one bucket
    per (edge, evidence) pair using the rubric in the classification
    prompt — never a free-form float.
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
# contributions to (α + β) per record — i.e. how many virtual Bernoulli
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
# These are pre-calibration defaults — once a labeled holdout of
# ≥50 trials exists, refit via ``calibration.py`` to minimize Brier.
EVIDENCE_TYPE_N_EFF: dict[EvidenceType, float] = {
    EvidenceType.CLINICAL_PHASE3:    15.0,
    EvidenceType.CLINICAL_PHASE2:     6.0,
    EvidenceType.CLINICAL_PHASE1:     2.0,
    EvidenceType.GENETIC_MR:         10.0,
    EvidenceType.GENETIC_GWAS:        4.0,
    EvidenceType.PRECLINICAL_IN_VIVO: 2.0,
    EvidenceType.PRECLINICAL_IN_VITRO: 1.0,
    EvidenceType.COMPUTATIONAL:       0.3,
    EvidenceType.LITERATURE:          0.2,
}


def p_obs_for_bucket(bucket: SupportBucket) -> float:
    return BUCKET_TO_P_OBS[bucket]


def effective_n_for_evidence(
    source_type: EvidenceType,
    quality_score: float = 1.0,
) -> float:
    """Effective virtual sample size for one evidence record.

    ``quality_score`` ∈ [0, 1] discounts the base N_eff — used to fold in
    the trial-level classification confidence (e.g., when the LLM's
    classification rubric tier was low, the evidence is downweighted).
    Default 1.0 means "no discount" — appropriate for non-LLM-derived
    evidence (LINCS signatures, GWAS hits, etc.) where there is no
    classification step to be uncertain about.
    """
    if not 0.0 <= quality_score <= 1.0:
        raise ValueError(
            f"quality_score must be in [0, 1], got {quality_score!r}"
        )
    base = EVIDENCE_TYPE_N_EFF[source_type]
    return base * quality_score


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
    (which discards strength information), we can opt to flip — though
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
