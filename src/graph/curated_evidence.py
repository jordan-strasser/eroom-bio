"""Round-25 helper: emit EvidenceRecords for curated-database knowledge.

Before round 25, the populator hand-set edge priors (Beta(2,1), Beta(5,1),
etc.) for facts derived from curated biomedical databases — Open Targets
drug-target lookups, ChEMBL action_type inference, the mAb-target table,
OT-association scores, endpoint-type classification. Those priors were
real evidence (vetted by curators), but they lived outside the
EvidenceRecord channel, so the prediction engine's trust-weighted geomean
couldn't tell "we know X from OT" from "we picked a default at populate
time."

Round 25 replaces hand-set priors with explicit ``EvidenceRecord``s
carrying ``source_type=DATABASE_CURATED``. Edge beliefs all start at
Beta(1, 1) and accumulate evidence — curated or trial-attributed — via
the same ``apply_virtual_evidence`` recipe. The trust-weighted geomean
then weights edges by how much evidence (of any kind) backs them.

This module collects the constructors and bucket-mapping logic so the
populator never builds ``EvidenceRecord``s by hand.
"""

from __future__ import annotations

from datetime import datetime, timezone

from src.graph.models import (
    EdgeBeliefState,
    EvidenceRecord,
    EvidenceType,
)
from src.inference.beliefs import (
    SupportBucket,
    apply_virtual_evidence,
    effective_n_for_evidence,
    p_obs_for_bucket,
)


def make_curated_record(
    *,
    source_id: str,
    bucket: SupportBucket,
    source_type: EvidenceType = EvidenceType.DATABASE_OT_DIRECT,
    notes: str | None = None,
    quality_score: float = 1.0,
    context: dict | None = None,
) -> EvidenceRecord:
    """Construct one curated EvidenceRecord.

    ``source_type`` selects which DATABASE_* tier's n_eff this record
    contributes. Each populator call site picks the right tier (see
    ``src/graph/populate.py``). Defaults to OT_DIRECT since that's the
    most-used path.

    ``source_id`` should encode the originating database AND the
    specific record id so the provenance is traceable. Examples:
      - ``"opentargets:CHEMBL1743072"`` for an OT drug-target binding
      - ``"chembl_rest:CHEMBL1743072"`` for ChEMBL action_type inference
      - ``"mab_table:bevacizumab"`` for the mAb curated-table lookup
      - ``"opentargets_assoc:ENSG00000112715__EFO_0005842"`` for an OT
        target-disease association
      - ``"endpoint_type_prior:DFS_colorectal_cancer"`` for the
        endpoint-class classifier

    ``bucket`` selects p_obs (strong_support → 0.95, etc).
    ``quality_score`` discounts n_eff — used when the curated source
    is itself uncertain (OT association with low evidence_count).
    """
    return EvidenceRecord(
        source_id=source_id,
        source_type=source_type,
        support=bucket.value,
        quality_score=quality_score,
        timestamp=datetime.now(timezone.utc),
        notes=notes,
        context=context or {},
    )


def belief_from_curated_record(record: EvidenceRecord) -> EdgeBeliefState:
    """Return an EdgeBeliefState seeded by one curated record applied
    to a Beta(1, 1) prior.

    Used at edge-creation time: the populator decides what curated fact
    backs a new edge, builds the record, and passes the resulting
    belief to ``graph.add_edge``. The record is stored on the belief's
    ``evidence`` list so the populator's call is replayable.
    """
    n = effective_n_for_evidence(record.source_type, record.quality_score)
    p_obs = p_obs_for_bucket(SupportBucket(record.support))
    belief = apply_virtual_evidence(EdgeBeliefState(), n_eff=n, p_obs=p_obs)
    belief.evidence.append(record)
    return belief


# ── Bucket-mapping helpers for specific curated sources ────────────────


def ot_association_score_to_bucket(ot_score: float) -> SupportBucket:
    """Map an Open Targets target-disease association score to a bucket.

    Thresholds chosen so the existing populate.py callers preserve
    direction: the old ``score_to_prior(score, count)`` with high score
    produced Beta heavily weighted toward success; low score weighted
    toward contradiction. New mapping keeps that direction-with-strength
    intuition through bucket choice. n_eff is supplied by the
    DATABASE_CURATED tier (3.0); quality_score scales by evidence_count.
    """
    if ot_score >= 0.75:
        return SupportBucket.STRONG_SUPPORT
    if ot_score >= 0.5:
        return SupportBucket.MODERATE_SUPPORT
    if ot_score >= 0.25:
        return SupportBucket.WEAK_SUPPORT
    if ot_score >= 0.05:
        return SupportBucket.AMBIGUOUS
    return SupportBucket.WEAK_CONTRADICT


def ot_score_quality(evidence_count: int) -> float:
    """Map OT association evidence_count → quality_score discount.

    More supporting data sources behind the score → higher confidence
    in it. Caps at 1.0 once we have 5+ sources (anything beyond is
    diminishing returns). Floor at 0.2 for single-source entries.
    """
    if evidence_count <= 0:
        return 0.0  # caller should skip rather than emit
    return min(1.0, max(0.2, evidence_count / 5.0))


def endpoint_class_to_bucket(
    *, accepted: bool, surrogate: bool, exploratory: bool
) -> SupportBucket:
    """Map an endpoint class (regulatory status) to a curated bucket.

    Accepted endpoints (OS, DFS in CRC, RFS) → strong: well-established
    that this endpoint captures the disease. Surrogate (PFS, ORR) →
    moderate: literature-supported but not always predictive. Exploratory
    → weak: a single trial or single biomarker, claim weaker.
    """
    if accepted:
        return SupportBucket.STRONG_SUPPORT
    if surrogate:
        return SupportBucket.MODERATE_SUPPORT
    if exploratory:
        return SupportBucket.WEAK_SUPPORT
    # Unknown class — be conservative (matches old default).
    return SupportBucket.WEAK_SUPPORT
