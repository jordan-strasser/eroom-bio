"""Tests for the principled Beta-Binomial conjugate-update primitives."""

from __future__ import annotations

import pytest

from src.graph.models import EdgeBeliefState, EvidenceType
from src.inference.beliefs import (
    BUCKET_TO_P_OBS,
    EVIDENCE_TYPE_N_EFF,
    SupportBucket,
    apply_virtual_evidence,
    bucket_to_direction,
    effective_n_for_evidence,
    flip_bucket,
    p_obs_for_bucket,
)
from src.graph.models import EvidenceDirection


class TestBucketTable:
    def test_all_buckets_have_p_obs(self):
        for b in SupportBucket:
            assert b in BUCKET_TO_P_OBS

    def test_p_obs_strictly_in_open_unit_interval(self):
        for p in BUCKET_TO_P_OBS.values():
            assert 0.0 < p < 1.0

    def test_p_obs_monotone_in_strength(self):
        ordered = [
            SupportBucket.STRONG_CONTRADICT,
            SupportBucket.MODERATE_CONTRADICT,
            SupportBucket.WEAK_CONTRADICT,
            SupportBucket.AMBIGUOUS,
            SupportBucket.WEAK_SUPPORT,
            SupportBucket.MODERATE_SUPPORT,
            SupportBucket.STRONG_SUPPORT,
        ]
        ps = [BUCKET_TO_P_OBS[b] for b in ordered]
        assert ps == sorted(ps), f"p_obs not monotone: {ps}"

    def test_p_obs_symmetric_around_ambiguous(self):
        # For each support/contradict pair, p_obs(support) + p_obs(contradict) == 1.
        pairs = [
            (SupportBucket.STRONG_SUPPORT, SupportBucket.STRONG_CONTRADICT),
            (SupportBucket.MODERATE_SUPPORT, SupportBucket.MODERATE_CONTRADICT),
            (SupportBucket.WEAK_SUPPORT, SupportBucket.WEAK_CONTRADICT),
        ]
        for support, contradict in pairs:
            assert (
                BUCKET_TO_P_OBS[support] + BUCKET_TO_P_OBS[contradict]
                == pytest.approx(1.0)
            )

    def test_ambiguous_is_exactly_half(self):
        assert BUCKET_TO_P_OBS[SupportBucket.AMBIGUOUS] == 0.5


class TestNEffTable:
    def test_all_evidence_types_present(self):
        for et in EvidenceType:
            assert et in EVIDENCE_TYPE_N_EFF

    def test_n_eff_strictly_positive(self):
        for n in EVIDENCE_TYPE_N_EFF.values():
            assert n > 0

    def test_clinical_dominates_preclinical(self):
        assert (
            EVIDENCE_TYPE_N_EFF[EvidenceType.CLINICAL_PHASE3]
            > EVIDENCE_TYPE_N_EFF[EvidenceType.PRECLINICAL_IN_VIVO]
        )
        assert (
            EVIDENCE_TYPE_N_EFF[EvidenceType.CLINICAL_PHASE2]
            > EVIDENCE_TYPE_N_EFF[EvidenceType.PRECLINICAL_IN_VITRO]
        )

    def test_genetic_mr_above_gwas(self):
        assert (
            EVIDENCE_TYPE_N_EFF[EvidenceType.GENETIC_MR]
            > EVIDENCE_TYPE_N_EFF[EvidenceType.GENETIC_GWAS]
        )


class TestEffectiveN:
    def test_default_quality_returns_base(self):
        assert effective_n_for_evidence(EvidenceType.CLINICAL_PHASE3) == pytest.approx(
            EVIDENCE_TYPE_N_EFF[EvidenceType.CLINICAL_PHASE3]
        )

    def test_quality_discounts_n_eff(self):
        base = EVIDENCE_TYPE_N_EFF[EvidenceType.CLINICAL_PHASE2]
        assert effective_n_for_evidence(
            EvidenceType.CLINICAL_PHASE2, quality_score=0.5
        ) == pytest.approx(base * 0.5)

    def test_quality_zero_yields_zero(self):
        assert effective_n_for_evidence(
            EvidenceType.LITERATURE, quality_score=0.0
        ) == 0.0

    def test_quality_out_of_range_raises(self):
        with pytest.raises(ValueError):
            effective_n_for_evidence(EvidenceType.LITERATURE, quality_score=1.5)
        with pytest.raises(ValueError):
            effective_n_for_evidence(EvidenceType.LITERATURE, quality_score=-0.1)


class TestBucketToDirection:
    @pytest.mark.parametrize("bucket", [
        SupportBucket.STRONG_SUPPORT,
        SupportBucket.MODERATE_SUPPORT,
        SupportBucket.WEAK_SUPPORT,
    ])
    def test_support_buckets_map_to_supporting(self, bucket):
        assert bucket_to_direction(bucket) == EvidenceDirection.SUPPORTING

    @pytest.mark.parametrize("bucket", [
        SupportBucket.STRONG_CONTRADICT,
        SupportBucket.MODERATE_CONTRADICT,
        SupportBucket.WEAK_CONTRADICT,
    ])
    def test_contradict_buckets_map_to_contradicting(self, bucket):
        assert bucket_to_direction(bucket) == EvidenceDirection.CONTRADICTING

    def test_ambiguous_maps_to_ambiguous(self):
        assert bucket_to_direction(SupportBucket.AMBIGUOUS) == EvidenceDirection.AMBIGUOUS


class TestFlipBucket:
    def test_flip_is_involution(self):
        for b in SupportBucket:
            assert flip_bucket(flip_bucket(b)) == b

    def test_ambiguous_is_self_flip(self):
        assert flip_bucket(SupportBucket.AMBIGUOUS) == SupportBucket.AMBIGUOUS


class TestApplyVirtualEvidence:
    def test_uniform_prior_strong_support(self):
        b = EdgeBeliefState()
        n_eff = 5.0
        p_obs = BUCKET_TO_P_OBS[SupportBucket.STRONG_SUPPORT]
        post = apply_virtual_evidence(b, n_eff=n_eff, p_obs=p_obs)
        assert post.alpha == pytest.approx(1.0 + n_eff * p_obs)
        assert post.beta == pytest.approx(1.0 + n_eff * (1.0 - p_obs))

    def test_ambiguous_evidence_preserves_mean(self):
        # Ambiguous evidence (p_obs=0.5) on a uniform prior leaves the
        # posterior mean at 0.5 — variance shrinks but the point estimate
        # stays put.
        b = EdgeBeliefState()
        post = apply_virtual_evidence(b, n_eff=10.0, p_obs=0.5)
        assert post.alpha == post.beta
        assert post.expected_probability == pytest.approx(0.5)
        assert post.variance < b.variance

    def test_does_not_mutate_input(self):
        b = EdgeBeliefState(alpha=2.0, beta=2.0)
        _ = apply_virtual_evidence(b, n_eff=5.0, p_obs=0.9)
        assert b.alpha == 2.0
        assert b.beta == 2.0

    def test_evidence_strength_increases(self):
        b = EdgeBeliefState(alpha=1.0, beta=1.0)
        post = apply_virtual_evidence(b, n_eff=5.0, p_obs=0.9)
        assert post.evidence_strength > b.evidence_strength
        # And by exactly N_eff, since (α+β) gains exactly N_eff each update.
        assert post.evidence_strength == pytest.approx(b.evidence_strength + 5.0)

    def test_repeated_strong_support_drives_mean_up(self):
        b = EdgeBeliefState()
        for _ in range(10):
            b = apply_virtual_evidence(b, n_eff=2.0, p_obs=0.95)
        assert b.expected_probability > 0.9

    def test_negative_n_eff_rejected(self):
        with pytest.raises(ValueError):
            apply_virtual_evidence(EdgeBeliefState(), n_eff=-1.0, p_obs=0.5)

    def test_p_obs_out_of_range_rejected(self):
        with pytest.raises(ValueError):
            apply_virtual_evidence(EdgeBeliefState(), n_eff=1.0, p_obs=1.5)
        with pytest.raises(ValueError):
            apply_virtual_evidence(EdgeBeliefState(), n_eff=1.0, p_obs=-0.1)


class TestPObsHelper:
    def test_p_obs_for_bucket_matches_table(self):
        for b in SupportBucket:
            assert p_obs_for_bucket(b) == BUCKET_TO_P_OBS[b]
