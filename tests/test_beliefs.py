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
    modulation_bucket,
    p_obs_for_bucket,
    pool_hierarchical,
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

    def test_database_curated_tiers_ordered_by_source_quality(self):
        """Per-source database tiers ordered by curation depth +
        primary-vs-aggregate character.

        Round-28 bumped the OT-direct / ChEMBL / mAb-table tier well
        above the round-25 value because the records assert MOLECULAR
        BINDING (a fact), not a probabilistic clinical outcome. OT-direct
        gets a small edge over ChEMBL / mAb-table because it aggregates
        multiple primary sources (ChEMBL + IUPHAR + DGIdb + drug
        labels), while ChEMBL alone or a hand-curated mAb table is a
        single source.

        Aggregate / heuristic tiers (OT-association, endpoint-prior)
        sit below the binding tier; pathway annotation is
        single-curator; LINCS is in-vitro perturbation; fallback +
        cross-reference are derived / heuristic.
        """
        assert (
            EVIDENCE_TYPE_N_EFF[EvidenceType.DATABASE_OT_DIRECT]
            >= EVIDENCE_TYPE_N_EFF[EvidenceType.DATABASE_CHEMBL]
        )
        assert (
            EVIDENCE_TYPE_N_EFF[EvidenceType.DATABASE_CHEMBL]
            == EVIDENCE_TYPE_N_EFF[EvidenceType.DATABASE_MAB_TABLE]
        )
        assert (
            EVIDENCE_TYPE_N_EFF[EvidenceType.DATABASE_CHEMBL]
            > EVIDENCE_TYPE_N_EFF[EvidenceType.DATABASE_OT_ASSOCIATION]
        )
        assert (
            EVIDENCE_TYPE_N_EFF[EvidenceType.DATABASE_OT_ASSOCIATION]
            > EVIDENCE_TYPE_N_EFF[EvidenceType.DATABASE_REACTOME_GO]
        )
        assert (
            EVIDENCE_TYPE_N_EFF[EvidenceType.DATABASE_REACTOME_GO]
            > EVIDENCE_TYPE_N_EFF[EvidenceType.DATABASE_LINCS]
        )
        assert (
            EVIDENCE_TYPE_N_EFF[EvidenceType.DATABASE_LINCS]
            > EVIDENCE_TYPE_N_EFF[EvidenceType.DATABASE_FALLBACK]
        )
        assert (
            EVIDENCE_TYPE_N_EFF[EvidenceType.DATABASE_FALLBACK]
            > EVIDENCE_TYPE_N_EFF[EvidenceType.DATABASE_CROSS_REFERENCE]
        )

    def test_database_binding_tier_below_phase3(self):
        """Round-28: curated binding records should sit below a single
        Phase-3 trial but rival or exceed Phase-2 — molecular binding
        is a cross-checked fact, not a noisy clinical signal."""
        for t in (
            EvidenceType.DATABASE_OT_DIRECT,
            EvidenceType.DATABASE_CHEMBL,
            EvidenceType.DATABASE_MAB_TABLE,
        ):
            assert (
                EVIDENCE_TYPE_N_EFF[EvidenceType.CLINICAL_PHASE3]
                > EVIDENCE_TYPE_N_EFF[t]
            )
            assert (
                EVIDENCE_TYPE_N_EFF[t]
                > EVIDENCE_TYPE_N_EFF[EvidenceType.CLINICAL_PHASE2]
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
        # posterior mean at 0.5—variance shrinks but the point estimate
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


class TestModulationBucket:
    """Mapping from v0.3.0 LLM modulation output (direction, confidence)
    to the seven-bucket SupportBucket system."""

    def test_strong_support_at_high_confidence_amplifies(self):
        assert modulation_bucket("amplifies", 0.95) == SupportBucket.STRONG_SUPPORT
        assert modulation_bucket("amplifies", 0.85) == SupportBucket.STRONG_SUPPORT

    def test_strong_contradict_at_high_confidence_suppresses(self):
        assert modulation_bucket("suppresses", 0.95) == SupportBucket.STRONG_CONTRADICT
        assert modulation_bucket("suppresses", 0.85) == SupportBucket.STRONG_CONTRADICT

    def test_moderate_band(self):
        assert modulation_bucket("amplifies", 0.84) == SupportBucket.MODERATE_SUPPORT
        assert modulation_bucket("amplifies", 0.70) == SupportBucket.MODERATE_SUPPORT
        assert modulation_bucket("suppresses", 0.75) == SupportBucket.MODERATE_CONTRADICT

    def test_weak_band(self):
        assert modulation_bucket("amplifies", 0.69) == SupportBucket.WEAK_SUPPORT
        assert modulation_bucket("amplifies", 0.55) == SupportBucket.WEAK_SUPPORT
        assert modulation_bucket("suppresses", 0.60) == SupportBucket.WEAK_CONTRADICT

    def test_ambiguous_below_floor(self):
        assert modulation_bucket("amplifies", 0.54) == SupportBucket.AMBIGUOUS
        assert modulation_bucket("amplifies", 0.0) == SupportBucket.AMBIGUOUS

    def test_neutral_always_ambiguous(self):
        """Neutral is ALWAYS AMBIGUOUS regardless of confidence. Trial
        failure has many possible explanations beyond "the modulator did
        nothing" (wrong dose / population / endpoint, underpowered, AE
        confounders). High confidence on neutral still does work via the
        Beta-Binomial path — AMBIGUOUS at high n_eff shrinks the posterior
        toward 0.5, encoding "strong evidence we don't know" rather than
        falsifying the modulation hypothesis."""
        assert modulation_bucket("neutral", 0.95) == SupportBucket.AMBIGUOUS
        assert modulation_bucket("neutral", 0.85) == SupportBucket.AMBIGUOUS
        assert modulation_bucket("neutral", 0.55) == SupportBucket.AMBIGUOUS
        assert modulation_bucket("neutral", 0.0) == SupportBucket.AMBIGUOUS

    def test_unknown_direction_is_ambiguous(self):
        # Defensive: garbage direction string doesn't crash, falls to AMBIGUOUS.
        assert modulation_bucket("sideways", 0.9) == SupportBucket.AMBIGUOUS

    def test_symmetric_around_ambiguous(self):
        """Bucket strengths should mirror across amplifies/suppresses for
        the same confidence — the existing flip_bucket relationship."""
        for conf in (0.55, 0.70, 0.85, 0.95):
            amp = modulation_bucket("amplifies", conf)
            sup = modulation_bucket("suppresses", conf)
            assert flip_bucket(amp) == sup


class TestPoolHierarchical:
    """Fixed-concentration hierarchical Beta-Binomial partial pooling — the
    cross-indication / cross-population backoff substrate (Phase A). A specific
    (leaf) belief borrows strength from a coarser ancestor acting as a capped
    prior; a well-evidenced leaf reclaims specificity."""

    def test_empty_returns_none(self):
        assert pool_hierarchical([]) is None

    def test_single_level_is_unchanged(self):
        b = EdgeBeliefState(alpha=8.0, beta=2.0)
        pooled = pool_hierarchical([b])
        assert pooled.alpha == 8.0 and pooled.beta == 2.0

    def test_carries_leaf_evidence_for_faithful_self_exclusion(self):
        """The pooled belief must carry the LEAF level's evidence records (not the
        ancestor's), so provenance self-exclusion can linearly remove a held-out
        trial's leaf contribution while the capped ancestor mass stays a constant
        prior. Without this, backoff edges silently survive self-exclusion."""
        from datetime import datetime, timezone
        from src.graph.models import EvidenceRecord, EvidenceType

        rec = EvidenceRecord(
            source_id="NCT00000001",
            source_type=EvidenceType.CLINICAL_PHASE3,
            support=SupportBucket.STRONG_SUPPORT.value,
            timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
        )
        leaf = EdgeBeliefState(alpha=4.0, beta=2.0, evidence=[rec])
        parent = EdgeBeliefState(alpha=70.0, beta=20.0)  # other-indication evidence
        pooled = pool_hierarchical([leaf, parent], prior_strength=20.0)
        assert pooled.evidence == [rec]  # leaf records carried, ancestor not unioned
        # The ancestor mass is a real prior offset (strength grew past the leaf's).
        assert pooled.evidence_strength > leaf.evidence_strength

    def test_sparse_leaf_shrinks_toward_rich_parent(self):
        """One whisper of leaf evidence barely moves a rich (capped) parent —
        the her2_positive_breast (≈empty) ← breast_cancer (strength 94) case.
        Before the fix the leaf shadowed the parent; now it borrows it."""
        leaf = EdgeBeliefState(alpha=1.5, beta=1.0)      # strength ~0.5, mean 0.6
        parent = EdgeBeliefState(alpha=75.0, beta=20.0)  # strength ~93, mean ~0.79
        pooled = pool_hierarchical([leaf, parent], prior_strength=20.0)
        # Pulled close to the parent's mean, NOT the leaf's raw mean.
        assert abs(pooled.expected_probability - parent.expected_probability) < 0.05
        assert pooled.expected_probability > leaf.expected_probability + 0.1

    def test_rich_leaf_retains_specificity_over_weak_parent(self):
        """A well-evidenced leaf (rheumatoid_arthritis, strength 16) is not
        dragged back to a weak arthritis parent (strength 1)."""
        leaf = EdgeBeliefState(alpha=14.0, beta=4.0)     # strength 16, mean ~0.78
        parent = EdgeBeliefState(alpha=1.6, beta=1.4)    # strength 1, mean ~0.53
        pooled = pool_hierarchical([leaf, parent], prior_strength=20.0)
        assert abs(pooled.expected_probability - leaf.expected_probability) < 0.05

    def test_parent_prior_is_capped(self):
        """A huge parent cannot steamroll a moderately-evidenced leaf forever —
        beyond ~prior_strength of its own evidence, the leaf dominates."""
        # leaf: ~30 obs at mean 0.4; parent: 400 obs at mean 0.9
        leaf = EdgeBeliefState(alpha=13.0, beta=19.0)
        parent = EdgeBeliefState(alpha=360.0, beta=40.0)
        capped = pool_hierarchical([leaf, parent], prior_strength=20.0)
        # With the cap the leaf (30 own obs) outweighs the 20-capped parent,
        # pulling the blend well below the parent's 0.9 (≈0.6 precision-weighted).
        assert capped.expected_probability < 0.7
        # Without a cap (huge prior_strength) the parent dominates instead —
        # proves the cap is what lets specificity win.
        uncapped = pool_hierarchical([leaf, parent], prior_strength=10_000.0)
        assert uncapped.expected_probability > 0.82

    def test_more_leaf_evidence_monotonically_shifts_toward_leaf(self):
        """As the leaf accumulates its own evidence, the pooled mean moves
        monotonically from the parent's mean toward the leaf's mean."""
        parent = EdgeBeliefState(alpha=40.0, beta=10.0)  # mean 0.8
        prev = None
        for n in (1, 5, 20, 100):
            leaf = EdgeBeliefState(alpha=1.0 + 0.3 * n, beta=1.0 + 0.7 * n)  # mean→0.3
            pooled = pool_hierarchical([leaf, parent], prior_strength=20.0)
            if prev is not None:
                assert pooled.expected_probability <= prev + 1e-9
            prev = pooled.expected_probability

    def test_env_override_changes_cap(self, monkeypatch):
        # Leaf leans LOW (mean 0.25, 6 own obs); parent leans high (mean 0.9).
        leaf = EdgeBeliefState(alpha=2.0, beta=6.0)
        parent = EdgeBeliefState(alpha=90.0, beta=10.0)
        monkeypatch.setenv("EROOM_POOL_PRIOR_STRENGTH", "2.0")
        tight = pool_hierarchical([leaf, parent])
        monkeypatch.setenv("EROOM_POOL_PRIOR_STRENGTH", "200.0")
        loose = pool_hierarchical([leaf, parent])
        # A looser (larger) cap lets more of the rich high-mean parent through,
        # so the low-leaning leaf is overridden and the blend sits higher; a
        # tight cap lets the leaf pull it down.
        assert loose.expected_probability > tight.expected_probability + 0.2
