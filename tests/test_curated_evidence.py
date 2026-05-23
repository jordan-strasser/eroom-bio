"""Tests for the round-25 curated-evidence helper."""

from __future__ import annotations

import pytest

from src.graph.curated_evidence import (
    belief_from_curated_record,
    endpoint_class_to_bucket,
    make_curated_record,
    ot_association_score_to_bucket,
    ot_score_quality,
)
from src.graph.models import EvidenceType
from src.inference.beliefs import SupportBucket


class TestMakeCuratedRecord:
    def test_default_source_type_is_ot_direct(self):
        r = make_curated_record(
            source_id="opentargets:CHEMBL1743072",
            bucket=SupportBucket.STRONG_SUPPORT,
        )
        assert r.source_type == EvidenceType.DATABASE_OT_DIRECT

    def test_explicit_source_type_passed_through(self):
        r = make_curated_record(
            source_id="lincs:KD_BRAF",
            bucket=SupportBucket.MODERATE_SUPPORT,
            source_type=EvidenceType.DATABASE_LINCS,
        )
        assert r.source_type == EvidenceType.DATABASE_LINCS

    def test_preserves_source_id_and_bucket(self):
        r = make_curated_record(
            source_id="mab_table:bevacizumab",
            bucket=SupportBucket.STRONG_SUPPORT,
            notes="bev → VEGFA per curated antibody table",
        )
        assert r.source_id == "mab_table:bevacizumab"
        assert r.support == "strong_support"
        assert "VEGFA" in (r.notes or "")

    def test_quality_score_passes_through(self):
        r = make_curated_record(
            source_id="opentargets_assoc:X__Y",
            bucket=SupportBucket.MODERATE_SUPPORT,
            quality_score=0.4,
        )
        assert r.quality_score == 0.4

    def test_quality_score_validation(self):
        with pytest.raises(ValueError):
            make_curated_record(
                source_id="x",
                bucket=SupportBucket.WEAK_SUPPORT,
                quality_score=1.5,
            )


class TestBeliefFromCuratedRecord:
    def test_strong_support_pushes_posterior_toward_one(self):
        r = make_curated_record(
            source_id="opentargets:CHEMBL1743072",
            bucket=SupportBucket.STRONG_SUPPORT,
        )
        b = belief_from_curated_record(r)
        # OT_DIRECT default tier: n_eff=12 (round-28). Starts at
        # Beta(1,1), applies p_obs=0.95 → α = 1 + 12·0.95 = 12.4,
        # β = 1 + 12·0.05 = 1.6.
        assert b.alpha == pytest.approx(12.4)
        assert b.beta == pytest.approx(1.6)
        assert b.expected_probability == pytest.approx(12.4 / 14.0)

    def test_moderate_support_lands_in_optimistic_range(self):
        # OT_ASSOCIATION tier (n_eff=2) at MODERATE_SUPPORT (p_obs=0.80).
        r = make_curated_record(
            source_id="opentargets_assoc:X__Y",
            bucket=SupportBucket.MODERATE_SUPPORT,
            source_type=EvidenceType.DATABASE_OT_ASSOCIATION,
        )
        b = belief_from_curated_record(r)
        # α = 1 + 2·0.80 = 2.6, β = 1 + 2·0.20 = 1.4.
        assert b.expected_probability == pytest.approx(2.6 / 4.0)

    def test_ambiguous_stays_centered(self):
        r = make_curated_record(
            source_id="opentargets_assoc:X__Y",
            bucket=SupportBucket.AMBIGUOUS,
        )
        b = belief_from_curated_record(r)
        # p_obs=0.5 → symmetric update
        assert b.expected_probability == pytest.approx(0.5)

    def test_quality_score_discounts_n_eff(self):
        full = belief_from_curated_record(make_curated_record(
            source_id="x", bucket=SupportBucket.STRONG_SUPPORT,
        ))
        half = belief_from_curated_record(make_curated_record(
            source_id="x", bucket=SupportBucket.STRONG_SUPPORT, quality_score=0.5,
        ))
        # OT_DIRECT n_eff=12; with half quality evidence_strength halves.
        assert full.evidence_strength == pytest.approx(12.0)
        assert half.evidence_strength == pytest.approx(6.0)

    def test_evidence_list_records_the_record(self):
        r = make_curated_record(
            source_id="mab_table:bevacizumab",
            bucket=SupportBucket.STRONG_SUPPORT,
        )
        b = belief_from_curated_record(r)
        assert len(b.evidence) == 1
        assert b.evidence[0].source_id == "mab_table:bevacizumab"


class TestOTScoreMapping:
    @pytest.mark.parametrize("score, expected_bucket", [
        (0.95, SupportBucket.STRONG_SUPPORT),
        (0.75, SupportBucket.STRONG_SUPPORT),
        (0.74, SupportBucket.MODERATE_SUPPORT),
        (0.50, SupportBucket.MODERATE_SUPPORT),
        (0.49, SupportBucket.WEAK_SUPPORT),
        (0.25, SupportBucket.WEAK_SUPPORT),
        (0.10, SupportBucket.AMBIGUOUS),
        (0.05, SupportBucket.AMBIGUOUS),
        (0.0, SupportBucket.WEAK_CONTRADICT),
    ])
    def test_score_to_bucket_monotonic(self, score, expected_bucket):
        assert ot_association_score_to_bucket(score) == expected_bucket

    @pytest.mark.parametrize("count, expected_min", [
        (0, 0.0),
        (1, 0.2),  # min floor for any positive count
        (5, 1.0),  # caps at 5+
        (20, 1.0),
        (100, 1.0),
    ])
    def test_evidence_count_quality(self, count, expected_min):
        assert ot_score_quality(count) == pytest.approx(expected_min)


class TestEndpointClass:
    def test_accepted_endpoint_strong(self):
        assert endpoint_class_to_bucket(
            accepted=True, surrogate=False, exploratory=False,
        ) == SupportBucket.STRONG_SUPPORT

    def test_surrogate_endpoint_moderate(self):
        assert endpoint_class_to_bucket(
            accepted=False, surrogate=True, exploratory=False,
        ) == SupportBucket.MODERATE_SUPPORT

    def test_exploratory_endpoint_weak(self):
        assert endpoint_class_to_bucket(
            accepted=False, surrogate=False, exploratory=True,
        ) == SupportBucket.WEAK_SUPPORT
