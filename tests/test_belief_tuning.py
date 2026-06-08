"""Tests for the delta-adjust Beta-Binomial tuner — exactness is the whole point."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.graph.models import EvidenceRecord
from src.inference.beliefs import EvidenceType
from src.inference.belief_tuning import BeliefTuneConfig, retune_alpha_beta

_TS = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _trial_rec(nct, n_eff_applied, p_obs_applied, outcome):
    return EvidenceRecord(
        source_id=nct,
        source_type=EvidenceType.CLINICAL_PHASE3,
        support="moderate_contradict",
        timestamp=_TS,
        context={
            "n_eff_applied": n_eff_applied,
            "p_obs_applied": p_obs_applied,
            "outcome": outcome,
            "outcome_conditioned": True,
        },
    )


def _curated_rec():
    return EvidenceRecord(
        source_id="opentargets:CHEMBLX",
        source_type=EvidenceType.DATABASE_OT_DIRECT,
        support="strong_support",
        timestamp=_TS,
        context={},
    )


def test_exact_at_default():
    recs = [_trial_rec("NCT1", 2.0, 0.2, "failure"), _curated_rec()]
    a, b = retune_alpha_beta(10.0, 5.0, recs)  # default cfg, no drop
    assert (a, b) == pytest.approx((10.0, 5.0))


def test_p_obs_override_is_exact():
    recs = [_trial_rec("NCT1", 2.0, 0.2, "failure")]
    cfg = BeliefTuneConfig(outcome_p_obs={"failure": 0.4})
    a, b = retune_alpha_beta(10.0, 5.0, recs, cfg)
    # Δα = 2·(0.4−0.2)=+0.4 ; Δβ = 2·(0.6−0.8)=−0.4
    assert (a, b) == pytest.approx((10.4, 4.6))


def test_neff_scale_is_exact():
    recs = [_trial_rec("NCT1", 2.0, 0.2, "failure")]
    cfg = BeliefTuneConfig(neff_scale={"clinical": 2.0})
    a, b = retune_alpha_beta(10.0, 5.0, recs, cfg)
    # n_new=4 ; Δα=(4−2)·0.2=+0.4 ; Δβ=(4−2)·0.8=+1.6
    assert (a, b) == pytest.approx((10.4, 6.6))


def test_drop_removes_contribution_exactly():
    recs = [_trial_rec("NCT1", 2.0, 0.2, "failure"), _curated_rec()]
    a, b = retune_alpha_beta(10.0, 5.0, recs, drop_ncts={"NCT1"})
    # remove NCT1: Δα=−0.4, Δβ=−1.6 ; curated untouched
    assert (a, b) == pytest.approx((9.6, 3.4))


def test_curated_records_never_tuned():
    # a curated record has no n_eff_applied → no config or drop touches it
    recs = [_curated_rec()]
    cfg = BeliefTuneConfig(neff_scale={"other": 5.0}, outcome_p_obs={"failure": 0.9})
    a, b = retune_alpha_beta(7.0, 3.0, recs, cfg, drop_ncts={"opentargets:CHEMBLX"})
    assert (a, b) == pytest.approx((7.0, 3.0))


def test_floor_prevents_nonpositive():
    recs = [_trial_rec("NCT1", 100.0, 0.5, "failure")]
    # drop a huge contribution that would push β below zero → floored at 1e-6
    a, b = retune_alpha_beta(2.0, 2.0, recs, drop_ncts={"NCT1"})
    assert a >= 1e-6 and b >= 1e-6
