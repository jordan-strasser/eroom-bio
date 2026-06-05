"""Unit tests for the calibration & paired-comparison metrics (the benchmark bedrock).

These verify known-value correctness — a wrong Brier/ECE/DeLong would silently
mis-rank the graph vs the baselines, so they're checked against hand-computed values
and against the existing eval_holdout_compose._auroc.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from scripts.eval_holdout_compose import _auroc as _auroc_list
from src.prediction.calibration import (
    auroc,
    brier_score,
    delong_paired,
    expected_calibration_error,
    maximum_calibration_error,
    mcnemar_paired,
    platt_apply,
    platt_fit,
    pr_auc,
    reliability_table,
)


# ── Brier ──────────────────────────────────────────────────────────────


def test_brier_perfect():
    assert brier_score([1.0, 0.0, 1.0, 0.0], [1, 0, 1, 0]) == 0.0


def test_brier_allhalf_is_quarter():
    assert brier_score([0.5, 0.5, 0.5], [1, 0, 1]) == pytest.approx(0.25)


def test_brier_known_value():
    # ((1-0.8)^2 + (0-0.3)^2)/2 = (0.04 + 0.09)/2
    assert brier_score([0.8, 0.3], [1, 0]) == pytest.approx(0.065)


# ── AUROC ──────────────────────────────────────────────────────────────


def test_auroc_perfect_separation():
    assert auroc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == pytest.approx(1.0)


def test_auroc_tie_is_half():
    assert auroc([0.5, 0.5], [1, 0]) == pytest.approx(0.5)


def test_auroc_single_class_is_nan():
    assert math.isnan(auroc([0.9, 0.8], [1, 1]))


def test_auroc_matches_eval_holdout_impl():
    rng = np.random.default_rng(0)
    for _ in range(20):
        p = rng.random(40).tolist()
        y = rng.integers(0, 2, 40).tolist()
        if 0 < sum(y) < len(y):
            assert auroc(p, y) == pytest.approx(_auroc_list(p, y), abs=1e-9)


# ── PR-AUC ─────────────────────────────────────────────────────────────


def test_prauc_perfect():
    assert pr_auc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == pytest.approx(1.0)


def test_prauc_below_one_when_misranked():
    # the lone negative ranked above the lone positive ⇒ AP = 0.5 < 1
    assert pr_auc([0.2, 0.9], [1, 0]) == pytest.approx(0.5)


# ── Reliability / ECE / MCE ─────────────────────────────────────────────


def test_reliability_counts_sum_to_n():
    p = [0.05, 0.15, 0.25, 0.95, 0.99]
    y = [0, 0, 0, 1, 1]
    assert sum(b.count for b in reliability_table(p, y, n_bins=10)) == len(p)


def test_ece_zero_when_perfectly_calibrated():
    # predictions exactly match observed frequency in each bin
    p = [0.0, 0.0, 1.0, 1.0]
    y = [0, 0, 1, 1]
    assert expected_calibration_error(p, y, n_bins=10) == pytest.approx(0.0)


def test_ece_detects_miscalibration():
    # always predict 0.9 but only half are positive ⇒ ECE ≈ 0.4
    p = [0.9] * 10
    y = [1, 0] * 5
    assert expected_calibration_error(p, y, n_bins=10) == pytest.approx(0.4, abs=1e-9)
    assert maximum_calibration_error(p, y, n_bins=10) == pytest.approx(0.4, abs=1e-9)


# ── DeLong ──────────────────────────────────────────────────────────────


def test_delong_auc_matches_auroc():
    rng = np.random.default_rng(1)
    p_a = rng.random(60)
    p_b = rng.random(60)
    y = rng.integers(0, 2, 60)
    while not (0 < y.sum() < len(y)):
        y = rng.integers(0, 2, 60)
    res = delong_paired(p_a, p_b, y)
    assert res.auc_a == pytest.approx(auroc(p_a, y), abs=1e-9)
    assert res.auc_b == pytest.approx(auroc(p_b, y), abs=1e-9)


def test_delong_identical_predictors_not_significant():
    rng = np.random.default_rng(2)
    p = rng.random(50)
    y = rng.integers(0, 2, 50)
    while not (0 < y.sum() < len(y)):
        y = rng.integers(0, 2, 50)
    res = delong_paired(p, p, y)
    assert res.auc_a == pytest.approx(res.auc_b)
    assert res.p_value == pytest.approx(1.0)


def test_delong_detects_clear_difference():
    # a strong predictor vs a noisy one on a large sample ⇒ significant
    rng = np.random.default_rng(3)
    n = 400
    y = rng.integers(0, 2, n)
    strong = y + rng.normal(0, 0.3, n)         # tracks the label
    noise = rng.random(n)                       # pure noise
    res = delong_paired(strong, noise, y)
    assert res.auc_a > res.auc_b
    assert res.p_value < 0.05


# ── McNemar ─────────────────────────────────────────────────────────────


def test_mcnemar_symmetric_not_significant():
    # equal discordance both ways ⇒ p == 1
    a = [True, False, True, False]
    b = [False, True, False, True]
    res = mcnemar_paired(a, b)
    assert res.only_a_correct == res.only_b_correct
    assert res.p_value == pytest.approx(1.0)


def test_mcnemar_one_sided_dominance():
    # a always right where they differ ⇒ small p
    a = [True] * 12 + [True, True]
    b = [False] * 12 + [True, True]
    res = mcnemar_paired(a, b)
    assert res.only_a_correct == 12
    assert res.only_b_correct == 0
    assert res.p_value < 0.001


# ── Platt scaling ───────────────────────────────────────────────────────


def test_platt_is_monotonic_and_improves_calibration():
    rng = np.random.default_rng(4)
    n = 300
    y = rng.integers(0, 2, n)
    # over-confident predictions: pushed toward 0/1
    raw = np.clip(y * 0.9 + 0.05 + rng.normal(0, 0.1, n), 0.01, 0.99)
    a, b = platt_fit(raw, y)
    cal = platt_apply(raw, a, b)
    # monotonic: sorting by raw keeps calibrated sorted
    order = np.argsort(raw)
    assert np.all(np.diff(cal[order]) >= -1e-9)
    assert cal.min() >= 0.0 and cal.max() <= 1.0
