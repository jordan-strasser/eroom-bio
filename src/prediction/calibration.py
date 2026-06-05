"""Calibration & paired-comparison metrics — the bedrock of the prediction benchmark.

Per BENCHMARK.md Q1: the right accuracy target is **edge/chain calibration**, not
chain-AUROC→1. A counterfactual composes per-edge Beta beliefs, so it is trustworthy
iff each probability is calibrated (a 0.7 prediction holds ~70% of the time). These
are directly measurable — reliability diagrams, Brier, ECE — independent of the noisy
chain outcome.

Also here: discrimination beyond AUROC (PR-AUC for class imbalance) and the **paired**
significance tests (DeLong for AUROC, McNemar for accuracy) the benchmark needs at
n≈129, where 0.57 vs 0.61 is not distinguishable unpaired (BENCHMARK.md control #3).

Pure numeric — no graph/IO deps — so it is unit-testable and reusable by both the
holdout harness and the baselines harness.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


# ── Calibration ─────────────────────────────────────────────────────────


def brier_score(probs: list[float] | np.ndarray, labels: list[int] | np.ndarray) -> float:
    """Mean squared error of probabilistic predictions = E[(p - y)^2].

    The single proper scoring rule for calibration + sharpness. Lower is better;
    0.25 is the all-0.5 baseline, lower than the base-rate Brier means real signal.
    """
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=float)
    if p.size == 0:
        return float("nan")
    return float(np.mean((p - y) ** 2))


@dataclass
class ReliabilityBin:
    lo: float
    hi: float
    count: int
    mean_pred: float   # mean predicted probability in the bin
    frac_pos: float    # observed fraction of positives in the bin


def reliability_table(
    probs: list[float] | np.ndarray,
    labels: list[int] | np.ndarray,
    n_bins: int = 10,
) -> list[ReliabilityBin]:
    """Equal-width reliability bins. A calibrated model has mean_pred ≈ frac_pos in
    every populated bin (the diagonal of a reliability diagram)."""
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out: list[ReliabilityBin] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        # last bin is closed on the right so p==1.0 lands somewhere
        in_bin = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        c = int(in_bin.sum())
        if c == 0:
            out.append(ReliabilityBin(lo, hi, 0, float("nan"), float("nan")))
            continue
        out.append(ReliabilityBin(
            lo, hi, c, float(p[in_bin].mean()), float(y[in_bin].mean())
        ))
    return out


def expected_calibration_error(
    probs: list[float] | np.ndarray,
    labels: list[int] | np.ndarray,
    n_bins: int = 10,
) -> float:
    """ECE = Σ (bin_count / N) · |mean_pred − frac_pos|. The headline calibration
    scalar: weighted average gap between predicted and observed frequency."""
    p = np.asarray(probs, dtype=float)
    if p.size == 0:
        return float("nan")
    n = p.size
    ece = 0.0
    for b in reliability_table(probs, labels, n_bins):
        if b.count == 0:
            continue
        ece += (b.count / n) * abs(b.mean_pred - b.frac_pos)
    return float(ece)


def maximum_calibration_error(
    probs: list[float] | np.ndarray,
    labels: list[int] | np.ndarray,
    n_bins: int = 10,
) -> float:
    """MCE = max over populated bins of |mean_pred − frac_pos| (worst-case gap)."""
    gaps = [
        abs(b.mean_pred - b.frac_pos)
        for b in reliability_table(probs, labels, n_bins)
        if b.count > 0
    ]
    return float(max(gaps)) if gaps else float("nan")


# ── Discrimination ──────────────────────────────────────────────────────


def auroc(probs: list[float] | np.ndarray, labels: list[int] | np.ndarray) -> float:
    """Rank-based AUROC with 0.5 tie credit. Matches eval_holdout_compose._auroc
    and DeLong's AUC estimate (kept here so the module is self-contained)."""
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels)
    pos = p[y == 1]
    neg = p[y == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    # vectorized Mann-Whitney with tie handling
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    sp = p[order]
    i = 0
    n = p.size
    while i < n:
        j = i
        while j < n and sp[j] == sp[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    r_pos = ranks[y == 1].sum()
    n_pos, n_neg = pos.size, neg.size
    return float((r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def pr_auc(probs: list[float] | np.ndarray, labels: list[int] | np.ndarray) -> float:
    """Average precision (area under precision-recall). The right discrimination
    metric under class imbalance (BENCHMARK.md Metrics) — the ~71%-success corpus
    makes AUROC look better than the model's grip on the minority (failure) class."""
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=int)
    if y.sum() == 0 or y.sum() == y.size:
        return float("nan")
    order = np.argsort(-p, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    precision = tp / (tp + fp)
    recall = tp / y.sum()
    # AP = Σ (R_n − R_{n−1}) · P_n
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - recall_prev) * precision))


# ── Paired significance tests (DeLong, McNemar) ─────────────────────────


def _midrank(x: np.ndarray) -> np.ndarray:
    """Midranks (ties get the average rank). Core of the fast DeLong algorithm."""
    order = np.argsort(x)
    z = x[order]
    n = len(x)
    t = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j
    out = np.empty(n, dtype=float)
    out[order] = t
    return out


def _fast_delong(preds_sorted: np.ndarray, m: int):
    """Fast DeLong (Sun & Xu 2014). `preds_sorted` is k×N with the m positive
    samples in the first m columns. Returns (aucs[k], covariance[k×k])."""
    n = preds_sorted.shape[1] - m
    k = preds_sorted.shape[0]
    pos = preds_sorted[:, :m]
    neg = preds_sorted[:, m:]
    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m + n])
    for r in range(k):
        tx[r, :] = _midrank(pos[r, :])
        ty[r, :] = _midrank(neg[r, :])
        tz[r, :] = _midrank(preds_sorted[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.atleast_2d(np.cov(v01))
    sy = np.atleast_2d(np.cov(v10))
    cov = sx / m + sy / n
    return aucs, cov


@dataclass
class DelongResult:
    auc_a: float
    auc_b: float
    z: float
    p_value: float          # two-sided H0: auc_a == auc_b
    ci95_diff: tuple[float, float]   # CI on (auc_a − auc_b)


def delong_paired(
    probs_a: list[float] | np.ndarray,
    probs_b: list[float] | np.ndarray,
    labels: list[int] | np.ndarray,
) -> DelongResult:
    """Paired DeLong test that two AUROCs (on the SAME trials/labels) differ.
    The correct test for graph-vs-baseline at small n: it accounts for the
    correlation between two predictors scored on identical samples."""
    y = np.asarray(labels)
    order = np.argsort(-y, kind="mergesort")  # positives (1) first
    m = int((y == 1).sum())
    preds = np.vstack([np.asarray(probs_a, float)[order], np.asarray(probs_b, float)[order]])
    aucs, cov = _fast_delong(preds, m)
    var_diff = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    var_diff = max(var_diff, 0.0)
    diff = float(aucs[0] - aucs[1])
    if var_diff <= 1e-18:
        z = 0.0 if abs(diff) < 1e-12 else np.sign(diff) * np.inf
        p = 1.0 if abs(diff) < 1e-12 else 0.0
        half = 0.0
    else:
        se = np.sqrt(var_diff)
        z = diff / se
        p = float(2 * stats.norm.sf(abs(z)))
        half = 1.959963985 * se
    return DelongResult(float(aucs[0]), float(aucs[1]), float(z), p, (diff - half, diff + half))


@dataclass
class McNemarResult:
    only_a_correct: int   # a right, b wrong
    only_b_correct: int   # a wrong, b right
    statistic: float
    p_value: float


def mcnemar_paired(
    correct_a: list[bool] | np.ndarray,
    correct_b: list[bool] | np.ndarray,
) -> McNemarResult:
    """Paired McNemar test on per-trial correctness (accuracy at a fixed threshold).
    Uses the exact binomial on the discordant pairs (right at small n)."""
    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    n01 = int(np.sum(a & ~b))
    n10 = int(np.sum(~a & b))
    nd = n01 + n10
    if nd == 0:
        return McNemarResult(n01, n10, 0.0, 1.0)
    # exact two-sided binomial test, H0: p = 0.5 on discordant pairs
    p = float(stats.binomtest(min(n01, n10), nd, 0.5, alternative="two-sided").pvalue)
    stat = (abs(n01 - n10) - 1) ** 2 / nd  # continuity-corrected chi-square (reported)
    return McNemarResult(n01, n10, float(stat), p)


# ── Recalibration (Platt / isotonic) — for the calibration tuning pass ──


def platt_fit(probs: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Fit Platt scaling sigmoid(a·logit(p)+b). Returns (a, b). Use on a
    validation fold, apply to the test fold (never fit on the test fold)."""
    p = np.clip(np.asarray(probs, float), 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1e6, solver="lbfgs")
    lr.fit(logit, np.asarray(labels, int))
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def platt_apply(probs: np.ndarray, a: float, b: float) -> np.ndarray:
    p = np.clip(np.asarray(probs, float), 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p))
    return 1.0 / (1.0 + np.exp(-(a * logit + b)))
