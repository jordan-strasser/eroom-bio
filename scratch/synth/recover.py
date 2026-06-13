"""EM/VBEM recovery for the synthetic eroom corpus (eroom-em-derivation.md §3-§5).

Runs the noisy-AND × safety-OR × leak EM loop in three modes and scores the
recovered posteriors against the planted ground truth from ``generate.py``:

  (a) unrouted            — every failure uses the full §3.1 responsibilities
                            ("unknown" branch); the reason field is ignored.
                            This is the pre-A3 baseline: one outcome bit smeared
                            across the whole backbone with no censoring.
  (b) routed + censored   — §3.2 competing-risks routing on the *true* reason.
      perfect reasons       Safety deaths censor efficacy; efficacy deaths blame
                            within Φ and credit safety survival; business/leak
                            censors everything.
  (c) routed, noisy       — mode (b) on the reason_noise-corrupted reasons.

Modes (a) and (b) share identical machinery and priors; the ONLY difference is
how a failed trial's branch is determined — so (b)-vs-(a) isolates the value of
routing, and (c)-vs-(b) isolates robustness to imperfect taxonomy extraction.

Metrics vs planted truth: per-edge MAE & Pearson corr, 90% credible-interval
coverage, convergence iterations, and held-out AUROC on fresh trials drawn from
the same planted params.  The headline experiment sweeps ``reuse_rate``.

Self-contained: depends only on numpy/scipy and the sibling ``generate.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import beta as beta_dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate as G  # noqa: E402
from generate import (  # noqa: E402
    GroundTruth, Corpus, Trial,
    SUCCESS, R_SAFETY, R_EFFICACY, R_BUSINESS, R_UNKNOWN,
)

_EPS = 1e-9

# Class-specific priors (§7): mildly optimistic must-holds, low-base-rate gates.
# Deliberately generic — they do NOT peek at planted values.  At high reuse the
# data overwhelms them (MAE→0); at low reuse the posterior reverts to them (the
# starvation signature), which is exactly what we want to measure.
PRIOR_MUST_MEAN = 0.70
PRIOR_GATE_MEAN = 0.10


# ---------------------------------------------------------------------------
# EM
# ---------------------------------------------------------------------------
def _init_state(gt: GroundTruth, prior_strength: float):
    """Beta(α,β) for must-holds, Beta(a,b) for gates, from class-mean priors."""
    s = prior_strength
    alpha0 = np.full(gt.n_must, PRIOR_MUST_MEAN * s)
    beta0 = np.full(gt.n_must, (1.0 - PRIOR_MUST_MEAN) * s)
    a0 = np.full(gt.n_gate, PRIOR_GATE_MEAN * s)
    b0 = np.full(gt.n_gate, (1.0 - PRIOR_GATE_MEAN) * s)
    return alpha0, beta0, a0, b0


def _precompute_success_counts(corpus: Corpus):
    """Successes are fully observed (§3): all must-holds held (+α), all gates
    survived (+b).  Constant across iterations — accumulate once."""
    gt = corpus.gt
    succ_alpha = np.zeros(gt.n_must)
    succ_gate_b = np.zeros(gt.n_gate)
    for t in corpus.trials:
        if t.y == 1:
            succ_alpha[t.must_idx] += 1.0
            succ_gate_b[t.gate_idx] += 1.0
    return succ_alpha, succ_gate_b


def _branch_for(trial: Trial, mode: str) -> str:
    """Which §3.2 branch this failed trial routes to, per recovery mode."""
    if mode == "a":
        return R_UNKNOWN
    reason = trial.reason if mode == "b" else trial.reason_obs
    return reason


def em_recover(corpus: Corpus, mode: str, prior_strength: float = 2.0,
               fix_leak: float | None = None, max_iter: int = 500,
               tol: float = 1e-7):
    """Run the §4-§5 EM loop. Returns recovered posteriors + diagnostics."""
    gt = corpus.gt
    leak = gt.leak if fix_leak is None else fix_leak  # §4: leak fixed
    leak = float(np.clip(leak, _EPS, 1 - _EPS))

    alpha0, beta0, a0, b0 = _init_state(gt, prior_strength)
    succ_alpha, succ_gate_b = _precompute_success_counts(corpus)

    alpha = alpha0 + succ_alpha
    beta = beta0.copy()
    a = a0.copy()
    b = b0 + succ_gate_b

    fails = [t for t in corpus.trials if t.y == 0]

    prev_mean = np.concatenate([alpha / (alpha + beta), a / (a + b)])
    iters = 0
    for it in range(1, max_iter + 1):
        iters = it
        rbar = np.clip(alpha / (alpha + beta), _EPS, 1 - _EPS)
        qbar = np.clip(a / (a + b), _EPS, 1 - _EPS)

        # M-step accumulators reset to priors + (fixed) success counts
        acc_alpha = alpha0 + succ_alpha
        acc_beta = beta0.copy()
        acc_a = a0.copy()
        acc_b = b0 + succ_gate_b

        for t in fails:
            mi, gi = t.must_idx, t.gate_idx
            phi = float(np.prod(rbar[mi]))
            s = float(np.prod(1.0 - qbar[gi]))
            pi = phi * s * leak
            branch = _branch_for(t, mode)

            if branch == R_SAFETY:
                # condition on "≥1 gate fired": split blame across gates, censor
                # efficacy/measurement entirely (no must-hold update), censor leak.
                denom = max(1.0 - s, _EPS)
                rho_kill = np.clip(qbar[gi] / denom, 0.0, 1.0)
                np.add.at(acc_a, gi, rho_kill)
                np.add.at(acc_b, gi, 1.0 - rho_kill)

            elif branch == R_EFFICACY:
                # all gates survived (informative about safety in the good
                # direction), leak ok; the miss is inside Φ — branch-local denom.
                np.add.at(acc_b, gi, 1.0)
                denom = max(1.0 - phi, _EPS)
                rho_fail = np.clip((1.0 - rbar[mi]) / denom, 0.0, 1.0)
                p_hold = np.clip((rbar[mi] - phi) / denom, 0.0, 1.0)
                np.add.at(acc_alpha, mi, p_hold)
                np.add.at(acc_beta, mi, rho_fail)

            elif branch == R_BUSINESS:
                # operational / strategic kill — censor everything (leak fixed).
                pass

            else:  # R_UNKNOWN — full marginal responsibilities (§3.1), no censor
                denom = max(1.0 - pi, _EPS)
                rho_fail = np.clip((1.0 - rbar[mi]) / denom, 0.0, 1.0)
                p_hold = np.clip((rbar[mi] - pi) / denom, 0.0, 1.0)
                np.add.at(acc_alpha, mi, p_hold)
                np.add.at(acc_beta, mi, rho_fail)
                rho_kill = np.clip(qbar[gi] / denom, 0.0, 1.0)
                np.add.at(acc_a, gi, rho_kill)
                np.add.at(acc_b, gi, 1.0 - rho_kill)

        alpha, beta, a, b = acc_alpha, acc_beta, acc_a, acc_b
        cur_mean = np.concatenate([alpha / (alpha + beta), a / (a + b)])
        delta = float(np.max(np.abs(cur_mean - prev_mean)))
        prev_mean = cur_mean
        if delta < tol:
            break

    return dict(alpha=alpha, beta=beta, a=a, b=b, leak=leak, iters=iters,
                last_delta=delta)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def auroc(y, scores) -> float:
    """Rank-based AUROC (Mann-Whitney), tie-safe. NaN if one class absent."""
    y = np.asarray(y)
    scores = np.asarray(scores, dtype=float)
    pos = scores[y == 1]
    neg = scores[y == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks within ties
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2.0
            ranks[order[i:j + 1]] = avg
        i = j + 1
    r_pos = ranks[y == 1].sum()
    n_pos, n_neg = pos.size, neg.size
    return float((r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def recovered_means(res):
    rbar = res["alpha"] / (res["alpha"] + res["beta"])
    qbar = res["a"] / (res["a"] + res["b"])
    return rbar, qbar


def predict(gt: GroundTruth, res, test_trials, train_obs_must, train_obs_gate):
    """π̂ = ∏ r̄ · ∏(1-q̄) · ℓ. Unseen edges back off to the recovered class
    mean over observed edges of that class (§6 hierarchical class prior)."""
    rbar, qbar = recovered_means(res)
    leak = res["leak"]

    # class-mean backoff for edges never observed in training
    rbar_eff = rbar.copy()
    for cls in ("spine", "triangle", "pop"):
        m = (gt.must_class == cls)
        seen = m & train_obs_must
        backoff = rbar[seen].mean() if seen.any() else PRIOR_MUST_MEAN
        rbar_eff[m & ~train_obs_must] = backoff
    qbar_eff = qbar.copy()
    if (~train_obs_gate).any():
        seen_q = train_obs_gate
        backoff_q = qbar[seen_q].mean() if seen_q.any() else PRIOR_GATE_MEAN
        qbar_eff[~train_obs_gate] = backoff_q

    scores = np.empty(len(test_trials))
    ys = np.empty(len(test_trials), dtype=int)
    for i, t in enumerate(test_trials):
        phi = np.prod(rbar_eff[t.must_idx])
        s = np.prod(1.0 - qbar_eff[t.gate_idx])
        scores[i] = phi * s * leak
        ys[i] = t.y
    return ys, scores


def score_recovery(corpus: Corpus, res, test_trials, ci=0.90):
    gt = corpus.gt
    rbar, qbar = recovered_means(res)
    inc_m = corpus.must_incidence()
    inc_g = corpus.gate_incidence()
    obs_m = inc_m > 0
    obs_g = inc_g > 0

    # per-edge MAE / corr over OBSERVED edges (the recoverable ones)
    r_true = gt.r_must[obs_m]
    r_hat = rbar[obs_m]
    must_mae = float(np.mean(np.abs(r_hat - r_true))) if obs_m.any() else float("nan")
    must_corr = (float(np.corrcoef(r_hat, r_true)[0, 1])
                 if obs_m.sum() > 2 and np.std(r_hat) > 0 else float("nan"))

    q_true = gt.q_gate[obs_g]
    q_hat = qbar[obs_g]
    gate_mae = float(np.mean(np.abs(q_hat - q_true))) if obs_g.any() else float("nan")

    # 90% central credible-interval coverage of the planted value (must-holds)
    lo = beta_dist.ppf((1 - ci) / 2, res["alpha"][obs_m], res["beta"][obs_m])
    hi = beta_dist.ppf(1 - (1 - ci) / 2, res["alpha"][obs_m], res["beta"][obs_m])
    coverage = float(np.mean((gt.r_must[obs_m] >= lo) & (gt.r_must[obs_m] <= hi)))

    # held-out AUROC + correlation of π̂ with the TRUE π (composite recovery:
    # less noise-limited than AUROC, rises 0 -> 1 as edges leave their priors)
    ys, scores = predict(gt, res, test_trials, obs_m, obs_g)
    au = auroc(ys, scores)
    pi_true = np.array([np.prod(gt.r_must[t.must_idx]) *
                        np.prod(1.0 - gt.q_gate[t.gate_idx]) * gt.leak
                        for t in test_trials])
    pi_corr = (float(np.corrcoef(scores, pi_true)[0, 1])
               if np.std(scores) > 1e-9 else 0.0)

    # downward bias of must-hold estimates (the contamination signature)
    must_bias = float(np.mean(r_hat - r_true)) if obs_m.any() else float("nan")

    return dict(
        must_mae=must_mae, must_corr=must_corr, gate_mae=gate_mae,
        coverage=coverage, auroc=au, pi_corr=pi_corr, iters=res["iters"],
        must_bias=must_bias, n_obs_must=int(obs_m.sum()),
        test_pos_rate=float(np.mean(ys)),
    )


# ---------------------------------------------------------------------------
# experiment drivers
# ---------------------------------------------------------------------------
def oracle_auroc(gt: GroundTruth, test_trials) -> float:
    """AUROC achievable with PERFECT knowledge of the planted params. This is
    the model's intrinsic discrimination ceiling — the irreducible Bernoulli
    noise floor. Recovered-param AUROC can only ever approach this, never 1.0."""
    ys = np.array([t.y for t in test_trials])
    sc = np.array([np.prod(gt.r_must[t.must_idx]) *
                   np.prod(1.0 - gt.q_gate[t.gate_idx]) * gt.leak
                   for t in test_trials])
    return auroc(ys, sc)


def run_cell(seed, reuse, modes, n_trials, n_test, reason_noise,
             prior_strength, corruption, frac_nonmech):
    """One (reuse, seed) ground truth → train + test corpora → all modes."""
    gt = G.make_ground_truth(
        seed, n_trials=n_trials, reuse_rate=reuse, reason_noise=reason_noise,
        corruption=corruption, frac_nonmech_failure=frac_nonmech,
    )
    train = Corpus(gt=gt, trials=G.sample_trials(
        gt, n_trials, seed=seed + 10_000, reason_noise=reason_noise,
        corruption=corruption))
    test = G.sample_trials(gt, n_test, seed=seed + 20_000, reason_noise=0.0,
                           corruption=corruption)
    out = {}
    for mode in modes:
        res = em_recover(train, mode, prior_strength=prior_strength)
        m = score_recovery(train, res, test)
        m["oracle"] = oracle_auroc(gt, test)
        out[mode] = m
    return out


def aggregate(cells, modes, metric):
    """cells: list of {mode: metrics}. Return {mode: (mean, std)} for `metric`."""
    agg = {}
    for mode in modes:
        vals = np.array([c[mode][metric] for c in cells if not np.isnan(c[mode][metric])])
        agg[mode] = (float(vals.mean()) if vals.size else float("nan"),
                     float(vals.std()) if vals.size else float("nan"))
    return agg


def reuse_sweep(reuse_rates, modes, n_seeds, n_trials, n_test,
                reason_noise_c, prior_strength, corruption, frac_nonmech):
    """Headline experiment: AUROC & MAE vs reuse_rate for each mode."""
    results = {}
    for reuse in reuse_rates:
        cells = [run_cell(seed, reuse, modes, n_trials, n_test, reason_noise_c,
                          prior_strength, corruption, frac_nonmech)
                 for seed in range(n_seeds)]
        results[reuse] = {
            "auroc": aggregate(cells, modes, "auroc"),
            "must_mae": aggregate(cells, modes, "must_mae"),
            "must_corr": aggregate(cells, modes, "must_corr"),
            "coverage": aggregate(cells, modes, "coverage"),
            "iters": aggregate(cells, modes, "iters"),
            "must_bias": aggregate(cells, modes, "must_bias"),
            "pi_corr": aggregate(cells, modes, "pi_corr"),
            "test_pos_rate": aggregate(cells, modes, "test_pos_rate"),
            # oracle is mode-independent; take it off the first mode
            "oracle": aggregate(cells, modes[:1], "oracle")[modes[0]],
        }
    return results


def noise_sweep(noises, reuse, n_seeds, n_trials, n_test, prior_strength,
                corruption, frac_nonmech):
    """Hold reuse fixed; sweep reason_noise for mode (c). Compare to (a)/(b)."""
    results = {}
    for noise in noises:
        cells = [run_cell(seed, reuse, ["a", "b", "c"], n_trials, n_test, noise,
                          prior_strength, corruption, frac_nonmech)
                 for seed in range(n_seeds)]
        results[noise] = {
            "auroc": aggregate(cells, ["a", "b", "c"], "auroc"),
            "must_mae": aggregate(cells, ["a", "b", "c"], "must_mae"),
            "pi_corr": aggregate(cells, ["a", "b", "c"], "pi_corr"),
            "must_bias": aggregate(cells, ["a", "b", "c"], "must_bias"),
        }
    return results


# ---------------------------------------------------------------------------
# reporting helpers (ASCII — no matplotlib in this env)
# ---------------------------------------------------------------------------
def _fmt(mean_std, p=3):
    m, s = mean_std
    if np.isnan(m):
        return "   n/a "
    return f"{m:.{p}f}±{s:.{p}f}"


def ascii_curve(xs, series, title, ymin=None, ymax=None, width=51, height=12):
    """series: dict label -> list of y (aligned to xs). Returns a text block."""
    ys_all = [y for s in series.values() for y in s if not np.isnan(y)]
    if not ys_all:
        return title + "\n(no data)\n"
    ymin = min(ys_all) if ymin is None else ymin
    ymax = max(ys_all) if ymax is None else ymax
    if ymax - ymin < 1e-9:
        ymax = ymin + 1e-9
    marks = {"a": "a", "b": "b", "c": "c"}
    grid = [[" "] * width for _ in range(height)]
    n = len(xs)
    for label, ys in series.items():
        ch = marks.get(label, label[0])
        for i, y in enumerate(ys):
            if np.isnan(y):
                continue
            col = int(round(i * (width - 1) / max(1, n - 1)))
            row = int(round((ymax - y) / (ymax - ymin) * (height - 1)))
            row = min(max(row, 0), height - 1)
            grid[row][col] = ch
    lines = [title]
    for r, row in enumerate(grid):
        yval = ymax - r * (ymax - ymin) / (height - 1)
        lines.append(f"{yval:6.3f} |" + "".join(row))
    axis = "       +" + "-" * width
    lines.append(axis)
    xlabels = "        " + "".join(
        f"{x:<{max(1, width // n)}.2g}"[:max(1, width // n)] for x in xs)
    lines.append(xlabels)
    return "\n".join(lines) + "\n"


def print_recovery_table(results, reuse, modes):
    print(f"\n### Recovery table @ reuse_rate = {reuse}  (mean±sd over seeds)")
    cell = results[reuse]
    cols = [("AUROC", "auroc"), ("must MAE", "must_mae"), ("must corr", "must_corr"),
            ("90% cover", "coverage"), ("must bias", "must_bias"), ("iters", "iters")]
    header = f"{'mode':<22}" + "".join(f"{c[0]:>14}" for c in cols)
    print(header)
    print("-" * len(header))
    names = {"a": "(a) unrouted", "b": "(b) routed/perfect", "c": "(c) routed/noisy"}
    for mode in modes:
        row = f"{names.get(mode, mode):<22}"
        for _, key in cols:
            p = 1 if key == "iters" else 3
            row += f"{_fmt(cell[key][mode], p):>14}"
        print(row)


def em_correctness_unit_test(seeds=4):
    """Isolated proof the EM is implemented correctly: a hand-built corpus where
    y ~ Bernoulli(r_a) DIRECTLY (one must-hold/trial, no gates, leak=1) so each
    edge's reliability is exactly identifiable. The EM must recover planted r to
    the sampling-noise floor as observations per edge grow. (Sanity check #1.)"""
    rows = []
    for per_edge in [16, 64, 256, 1024]:
        maes, corrs = [], []
        for seed in range(seeds):
            rng = np.random.default_rng(seed)
            n_edges = 200
            r = rng.beta(4, 2, size=n_edges)
            gt = GroundTruth(r_must=r, must_class=np.array(["spine"] * n_edges),
                             n_spine=n_edges, n_tri=0, n_pop=0,
                             q_gate=np.zeros(0), leak=1.0,
                             cfg=dict(n_spine_per_trial=1, n_gate_per_trial=0,
                                      n_trials=n_edges * per_edge, leak=1.0))
            trials = []
            for a in range(n_edges):
                for _ in range(per_edge):
                    y = int(rng.random() < r[a])
                    reason = SUCCESS if y else R_EFFICACY
                    trials.append(Trial(must_idx=np.array([a]),
                                        gate_idx=np.zeros(0, dtype=int),
                                        y=y, reason=reason, reason_obs=reason))
            res = em_recover(Corpus(gt=gt, trials=trials), "b", prior_strength=2.0)
            rhat = res["alpha"] / (res["alpha"] + res["beta"])
            maes.append(float(np.mean(np.abs(rhat - r))))
            corrs.append(float(np.corrcoef(rhat, r)[0, 1]))
        # expected sampling SE of a proportion from per_edge Bernoulli draws
        se = float(np.mean(np.sqrt(r * (1 - r) / per_edge)))
        rows.append(dict(per_edge=per_edge, mae=np.mean(maes), corr=np.mean(corrs),
                         sampling_se=se))
    return rows


def main():
    ap = argparse.ArgumentParser(description="EM/VBEM recovery + sweeps.")
    ap.add_argument("--mode", choices=["sweep", "single", "noise", "all"],
                    default="all")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--n-trials", type=int, default=500)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--reuse", type=float, default=1.24)
    ap.add_argument("--reason-noise-c", type=float, default=0.30,
                    help="reason_noise level used for mode (c) in the reuse sweep")
    ap.add_argument("--prior-strength", type=float, default=2.0)
    ap.add_argument("--frac-nonmech", type=float, default=0.10)
    ap.add_argument("--corruption", choices=["to_unknown", "adversarial"],
                    default="to_unknown")
    ap.add_argument("--out", type=str, default="sweep_results.json")
    args = ap.parse_args()

    modes = ["a", "b", "c"]
    reuse_rates = [0.5, 1.0, 1.24, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
    noises = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    blob = {"config": vars(args)}

    if args.mode in ("sweep", "all"):
        print("== EM correctness unit test (y ~ Bern(r); sanity #1) ==", flush=True)
        ut = em_correctness_unit_test()
        print(f"{'obs/edge':>9}{'recovered MAE':>15}{'sampling SE':>14}{'corr':>8}")
        for row in ut:
            print(f"{row['per_edge']:>9}{row['mae']:>15.4f}"
                  f"{row['sampling_se']:>14.4f}{row['corr']:>8.3f}")
        ut_pass = ut[-1]["mae"] < 0.03 and ut[-1]["corr"] > 0.95
        print(f"[{'PASS' if ut_pass else 'FAIL'}] EM recovers planted r to "
              f"sampling-noise floor in the identifiable limit "
              f"(MAE@1024={ut[-1]['mae']:.4f}, corr={ut[-1]['corr']:.3f})")
        blob["unit_test"] = ut
        blob["unit_test_pass"] = bool(ut_pass)

    if args.mode in ("sweep", "all"):
        print("== Reuse-rate sweep ==", flush=True)
        rs = reuse_sweep(reuse_rates, modes, args.seeds, args.n_trials,
                         args.n_test, args.reason_noise_c, args.prior_strength,
                         args.corruption, args.frac_nonmech)
        blob["reuse_sweep"] = {str(k): v for k, v in rs.items()}

        # tables
        print("\n=== AUROC vs reuse_rate (mean±sd)   [base rate 0.500] ===")
        hdr = (f"{'reuse':>7}" + "".join(f"{'('+m+')':>14}" for m in modes)
               + f"{'oracle':>14}")
        print(hdr)
        for reuse in reuse_rates:
            row = f"{reuse:>7}"
            for m in modes:
                row += f"{_fmt(rs[reuse]['auroc'][m]):>14}"
            row += f"{_fmt(rs[reuse]['oracle']):>14}"
            print(row + ("   <- corpus" if reuse == 1.24 else ""))
        print("\n=== must-hold MAE vs reuse_rate (mean±sd) ===")
        print(hdr)
        for reuse in reuse_rates:
            row = f"{reuse:>7}"
            for m in modes:
                row += f"{_fmt(rs[reuse]['must_mae'][m]):>14}"
            print(row + ("   <- corpus" if reuse == 1.24 else ""))
        print("\n=== composite recovery corr(π̂, π_true) vs reuse_rate (mean±sd) ===")
        print(hdr)
        for reuse in reuse_rates:
            row = f"{reuse:>7}"
            for m in modes:
                row += f"{_fmt(rs[reuse]['pi_corr'][m]):>14}"
            print(row + ("   <- corpus" if reuse == 1.24 else ""))

        # ASCII curves
        au_series = {m: [rs[r]["auroc"][m][0] for r in reuse_rates] for m in modes}
        au_series["O"] = [rs[r]["oracle"][0] for r in reuse_rates]  # ceiling
        print()
        print(ascii_curve(
            reuse_rates, au_series,
            "AUROC vs reuse_rate  (a=unrouted b=routed/perfect c=routed/noisy "
            "O=oracle-ceiling; 1.24=corpus)", ymin=0.45, ymax=0.70))
        print(ascii_curve(
            reuse_rates,
            {m: [rs[r]["must_mae"][m][0] for r in reuse_rates] for m in modes},
            "must-hold MAE vs reuse_rate  (lower=better recovery)"))

        print_recovery_table(rs, 1.24, modes)

        # ---- what routing buys (b vs a), per reuse ----
        print("\n=== Value of routing (mode b - mode a), per reuse ===")
        print(f"{'reuse':>7}{'dAUROC(b-a)':>14}{'dMAE(a-b)':>12}"
              f"{'|bias|a':>10}{'|bias|b':>10}")
        for reuse in reuse_rates:
            d_au = rs[reuse]["auroc"]["b"][0] - rs[reuse]["auroc"]["a"][0]
            d_mae = rs[reuse]["must_mae"]["a"][0] - rs[reuse]["must_mae"]["b"][0]
            ba = abs(rs[reuse]["must_bias"]["a"][0])
            bb = abs(rs[reuse]["must_bias"]["b"][0])
            print(f"{reuse:>7}{d_au:>+14.3f}{d_mae:>+12.3f}{ba:>10.3f}{bb:>10.3f}"
                  + ("   <- corpus" if reuse == 1.24 else ""))

        # ---- sanity checks ----
        print("\n=== SANITY CHECKS ===")
        # (1) EM correctness — proven by the identifiable-limit unit test above
        ut = blob.get("unit_test", [{}])
        ut_pass = blob.get("unit_test_pass", False)
        print(f"[{'PASS' if ut_pass else 'FAIL'}] (1) EM recovers planted r to the "
              f"sampling-noise floor in the identifiable limit "
              f"(unit-test MAE@1024={ut[-1].get('mae', float('nan')):.4f}, "
              f"corr={ut[-1].get('corr', float('nan')):.3f})")
        # full-model recovery must IMPROVE monotonically with reuse (corr up,
        # MAE down) — the data-hungry but correct behaviour of the noisy-AND EM.
        corr_lo = rs[min(reuse_rates)]["must_corr"]["b"][0]
        corr_hi = rs[max(reuse_rates)]["must_corr"]["b"][0]
        s1b = corr_hi > corr_lo + 0.20
        print(f"      full-model mode-b recovery corr rises with reuse: "
              f"{corr_lo:.3f} (reuse {min(reuse_rates):g}) -> "
              f"{corr_hi:.3f} (reuse {max(reuse_rates):g})  [{'ok' if s1b else 'X'}]")
        # (2) routing never hurts recovery: MAE(b)<=MAE(a) and |bias(b)|<=|bias(a)|.
        #     AUROC is noise-limited near the oracle ceiling, so recovery metrics
        #     (MAE, bias) are the faithful test of "routing helps".
        s2 = True
        detail = []
        for reuse in reuse_rates:
            mae_a = rs[reuse]["must_mae"]["a"][0]
            mae_b = rs[reuse]["must_mae"]["b"][0]
            ba = abs(rs[reuse]["must_bias"]["a"][0])
            bb = abs(rs[reuse]["must_bias"]["b"][0])
            ok = (mae_b <= mae_a + 0.005) and (bb <= ba + 0.005)
            s2 = s2 and ok
            detail.append(f"r={reuse:>5}: MAE b{mae_b:.3f}<=a{mae_a:.3f}, "
                          f"|bias| b{bb:.3f}<=a{ba:.3f} {'ok' if ok else 'X'}")
        print(f"[{'PASS' if s2 else 'FAIL'}] (2) routing never hurts recovery "
              f"(MAE & |bias|: b<=a at every reuse)")
        for d in detail:
            print("        " + d)
        blob["sanity_reuse"] = dict(check1_em_correct=bool(ut_pass),
                                    check1b_recovery_rises=bool(s1b),
                                    check2_routing_helps=bool(s2))

    if args.mode in ("noise", "all"):
        print("\n== Reason-noise sweep @ reuse=%.2f ==" % args.reuse, flush=True)
        ns = noise_sweep(noises, args.reuse, args.seeds, args.n_trials,
                         args.n_test, args.prior_strength, args.corruption,
                         args.frac_nonmech)
        blob["noise_sweep"] = {str(k): v for k, v in ns.items()}
        hdr = f"{'noise':>7}" + "".join(f"{'('+m+')':>14}" for m in ["a", "b", "c"])
        for metric, label in [("auroc", "AUROC"),
                              ("pi_corr", "composite recovery corr(π̂,π_true)")]:
            print(f"\n=== {label} vs reason_noise @ reuse={args.reuse} "
                  f"(corruption={args.corruption}) ===")
            print(hdr)
            for noise in noises:
                row = f"{noise:>7}"
                for m in ["a", "b", "c"]:
                    row += f"{_fmt(ns[noise][metric][m]):>14}"
                print(row)
        print()
        print(ascii_curve(
            noises,
            {"a": [ns[n]["pi_corr"]["a"][0] for n in noises],
             "b": [ns[n]["pi_corr"]["b"][0] for n in noises],
             "c": [ns[n]["pi_corr"]["c"][0] for n in noises]},
            f"composite recovery corr vs reason_noise @ reuse={args.reuse}  "
            f"(c falls from b toward a as noise->1)"))

        # sanity 3: c -> a as noise -> 1
        au_c1 = ns[1.0]["auroc"]["c"][0]
        au_a1 = ns[1.0]["auroc"]["a"][0]
        s3 = abs(au_c1 - au_a1) < 0.02
        print(f"[{'PASS' if s3 else 'FAIL'}] (3) mode (c) -> (a) as noise->1 "
              f"-> AUROC c@1.0={au_c1:.3f} vs a@1.0={au_a1:.3f}")
        blob["sanity_noise"] = dict(c_at_1=au_c1, a_at_1=au_a1, check3_pass=bool(s3))

    if args.mode == "single":
        print(f"== Single config: reuse={args.reuse}, "
              f"reason_noise(c)={args.reason_noise_c} ==", flush=True)
        cells = [run_cell(seed, args.reuse, modes, args.n_trials, args.n_test,
                          args.reason_noise_c, args.prior_strength,
                          args.corruption, args.frac_nonmech)
                 for seed in range(args.seeds)]
        single = {"auroc": aggregate(cells, modes, "auroc"),
                  "must_mae": aggregate(cells, modes, "must_mae"),
                  "must_corr": aggregate(cells, modes, "must_corr"),
                  "coverage": aggregate(cells, modes, "coverage"),
                  "must_bias": aggregate(cells, modes, "must_bias"),
                  "iters": aggregate(cells, modes, "iters")}
        print_recovery_table({args.reuse: single}, args.reuse, modes)
        blob["single"] = single

    with open(args.out, "w") as fh:
        json.dump(blob, fh, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
