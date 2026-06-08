"""Get to the bottom of the pessimism: sweep the COMPOSITION knobs.

The Beta-Binomial edge-update tuning was a near-no-op (scripts/tune_beliefs.py),
so the systematic pessimism (preds of 0.25-0.45 → 65-80% success) must live
DOWNSTREAM of the edge beliefs — in how they're composed into a chain probability
and dragged by safety. This sweeps, on the SAME delta-adjust K-fold holdout:

  - softmin temperature `_SOFTMIN_T` (→0 = hard weakest-link; larger = toward mean)
  - aggregation `EROOM_AGG` (softmin / geomean / harmonic)
  - safety penalty cap (incl. OFF, to isolate safety's contribution)
  - informed prior (on/off, mean) for under-evidenced edges
  - **concentration-cap** (owner idea): when α+β>C, scale both by C/(α+β) — preserves
    the mean EXACTLY, caps overconfidence. Tested here because it interacts with the
    softmin (a wider weak edge isn't always the min).

One-factor-at-a-time off a default, + a couple combos, to attribute the pessimism.
Holdout reuses the validated delta-adjust (corr 0.98 vs re-attribution). Diagnostic
metrics are the full 5-fold holdout; the chosen winner is then confirmed val→test.
"""
from __future__ import annotations

import argparse
import os
from collections import defaultdict

import numpy as np

import src.prediction.path_query as pq
from src.graph.store import GraphStore
from src.inference.belief_tuning import DEFAULT_CONFIG, extract_contribs, retune_from_contribs
from src.prediction.calibration import brier_score, expected_calibration_error, reliability_table
from src.prediction.calibration import auroc as _auroc
from src.prediction.path_query import PredictionEngine
from scripts.eval_baselines import build_trials
from scripts.eval_holdout_compose import _binary_accuracy
from scripts.eval_holdout_kfold import _fold
from scripts.tune_beliefs import _predict


def collect_all(G):
    """All edges carrying a belief: (belief_dict, alpha0, beta0, contribs)."""
    out = []
    for _u, _v, _k, data in G.edges(keys=True, data=True):
        b = data.get("belief")
        if not b:
            continue
        out.append((b, float(b["alpha"]), float(b["beta"]),
                    extract_contribs(b.get("evidence") or [])))
    return out


def _cap(a, b, C):
    s = a + b
    if C and s > C:
        f = C / s
        return max(a * f, 1e-6), max(b * f, 1e-6)
    return a, b


def _patch(edges, drop, conc_cap):
    for b, a0, b0, contribs in edges:
        a, bb = retune_from_contribs(a0, b0, contribs, DEFAULT_CONFIG, drop)
        if conc_cap:
            a, bb = _cap(a, bb, conc_cap)
        b["alpha"], b["beta"] = a, bb


def _restore(edges):
    for b, a0, b0, _ in edges:
        b["alpha"], b["beta"] = a0, b0


# Default composition (current production knobs).
DEFAULT = dict(softmin_t=0.10, agg="softmin", informed=True, prior_mean=0.75,
               safety_cap=0.60, conc_cap=None, dlt_floor=0.15, min_belief=0.55)


def _apply(cfg):
    """Monkeypatch path_query to a composition config; returns a restore thunk."""
    saved = (pq._SOFTMIN_T, pq._INFORMED_PRIOR_A, pq._INFORMED_PRIOR_B,
             pq._SAFETY_DLT_FLOOR, PredictionEngine._SAFETY_PENALTY_CAP,
             PredictionEngine._SAFETY_PENALTY_MIN_BELIEF,
             os.environ.get("EROOM_AGG"), os.environ.get("EROOM_INFORMED_PRIOR"))
    pq._SOFTMIN_T = cfg["softmin_t"]
    pq._INFORMED_PRIOR_A = cfg["prior_mean"] * 2.0
    pq._INFORMED_PRIOR_B = (1.0 - cfg["prior_mean"]) * 2.0
    pq._SAFETY_DLT_FLOOR = cfg["dlt_floor"]
    PredictionEngine._SAFETY_PENALTY_CAP = cfg["safety_cap"]
    PredictionEngine._SAFETY_PENALTY_MIN_BELIEF = cfg["min_belief"]
    os.environ["EROOM_AGG"] = cfg["agg"]
    os.environ["EROOM_INFORMED_PRIOR"] = "1" if cfg["informed"] else "0"

    def restore():
        (pq._SOFTMIN_T, pq._INFORMED_PRIOR_A, pq._INFORMED_PRIOR_B,
         pq._SAFETY_DLT_FLOOR, PredictionEngine._SAFETY_PENALTY_CAP,
         PredictionEngine._SAFETY_PENALTY_MIN_BELIEF, agg, inf) = saved
        if agg is None:
            os.environ.pop("EROOM_AGG", None)
        else:
            os.environ["EROOM_AGG"] = agg
        if inf is None:
            os.environ.pop("EROOM_INFORMED_PRIOR", None)
        else:
            os.environ["EROOM_INFORMED_PRIOR"] = inf
    return restore


def holdout(g, edges, eval_trials, cfg, k, n_samples, fold_subset=None):
    by_fold = defaultdict(list)
    for t in eval_trials:
        by_fold[_fold(t.nct, k)].append(t)
    restore = _apply(cfg)
    preds = {}
    try:
        for f in sorted(by_fold):
            if fold_subset is not None and f not in fold_subset:
                continue
            rows = by_fold[f]
            _patch(edges, frozenset(t.nct for t in rows), cfg["conc_cap"])
            preds.update(_predict(g, rows, n_samples))
            _restore(edges)
    finally:
        restore()
    return preds


def _score(preds, by_nct):
    items = [(p, by_nct[n].y) for n, p in preds.items() if p is not None and n in by_nct]
    if len(items) < 4:
        return None
    p = [x for x, _ in items]
    y = [yy for _, yy in items]
    if len(set(y)) < 2:
        return None
    return dict(n=len(y), brier=brier_score(p, y), ece=expected_calibration_error(p, y),
                auroc=_auroc(p, y), acc=_binary_accuracy(p, y, 0.5)[0],
                mean_pred=float(np.mean(p)))


def make_grid():
    """Focused on the diagnosis: safety made SPECIFIC (kept ON), vs the OFF
    ceiling and the default floor. dlt_floor=0.0 ⇒ only failure-causing tox
    penalizes (tolerated AEs contribute nothing — kills the soft-or
    over-accumulation); min_belief↑ ⇒ fewer AEs clear the gate."""
    g = {"default": dict(DEFAULT)}
    g["safety=OFF (ceiling)"] = {**DEFAULT, "safety_cap": 0.0}
    g["agg=geomean"] = {**DEFAULT, "agg": "geomean"}
    g["dlt_floor=0.0"] = {**DEFAULT, "dlt_floor": 0.0}
    g["min_belief=0.65"] = {**DEFAULT, "min_belief": 0.65}
    g["min_belief=0.75"] = {**DEFAULT, "min_belief": 0.75}
    g["dlt0.0+mb0.65"] = {**DEFAULT, "dlt_floor": 0.0, "min_belief": 0.65}
    g["dlt0.0+mb0.75"] = {**DEFAULT, "dlt_floor": 0.0, "min_belief": 0.75}
    g["geomean+dlt0.0"] = {**DEFAULT, "agg": "geomean", "dlt_floor": 0.0}
    g["geomean+dlt0.0+mb0.65"] = {
        **DEFAULT, "agg": "geomean", "dlt_floor": 0.0, "min_belief": 0.65}
    return g


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--holdout-corpus", default=None)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--min-overlap", type=int, default=5)
    ap.add_argument("--n-samples", type=int, default=2000)
    ap.add_argument("--val-folds", default="0,1,2")
    ap.add_argument("--test-folds", default="3,4")
    args = ap.parse_args()

    np.random.seed(42)
    g = GraphStore()
    g.import_snapshot(args.graph)
    edges = collect_all(g._graph)  # noqa: SLF001
    trials = build_trials(g, args.corpus, args.holdout_corpus, args.min_overlap)
    eval_trials = [t for t in trials if t.is_eval]
    by_nct = {t.nct: t for t in eval_trials}
    base = np.mean([t.y for t in eval_trials])
    print(f"belief edges: {len(edges)} | scored slice: {len(eval_trials)} "
          f"(base rate {base:.3f})")
    print("\n── Composition sweep — FULL 5-fold delta-adjust holdout (diagnostic) ──")
    print(f"  {'config':<24}{'Brier':>8}{'ECE':>7}{'AUROC':>8}{'Acc':>7}{'mean_p':>8}")
    grid = make_grid()
    full = {}
    for name, cfg in grid.items():
        s = _score(holdout(g, edges, eval_trials, cfg, args.k, args.n_samples), by_nct)
        if not s:
            continue
        full[name] = s
        print(f"  {name:<24}{s['brier']:>8.3f}{s['ece']:>7.3f}{s['auroc']:>8.3f}"
              f"{s['acc']:>7.3f}{s['mean_pred']:>8.3f}")
    print(f"  (base rate {base:.3f}; mean_p << base ⇒ pessimistic. Brier↓ ECE↓ = better calibrated)")

    # Honest val→test for the best-by-val-Brier config.
    val = {int(x) for x in args.val_folds.split(",")}
    test = {int(x) for x in args.test_folds.split(",")}
    val_scores = {n: _score(holdout(g, edges, eval_trials, grid[n], args.k, args.n_samples, val), by_nct)
                  for n in grid}
    val_scores = {n: s for n, s in val_scores.items() if s}
    best = min(val_scores, key=lambda n: val_scores[n]["brier"])
    print(f"\n── Honest val→test (selection on val folds {sorted(val)}, scored once on test) ──")
    print(f"  best-on-val (min Brier): {best!r}")
    print(f"  {'config':<24}{'Brier':>8}{'ECE':>7}{'AUROC':>8}{'Acc':>7}")
    for name in ("default", best):
        s = _score(holdout(g, edges, eval_trials, grid[name], args.k, args.n_samples, test), by_nct)
        if s:
            print(f"  {name:<24}{s['brier']:>8.3f}{s['ece']:>7.3f}{s['auroc']:>8.3f}{s['acc']:>7.3f}")

    # Reliability of default vs winner (full holdout) — the pessimism diagram.
    for name in ("default", best):
        preds = holdout(g, edges, eval_trials, grid[name], args.k, args.n_samples)
        items = [(p, by_nct[n].y) for n, p in preds.items() if p is not None]
        print(f"\n  reliability — {name!r} (full holdout):")
        for b in reliability_table([p for p, _ in items], [y for _, y in items], 10):
            if b.count:
                print(f"    [{b.lo:.1f},{b.hi:.1f})  n={b.count:>3}  pred={b.mean_pred:.3f}  obs={b.frac_pos:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
