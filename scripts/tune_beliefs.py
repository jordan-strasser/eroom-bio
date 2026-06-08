"""Tune the Beta-Binomial edge-update parameters (n_eff, p_obs) — the owner's focus.

WHY this is fast: each attributor record persists `n_eff_applied`/`p_obs_applied`
in context, and the conjugate update is additive, so a candidate config's edge
beliefs are a DELTA-ADJUST of the stored ones — no re-attribution
(src/inference/belief_tuning.py). A K-fold holdout = drop a fold's trials'
records and re-predict; this reproduces eval_holdout_kfold's re-attribution
holdout (validated below against its dumped predictions).

DISCIPLINE (BENCHMARK.md): config is selected on VALIDATION folds only; the TEST
folds are scored once with the chosen config. Calibration first (Brier/ECE),
AUROC second. The winning config is then CONFIRMED by a true re-attribution
(eval_holdout_kfold) — delta-adjust holds the explain-away feedback fixed
(second-order), so the cheap sweep finds the config and re-attribution certifies it.

Usage:
    python -m scripts.tune_beliefs \
        --graph data/exports/onco_scale_500_annotated.json \
        --corpus onco_scale_500 --holdout-corpus onco_scale_248_add \
        --validate-against /tmp/graph_preds_n500.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.graph.store import GraphStore
from src.inference.belief_tuning import (
    BeliefTuneConfig,
    extract_contribs,
    retune_from_contribs,
)
from src.prediction.calibration import (
    brier_score,
    expected_calibration_error,
    reliability_table,
)
from src.prediction.calibration import auroc as _auroc
from src.prediction.path_query import predict_clinical_hypothesis
from scripts.eval_baselines import build_trials
from scripts.eval_holdout_compose import _binary_accuracy
from scripts.eval_holdout_kfold import _fold


def collect_tunable(G):
    """[(belief_dict, alpha0, beta0, contribs)] for every edge that carries
    delta-tunable (attributor-applied) records."""
    out = []
    for _u, _v, _k, data in G.edges(keys=True, data=True):
        b = data.get("belief")
        if not b:
            continue
        contribs = extract_contribs(b.get("evidence") or [])
        if contribs:
            out.append((b, float(b["alpha"]), float(b["beta"]), contribs))
    return out


def _patch(edges, cfg, drop):
    for b, a0, b0, contribs in edges:
        a, bb = retune_from_contribs(a0, b0, contribs, cfg, drop)
        b["alpha"], b["beta"] = a, bb


def _restore(edges):
    for b, a0, b0, _ in edges:
        b["alpha"], b["beta"] = a0, b0


def _predict(g, trials, n_samples):
    out = {}
    for t in trials:
        try:
            r = predict_clinical_hypothesis(
                g, t.chain["compound_id"], t.chain["indication_id"],
                n_samples=n_samples, **t.kwargs,
            )
            out[t.nct] = r.overall_probability if r.edge_contributions else None
        except KeyError:
            out[t.nct] = None
    return out


def holdout_preds(g, edges, eval_trials, cfg, k, n_samples, fold_subset=None):
    """K-fold delta-adjust holdout: for each fold, drop its trials' records, predict
    them. `fold_subset` restricts which folds are predicted (val vs test)."""
    by_fold = defaultdict(list)
    for t in eval_trials:
        by_fold[_fold(t.nct, k)].append(t)
    preds = {}
    for f in sorted(by_fold):
        if fold_subset is not None and f not in fold_subset:
            continue
        rows = by_fold[f]
        _patch(edges, cfg, frozenset(t.nct for t in rows))
        preds.update(_predict(g, rows, n_samples))
        _restore(edges)
    return preds


def _score(preds, trials_by_nct):
    items = [(p, trials_by_nct[n].y) for n, p in preds.items()
             if p is not None and n in trials_by_nct]
    if len(items) < 4:
        return None
    p = [x for x, _ in items]
    y = [yy for _, yy in items]
    if len(set(y)) < 2:
        return None
    return dict(n=len(y), brier=brier_score(p, y),
                ece=expected_calibration_error(p, y), auroc=_auroc(p, y),
                acc=_binary_accuracy(p, y, 0.5)[0])


def make_grid() -> dict[str, BeliefTuneConfig]:
    """Focused, principled grid. The pessimism (means too low) points at the
    outcome p_obs: failure=0.20 treats a failed trial as strong falsification, but
    the project's own stance is 'failure != falsification' — so raise it toward
    neutral. Also try strengthening success and down-weighting trial n_eff."""
    g: dict[str, BeliefTuneConfig] = {"default": BeliefTuneConfig()}
    for fp in (0.30, 0.35, 0.40, 0.45):
        g[f"fail={fp}"] = BeliefTuneConfig(outcome_p_obs={"failure": fp})
    g["succ=0.90"] = BeliefTuneConfig(outcome_p_obs={"success": 0.90})
    g["fail=0.35,succ=0.88"] = BeliefTuneConfig(
        outcome_p_obs={"failure": 0.35, "success": 0.88})
    g["fail=0.40,succ=0.90"] = BeliefTuneConfig(
        outcome_p_obs={"failure": 0.40, "success": 0.90})
    g["clin_neff×0.5"] = BeliefTuneConfig(neff_scale={"clinical": 0.5})
    g["clin×0.5,fail=0.35"] = BeliefTuneConfig(
        neff_scale={"clinical": 0.5}, outcome_p_obs={"failure": 0.35})
    g["clin×2.0,fail=0.40"] = BeliefTuneConfig(
        neff_scale={"clinical": 2.0}, outcome_p_obs={"failure": 0.40})
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
    ap.add_argument("--validate-against", default=None,
                    help="graph-preds dump from eval_baselines --dump-graph-preds; "
                         "checks the delta-adjust holdout matches re-attribution.")
    ap.add_argument("--val-folds", default="0,1,2")
    ap.add_argument("--test-folds", default="3,4")
    args = ap.parse_args()

    np.random.seed(42)
    g = GraphStore()
    g.import_snapshot(args.graph)
    edges = collect_tunable(g._graph)  # noqa: SLF001
    trials = build_trials(g, args.corpus, args.holdout_corpus, args.min_overlap)
    eval_trials = [t for t in trials if t.is_eval]
    by_nct = {t.nct: t for t in eval_trials}
    print(f"tunable edges (with applied records): {len(edges)}")
    print(f"scored slice: {len(eval_trials)} trials "
          f"(base rate {np.mean([t.y for t in eval_trials]):.3f})")

    # 1) Validate the cheap delta-adjust holdout vs the re-attribution dump.
    if args.validate_against:
        ref = json.loads(Path(args.validate_against).read_text()).get("holdout", {})
        da = holdout_preds(g, edges, eval_trials, BeliefTuneConfig(), args.k, args.n_samples)
        common = [n for n in da if n in ref and da[n] is not None and ref[n] is not None]
        if common:
            x = np.array([da[n] for n in common]); y = np.array([ref[n] for n in common])
            corr = float(np.corrcoef(x, y)[0, 1])
            print(f"\nVALIDATION vs re-attribution dump (n={len(common)}): "
                  f"corr={corr:.3f}, mean|Δ|={np.mean(np.abs(x - y)):.3f}, "
                  f"max|Δ|={np.max(np.abs(x - y)):.3f}")
            print("  (high corr ⇒ the cheap delta-adjust tracks the faithful "
                  "re-attribution holdout; tuning on it is sound)")

    val_folds = {int(x) for x in args.val_folds.split(",")}
    test_folds = {int(x) for x in args.test_folds.split(",")}

    # 2) Tune on VAL folds (calibration first).
    print(f"\n── Tuning on validation folds {sorted(val_folds)} (calibration first) ──")
    print(f"  {'config':<22}{'n':>5}{'Brier':>8}{'ECE':>7}{'AUROC':>8}{'Acc':>7}")
    grid = make_grid()
    val_scores = {}
    for name, cfg in grid.items():
        preds = holdout_preds(g, edges, eval_trials, cfg, args.k, args.n_samples, val_folds)
        s = _score(preds, by_nct)
        if not s:
            continue
        val_scores[name] = s
        print(f"  {name:<22}{s['n']:>5}{s['brier']:>8.3f}{s['ece']:>7.3f}"
              f"{s['auroc']:>8.3f}{s['acc']:>7.3f}")
    best = min(val_scores, key=lambda nme: val_scores[nme]["brier"])
    print(f"\n  best-on-val (min Brier): {best!r}")

    # 3) Score default + best on the held-out TEST folds (touched once).
    print(f"\n── TEST folds {sorted(test_folds)} (scored once) ──")
    print(f"  {'config':<22}{'n':>5}{'Brier':>8}{'ECE':>7}{'AUROC':>8}{'Acc':>7}")
    for name in ("default", best):
        preds = holdout_preds(g, edges, eval_trials, grid[name], args.k, args.n_samples, test_folds)
        s = _score(preds, by_nct)
        if s:
            print(f"  {name:<22}{s['n']:>5}{s['brier']:>8.3f}{s['ece']:>7.3f}"
                  f"{s['auroc']:>8.3f}{s['acc']:>7.3f}")

    # 4) Full-slice reliability of the best config (for the diagram).
    preds = holdout_preds(g, edges, eval_trials, grid[best], args.k, args.n_samples)
    items = [(p, by_nct[n].y) for n, p in preds.items() if p is not None]
    print(f"\n  reliability — best config {best!r} (full 5-fold holdout):")
    print(f"    {'bin':<13}{'count':>7}{'mean_pred':>11}{'obs_freq':>10}")
    for b in reliability_table([p for p, _ in items], [y for _, y in items], n_bins=10):
        if b.count:
            print(f"    [{b.lo:.1f},{b.hi:.1f}){'':<4}{b.count:>7}"
                  f"{b.mean_pred:>11.3f}{b.frac_pos:>10.3f}")
    print(f"\nNEXT: confirm {best!r} with a true re-attribution "
          f"(set the params + eval_holdout_kfold) — delta-adjust holds explain-away "
          f"fixed (2nd-order), re-attribution certifies.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
