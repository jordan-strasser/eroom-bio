"""Ablate the safety-penalty FORM — does the three-gate complexity earn its keep?

The per-AE contribution is severity × belief × trust × failure_causing_fraction,
soft-or'd. Each factor is a hand-set transform (severity scale is hand-tuned and
bypassed on the ~89% of AEs with no CTCAE grade; belief/trust forms are chosen,
not derived). This tests, on the SAME delta-adjust holdout, whether dropping each
factor hurts — and whether the simplest form (just failure_causing_fraction)
keeps the one safety true-positive in the corpus (torcetrapib, a failure-causing
cardiac tox) correctly flagged.

If a simpler form ties the full one on the holdout AND keeps torcetrapib, delete
the complexity (Occam + the owner's ad-hoc-ness concern).
"""
from __future__ import annotations

import argparse
import math

import numpy as np

from src.graph.store import GraphStore
from src.prediction.path_query import PredictionEngine, _ae_severity_weight, predict_clinical_hypothesis
from src.prediction.calibration import brier_score, expected_calibration_error
from src.prediction.calibration import auroc as _auroc
from scripts.eval_baselines import build_trials
from scripts.eval_holdout_compose import _binary_accuracy, _target_index
from scripts.tune_composition import collect_all, _patch, _restore
from scripts.tune_beliefs import _predict
from scripts.eval_holdout_kfold import _fold


def _contrib(r, form):
    sev = _ae_severity_weight(r.severity_range, serious=r.serious)
    bel = max(0.0, min(1.0, (r.belief_probability - 0.5) / 0.5))
    tru = min(1.0, math.log(r.evidence_strength + 1.0) / math.log(50.0))
    frac = r.failure_causing_fraction
    return {
        "full": sev * bel * tru * frac,           # = production (floor=0)
        "drop_trust": sev * bel * frac,
        "drop_belief": sev * tru * frac,
        "drop_severity": bel * tru * frac,
        "sevxfrac": sev * frac,
        "dlt_only": frac,                          # simplest: just failure-causing mass
    }[form]


def make_penalty(form, aggregate="softor"):
    def fn(self, risks):
        if not risks:
            return 0.0
        cs = [_contrib(r, form) for r in risks]
        if aggregate == "max":
            penalty = max(cs)
        else:
            penalty = 1.0
            for c in cs:
                penalty *= (1.0 - c)
            penalty = 1.0 - penalty
        return min(self._SAFETY_PENALTY_CAP, penalty)
    return fn


def holdout(g, edges, eval_trials, form, agg, k, n_samples):
    from collections import defaultdict
    by_fold = defaultdict(list)
    for t in eval_trials:
        by_fold[_fold(t.nct, k)].append(t)
    saved = PredictionEngine._penalty_from_risks
    PredictionEngine._penalty_from_risks = make_penalty(form, agg)
    preds = {}
    try:
        for f in sorted(by_fold):
            rows = by_fold[f]
            _patch(edges, frozenset(t.nct for t in rows), None)
            preds.update(_predict(g, rows, n_samples))
            _restore(edges)
    finally:
        PredictionEngine._penalty_from_risks = saved
    return preds


def torcetrapib(g, form, agg, n_samples):
    """P(success) + safety_penalty for the corpus's one failure-causing-tox case."""
    tid = _target_index(g).get("CETP")
    if not tid or "cardiovascular_diseases" not in g._graph:  # noqa: SLF001
        return None
    saved = PredictionEngine._penalty_from_risks
    PredictionEngine._penalty_from_risks = make_penalty(form, agg)
    try:
        r = predict_clinical_hypothesis(g, None, "cardiovascular_diseases",
                                        target_id=tid, n_samples=n_samples)
    except KeyError:
        return None
    finally:
        PredictionEngine._penalty_from_risks = saved
    if not r.edge_contributions:
        return None
    return r.overall_probability, r.safety_penalty


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--holdout-corpus", default=None)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--min-overlap", type=int, default=5)
    ap.add_argument("--n-samples", type=int, default=2000)
    args = ap.parse_args()
    np.random.seed(42)
    g = GraphStore()
    g.import_snapshot(args.graph)
    edges = collect_all(g._graph)  # noqa: SLF001
    trials = build_trials(g, args.corpus, args.holdout_corpus, args.min_overlap)
    eval_trials = [t for t in trials if t.is_eval]
    by_nct = {t.nct: t for t in eval_trials}
    base = np.mean([t.y for t in eval_trials])
    print(f"scored slice: {len(eval_trials)} (base rate {base:.3f})")
    print("\n── Safety-penalty FORM ablation (n=129 holdout) + torcetrapib guardrail ──")
    print(f"  {'form':<22}{'agg':<8}{'Brier':>8}{'ECE':>7}{'AUROC':>8}{'Acc':>7}"
          f"{'torce_P':>9}{'torce_pen':>10}")
    forms = [("full", "softor"), ("drop_trust", "softor"), ("drop_belief", "softor"),
             ("drop_severity", "softor"), ("sevxfrac", "softor"), ("dlt_only", "softor"),
             ("full", "max"), ("sevxfrac", "max")]
    for form, agg in forms:
        preds = holdout(g, edges, eval_trials, form, agg, args.k, args.n_samples)
        items = [(p, by_nct[n].y) for n, p in preds.items() if p is not None]
        pr = [p for p, _ in items]; y = [yy for _, yy in items]
        br, ece, au = brier_score(pr, y), expected_calibration_error(pr, y), _auroc(pr, y)
        acc = _binary_accuracy(pr, y, 0.5)[0]
        tor = torcetrapib(g, form, agg, args.n_samples)
        ts = f"{tor[0]:.3f}" if tor else "—"
        tp = f"{tor[1]:.3f}" if tor else "—"
        flag = "" if (tor and tor[0] < 0.5) else "  ⚠ torce NOT flagged"
        print(f"  {form:<22}{agg:<8}{br:>8.3f}{ece:>7.3f}{au:>8.3f}{acc:>7.3f}"
              f"{ts:>9}{tp:>10}{flag}")
    print("\n  torce_P<0.5 = correctly predicts failure (the safety thesis). If a SIMPLE")
    print("  form ties 'full' on the holdout AND keeps torce_P<0.5, the complexity is")
    print("  unjustified and should be deleted.")

    # Where do the pessimistic (0.2-0.4) trials go under soft-or vs max?
    from src.prediction.calibration import reliability_table
    so = holdout(g, edges, eval_trials, "full", "softor", args.k, args.n_samples)
    mx = holdout(g, edges, eval_trials, "full", "max", args.k, args.n_samples)
    for label, preds in (("soft-or", so), ("max", mx)):
        items = [(p, by_nct[n].y) for n, p in preds.items() if p is not None]
        print(f"\n  reliability — full / {label}:")
        for b in reliability_table([p for p, _ in items], [y for _, y in items], 10):
            if b.count:
                print(f"    [{b.lo:.1f},{b.hi:.1f})  n={b.count:>3}  pred={b.mean_pred:.3f}  obs={b.frac_pos:.3f}")
    # Migration of the trials soft-or scored in [0.2,0.4): where do they land + were they successes?
    low = [(n, so[n]) for n in so if so[n] is not None and 0.2 <= so[n] < 0.4 and n in mx and mx[n] is not None]
    n_succ = sum(by_nct[n].y for n, _ in low)
    print(f"\n  MIGRATION — {len(low)} trials soft-or rated [0.2,0.4) "
          f"(actual success rate {n_succ}/{len(low)} = {n_succ/max(1,len(low)):.2f}):")
    print(f"    {'nct':<13}{'soft-or':>9}{'max':>8}{'label':>9}")
    for n, sp in sorted(low, key=lambda x: x[1]):
        print(f"    {n:<13}{sp:>9.3f}{mx[n]:>8.3f}{'success' if by_nct[n].y else 'FAILURE':>9}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
