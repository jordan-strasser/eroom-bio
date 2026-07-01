"""Parts 2-3: LOIO + the direction-pooling ablation on the DRD2 graph.

Reuses the honest per-fold re-attribution harness (attributor_main with
exclude_from_attribution) + eval_holdout_compose helpers. For each held-out
indication Y: re-attribute the pre-attribution initial.json excluding Y's trials,
then predict Y's clean-labeled trials under THREE direction modes from the SAME
re-attributed graph:
  backoff             — child shrunk toward both-direction pooled parent
  same_direction_only — only same-direction evidence (no cross-+/- pooling)
  flat                — direction ignored
Part 3 hypothesis test: pooled AUROC(backoff) vs AUROC(same_direction_only)
(pooling +/- adds value?) and vs AUROC(flat) (direction machinery is signal?),
with a paired DeLong. Plus within-direction transfer (hold PD; source only
agonist non-PD evidence).

Usage:
  python -m scratch.drd2.loio_direction_eval
"""
from __future__ import annotations

import asyncio
import json
import tempfile
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
from scipy import stats

from src.annotation.attributor import _main as attributor_main
from src.graph.store import GraphStore
from src.prediction.path_query import (
    DIRECTION_BACKOFF, DIRECTION_FLAT, DIRECTION_SAME_ONLY,
    predict_clinical_hypothesis, seed_prediction_rng,
)
from scripts.eval_holdout_compose import (
    ANN_DIR, _auroc, _corpus_ncts, _load_canonicalization_cache,
    _overlap_count, _resolve_label, _training_used_nodes, _trial_conditions,
    resolve_chain,
)

INITIAL = "data/exports/drd2_subset_initial.json"
ANNOTATED = "data/exports/drd2_subset_annotated.json"
CORPUS = "drd2_subset"
MODES = (DIRECTION_BACKOFF, DIRECTION_SAME_ONLY, DIRECTION_FLAT)
N_SAMPLES = 4000
MIN_BOTH_CLASS = 3
MIN_OVERLAP = 4

DIRECTION_OF = {  # from the corpus roster
    "pramipexol": "agonist", "ropinirole": "agonist", "ropinirol": "agonist",
    "rotigotine": "agonist", "apomorphine": "agonist", "cabergoline": "agonist",
    "haloperidol": "antagonist", "risperidone": "antagonist",
    "olanzapine": "antagonist", "metoclopramide": "antagonist",
    "prochlorperazine": "antagonist",
}


# ── paired DeLong (Sun & Xu fast algorithm) ───────────────────────────────
def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N); T2[J] = T
    return T2


def delong_paired(y_true, p_a, p_b):
    """Two-sided paired DeLong for AUC(a) vs AUC(b) on the same labels.
    Returns (auc_a, auc_b, z, p). NaN-safe on degenerate folds."""
    y = np.asarray(y_true, float)
    m = int(y.sum()); n = len(y) - m
    if m == 0 or n == 0:
        return (float("nan"),) * 4
    order = (-y).argsort(kind="mergesort")
    preds = np.vstack([np.asarray(p_a, float)[order], np.asarray(p_b, float)[order]])
    k = 2
    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m + n])
    for r in range(k):
        tx[r] = _midrank(preds[r, :m])
        ty[r] = _midrank(preds[r, m:])
        tz[r] = _midrank(preds[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01); sy = np.cov(v10)
    cov = sx / m + sy / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return float(aucs[0]), float(aucs[1]), 0.0, 1.0
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(aucs[0]), float(aucs[1]), float(z), float(p)


def _drug_dir(cid: str | None) -> str:
    for d, dr in DIRECTION_OF.items():
        if d in (cid or "").lower():
            return dr
    return "unknown"


def _predict(g, chain, kwargs, mode):
    seed_prediction_rng(42)
    try:
        r = predict_clinical_hypothesis(
            g, chain["compound_id"], chain["indication_id"],
            n_samples=N_SAMPLES, direction_mode=mode, **kwargs)
    except KeyError:
        return None
    return r.overall_probability if r.edge_contributions else None


async def _reattribute(exclude: list[str]) -> GraphStore:
    with tempfile.NamedTemporaryFile(suffix="_loio.json", delete=False) as fh:
        tmp = fh.name
    try:
        await attributor_main(str(ANN_DIR), INITIAL, tmp,
                              exclude_from_attribution=exclude)
        g = GraphStore(); g.import_snapshot(tmp)
    finally:
        Path(tmp).unlink(missing_ok=True)
    return g


async def main() -> int:
    full = GraphStore(); full.import_snapshot(ANNOTATED)
    training_used = _training_used_nodes(full)
    canon = _load_canonicalization_cache()

    scorable = []  # (nct,label01,chain,kwargs,indication,direction)
    for nct in _corpus_ncts(CORPUS):
        ext_p = ANN_DIR / f"{nct}_extraction.json"
        if not ext_p.exists():
            continue
        extraction = json.loads(ext_p.read_text())
        cls_p = ANN_DIR / f"{nct}_classification.json"
        classification = json.loads(cls_p.read_text()) if cls_p.exists() else None
        label = _resolve_label(extraction, classification)
        if label not in ("success", "failure"):
            continue
        chain = resolve_chain(extraction, _trial_conditions(extraction), full, canon)
        if (chain["target_id"] not in training_used
                or chain["indication_id"] not in training_used):
            continue
        if _overlap_count(chain, training_used) < MIN_OVERLAP:
            continue
        kwargs = {k: chain[k] for k in
                  ("target_id", "mechanism_id", "biology_id", "endpoint_id",
                   "population_id") if chain[k] and chain[k] != "UNKNOWN"}
        scorable.append((nct, 1 if label == "success" else 0, chain, kwargs,
                         chain["indication_id"], _drug_dir(chain["compound_id"])))

    by_ind = defaultdict(list)
    for row in scorable:
        by_ind[row[4]].append(row)
    eval_inds = {ind: rows for ind, rows in by_ind.items()
                 if len(rows) >= MIN_BOTH_CLASS
                 and any(r[1] == 1 for r in rows) and any(r[1] == 0 for r in rows)}
    print(f"scorable trials: {len(scorable)}; base rate "
          f"{mean(r[1] for r in scorable):.3f}")
    print(f"LOIO-evaluable indications (>={MIN_BOTH_CLASS} both-class): "
          f"{sorted((i, len(r), sum(x[1] for x in r)) for i, r in eval_inds.items())}\n")

    # ── in-sample (full graph, backoff) for the leakage gap ──
    insample = {m: {} for m in MODES}
    for nct, y, chain, kwargs, ind, _d in scorable:
        for mode in MODES:
            insample[mode][nct] = _predict(full, chain, kwargs, mode)

    # ── LOIO per indication ──
    holdout = {m: {} for m in MODES}
    ind_all_ncts = defaultdict(set)
    for nct, _y, _c, _k, ind, _d in scorable:
        ind_all_ncts[ind].add(nct)
    for ts in full.trial_subgraphs.values():
        for ch in ts.chains:
            if ch.indication_id in eval_inds:
                ind_all_ncts[ch.indication_id].add(ts.trial_id)

    per_ind = {m: {} for m in MODES}
    for ind in sorted(eval_inds, key=lambda i: -len(eval_inds[i])):
        rows = eval_inds[ind]
        hold = sorted(ind_all_ncts[ind])
        print(f"  LOIO {ind[:28]:28s} hold {len(hold)}, score {len(rows)} "
              f"({sum(r[1] for r in rows)}+/{sum(1-r[1] for r in rows)}-)...", flush=True)
        gy = await _reattribute(hold)
        for mode in MODES:
            ys, ps = [], []
            for nct, y, chain, kwargs, _ind, _d in rows:
                p = _predict(gy, chain, kwargs, mode)
                if p is not None:
                    holdout[mode][nct] = p; ys.append(y); ps.append(p)
            if len(set(ys)) == 2:
                per_ind[mode][ind] = (_auroc(ps, ys), len(ys))

    # ── within-direction transfer: hold PD, source only agonist non-PD ──
    within = {}
    pd_key = next((i for i in eval_inds if "parkinson" in i), None)
    if pd_key:
        antagonist_ncts = [r[0] for r in scorable if r[5] == "antagonist"]
        exclude = sorted(set(ind_all_ncts[pd_key]) | set(antagonist_ncts))
        print(f"\n  within-direction: hold PD + all antagonist ({len(exclude)} excl), "
              f"predict PD from agonist-only...", flush=True)
        gpd = await _reattribute(exclude)
        for mode in MODES:
            ys, ps = [], []
            for nct, y, chain, kwargs, ind, _d in eval_inds[pd_key]:
                p = _predict(gpd, chain, kwargs, mode)
                if p is not None:
                    ys.append(y); ps.append(p)
            within[mode] = (_auroc(ps, ys) if len(set(ys)) == 2 else None, len(ys))

    # ── pooled AUROC + ablation DeLong (paired on the intersection) ──
    def pooled(dct):
        ncts = [r[0] for r in scorable if r[0] in dct and dct[r[0]] is not None]
        y = [1 if r[1] == 1 else 0 for r in scorable if r[0] in ncts]
        # keep order aligned
        y = [next(r[1] for r in scorable if r[0] == n) for n in ncts]
        p = [dct[n] for n in ncts]
        return ncts, y, p

    print("\n================ RESULTS ================")
    print(f"base rate (success): {mean(r[1] for r in scorable):.3f} "
          f"(n={len(scorable)}; {sum(r[1] for r in scorable)}+/"
          f"{sum(1-r[1] for r in scorable)}-)")
    out = {"base_rate": mean(r[1] for r in scorable), "n": len(scorable),
           "per_indication": {}, "pooled": {}, "within_direction": {},
           "leakage_gap": {}, "ablation": {}}

    print("\nPer-indication holdout AUROC (n):")
    for ind in sorted(eval_inds):
        cells = "  ".join(
            f"{m.split('_')[0]}={per_ind[m].get(ind, (float('nan'),0))[0]:.3f}"
            for m in MODES)
        n = per_ind[DIRECTION_BACKOFF].get(ind, (0, 0))[1]
        base = mean(r[1] for r in eval_inds[ind])
        print(f"  {ind[:26]:26s} n={n:2d} base={base:.2f}  {cells}")
        out["per_indication"][ind] = {
            m: per_ind[m].get(ind, (None, 0)) for m in MODES}

    print("\nPooled holdout AUROC + leakage gap:")
    pooled_series = {}
    for m in MODES:
        ncts, y, p = pooled(holdout[m])
        au = _auroc(p, y) if len(set(y)) == 2 else float("nan")
        pooled_series[m] = (ncts, y, p, au)
        # leakage gap vs in-sample on same ncts
        ins_p = [insample[m][n] for n in ncts if insample[m].get(n) is not None]
        ins_y = [y[i] for i, n in enumerate(ncts) if insample[m].get(n) is not None]
        ins_au = _auroc(ins_p, ins_y) if len(set(ins_y)) == 2 else float("nan")
        print(f"  {m:22s} holdout AUROC={au:.3f}  in-sample={ins_au:.3f}  "
              f"gap={ins_au-au:+.3f}  (n={len(y)})")
        out["pooled"][m] = {"auroc": au, "in_sample": ins_au, "n": len(y)}

    # ablation: paired DeLong on the common NCT set
    common = [n for n in pooled_series[DIRECTION_BACKOFF][0]
              if all(n in holdout[m] for m in MODES)]
    yv = [next(r[1] for r in scorable if r[0] == n) for n in common]
    print(f"\nAblation (paired DeLong, common n={len(common)}, "
          f"{sum(yv)}+/{len(yv)-sum(yv)}-):")
    for a, b in ((DIRECTION_BACKOFF, DIRECTION_SAME_ONLY),
                 (DIRECTION_BACKOFF, DIRECTION_FLAT),
                 (DIRECTION_SAME_ONLY, DIRECTION_FLAT)):
        pa = [holdout[a][n] for n in common]; pb = [holdout[b][n] for n in common]
        au_a, au_b, z, pval = delong_paired(yv, pa, pb)
        print(f"  {a.split('_')[0]:8s} vs {b.split('_')[0]:8s}: "
              f"AUROC {au_a:.3f} vs {au_b:.3f}  Δ={au_a-au_b:+.3f}  "
              f"z={z:+.2f} p={pval:.3f}")
        out["ablation"][f"{a}_vs_{b}"] = {"auroc_a": au_a, "auroc_b": au_b,
                                          "delta": au_a - au_b, "z": z, "p": pval}

    print("\nWithin-direction transfer (hold PD, agonist-only source):")
    for m in MODES:
        au, n = within.get(m, (None, 0))
        print(f"  {m:22s} AUROC={au if au is None else round(au,3)} (n={n})")
        out["within_direction"][m] = {"auroc": au, "n": n}

    Path("scratch/drd2/loio_results.json").write_text(json.dumps(out, indent=2, default=str))
    print("\nwrote scratch/drd2/loio_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
