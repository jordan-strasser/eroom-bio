"""Scalar vs (s,t)-field SELF-EXCLUDED AUROC — the localization probe.

Eval convention (owner, 2026-06-09): *always report scalar AND field AUROC
together.* This is the tool that does it, fully leave-one-out (out-of-sample),
on the train graph's own trials.

For each train trial with a binary CT.gov label we recompute every backbone
edge's belief with that trial's evidence removed, two ways:

  SCALAR  — provenance delta-adjust (`_belief_excluding_set`): the marginal Beta
            with the trial's records replayed back out.
  FIELD   — the (s,t) belief field with the trial's anchors removed
            (`BeliefField.without_trial`) AND its marginal fallback swapped to
            the scalar-LOO Beta (the docstring's "fully clean" LOO: anchors are
            dropped by nct, but the fallback otherwise still carries the trial).
            Queried at the trial's OWN per-edge (s,t) — the node descriptions its
            chain actually traversed (`build_st_desc_map`), embedded with BioLORD.

Then AUROC (success>failure) at three levels, scalar beside field:
  1. per edge type   — does localization make mechanism_affects MORE predictive
                       and the pooled/DB edges LESS anti-predictive?
  2. trial softmin    — the production weakest-link aggregate (decisive chain).
  3. learned logistic — 5-fold CV over the 6 per-edge-type beliefs (the
                        chain-is-not-useless result), scalar features vs field.

`--bandwidth b[,b2,...]` sweeps the field kernel width at query time (no
re-materialization) to tune sharing within an edge's sub-region.

Run:
  EROOM_PRIVATE_ROOT=../eroom-enterprise/artifacts python -m scripts.scalar_vs_field_auroc \
      --graph data/exports/multi_500_annotated.json \
      --field ../eroom-enterprise/artifacts/multi_500_annotated_belief_field.json \
      --bandwidth 0.25,0.15,0.40
"""
from __future__ import annotations

import argparse
import statistics as stx
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.graph import biolord_embeddings as BE
from src.graph.store import GraphStore
from src.inference.belief_field import BeliefField, expected_p
from src.prediction.calibration import auroc
from src.prediction.field_prediction import build_st_desc_map, load_edge_fields
from src.prediction.path_query import PredictionEngine, _aggregate_samples, _trust_weight

import scripts.holdout_thesis_analysis as H

ANN = Path("data/annotations")
# 6 backbone edge types that carry a materialized field (responds_differently
# has none). Order is the logistic feature order.
EDGE_TYPES = [
    "affects", "modulates_via", "mechanism_affects",
    "biology_drives", "reflects_biology", "endpoint_captures",
]


# ── in-process BioLORD cache (read disk once; embed_text re-reads per call) ──
def make_embedder():
    disk = BE._load_cache(BE.DEFAULT_CACHE_PATH)
    mem: dict[str, list[float]] = {}
    miss = {"n": 0}

    def emb(text: str) -> list[float]:
        k = BE._normalize_key(text)
        v = mem.get(k)
        if v is not None:
            return v
        v = disk.get(k)
        if v is None:
            miss["n"] += 1
            v = BE.embed_text(text)  # cache miss → model (should be rare; all node descs were embedded at build)
        mem[k] = v
        return v

    return emb, miss


def _field_loo_mean(field: BeliefField, nct: str, scalar_loo_a: float,
                    scalar_loo_b: float, s_vec, t_vec, bandwidth: float | None) -> float:
    """Fully-clean field LOO mean at (s,t): drop the trial's anchors AND replace
    the marginal fallback with the scalar-LOO Beta so a far-from-anchor query
    falls back to the leave-one-out pooled mean, not the leaky full marginal."""
    loo = field.without_trial(nct)
    clean = BeliefField(
        anchors=loo.anchors,
        bandwidth=bandwidth if bandwidth is not None else loo.bandwidth,
        marginal_alpha=scalar_loo_a,
        marginal_beta=scalar_loo_b,
        fallback_strength=loo.fallback_strength,
    )
    return expected_p(clean, s_vec, t_vec)


def trial_features(store, nct, field_map, emb, bandwidth):
    """Per-trial self-excluded edge beliefs, scalar & field.

    Returns dict with:
      per_type: {et: (scalar_mean, field_mean)}  averaged over the trial's edges
      softmin:  (scalar_overall, field_overall)  for the decisive (min-scalar) chain
    or None if the trial has no resolvable chain.
    """
    sg = store.trial_subgraphs.get(nct)
    if not sg or not sg.chains:
        return None
    engine = PredictionEngine(store)
    st_map = build_st_desc_map(store, nct)

    pt_scalar: dict[str, list[float]] = defaultdict(list)
    pt_field: dict[str, list[float]] = defaultdict(list)
    chain_aggs: list[tuple[float, float]] = []
    seen = set()
    for ch in sg.chains:
        ckey = (ch.compound_id, ch.target_id, ch.mechanism_id,
                ch.biology_id, ch.endpoint_id, ch.subgroup_population_id)
        if ckey in seen:
            continue
        seen.add(ckey)
        try:
            res = engine.predict(ch, n_samples=1)  # only need edge_contributions; overall recomputed below
        except Exception:  # noqa: BLE001
            continue
        if not res.edge_contributions:
            continue
        s_means, f_means, weights = [], [], []
        for ec in res.edge_contributions:
            et = ec.edge_type.value
            b = H._belief_excluding_set(ec.belief, {nct}, et)
            denom = b.alpha + b.beta
            s_mean = b.alpha / denom if denom > 0 else 0.5
            f_mean = s_mean
            fld = field_map.get((ec.source_id, ec.target_id, et))
            pair = st_map.get((ec.source_id, ec.target_id, et))
            if fld is not None and pair and pair[0] and pair[1]:
                f_mean = _field_loo_mean(fld, nct, b.alpha, b.beta,
                                         emb(pair[0]), emb(pair[1]), bandwidth)
            pt_scalar[et].append(s_mean)
            pt_field[et].append(f_mean)
            s_means.append(s_mean)
            f_means.append(f_mean)
            weights.append(_trust_weight(b))
        if s_means:
            s_ov = float(_aggregate_samples([np.array([m]) for m in s_means], weights)[0])
            f_ov = float(_aggregate_samples([np.array([m]) for m in f_means], weights)[0])
            chain_aggs.append((s_ov, f_ov))
    if not chain_aggs:
        return None
    decisive = min(chain_aggs, key=lambda x: x[0])  # weakest-link chain by scalar
    return {
        "per_type": {et: (stx.mean(pt_scalar[et]), stx.mean(pt_field[et]))
                     for et in pt_scalar},
        "softmin": decisive,
    }


def _auroc_or_na(probs, y):
    if len(set(y)) < 2 or len(y) < 5:
        return None
    return auroc(probs, y)


def _learned_cv_auroc(rows, key, y, seed=0):
    """5-fold stratified CV logistic over per-edge-type beliefs.
    key 'scalar'/'field' -> 6 features (that component); 'both' -> 12 features
    (scalar AND field per edge type — the complementarity test: does the field
    add signal the scalar lacks where it's saturated, e.g. modulates_via?).
    Missing edge type -> 0.5. Returns (out-of-fold AUROC, full-fit weights)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    if key == "both":
        X = np.array([[r["per_type"].get(et, (0.5, 0.5))[c]
                       for et in EDGE_TYPES for c in (0, 1)] for r in rows])
        names = [f"{et}:{'sc' if c == 0 else 'fld'}"
                 for et in EDGE_TYPES for c in (0, 1)]
    else:
        comp = 0 if key == "scalar" else 1
        X = np.array([[r["per_type"].get(et, (0.5, 0.5))[comp] for et in EDGE_TYPES]
                      for r in rows])
        names = list(EDGE_TYPES)
    yv = np.array(y)
    oof = np.zeros(len(yv))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, yv):
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X[tr], yv[tr])
        oof[te] = lr.predict_proba(X[te])[:, 1]
    return auroc(list(oof), list(yv)), dict(zip(
        names, np.round(LogisticRegression(max_iter=1000).fit(X, yv).coef_[0], 3)))


def run(store, field_map, ncts_labels, emb, bandwidth, miss):
    rows, y = [], []
    for nct, lab in ncts_labels:
        feats = trial_features(store, nct, field_map, emb, bandwidth)
        if feats is None:
            continue
        rows.append(feats)
        y.append(1 if lab == "success" else 0)

    bw = bandwidth if bandwidth is not None else "default(0.25)"
    print(f"\n{'='*72}\nBANDWIDTH = {bw}   (scored trials: {len(y)}, "
          f"success={sum(y)}, failure={len(y)-sum(y)}, embed-misses={miss['n']})\n{'='*72}")

    # 1. per edge type
    print(f"\n  per-edge-type self-excluded AUROC (mean belief over trial's edges of that type)")
    print(f"  {'edge_type':22s} {'n':>4s} {'scalar':>8s} {'field':>8s} {'Δ':>7s}")
    for et in EDGE_TYPES:
        sub = [(r["per_type"][et], yy) for r, yy in zip(rows, y) if et in r["per_type"]]
        if len(sub) < 5:
            continue
        s_probs = [p[0][0] for p in sub]
        f_probs = [p[0][1] for p in sub]
        yy = [p[1] for p in sub]
        sa, fa = _auroc_or_na(s_probs, yy), _auroc_or_na(f_probs, yy)
        if sa is None:
            continue
        d = (fa - sa) if fa is not None else 0.0
        print(f"  {et:22s} {len(sub):>4d} {sa:>8.3f} {fa:>8.3f} {d:>+7.3f}")

    # 2. trial softmin (production weakest-link)
    s_soft = [r["softmin"][0] for r in rows]
    f_soft = [r["softmin"][1] for r in rows]
    sa, fa = _auroc_or_na(s_soft, y), _auroc_or_na(f_soft, y)
    print(f"\n  trial softmin (weakest-link, decisive chain): "
          f"scalar {sa:.3f}  |  field {fa:.3f}  |  Δ {fa-sa:+.3f}")

    # 2b. mechanism_affects-only
    ma = [(r["per_type"]["mechanism_affects"], yy)
          for r, yy in zip(rows, y) if "mechanism_affects" in r["per_type"]]
    if len(ma) >= 5:
        sa = _auroc_or_na([p[0][0] for p in ma], [p[1] for p in ma])
        fa = _auroc_or_na([p[0][1] for p in ma], [p[1] for p in ma])
        print(f"  mechanism_affects-only:                        "
              f"scalar {sa:.3f}  |  field {fa:.3f}  |  Δ {fa-sa:+.3f}")

    # 3. learned 5-fold-CV logistic over the 6 edge types
    s_auc, s_coef = _learned_cv_auroc(rows, "scalar", y)
    f_auc, f_coef = _learned_cv_auroc(rows, "field", y)
    b_auc, _ = _learned_cv_auroc(rows, "both", y)  # complementarity: scalar+field
    print(f"\n  learned logistic (5-fold CV over 6 edge beliefs): "
          f"scalar {s_auc:.3f}  |  field {f_auc:.3f}  |  Δ {f_auc-s_auc:+.3f}")
    print(f"    scalar+field combined (12 feats): {b_auc:.3f}  "
          f"(Δ vs scalar {b_auc-s_auc:+.3f} — does the field add what the scalar lacks?)")
    print(f"    scalar weights: {s_coef}")
    print(f"    field  weights: {f_coef}")
    return {"n": len(y), "softmin_scalar": _auroc_or_na(s_soft, y),
            "softmin_field": _auroc_or_na(f_soft, y),
            "learned_scalar": s_auc, "learned_field": f_auc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--field", required=True, help="materialized belief-field snapshot")
    ap.add_argument("--bandwidth", default="", help="comma list to sweep, e.g. 0.25,0.15,0.40 (blank=field default)")
    a = ap.parse_args()

    store = GraphStore()
    store.import_snapshot(a.graph)
    field_map = load_edge_fields(a.field)
    emb, miss = make_embedder()

    ncts_labels = []
    for nct in store.trial_subgraphs:
        ext = H._load_json(ANN / f"{nct}_extraction.json")
        cls = H._load_json(ANN / f"{nct}_classification.json")
        lab = H._resolve_label(ext, cls) if ext else None
        if lab in ("success", "failure"):
            ncts_labels.append((nct, lab))
    print(f"graph={a.graph}\nfield={a.field}  ({len(field_map)} edges)\n"
          f"binary-labeled trials: {len(ncts_labels)}")

    bands = [None] if not a.bandwidth.strip() else [float(x) for x in a.bandwidth.split(",")]
    summary = []
    for bw in bands:
        miss["n"] = 0
        summary.append((bw, run(store, field_map, ncts_labels, emb, bw, miss)))

    if len(summary) > 1:
        print(f"\n{'='*72}\nBANDWIDTH SWEEP SUMMARY (trial softmin + learned, scalar|field)\n{'='*72}")
        print(f"  {'bw':>8s} {'soft_sc':>8s} {'soft_fld':>8s} {'learn_sc':>9s} {'learn_fld':>9s}")
        for bw, s in summary:
            print(f"  {str(bw):>8s} {s['softmin_scalar']:>8.3f} {s['softmin_field']:>8.3f} "
                  f"{s['learned_scalar']:>9.3f} {s['learned_field']:>9.3f}")


if __name__ == "__main__":
    main()
