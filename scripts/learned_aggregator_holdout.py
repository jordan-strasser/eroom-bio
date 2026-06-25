"""TRUE forward-holdout validation of the learned-aggregation predictor.

The probe (`scalar_vs_field_auroc.py`) showed a 5-fold-CV logistic over the 6
scalar per-edge-type beliefs hits 0.766 self-excluded AUROC — but that's
LOO-on-train (optimistic: merge/canon saw every trial). This is the honest test:
FIT the aggregator on the train trials and PREDICT a disjoint 2021-26 holdout
that contributed ZERO evidence (built with --exclude-from-attribution).

One graph (multi_500_holdout_annotated.json = 472 train + 42 excluded holdout),
so beliefs live in one regime:
  TRAIN trial → features = self-excluded edge beliefs (LOO: remove its own
                evidence, mimicking the unseen-trial regime).
  HOLDOUT     → features = full edge beliefs (already evidence-free). Per-trial
                contamination check: a holdout edge must carry NO self-record.

Reports, on the SAME holdout trials (success>failure AUROC + Brier + acc):
  base rate                — the trivial predictor
  softmin (scalar)         — the production hand-tuned weakest-link
  learned logistic         — fit on train self-excluded features
The field is OMITTED: it's degenerate on a merged graph (see
project_predictor_signal_finding / scalar_vs_field_auroc) — all anchors collapse
to node coords, so it equals/under-performs the scalar.

Run:
  python -m scripts.learned_aggregator_holdout \
      --graph data/exports/multi_500_holdout_annotated.json \
      --holdout-ncts data/corpora/holdout_2021_2026.txt
"""
from __future__ import annotations

import argparse
import statistics as stx
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.graph.store import GraphStore
from src.prediction.calibration import auroc, brier_score, expected_calibration_error
from src.prediction.path_query import PredictionEngine, _aggregate_samples, _trust_weight

import scripts.holdout_thesis_analysis as H
from scripts.scalar_vs_field_auroc import EDGE_TYPES

ANN = Path("data/annotations")


def trial_row(store, nct, engine, *, exclude_self: bool):
    """Return (per_type_mean: {et: belief}, softmin, n_self_records) for a trial.

    exclude_self=True applies the LOO delta-adjust (train trials); False uses the
    full belief (holdout trials, already evidence-free). n_self_records counts
    edges that still carry the trial's own NCT as a source (contamination)."""
    sg = store.trial_subgraphs.get(nct)
    if not sg or not sg.chains:
        return None
    pt: dict[str, list[float]] = defaultdict(list)
    aggs: list[float] = []
    n_self = 0
    seen = set()
    for ch in sg.chains:
        ckey = (ch.compound_id, ch.target_id, ch.mechanism_id,
                ch.biology_id, ch.endpoint_id, ch.subgroup_population_id)
        if ckey in seen:
            continue
        seen.add(ckey)
        try:
            res = engine.predict(ch, n_samples=1)
        except Exception:  # noqa: BLE001
            continue
        if not res.edge_contributions:
            continue
        means, weights = [], []
        for ec in res.edge_contributions:
            et = ec.edge_type.value
            if any(ev.source_id == nct for ev in ec.belief.evidence):
                n_self += 1
            b = H._belief_excluding_set(ec.belief, {nct}, et) if exclude_self else ec.belief
            denom = b.alpha + b.beta
            m = b.alpha / denom if denom > 0 else 0.5
            pt[et].append(m)
            means.append(m)
            weights.append(_trust_weight(b))
        if means:
            aggs.append(float(_aggregate_samples([np.array([x]) for x in means])[0]))
    if not aggs:
        return None
    return {et: stx.mean(v) for et, v in pt.items()}, min(aggs), n_self


def feature_vec(per_type: dict[str, float]) -> list[float]:
    return [per_type.get(et, 0.5) for et in EDGE_TYPES]


def _metrics(name, probs, y, threshold=0.5):
    base = sum(y) / len(y)
    acc = sum((p >= threshold) == (yy == 1) for p, yy in zip(probs, y)) / len(y)
    print(f"  {name:24s} AUROC {auroc(probs, y):.3f} | Brier {brier_score(probs, y):.3f} "
          f"| ECE {expected_calibration_error(probs, y):.3f} | acc@{threshold:.2f} {acc:.3f} "
          f"| base {base:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--holdout-ncts", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    store = GraphStore()
    store.import_snapshot(a.graph)
    engine = PredictionEngine(store)
    holdout = {c for ln in Path(a.holdout_ncts).read_text().splitlines()
               if (c := ln.split("#", 1)[0].strip())}

    def label(nct):
        ext = H._load_json(ANN / f"{nct}_extraction.json")
        cls = H._load_json(ANN / f"{nct}_classification.json")
        return H._resolve_label(ext, cls) if ext else None

    Xtr, ytr = [], []
    Xte, yte, soft_te = [], [], []
    contaminated = []
    n_train_seen = n_holdout_seen = 0
    for nct in store.trial_subgraphs:
        lab = label(nct)
        if lab not in ("success", "failure"):
            continue
        is_holdout = nct in holdout
        row = trial_row(store, nct, engine, exclude_self=not is_holdout)
        if row is None:
            continue
        per_type, softmin, n_self = row
        y = 1 if lab == "success" else 0
        if is_holdout:
            n_holdout_seen += 1
            if n_self:
                contaminated.append((nct, n_self))
            Xte.append(feature_vec(per_type))
            yte.append(y)
            soft_te.append(softmin)
        else:
            n_train_seen += 1
            Xtr.append(feature_vec(per_type))
            ytr.append(y)

    print(f"graph={a.graph}")
    print(f"train (fit) binary trials: {n_train_seen}   "
          f"holdout (test) binary trials: {n_holdout_seen} "
          f"(success={sum(yte)}, failure={len(yte)-sum(yte)})")
    if contaminated:
        print(f"  ⚠ CONTAMINATION — {len(contaminated)} holdout trials carry self-evidence "
              f"(NOT out-of-sample): {contaminated[:6]}")
    else:
        print("  ✓ no holdout trial carries self-evidence (clean out-of-sample)")
    if len(set(yte)) < 2 or len(yte) < 8:
        print("  too few/one-class holdout trials to score")
        return

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    Xtr_a, Xte_a = np.array(Xtr), np.array(Xte)
    print(f"\nHOLDOUT METRICS (n={len(yte)}):")
    _metrics("base rate (constant)", [sum(yte) / len(yte)] * len(yte), yte)
    _metrics("softmin scalar (prod)", soft_te, yte)

    lr = LogisticRegression(max_iter=2000)
    lr.fit(Xtr_a, ytr)
    p_raw = lr.predict_proba(Xte_a)[:, 1]
    _metrics("learned logistic", list(p_raw), yte)

    sc = StandardScaler().fit(Xtr_a)
    lrs = LogisticRegression(max_iter=2000).fit(sc.transform(Xtr_a), ytr)
    p_std = lrs.predict_proba(sc.transform(Xte_a))[:, 1]
    _metrics("learned logistic (std)", list(p_std), yte)

    coef = dict(zip(EDGE_TYPES, np.round(lr.coef_[0], 3)))
    print(f"\n  learned weights (raw): {coef}")
    print(f"  (LOO-on-train cross-val was 0.766; this is the honest forward-holdout number)")


if __name__ == "__main__":
    main()
