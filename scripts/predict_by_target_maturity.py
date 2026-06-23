"""Does the chain predict where its beliefs are DIFFERENTIATED (novel/sparse
targets) vs SATURATED (mature targets)?

The 2021-26 holdout failed because it was dominated by mature targets (CDK4/6,
AR, ESR1, FGFR) whose graph beliefs are uniformly high and undifferentiated. The
fair-shot hypothesis: the mechanistic chain should predict best where its beliefs
actually distinguish trials — under-studied targets.

This is the CHEAP first cut (no build): stratify the train trials' self-excluded
(LOO) prediction by a STRUCTURAL covariate — target maturity = how many distinct
train trials hit the same target gene. Target maturity is known a priori
(independent of the trial's outcome), so stratifying on it is honest subgroup
analysis, NOT predictor-value subset-mining. If sparse-target trials show higher
AUROC, curate a novel-target holdout corpus to confirm out-of-sample.

Run: python -m scripts.predict_by_target_maturity --graph data/exports/multi_500_annotated.json
"""
from __future__ import annotations

import argparse
import statistics as stx
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.graph.store import GraphStore
from src.prediction.calibration import auroc
from src.prediction.path_query import PredictionEngine, _aggregate_samples, _trust_weight

import scripts.holdout_thesis_analysis as H

ANN = Path("data/annotations")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--bins", type=int, default=3, help="number of maturity strata (terciles default)")
    a = ap.parse_args()
    store = GraphStore()
    store.import_snapshot(a.graph)
    engine = PredictionEngine(store)

    # structural index: target_id -> set of trials touching it (any chain)
    target_trials: dict[str, set[str]] = defaultdict(set)
    mech_trials: dict[str, set[str]] = defaultdict(set)
    for nct, sg in store.trial_subgraphs.items():
        for ch in sg.chains:
            if ch.target_id and ch.target_id != "UNKNOWN":
                target_trials[ch.target_id].add(nct)
            if ch.mechanism_id and ch.mechanism_id != "UNKNOWN":
                mech_trials[ch.mechanism_id].add(nct)

    def label(nct):
        ext = H._load_json(ANN / f"{nct}_extraction.json")
        cls = H._load_json(ANN / f"{nct}_classification.json")
        return H._resolve_label(ext, cls) if ext else None

    rows = []  # (nct, y, softmin_loo, target_maturity, mech_maturity)
    for nct, sg in store.trial_subgraphs.items():
        lab = label(nct)
        if lab not in ("success", "failure") or not sg.chains:
            continue
        # decisive chain by self-excluded softmin; record its target's maturity
        best = None
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
                b = H._belief_excluding_set(ec.belief, {nct}, ec.edge_type.value)
                d = b.alpha + b.beta
                means.append(b.alpha / d if d > 0 else 0.5)
                weights.append(_trust_weight(b))
            soft = float(_aggregate_samples([np.array([m]) for m in means])[0])
            tmat = len(target_trials.get(ch.target_id, set()) - {nct})
            mmat = len(mech_trials.get(ch.mechanism_id, set()) - {nct})
            if best is None or soft < best[0]:
                best = (soft, tmat, mmat)
        if best is None:
            continue
        rows.append((nct, 1 if lab == "success" else 0, best[0], best[1], best[2]))

    print(f"graph={a.graph}  binary trials={len(rows)}\n")

    def stratify(name, idx):
        vals = sorted(r[idx] for r in rows)
        n = len(vals)
        cuts = [vals[int(k * n / a.bins)] for k in range(1, a.bins)]
        print(f"=== stratified by {name} (cuts at {cuts}) ===")
        print(f"  {'stratum':18s} {'n':>4s} {'succ':>4s} {'fail':>4s} {'base':>6s} "
              f"{'softmin_AUROC':>14s} {'mean_soft':>10s}")
        edges = [-1] + cuts + [10**9]
        for lo, hi in zip(edges, edges[1:]):
            grp = [r for r in rows if lo < r[idx] <= hi]
            if len(grp) < 6:
                continue
            y = [r[1] for r in grp]
            soft = [r[2] for r in grp]
            base = sum(y) / len(y)
            au = auroc(soft, y) if len(set(y)) > 1 else float("nan")
            rng = f"({lo+1 if lo>=0 else 0}-{hi if hi<10**8 else '∞'})"
            print(f"  {name[:6]+rng:18s} {len(grp):>4d} {sum(y):>4d} {len(y)-sum(y):>4d} "
                  f"{base:>6.2f} {au:>14.3f} {stx.mean(soft):>10.3f}")
        # overall for reference
        y = [r[1] for r in rows]; soft = [r[2] for r in rows]
        print(f"  {'ALL':18s} {len(rows):>4d} {sum(y):>4d} {len(y)-sum(y):>4d} "
              f"{sum(y)/len(y):>6.2f} {auroc(soft,y):>14.3f} {stx.mean(soft):>10.3f}\n")

    stratify("target_maturity", 3)
    stratify("mech_maturity", 4)


if __name__ == "__main__":
    main()
