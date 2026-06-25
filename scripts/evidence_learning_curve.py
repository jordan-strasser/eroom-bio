"""Does forward-holdout AUROC rise as the training corpus grows (n=10→500)?

The owner's expectation: AUROC should increase ~monotonically with #trials. Test
it WITHOUT 5 rebuilds by exploiting that Beta beliefs are additive over evidence
records: for a keep-first-k corpus, recompute each holdout edge's belief from
only the first-k train trials' records (+ the always-on DB priors), then score
the SAME 34 holdout trials. Decisive chain per holdout trial is fixed (chosen at
full evidence) so the curve isn't confounded by chain re-selection.

CAVEAT: graph TOPOLOGY (which nodes merged) is frozen at n=500; only the evidence
is subset. So this tests belief-ACCUMULATION monotonicity, not a true from-scratch
n=k build (which would also have less merging). A flat curve here is strong
evidence more trials don't help; a rising curve warrants confirming with real
nested builds.

Run: python -m scripts.evidence_learning_curve \
       --graph data/exports/multi_500_holdout_annotated.json \
       --holdout-ncts data/corpora/holdout_2021_2026.txt
"""
from __future__ import annotations

import argparse
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
    ap.add_argument("--holdout-ncts", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    store = GraphStore()
    store.import_snapshot(a.graph)
    engine = PredictionEngine(store)
    holdout = {c for ln in Path(a.holdout_ncts).read_text().splitlines()
               if (c := ln.split("#", 1)[0].strip())}

    def label(nct):
        e = H._load_json(ANN / f"{nct}_extraction.json")
        c = H._load_json(ANN / f"{nct}_classification.json")
        return H._resolve_label(e, c) if e else None

    # ordered train NCTs (random, fixed seed — accumulation SHAPE not order matters)
    train = [n for n in store.trial_subgraphs if n not in holdout]
    rng = np.random.default_rng(a.seed)
    rng.shuffle(train)

    # decisive chain (min overall at full evidence) per binary holdout trial → its edges
    test = []  # (nct, y, [(belief, et), ...])
    for nct in holdout:
        lab = label(nct)
        if lab not in ("success", "failure"):
            continue
        sg = store.trial_subgraphs.get(nct)
        if not sg or not sg.chains:
            continue
        best = None
        seen = set()
        for ch in sg.chains:
            ck = (ch.compound_id, ch.target_id, ch.mechanism_id,
                  ch.biology_id, ch.endpoint_id, ch.subgroup_population_id)
            if ck in seen:
                continue
            seen.add(ck)
            try:
                res = engine.predict(ch, n_samples=300)
            except Exception:  # noqa: BLE001
                continue
            if not res.edge_contributions:
                continue
            if best is None or res.overall_probability < best[0]:
                best = (res.overall_probability,
                        [(ec.belief, ec.edge_type.value) for ec in res.edge_contributions])
        if best:
            test.append((nct, 1 if lab == "success" else 0, best[1]))

    y = [t[1] for t in test]
    print(f"graph={a.graph}\nholdout binary trials={len(test)} (succ {sum(y)}/fail {len(y)-sum(y)})")
    print(f"train pool={len(train)}\n")
    print(f"  {'k_trials':>9s} {'AUROC':>7s} {'mean_eff':>9s}  (efficacy softmin, keep-first-k evidence)")

    grid = [5, 10, 25, 50, 100, 150, 250, 350, len(train)]
    for k in grid:
        exclude = set(train[k:])  # keep first-k train NCTs (+ DB priors); drop the rest
        probs = []
        for _nct, _y, edges in test:
            means, w = [], []
            for belief, et in edges:
                b = H._belief_excluding_set(belief, exclude, et)
                if b.evidence_strength <= 0.0:
                    continue
                d = b.alpha + b.beta
                means.append(b.alpha / d if d > 0 else 0.5)
                w.append(_trust_weight(b))
            if means:
                probs.append(float(_aggregate_samples([np.array([m]) for m in means])[0]))
            else:
                probs.append(0.5)
        au = auroc(probs, y) if len(set(y)) > 1 else float("nan")
        print(f"  {k:>9d} {au:>7.3f} {float(np.mean(probs)):>9.3f}")


if __name__ == "__main__":
    main()
