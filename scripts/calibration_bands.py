"""In-sample 3-band calibration: per-trial prediction from a trial's OWN data.

For every trial in the graph, predict its STATED chain(s) in-sample (the trial's
own evidence is on the edges) and bucket the predicted P(success) by the trial's
true 3-way outcome. The calibration targets (owner, 2026-06-09):
    success   mean > 0.70
    ambiguous mean 0.45-0.55
    failure   mean < 0.40
This is the in-sample BASELINE (a sanity/calibration check), NOT generalization —
out-of-sample is the harder test and is at the multi-causal ceiling. Re-run after
changing a knob (env: EROOM_SOFTMIN_T, EROOM_PRIOR_MEAN/STRENGTH, EROOM_AGG,
EROOM_SAFETY_*) to see the effect on the bands.

Run: python -m scripts.calibration_bands --graph data/exports/multi_500_annotated.json
"""
from __future__ import annotations

import argparse
import statistics as st
from collections import defaultdict
from pathlib import Path

from src.graph.store import GraphStore
import scripts.holdout_thesis_analysis as H

ANN = Path("data/annotations")
TARGET = {"success": ("> 0.70", lambda p: p > 0.70),
          "ambiguous": ("0.45-0.55", lambda p: 0.45 <= p <= 0.55),
          "failure": ("< 0.40", lambda p: p < 0.40)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--n-samples", type=int, default=1500)
    a = ap.parse_args()
    g = GraphStore()
    g.import_snapshot(a.graph)

    by_class: dict[str, list[float]] = defaultdict(list)
    n_uncovered = 0
    for nct in g.trial_subgraphs:
        ext = H._load_json(ANN / f"{nct}_extraction.json")
        cls = H._load_json(ANN / f"{nct}_classification.json")
        lab = H._resolve_label(ext, cls) if ext else None
        if lab not in ("success", "failure", "ambiguous"):
            continue
        res = H._decisive_result(g, nct, a.n_samples)
        if res is None:
            n_uncovered += 1
            continue
        by_class[lab].append(res.overall_probability)

    print(f"\nIN-SAMPLE 3-BAND CALIBRATION (per-trial, min-overall stated chain)")
    print(f"graph={a.graph}  uncovered={n_uncovered}\n")
    print(f"{'class':10s} {'n':>4s} {'mean':>6s} {'median':>7s} {'p10':>6s} {'p90':>6s} "
          f"{'in-band':>8s}   target")
    for cls_name in ("success", "ambiguous", "failure"):
        ps = sorted(by_class.get(cls_name, []))
        if not ps:
            continue
        tgt, hit = TARGET[cls_name]
        inb = sum(hit(p) for p in ps) / len(ps)
        p10 = ps[int(0.1 * len(ps))]
        p90 = ps[min(int(0.9 * len(ps)), len(ps) - 1)]
        print(f"{cls_name:10s} {len(ps):>4d} {st.mean(ps):>6.3f} {st.median(ps):>7.3f} "
              f"{p10:>6.3f} {p90:>6.3f} {inb*100:>6.0f}%   {tgt}")
    s, f = by_class.get("success"), by_class.get("failure")
    if s and f:
        print(f"\nseparation (mean success − mean failure): {st.mean(s) - st.mean(f):+.3f}")
        # how many failures sit ABOVE the success band floor (the costly overlap)
        over = sum(1 for p in f if p > 0.55) / len(f)
        print(f"failures predicted >0.55 (false-confidence): {over*100:.0f}%")


if __name__ == "__main__":
    main()
