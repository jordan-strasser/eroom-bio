"""Phase C measure-first: how should the predictor aggregate the fan-out's
multiple pathway-chains per trial?

With the Option-2 pathway fan-out a trial decomposes into MANY stated chains
(one per Reactome pathway). `predict_clinical_hypothesis` currently returns the
``min`` (most pessimistic) overall_probability across them — so a broader gene
footprint mechanically lowers P(success) (the worst pathway dominates). Before
implementing any "marginalize over pathways" change, MEASURE which per-trial
aggregation best separates success from failure. The data picks the
marginalization; we don't assume.

Aggregations compared (per trial, over its stated chains' overall_probability):
  min        — current behavior (weakest pathway-chain wins)
  mean       — flat marginal over pathways
  max        — best pathway-chain (drug works through SOME pathway)
  median     — robust-central pathway
  best_evid  — the chain with the most total edge evidence (trust the
               best-accumulated pathway — the fan-out's cross-trial substrate)

IN-SAMPLE (the trial's own evidence is in the beliefs) — so the ABSOLUTE AUROC is
optimistic, but the RELATIVE ranking across aggregations is robust (the optimism
is shared). The honest absolute belongs to the kfold / exclude-from-attribution
harness; this answers only "which aggregation, and does min bias the fan-out?".

Usage: python -m scripts.phasec_aggregation_diagnostic \
         --graph data/exports/phasec250_annotated.json \
         --annotations data/annotations
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median

from scripts.eval_holdout_compose import _auroc, _resolve_label
from src.graph.store import GraphStore
from src.prediction.path_query import PredictionEngine


def _load_label(annotations: Path, nct: str) -> str | None:
    ext_p = annotations / f"{nct}_extraction.json"
    cls_p = annotations / f"{nct}_classification.json"
    if not ext_p.exists():
        return None
    extraction = json.loads(ext_p.read_text())
    classification = json.loads(cls_p.read_text()) if cls_p.exists() else None
    return _resolve_label(extraction, classification)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--annotations", default="data/annotations")
    ap.add_argument("--n-samples", type=int, default=2000)
    args = ap.parse_args()

    g = GraphStore()
    g.import_snapshot(args.graph)
    engine = PredictionEngine(g)
    annotations = Path(args.annotations)

    aggs = {
        "min": min,
        "mean": mean,
        "max": max,
        "median": median,
    }
    # per-aggregation probability lists (aligned with `labels`)
    series: dict[str, list[float]] = {k: [] for k in aggs}
    series["best_evid"] = []
    labels: list[int] = []

    n_trials = 0
    fanout_sizes: list[int] = []
    for ts in g.trial_subgraphs.values():
        label = _load_label(annotations, ts.trial_id)
        if label not in ("success", "failure"):
            continue
        # predict each stated chain; keep (overall_probability, total evidence)
        scored: list[tuple[float, float]] = []
        for ch in ts.chains:
            try:
                res = engine.predict(ch, n_samples=args.n_samples)
            except KeyError:
                continue  # chain references a node pruned from the merged graph
            if not res.edge_contributions:
                continue
            ev = sum(c.belief.evidence_strength for c in res.edge_contributions)
            scored.append((res.overall_probability, ev))
        if not scored:
            continue
        n_trials += 1
        fanout_sizes.append(len(scored))
        probs = [p for p, _ in scored]
        for name, fn in aggs.items():
            series[name].append(float(fn(probs)))
        # best_evidence: overall_probability of the most-evidenced chain
        series["best_evid"].append(max(scored, key=lambda t: t[1])[0])
        labels.append(1 if label == "success" else 0)

    pos = sum(labels)
    print(f"trials scored: {n_trials}  (success={pos}, failure={len(labels) - pos})")
    print(f"fan-out chains/trial: mean {mean(fanout_sizes):.1f} "
          f"median {int(median(fanout_sizes))} max {max(fanout_sizes)}")
    print(f"base rate (success): {pos / len(labels):.3f}\n")
    print("per-trial aggregation → IN-SAMPLE AUROC (relative ranking is the signal):")
    rows = sorted(
        ((name, _auroc(series[name], labels)) for name in series),
        key=lambda kv: (kv[1] if kv[1] == kv[1] else -1), reverse=True,
    )
    for name, au in rows:
        mark = "  ← current" if name == "min" else ""
        print(f"  {name:10s} {au:.3f}{mark}")


if __name__ == "__main__":
    main()
