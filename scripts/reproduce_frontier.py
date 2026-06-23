"""One tracked command to reproduce the frontier headline numbers DIRECTLY from
the built graph — no untracked scratch dump (onco_graph_preds.json) involved.

It runs the two tracked eval harnesses, seeded for exact reproducibility:
  1. scripts.eval_holdout_kfold  → holdout AUROC + in-sample (re-attributes the
     initial per fold, excluding the fold, then predicts — leak-free).
  2. scripts.eval_baselines --with-graph → ΔAUROC of the graph vs the 4-field
     design baseline on the SAME folds, paired DeLong.

With the frontier config baked into src/config.py, NO env vars are needed (the
old `EROOM_ROUTING=1 EROOM_BIO_ONTOLOGY=1` are now the unconditional defaults).

Reproduces (seed 42): holdout AUROC 0.701 / in-sample 0.768 / acc 0.713, and
ΔAUROC(graph − design) ≈ +0.01 (DeLong p≈0.84 — a tie head-to-head; the graph's
incremental alpha is shown by the stacked tests in ONCOLOGY_DIG_RESULTS.md).

Usage:
    python -m scripts.reproduce_frontier
    python -m scripts.reproduce_frontier --initial ... --annotated ... --corpus ... --seed 42
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_DEF_INITIAL = "data/exports/onco_scale_500_rekeyed_initial.json"
_DEF_ANNOTATED = "data/exports/onco_scale_500_frontier_annotated.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--initial", default=_DEF_INITIAL,
                    help="pre-attribution initial.json (re-attributed per fold)")
    ap.add_argument("--annotated", default=_DEF_ANNOTATED,
                    help="trained annotated.json (chain resolution + in-sample)")
    ap.add_argument("--corpus", default="onco_scale_500",
                    help="corpus name under data/corpora/")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42,
                    help="prediction-RNG seed for exact reproducibility")
    args = ap.parse_args()

    for p in (args.initial, args.annotated):
        if not Path(p).exists():
            sys.exit(
                f"missing snapshot: {p}\n"
                "Build it first with scripts.build_graph (a fresh checkout has no "
                "exports — they are regenerable, not tracked)."
            )

    py = sys.executable
    common = ["--corpus", args.corpus, "--k", str(args.k), "--seed", str(args.seed)]

    print("=" * 70)
    print("1/2  Holdout AUROC  (scripts.eval_holdout_kfold)")
    print("=" * 70, flush=True)
    subprocess.run(
        [py, "-m", "scripts.eval_holdout_kfold",
         "--initial", args.initial, "--annotated", args.annotated, *common],
        check=True,
    )

    print("\n" + "=" * 70)
    print("2/2  ΔAUROC vs 4-field design baseline, paired DeLong  "
          "(scripts.eval_baselines --with-graph)")
    print("=" * 70, flush=True)
    subprocess.run(
        [py, "-m", "scripts.eval_baselines", "--with-graph",
         "--graph", args.annotated, "--initial", args.initial, *common],
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
