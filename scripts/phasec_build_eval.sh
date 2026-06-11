#!/bin/zsh
# Phase C learning-curve driver: build a CLEAN Option-2 fan-out at N trials
# (cached annotations → 0 LLM cost), write its corpus, run the honest K-fold
# holdout. Usage: scripts/phasec_build_eval.sh <N> <area>
set -e
cd /Users/jordanstrasser/Code/eroom-bio/eroom
source .venv/bin/activate
N="${1:?need N}"; AREA="${2:?need area}"

echo "=== build $AREA (n=$N, clean fan-out, cached annotations) ==="
python -m scripts.build_graph --corpus multi_500 --max-trials "$N" \
  --keep-annotations --area "$AREA" --bottom-up --allow-partial-subgraphs \
  > "data/dev/${AREA}_build.log" 2>&1
echo "build exit=$?  tail:"; tail -3 "data/dev/${AREA}_build.log"
[ -f "data/exports/${AREA}_annotated.json" ] || { echo "NO ANNOTATED GRAPH"; exit 1; }

python - "$AREA" <<'PY'
import sys
from pathlib import Path
from src.graph.store import GraphStore
area=sys.argv[1]
g=GraphStore(); g.import_snapshot(f'data/exports/{area}_annotated.json')
ncts=sorted(g.trial_subgraphs.keys())
Path(f'data/corpora/{area}.txt').write_text('\n'.join(ncts)+'\n')
print(f'corpus {area}: {len(ncts)} NCTs in graph')
PY

echo "=== honest K-fold holdout n=$N ==="
python -m scripts.eval_holdout_kfold \
  --initial "data/exports/${AREA}_initial.json" \
  --annotated "data/exports/${AREA}_annotated.json" \
  --corpus "$AREA" --k 5 --n-samples 2000 2>&1 \
  | grep -v "MEAN pooling\|Creating a new\|already attributed\|Saved annotated\|Processed\|Found [0-9]\|efficacy edge update\|Skipped\|Largest changes\|->.*causes_ae\|->.*modulates\|Loading graph\|Loaded:\|Excluding\|tmp.*kfold" \
  | tail -16
