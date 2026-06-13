#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
ANN=data/annotations
INIT=data/exports/multi_500_initial.json
REKEY=data/exports/b1_rekeyed_initial.json
OUT=scratch/diagnostics

echo "=== [1/2] BASELINE re-attribution (content-hash biology) ==="
$PY -m src.annotation.attributor --input "$ANN" --graph "$INIT" \
    --output data/exports/b1_baseline_annotated.json > "$OUT/b1_baseline_attr.log" 2>&1
echo "baseline done: $(ls -la data/exports/b1_baseline_annotated.json | awk '{print $5}') bytes"

echo "=== [2/2] B1 re-attribution (GO-BP biology, gate 0.60) ==="
$PY -m src.annotation.attributor --input "$ANN" --graph "$REKEY" \
    --output data/exports/b1_ontology_annotated.json > "$OUT/b1_ontology_attr.log" 2>&1
echo "b1 done: $(ls -la data/exports/b1_ontology_annotated.json | awk '{print $5}') bytes"
echo "=== B1 A/B ATTRIBUTION COMPLETE ==="
