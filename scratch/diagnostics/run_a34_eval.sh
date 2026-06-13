#!/usr/bin/env bash
# A3/A4 aggregate verification runner. Re-attributes multi_500_initial under
# EROOM_ROUTING off (baseline) and on (routed), both from the SAME initial so
# the only difference is the flag, then probes the contamination/spread metrics.
# The k-fold holdout (heavy) is launched separately after this validates.
set -euo pipefail
cd "$(dirname "$0")/../.."

PY=.venv/bin/python
INIT=data/exports/multi_500_initial.json
ANN=data/annotations
OUT=scratch/diagnostics
BASE=data/exports/multi_500_baseline_reattr.json
ROUTED=data/exports/multi_500_routed_reattr.json

echo "=== [1/3] baseline re-attribution (EROOM_ROUTING off) ==="
EROOM_ROUTING=0 $PY -m src.annotation.attributor \
    --input "$ANN" --graph "$INIT" --output "$BASE" \
    > "$OUT/a34_baseline_reattr.log" 2>&1
echo "baseline done: $(ls -la $BASE | awk '{print $5}') bytes"

echo "=== [2/3] routed re-attribution (EROOM_ROUTING on) ==="
EROOM_ROUTING=1 $PY -m src.annotation.attributor \
    --input "$ANN" --graph "$INIT" --output "$ROUTED" \
    > "$OUT/a34_routed_reattr.log" 2>&1
echo "routed done: $(ls -la $ROUTED | awk '{print $5}') bytes"

echo "=== [3/3] contamination (#2) + efficacy-spread (#3) probe ==="
$PY -m scratch.diagnostics.probe_routing_metrics "$BASE" "$ROUTED" \
    | tee "$OUT/a34_metrics.txt"

echo "=== REATTR+PROBE COMPLETE ==="
