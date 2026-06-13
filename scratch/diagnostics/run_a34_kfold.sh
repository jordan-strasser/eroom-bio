#!/usr/bin/env bash
# A3/A4 honest k-fold holdout: baseline (routing off) vs routed (routing on).
# Same initial, same corpus, same folds (md5(nct)%k) — only the flag differs.
# Each fold re-attributes the initial EXCLUDING that fold (leakage guard), then
# predicts the held-out trials. The flag is read at attribute() call time.
set -euo pipefail
cd "$(dirname "$0")/../.."

PY=.venv/bin/python
INIT=data/exports/multi_500_initial.json
OUT=scratch/diagnostics

echo "=== BASELINE k-fold (EROOM_ROUTING off) ==="
EROOM_ROUTING=0 $PY -m scripts.eval_holdout_kfold \
    --initial "$INIT" \
    --annotated data/exports/multi_500_baseline_reattr.json \
    --corpus multi_500 --k 5 \
    > "$OUT/a34_kfold_baseline.txt" 2>&1
echo "baseline k-fold done"
grep -E "AUROC|gap|binary acc|Excluding [0-9]+ NCT" "$OUT/a34_kfold_baseline.txt" | head -20 || true

echo "=== ROUTED k-fold (EROOM_ROUTING on) ==="
EROOM_ROUTING=1 $PY -m scripts.eval_holdout_kfold \
    --initial "$INIT" \
    --annotated data/exports/multi_500_routed_reattr.json \
    --corpus multi_500 --k 5 \
    > "$OUT/a34_kfold_routed.txt" 2>&1
echo "routed k-fold done"
grep -E "AUROC|gap|binary acc|Excluding [0-9]+ NCT" "$OUT/a34_kfold_routed.txt" | head -20 || true

echo "=== KFOLD COMPLETE ==="
