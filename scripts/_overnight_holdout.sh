#!/bin/bash
# Overnight north-star holdout pipeline: build out-of-sample demo graph → run 3-axis+ablation harness.
# Robust to partial failures so a few bad extractions can't abort the whole run.
cd /Users/jordanstrasser/Code/eroom-bio/eroom || exit 1
source .venv/bin/activate 2>/dev/null

echo "=== PIPELINE START $(date) ==="

# 1. commit the frozen holdout corpus (reproducible spec)
git add data/corpora/holdout_2021_2026.txt 2>/dev/null
git commit -q -m "data: 2021-2026 holdout corpus (42 trials, coverage-checked, 0 train overlap)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>" 2>/dev/null || echo "(corpus already committed)"

# exclude-from-attribution = every holdout NCT (out-of-sample, even though 0 overlap)
EXCLUDE=$(grep -vE '^#|^$' data/corpora/holdout_2021_2026.txt | awk '{print $1}' | paste -sd, -)
echo "holdout exclude list: $(echo "$EXCLUDE" | tr ',' '\n' | wc -l | tr -d ' ') NCTs"

# 2. build the out-of-sample demo graph (extract 42 → populate → merge into base → attribute EXCLUDING holdout)
echo "=== BUILD START $(date) ==="
python -m scripts.build_graph \
  --base-snapshot data/exports/multi_500_annotated.json \
  --add-corpus holdout_2021_2026 --area multi_500_holdout \
  --exclude-from-attribution "$EXCLUDE" \
  --keep-annotations --include-terminated --concurrency 8 \
  --allow-partial-classify --allow-partial-subgraphs \
  > /tmp/holdout_build.log 2>&1
echo "=== BUILD exit=$? $(date) ==="
grep -E "credit balance|Aborting|classify success|backbone complete|chain coverage|Step 4.5|Final snapshot" /tmp/holdout_build.log | tail -8

# 3. commit the paid annotations regardless of harness outcome (protect the spend)
git add data/annotations/ 2>/dev/null
git commit -q -m "data: 2021-2026 holdout extractions + classifications (paid)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>" 2>/dev/null || echo "(no new annotations to commit)"

# 4. run the 3-axis + ablation harness if the graph was produced
if [ -f data/exports/multi_500_holdout_annotated.json ]; then
  echo "=== HARNESS START $(date) ==="
  python -m scripts.holdout_thesis_analysis \
    --graph data/exports/multi_500_holdout_annotated.json \
    --holdout-ncts data/corpora/holdout_2021_2026.txt \
    --threshold 0.5 > /tmp/holdout_results.txt 2>&1
  echo "=== HARNESS exit=$? $(date) ==="
  # also run the cross-indication census on the demo graph for the Axis-1 record
  python -m scripts.cross_indication_census \
    --graph data/exports/multi_500_holdout_annotated.json \
    > /tmp/holdout_census.txt 2>&1
else
  echo "=== NO DEMO GRAPH — build failed; see /tmp/holdout_build.log ==="
fi

echo "=== PIPELINE_DONE $(date) ==="
