"""Validate the A.4 embedding-merger threshold against the A.5 gold set.

The merger collapses two biology nodes when BioLORD cosine(desc_a, desc_b) >=
threshold. The gold set says which pairs are truly `merge` vs not (parent_of /
child_of / sibling / unrelated should NOT be merged). This sweeps the threshold
and reports precision/recall/F1 for the merge decision, so we pick a threshold
that catches semantic twins without collapsing siblings or parent/child pairs.

Usage:
    python -m scripts.eval_merge_threshold
"""

from __future__ import annotations

import json
from pathlib import Path

GOLD = Path("data/eval/biology_gold_pairs.jsonl")


def main() -> int:
    from src.graph.biolord_embeddings import cosine_similarity, embed_text

    rows = [json.loads(ln) for ln in GOLD.read_text().splitlines() if ln.strip()]
    # Positive = should-merge; negative = everything else.
    scored: list[tuple[float, bool, str]] = []
    cache: dict[str, list[float]] = {}

    def emb(t: str) -> list[float]:
        if t not in cache:
            cache[t] = embed_text(t)
        return cache[t]

    for r in rows:
        a, b, label = r.get("text_a", ""), r.get("text_b", ""), r.get("label")
        if not a.strip() or not b.strip() or label is None:
            continue
        cos = cosine_similarity(emb(a), emb(b))
        scored.append((cos, label == "merge", label))

    n_pos = sum(1 for _, p, _ in scored if p)
    print(f"gold pairs scored: {len(scored)}  (merge={n_pos}, non-merge={len(scored) - n_pos})")

    # cosine distribution by gold label — shows separability of the boundary
    print("\nmean BioLORD cosine by gold label:")
    for lab in ["merge", "parent_of", "child_of", "sibling", "unrelated"]:
        vals = [c for c, _, l in scored if l == lab]
        if vals:
            print(f"  {lab:10} n={len(vals):3}  mean={sum(vals) / len(vals):.3f}  "
                  f"min={min(vals):.3f}  max={max(vals):.3f}")

    print("\nthreshold sweep (merge decision = cosine >= t):")
    print("   t     precision  recall   F1    (TP/FP/FN)")
    best = (0.0, 0.0)  # (f1, threshold)
    for t in [round(0.70 + 0.02 * i, 2) for i in range(16)]:  # 0.70..1.00
        tp = sum(1 for c, p, _ in scored if c >= t and p)
        fp = sum(1 for c, p, _ in scored if c >= t and not p)
        fn = sum(1 for c, p, _ in scored if c < t and p)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        if f1 > best[0]:
            best = (f1, t)
        marker = "  <- default 0.92" if abs(t - 0.92) < 1e-9 else ""
        print(f"  {t:.2f}    {prec:.3f}     {rec:.3f}   {f1:.3f}  ({tp}/{fp}/{fn}){marker}")
    print(f"\nbest F1 = {best[0]:.3f} at threshold {best[1]:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
