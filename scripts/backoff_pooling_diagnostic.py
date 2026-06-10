"""Phase-A diagnostic: does hierarchical backoff pooling actually fire, and at
what evidence-strength scale?

The current resolvers (`_resolve_indication_edge`, `_resolve_responds_differently`)
return the FIRST evidenced level most-specific-first, so a sparse leaf shadows a
rich parent. The Phase-A fix pools them. This script answers two questions on a
built graph BEFORE the fix:

1. FIRE RATE — how many chain (src, leaf-indication) / (leaf-pop, indication)
   pairs have BOTH the leaf edge AND a coarser ancestor edge carrying evidence?
   Those are exactly the pairs whose prediction changes under pooling. If ~0,
   the fix is inert on this corpus.
2. SCALE — the evidence_strength distribution of the leaf vs the ancestor in
   those firing pairs, to pick the prior-cap τ in the right order of magnitude
   (τ ≈ "a few strong trials' worth" of cross-population transfer credit).

Usage: python -m scripts.backoff_pooling_diagnostic [path_to_annotated.json]
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from statistics import median


def _strength(belief: dict) -> float:
    return float(belief.get("alpha", 1.0)) + float(belief.get("beta", 1.0)) - 2.0


def _mean(belief: dict) -> float:
    a = float(belief.get("alpha", 1.0))
    b = float(belief.get("beta", 1.0))
    return a / (a + b) if (a + b) > 0 else 0.5


def _population_ancestors(pop_id: str) -> list[str]:
    axes = pop_id.split("__")
    if len(axes) <= 1:
        return [pop_id]
    out = [pop_id]
    for k in range(len(axes) - 1, 0, -1):
        for combo in combinations(axes, k):
            out.append("__".join(combo))
    return out


def _pctiles(xs: list[float]) -> str:
    if not xs:
        return "(none)"
    xs = sorted(xs)
    n = len(xs)

    def q(p: float) -> float:
        return xs[min(n - 1, int(p * n))]

    return (
        f"n={n} min={xs[0]:.1f} p25={q(0.25):.1f} med={median(xs):.1f} "
        f"p75={q(0.75):.1f} p90={q(0.90):.1f} max={xs[-1]:.1f}"
    )


def main(path: str) -> None:
    d = json.load(open(path))
    g = d["graph"]
    edges = g["edges"]

    # SUBTYPE_OF: leaf -> parent (indication hierarchy)
    subtype_parent: dict[str, list[str]] = {}
    for e in edges:
        if e["edge_type"] == "subtype_of":
            subtype_parent.setdefault(e["source"], []).append(e["target"])

    # Index belief by (src, tgt, edge_type)
    belief_idx: dict[tuple[str, str, str], dict] = {}
    for e in edges:
        belief_idx[(e["source"], e["target"], e["edge_type"])] = e.get("belief", {})

    def indication_ancestors(ind: str, max_depth: int = 5) -> list[str]:
        out = [ind]
        seen = {ind}
        cur = ind
        for _ in range(max_depth):
            parents = [p for p in subtype_parent.get(cur, []) if p not in seen]
            if not parents:
                break
            cur = parents[0]
            seen.add(cur)
            out.append(cur)
        return out

    # ── Indication-targeted edges (biology_drives, endpoint_captures) ──
    for et in ("biology_drives", "endpoint_captures"):
        fire = 0
        total_leaf_evidenced = 0
        leaf_str: list[float] = []
        anc_str: list[float] = []
        examples: list[str] = []
        for (src, tgt, etype), belief in belief_idx.items():
            if etype != et:
                continue
            if _strength(belief) <= 0:
                continue
            ancs = indication_ancestors(tgt)
            if len(ancs) <= 1:
                continue  # leaf has no parent in the hierarchy
            total_leaf_evidenced += 1
            # nearest evidenced strict ancestor
            for anc in ancs[1:]:
                ab = belief_idx.get((src, anc, et))
                if ab and _strength(ab) > 0:
                    fire += 1
                    leaf_str.append(_strength(belief))
                    anc_str.append(_strength(ab))
                    if len(examples) < 6:
                        examples.append(
                            f"  {src[:24]:24s} {tgt[:18]:18s}(str {_strength(belief):.0f},"
                            f" mean {_mean(belief):.2f}) <- {anc[:18]:18s}"
                            f"(str {_strength(ab):.0f}, mean {_mean(ab):.2f})"
                        )
                    break
        print(f"\n=== {et} ===")
        print(f"leaf edges with evidence & a hierarchy parent: {total_leaf_evidenced}")
        print(f"  of those, FIRING (parent also evidenced):    {fire}")
        print(f"  leaf strength   {_pctiles(leaf_str)}")
        print(f"  parent strength {_pctiles(anc_str)}")
        for ex in examples:
            print(ex)

    # ── Population-targeted edges (responds_differently) ──
    fire = 0
    total_leaf_evidenced = 0
    leaf_str = []
    anc_str = []
    examples = []
    for (src, tgt, etype), belief in belief_idx.items():
        if etype != "responds_differently":
            continue
        if _strength(belief) <= 0:
            continue
        ancs = _population_ancestors(src)
        if len(ancs) <= 1:
            continue
        total_leaf_evidenced += 1
        for anc in ancs[1:]:
            ab = belief_idx.get((anc, tgt, "responds_differently"))
            if ab and _strength(ab) > 0:
                fire += 1
                leaf_str.append(_strength(belief))
                anc_str.append(_strength(ab))
                if len(examples) < 6:
                    examples.append(
                        f"  {src[:30]:30s}(str {_strength(belief):.0f}) <- "
                        f"{anc[:24]:24s}(str {_strength(ab):.0f})"
                    )
                break
    print("\n=== responds_differently ===")
    print(f"leaf edges with evidence & a hierarchy parent: {total_leaf_evidenced}")
    print(f"  of those, FIRING (parent also evidenced):    {fire}")
    print(f"  leaf strength   {_pctiles(leaf_str)}")
    print(f"  parent strength {_pctiles(anc_str)}")
    for ex in examples:
        print(ex)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/exports/multi_500_annotated.json")
