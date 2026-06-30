"""Cross-indication bridge census — does the biology abstraction bridge diseases?

Enumerates every ``BiologyNode`` that drives ≥2 canonical indications (a
structural cross-indication bridge) and every ``mechanism_affects`` edge
co-updated by trials from ≥2 indications. Ranks by whether the bridge spans
*therapeutic areas* (oncology ↔ other — the north-star claim) and by the
strong/weak belief contrast across indications (same pathway, opposite
clinical fate by disease — the differentiated finding a flat model can't make).

    python -m scripts.bridge_census --graph data/exports/onco_scale_500_enr_annotated.json
    python -m scripts.bridge_census --graph <g> --cross-area-only --min-spread 0.2
    python -m scripts.bridge_census --graph <g> --json out.json

See ``src/prediction/provenance.py`` for the analysis itself.
"""

from __future__ import annotations

import argparse
import json

from src.graph.models import EdgeType
from src.graph.store import GraphStore
from src.prediction.bridges import (
    BiologyBridge,
    find_biology_bridges,
    find_mechanism_bridges,
)


def _has_real_nononco_side(bridge: BiologyBridge) -> bool:
    """A *clean* cross-area bridge: an oncology side AND a non-oncology side
    that is a genuine disease (not a tox/procedure/non-answer slug)."""
    has_onco = any(s.is_oncology for s in bridge.sides)
    has_real_other = any(
        (not s.is_oncology) and (not s.is_non_disease) for s in bridge.sides
    )
    return has_onco and has_real_other


def _fmt_side(s) -> str:
    flag = "  ⚠non-disease" if s.is_non_disease else ""
    nct = s.source_ncts[0] if s.source_ncts else "—"
    more = f" +{len(s.source_ncts) - 1}" if len(s.source_ncts) > 1 else ""
    return (
        f"      {s.indication_id[:34]:<34} {s.area:<22} "
        f"E[p]={s.expected_probability:.2f}  n_eff={s.evidence_strength:5.1f}  "
        f"rec={s.n_records:<3} {nct}{more}{flag}"
    )


def _print_bridge(idx: int, b: BiologyBridge) -> None:
    print(
        f"\n[{idx}] {b.biology_name}   ({b.biology_id}, {b.id_scheme})"
        f"   spread={b.belief_spread:.2f}  areas={b.n_areas}  inds={b.n_indications}"
    )
    for s in b.sides:
        print(_fmt_side(s))
    if b.strongest is not None and b.weakest is not None and b.belief_spread > 0:
        print(
            f"      → STRONG in {b.strongest.indication_id} "
            f"({b.strongest.expected_probability:.2f}) vs WEAK in "
            f"{b.weakest.indication_id} ({b.weakest.expected_probability:.2f})"
        )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--graph", required=True, help="trained annotated.json snapshot")
    ap.add_argument(
        "--cross-area-only",
        action="store_true",
        help="only show bridges spanning oncology and a real non-onco disease",
    )
    ap.add_argument(
        "--min-spread",
        type=float,
        default=0.0,
        help="only show biology bridges with belief spread ≥ this (the "
        "strong/weak contrast; default 0 = all)",
    )
    ap.add_argument("--limit", type=int, default=30, help="max bridges per section")
    ap.add_argument(
        "--spread-evidence-floor",
        type=float,
        default=1.0,
        help="min evidence_strength for a side to count in the spread (so a "
        "single weak observation isn't read as a contradiction)",
    )
    ap.add_argument("--json", default=None, help="also dump full results to this path")
    args = ap.parse_args()

    store = GraphStore()
    store.import_snapshot(args.graph)

    # Edge-source breakdown (context for the BiologyNode-only restriction).
    bd = store.get_edges_by_type(EdgeType.BIOLOGY_DRIVES)
    src_bio = sum(
        1 for e in bd if store.get_node(e["source_id"]).get("node_type") == "BiologyNode"
    )

    bio_bridges = find_biology_bridges(
        store, spread_evidence_floor=args.spread_evidence_floor
    )
    mech_bridges = find_mechanism_bridges(store)

    cross_area = [b for b in bio_bridges if _has_real_nononco_side(b)]

    print("=" * 78)
    print(f"BRIDGE CENSUS — {args.graph}")
    print("=" * 78)
    print(
        f"biology_drives edges: {len(bd)}  "
        f"(BiologyNode-sourced: {src_bio} / TargetNode: {len(bd) - src_bio})"
    )
    print(f"BiologyNodes driving ≥2 canonical indications: {len(bio_bridges)}")
    print(
        f"  └─ spanning oncology AND a real non-onco disease: {len(cross_area)}"
        "   ← north-star bridges"
    )
    print(
        f"mechanism_affects edges co-updated by ≥2 indications: {len(mech_bridges)}"
    )

    # ── Section 1: cross-area biology bridges (the north-star gold) ──
    sel = sorted(
        cross_area,
        key=lambda b: (b.belief_spread, b.n_areas, b.n_indications),
        reverse=True,
    )
    print("\n" + "─" * 78)
    print("CROSS-AREA BIOLOGY BRIDGES (oncology ↔ other disease)")
    print("─" * 78)
    if not sel:
        print("  (none — the abstraction does not bridge areas on this corpus)")
    for i, b in enumerate(sel[: args.limit], 1):
        _print_bridge(i, b)

    # ── Section 2: high-contrast bridges (strong here / weak there) ──
    if not args.cross_area_only:
        contrast = [
            b
            for b in bio_bridges
            if b.belief_spread >= max(args.min_spread, 0.15)
            and b not in cross_area
        ]
        contrast.sort(key=lambda b: b.belief_spread, reverse=True)
        print("\n" + "─" * 78)
        print("HIGH-CONTRAST BRIDGES (same biology, strong in one disease / weak in another)")
        print("─" * 78)
        if not contrast:
            print("  (none above the spread threshold)")
        for i, b in enumerate(contrast[: args.limit], 1):
            _print_bridge(i, b)

    # ── Section 3: mechanism bridges (one belief, multiple diseases) ──
    print("\n" + "─" * 78)
    print("MECHANISM BRIDGES (one mechanism_affects belief, trials from ≥2 indications)")
    print("─" * 78)
    mech_sel = sorted(
        mech_bridges,
        key=lambda m: (m.spans_oncology_and_other, len(m.indications)),
        reverse=True,
    )
    if args.cross_area_only:
        mech_sel = [m for m in mech_sel if m.spans_oncology_and_other]
    if not mech_sel:
        print("  (none)")
    for i, m in enumerate(mech_sel[: args.limit], 1):
        tag = "  ← cross-area" if m.spans_oncology_and_other else ""
        print(
            f"\n[{i}] {m.mechanism_name} → {m.biology_name}   "
            f"E[p]={m.expected_probability:.2f} n_eff={m.evidence_strength:.1f}{tag}"
        )
        for ind, ncts in sorted(m.ncts_by_indication.items()):
            print(f"      {ind[:36]:<36} {', '.join(ncts[:4])}")

    if args.json:
        payload = {
            "biology_bridges": [b.model_dump() for b in bio_bridges],
            "mechanism_bridges": [m.model_dump() for m in mech_bridges],
        }
        with open(args.json, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
