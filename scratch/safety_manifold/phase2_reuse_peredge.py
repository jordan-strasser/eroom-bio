"""Phase 2 measurement, per-EDGE granularity (apples-to-apples with field P5).

The field's reuse bar is per-parameter (per edge), not per-entity-profile. At
the profile level almost every entity clears >=8 (a 12-AE compound sums past 8
trivially), which hides where borrowing matters. Per (entity, AE) edge:

  own_mass   = alpha+beta-2 from the entity's own exact-id evidence
  borrowed   = kernel-weighted neighbor evidence
  multiplier = borrowed / own         (field-comparable cross-node multiplier)
  %>=8       = share of edges whose effective mass clears 8, exact vs +borrow

Reported overall and on the SPARSE subset (own < 8) where exact-id is starved
and borrowing should concentrate. Plus the novel-entity inheritance (exclude_self).
"""
from __future__ import annotations

import sys
import statistics

sys.path.insert(0, "scratch/safety_manifold")
from geometry import load_geometry, _belief_strength  # noqa: E402
from borrow import SafetyManifold  # noqa: E402

BAR = 8.0


def _pct(x, n):
    return 100 * x / n if n else 0.0


def compound_edges(geo, man):
    """yield (own_mass, borrowed_mass) per causes_ae edge with own evidence."""
    for cid in geo.compound_fp:
        for ae_id, bel in geo.causes_ae.get(cid, {}).items():
            if _belief_strength(bel) < 1.0:
                continue
            bb = man.borrowed_causes_ae(cid, ae_id)
            yield bb.own_mass, bb.borrowed_mass


def target_edges(geo, man):
    for tid in geo.target_pathways:
        if not geo.target_pathways[tid]:
            continue
        for ae_id, bel in geo.target_ae.get(tid, {}).items():
            if ae_id.startswith("AE:soc:") or _belief_strength(bel) < 1.0:
                continue
            bb = man.borrowed_target_ae(tid, ae_id)
            yield bb.own_mass, bb.borrowed_mass


def summarize(edges):
    edges = [(o, b) for o, b in edges if o > 0]
    n = len(edges)
    mult = [b / o for o, b in edges]
    own_ge8 = sum(1 for o, b in edges if o >= BAR)
    brw_ge8 = sum(1 for o, b in edges if (o + b) >= BAR)
    sparse = [(o, b) for o, b in edges if o < BAR]
    sparse_mult = [b / o for o, b in sparse]
    sparse_rescued = sum(1 for o, b in sparse if (o + b) >= BAR)
    return {
        "n": n,
        "mult_med": statistics.median(mult) if mult else 0,
        "mult_mean": statistics.mean(mult) if mult else 0,
        "own_ge8": _pct(own_ge8, n),
        "brw_ge8": _pct(brw_ge8, n),
        "n_sparse": len(sparse),
        "sparse_mult_med": statistics.median(sparse_mult) if sparse_mult else 0,
        "sparse_mult_mean": statistics.mean(sparse_mult) if sparse_mult else 0,
        "sparse_rescued_pct": _pct(sparse_rescued, len(sparse)),
    }


if __name__ == "__main__":
    snap = sys.argv[1] if len(sys.argv) > 1 else "data/exports/multi_500_annotated.json"
    geo = load_geometry(snap)
    print(f"snapshot: {snap}\n")

    print("=== COMPOUND manifold, per causes_ae edge ===")
    print(f"{'bw':>5} {'smin':>5} {'n_edge':>7} {'mult_med':>9} {'mult_mean':>10} "
          f"{'own%≥8':>7} {'brw%≥8':>7} | {'sparse':>6} {'spMult_md':>10} {'spRescue':>9}")
    for bw, smin in [(0.3, 0.3), (0.5, 0.3), (0.5, 0.2), (0.5, 0.4)]:
        man = SafetyManifold(geo, bw_compound=bw, simmin_compound=smin)
        s = summarize(compound_edges(geo, man))
        print(f"{bw:>5.2f} {smin:>5.2f} {s['n']:>7} {s['mult_med']:>9.3f} {s['mult_mean']:>10.3f} "
              f"{s['own_ge8']:>6.0f}% {s['brw_ge8']:>6.0f}% | {s['n_sparse']:>6} "
              f"{s['sparse_mult_med']:>10.3f} {s['sparse_rescued_pct']:>8.0f}%")

    print("\n=== TARGET manifold, per target_associated_ae edge ===")
    print(f"{'bw':>5} {'smin':>5} {'n_edge':>7} {'mult_med':>9} {'mult_mean':>10} "
          f"{'own%≥8':>7} {'brw%≥8':>7} | {'sparse':>6} {'spMult_md':>10} {'spRescue':>9}")
    for bw, smin in [(0.25, 0.1), (0.4, 0.1), (0.4, 0.05), (0.4, 0.2)]:
        man = SafetyManifold(geo, bw_target=bw, simmin_target=smin)
        s = summarize(target_edges(geo, man))
        print(f"{bw:>5.2f} {smin:>5.2f} {s['n']:>7} {s['mult_med']:>9.3f} {s['mult_mean']:>10.3f} "
              f"{s['own_ge8']:>6.0f}% {s['brw_ge8']:>6.0f}% | {s['n_sparse']:>6} "
              f"{s['sparse_mult_med']:>10.3f} {s['sparse_rescued_pct']:>8.0f}%")
