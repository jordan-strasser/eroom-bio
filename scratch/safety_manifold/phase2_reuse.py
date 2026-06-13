"""Phase 2 measurement — effective reuse the kernel manufactures.

Scoreboard (vs the BioLORD field's 0.17x cross-trial multiplier):
  cross-node multiplier = borrowed_mass / own_mass  (per entity, kernel-weighted
  neighbor evidence relative to the entity's own exact-id evidence). A real,
  aligned kernel should exceed 1x.

Also: % of entities whose AE-profile effective mass clears >= 8 (the reuse bar
from the synth harness), own-only vs own+borrowed; and the novel-entity preview
(exclude_self: what an entirely held-out entity inherits from neighbors alone).
Bandwidth/sim_min are swept and recorded.
"""
from __future__ import annotations

import sys
import statistics

sys.path.insert(0, "scratch/safety_manifold")
from geometry import load_geometry, _belief_strength  # noqa: E402
from borrow import SafetyManifold  # noqa: E402

REUSE_BAR = 8.0


def _frac_ge(vals, bar):
    return sum(1 for v in vals if v >= bar) / len(vals) if vals else 0.0


def measure_compound(geo, bw, smin):
    man = SafetyManifold(geo, bw_compound=bw, simmin_compound=smin)
    cids = [c for c in geo.compound_fp
            if any(_belief_strength(b) >= 1.0 for b in geo.causes_ae.get(c, {}).values())]
    mults, own_only, with_borrow, nov = [], [], [], []
    n_with_nb = 0
    for c in cids:
        own, brc = man.compound_profile_reuse(c)
        if man.compound_neighbors(c):
            n_with_nb += 1
        if own > 0:
            mults.append(brc / own)
        own_only.append(own)
        with_borrow.append(own + brc)
        # novel-entity: drop own evidence, keep only borrowed
        nov_mass = sum(man.borrowed_causes_ae(c, ae, exclude_self=True).borrowed_mass
                       for ae in geo.causes_ae.get(c, {}))
        nov.append(nov_mass)
    return {
        "n": len(cids), "n_with_nb": n_with_nb,
        "mult_median": statistics.median(mults) if mults else 0.0,
        "mult_mean": statistics.mean(mults) if mults else 0.0,
        "ge8_own": _frac_ge(own_only, REUSE_BAR),
        "ge8_borrow": _frac_ge(with_borrow, REUSE_BAR),
        "nov_ge1": _frac_ge(nov, 1.0),
        "nov_median": statistics.median(nov) if nov else 0.0,
    }


def measure_target(geo, bw, smin):
    man = SafetyManifold(geo, bw_target=bw, simmin_target=smin)
    tids = [t for t in geo.target_pathways
            if geo.target_pathways[t]
            and any(_belief_strength(b) >= 1.0
                    for ae, b in geo.target_ae.get(t, {}).items()
                    if not ae.startswith("AE:soc:"))]
    mults, own_only, with_borrow, nov = [], [], [], []
    n_with_nb = 0
    for t in tids:
        own, brc = man.target_profile_reuse(t)
        if man.target_neighbors(t):
            n_with_nb += 1
        if own > 0:
            mults.append(brc / own)
        own_only.append(own)
        with_borrow.append(own + brc)
        nov_mass = sum(man.borrowed_target_ae(t, ae, exclude_self=True).borrowed_mass
                       for ae in geo.target_ae.get(t, {}) if not ae.startswith("AE:soc:"))
        nov.append(nov_mass)
    return {
        "n": len(tids), "n_with_nb": n_with_nb,
        "mult_median": statistics.median(mults) if mults else 0.0,
        "mult_mean": statistics.mean(mults) if mults else 0.0,
        "ge8_own": _frac_ge(own_only, REUSE_BAR),
        "ge8_borrow": _frac_ge(with_borrow, REUSE_BAR),
        "nov_ge1": _frac_ge(nov, 1.0),
        "nov_median": statistics.median(nov) if nov else 0.0,
    }


def _print(title, rows):
    print(f"\n=== {title} ===")
    print(f"{'bw':>5} {'simmin':>7} {'n':>4} {'%nb':>5} "
          f"{'mult_med':>9} {'mult_mean':>10} {'%≥8 own':>8} {'%≥8 brw':>8} "
          f"{'%nov≥1':>7} {'nov_med':>8}")
    for (bw, smin), r in rows:
        print(f"{bw:>5.2f} {smin:>7.2f} {r['n']:>4} "
              f"{100*r['n_with_nb']/r['n']:>4.0f}% "
              f"{r['mult_median']:>9.3f} {r['mult_mean']:>10.3f} "
              f"{100*r['ge8_own']:>7.0f}% {100*r['ge8_borrow']:>7.0f}% "
              f"{100*r['nov_ge1']:>6.0f}% {r['nov_median']:>8.2f}")


if __name__ == "__main__":
    snap = sys.argv[1] if len(sys.argv) > 1 else "data/exports/multi_500_annotated.json"
    geo = load_geometry(snap)
    print(f"snapshot: {snap}")

    crows = []
    for bw in (0.2, 0.3, 0.5):
        for smin in (0.2, 0.3, 0.4):
            crows.append(((bw, smin), measure_compound(geo, bw, smin)))
    _print("COMPOUND manifold (Tanimoto)", crows)

    trows = []
    for bw in (0.2, 0.25, 0.4):
        for smin in (0.05, 0.1, 0.2):
            trows.append(((bw, smin), measure_target(geo, bw, smin)))
    _print("TARGET manifold (pathway Jaccard)", trows)
