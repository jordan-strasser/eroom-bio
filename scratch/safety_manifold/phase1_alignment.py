"""Phase 1 — the alignment gate.

Does domain geometry predict AE sharing? This is the safety analog of the
B1 gate: build a manifold's borrowing only if proximity on it predicts
shared adverse events. If structure (or pathway) proximity does NOT track
AE-profile similarity, pooling over it would be WRONG -> skip, keep exact-id.

Compound manifold : Tanimoto over Morgan/ECFP4   vs  AE-profile similarity
Target manifold   : Jaccard over Reactome/GO pathways vs on-target AE-profile sim

For each: Pearson + Spearman corr over all pairs, a similarity-threshold
sweep (near vs far bins), a permutation null (shuffle AE profiles across
entities), and IDF-weighting to discount ubiquitous AEs (nausea/fatigue).
The compound test is reported BOTH including and excluding same-target pairs,
so we can see whether structure predicts AE sharing as genuine off-target
signal, not just a proxy for shared target.
"""
from __future__ import annotations

import math
import sys
from collections import defaultdict

sys.path.insert(0, "scratch/safety_manifold")
from geometry import (  # noqa: E402
    load_geometry, tanimoto, jaccard, cosine, _belief_strength, _belief_mean,
)

MIN_EV = 1.0        # AE counts only if its belief carries >=1 evidence pseudo-count
MIN_PROFILE = 2     # an entity needs >=2 qualifying AEs to enter a pairwise test


def _spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    return _pearson(ranks(xs), ranks(ys))


def _pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def _idf(profiles):
    """AE -> inverse-document-frequency over the entity profiles."""
    df = defaultdict(int)
    N = len(profiles)
    for prof in profiles.values():
        for ae in prof:
            df[ae] += 1
    return {ae: math.log((N + 1) / (c + 0.5)) for ae, c in df.items()}, N


def _idf_jaccard(a: set, b: set, idf: dict) -> float:
    if not a or not b:
        return 0.0
    inter = sum(idf.get(k, 0.0) for k in (a & b))
    union = sum(idf.get(k, 0.0) for k in (a | b))
    return inter / union if union else 0.0


def _build_pairs(entity_sim, profiles, idf, same_group=None):
    """Return per-pair (geo_sim, plain_jac, idf_jac, cos, same_grp)."""
    ents = sorted(profiles.keys())
    rows = []
    for i in range(len(ents)):
        a = ents[i]
        for j in range(i + 1, len(ents)):
            b = ents[j]
            gs = entity_sim(a, b)
            if gs is None:
                continue
            sa, sb = set(profiles[a]), set(profiles[b])
            pj = jaccard(sa, sb)
            ij = _idf_jaccard(sa, sb, idf)
            cs = cosine(profiles[a], profiles[b])
            sg = None
            if same_group is not None:
                sg = bool(same_group(a, b))
            rows.append((gs, pj, ij, cs, sg))
    return rows


def _report(name, rows, idx_geo=0, idx_ae=2, label="idf-Jaccard"):
    xs = [r[idx_geo] for r in rows]
    ys = [r[idx_ae] for r in rows]
    pear = _pearson(xs, ys)
    spear = _spearman(xs, ys)
    print(f"\n--- {name}: geo-sim vs {label} (n_pairs={len(rows)}) ---")
    print(f"  Pearson r = {pear:+.3f}   Spearman rho = {spear:+.3f}")
    # threshold sweep on geometry
    print(f"  {'geo-sim bin':>14s} {'n':>6s} {'mean AE-sim':>12s}")
    bins = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 1.01)]
    for lo, hi in bins:
        sub = [r[idx_ae] for r in rows if lo <= r[idx_geo] < hi]
        if sub:
            print(f"  [{lo:.2f},{hi:.2f}) {len(sub):>6d} {sum(sub)/len(sub):>12.4f}")
    return pear, spear


def _permutation_null(rows, profiles, entity_sim, idf, n_perm=200, seed=12345):
    """Shuffle AE profiles across entities; recompute Pearson(geo, idf-jac).
    Returns (observed, null_mean, null_p) where p = P(null >= observed)."""
    ents = sorted(profiles.keys())
    # observed
    obs = _pearson([r[0] for r in rows], [r[2] for r in rows])
    # precompute geo sims and the entity index of each pair
    pair_idx = []
    for i in range(len(ents)):
        for j in range(i + 1, len(ents)):
            gs = entity_sim(ents[i], ents[j])
            if gs is not None:
                pair_idx.append((i, j, gs))
    prof_list = [set(profiles[e]) for e in ents]
    rng = _Lcg(seed)
    geos = [g for _, _, g in pair_idx]
    null_vals = []
    ge = 0
    for _ in range(n_perm):
        perm = list(range(len(ents)))
        for k in range(len(perm) - 1, 0, -1):
            m = rng.randint(k + 1)
            perm[k], perm[m] = perm[m], perm[k]
        ys = []
        for (i, j, _g) in pair_idx:
            sa, sb = prof_list[perm[i]], prof_list[perm[j]]
            ys.append(_idf_jaccard(sa, sb, idf))
        nv = _pearson(geos, ys)
        null_vals.append(nv)
        if nv >= obs:
            ge += 1
    null_mean = sum(null_vals) / len(null_vals)
    p = (ge + 1) / (n_perm + 1)
    return obs, null_mean, p


class _Lcg:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF
    def randint(self, n):
        self.s = (1103515245 * self.s + 12345) & 0x7FFFFFFF
        return self.s % n


def compound_alignment(geo):
    print("\n" + "=" * 70)
    print("COMPOUND MANIFOLD — Tanimoto (ECFP4) vs AE-profile similarity")
    print("=" * 70)
    profiles = {}
    for cid in geo.compound_fp:
        prof = {ae: _belief_mean(b) for ae, b in geo.causes_ae.get(cid, {}).items()
                if _belief_strength(b) >= MIN_EV}
        if len(prof) >= MIN_PROFILE:
            profiles[cid] = prof
    print(f"compounds with fp AND >= {MIN_PROFILE} evidenced AEs: {len(profiles)}")
    idf, N = _idf(profiles)

    def csim(a, b):
        return tanimoto(geo.compound_fp[a], geo.compound_fp[b])

    def same_target(a, b):
        return bool(geo.binds.get(a, set()) & geo.binds.get(b, set()))

    rows = _build_pairs(csim, profiles, idf, same_group=same_target)
    _report("ALL pairs", rows, label="idf-Jaccard")
    _report("ALL pairs (plain Jaccard)", rows, idx_ae=1, label="plain-Jaccard")
    obs, nm, p = _permutation_null(rows, profiles, csim, idf)
    print(f"\n  permutation null (idf-Jaccard): obs r={obs:+.3f}  null mean={nm:+.3f}  p={p:.4f}")

    # genuine off-target: different-target pairs only
    diff = [r for r in rows if r[4] is False]
    same = [r for r in rows if r[4] is True]
    print(f"\n  same-target pairs: {len(same)}   different-target pairs: {len(diff)}")
    if diff:
        _report("DIFFERENT-target pairs (off-target signal)", diff, label="idf-Jaccard")
    if same:
        _report("SAME-target pairs (incl on-target)", same, label="idf-Jaccard")
    return rows, obs, p


def target_alignment(geo):
    print("\n" + "=" * 70)
    print("TARGET MANIFOLD — pathway co-membership vs on-target AE-profile sim")
    print("=" * 70)
    profiles = {}
    for tid, pw in geo.target_pathways.items():
        if not pw:
            continue
        prof = {ae: _belief_mean(b) for ae, b in geo.target_ae.get(tid, {}).items()
                if not ae.startswith("AE:soc:") and _belief_strength(b) >= MIN_EV}
        if len(prof) >= MIN_PROFILE:
            profiles[tid] = prof
    print(f"targets with pathways AND >= {MIN_PROFILE} evidenced on-target AEs: {len(profiles)}")
    if len(profiles) < 5:
        print("  (too few targets carry both a pathway set and an evidenced AE profile)")
    idf, N = _idf(profiles)

    def tsim(a, b):
        return jaccard(geo.target_pathways[a], geo.target_pathways[b])

    rows = _build_pairs(tsim, profiles, idf)
    if rows:
        _report("ALL target pairs", rows, label="idf-Jaccard")
        _report("ALL target pairs (plain Jaccard)", rows, idx_ae=1, label="plain-Jaccard")
        obs, nm, p = _permutation_null(rows, profiles, tsim, idf)
        print(f"\n  permutation null (idf-Jaccard): obs r={obs:+.3f}  null mean={nm:+.3f}  p={p:.4f}")
        return rows, obs, p
    return rows, 0.0, 1.0


if __name__ == "__main__":
    snap = sys.argv[1] if len(sys.argv) > 1 else "data/exports/multi_500_annotated.json"
    print(f"snapshot: {snap}")
    geo = load_geometry(snap)
    compound_alignment(geo)
    target_alignment(geo)
