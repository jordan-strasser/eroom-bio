"""Phase 4 — held-out novel-entity test (the value prop) + invariance.

The product claim: a NOVEL program (compound/target the graph has never seen)
inherits a calibrated AE liability from its manifold neighbors. We test it the
only honest way — hold the entity out ENTIRELY and predict its AE profile from
neighbors alone:

  COMPOUND: hold out compound c, predict which SOCs it will show from its
            STRUCTURE-neighbors (Tanimoto-weighted), trial-DISJOINT (a neighbor
            sharing any trial with c is dropped — combo arms attribute the same
            trial's AEs to multiple compounds, which would leak).
  TARGET:   hold out target t, predict its on-target SOCs from PATHWAY-neighbors.

Scoreboard per entity-type, manifold vs the base-rate prior:
  - discrimination (AUROC over all entity×SOC pairs)
  - calibration (reliability table: predicted bin -> observed frequency)
Plus the exact-target invariance (within-target target_associated_ae SD,
must stay ~0.048) and the cross-node reuse multiplier summary.
"""
from __future__ import annotations

import sys
from collections import defaultdict

sys.path.insert(0, "scratch/safety_manifold")
from geometry import load_geometry, tanimoto, jaccard, _belief_mean, _belief_strength  # noqa: E402

AE_PRESENT = 0.55
SIMMIN_C = 0.25
SIMMIN_T = 0.05
BW_C = 0.4
BW_T = 0.4
import math


def kern(sim, bw, smin):
    return math.exp((sim - 1) / bw) if sim >= smin else 0.0


def soc_map(geo):
    return {nid: (n.get("soc_id") or "").strip()
            for nid, n in geo.nodes.items()
            if n.get("node_type") == "AdverseEventNode" and not nid.startswith("AE:soc:")
            and (n.get("soc_id") or "").strip()}


def compound_trials(geo):
    """compound -> set(trial ids) from its causes_ae evidence records."""
    out = defaultdict(set)
    for cid, aes in geo.causes_ae.items():
        for ae, bel in aes.items():
            for rec in bel.get("evidence", []):
                sid = rec.get("source_id")
                if sid:
                    out[cid].add(sid)
    return out


def auroc(pairs):
    """pairs = [(score, label0/1)]; rank-based AUROC."""
    pos = [s for s, y in pairs if y == 1]
    neg = [s for s, y in pairs if y == 0]
    if not pos or not neg:
        return float("nan")
    ordered = sorted(pairs, key=lambda p: p[0])
    rank = {}
    i = 0
    r = 1
    vals = [p[0] for p in ordered]
    # average ranks for ties
    ranks = [0] * len(ordered)
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and ordered[j + 1][0] == ordered[i][0]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    sum_pos = sum(rk for rk, p in zip(ranks, ordered) if p[1] == 1)
    n_pos, n_neg = len(pos), len(neg)
    return (sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def calibration(pairs, bins=5):
    rows = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sub = [y for s, y in pairs if (lo <= s < hi or (b == bins - 1 and s == 1.0))]
        if sub:
            rows.append((lo, hi, len(sub), sum(sub) / len(sub)))
    return rows


def per_soc_auroc(entity_scores, socs):
    """entity_scores[(entity,soc)] = (score, truth). Returns mean within-SOC
    AUROC over SOCs that have both a present and an absent entity. The base-rate
    prior is CONSTANT within a SOC, so its per-SOC AUROC is exactly 0.5 — any
    manifold value above 0.5 is genuine compound-specific discrimination beyond
    the base rate (the cross-SOC base-rate trap is removed)."""
    aurocs = []
    for sc in socs:
        pairs = [(s, y) for (e, s2), (s, y) in entity_scores.items() if s2 == sc]
        if not pairs:
            continue
        a = auroc(pairs)
        if a == a:  # not nan (has both classes)
            aurocs.append(a)
    return (sum(aurocs) / len(aurocs)) if aurocs else float("nan"), len(aurocs)


def compound_novel_test(geo):
    soc = soc_map(geo)
    socs = sorted(set(soc.values()))
    ctrials = compound_trials(geo)
    csoc = {}
    for cid, aes in geo.causes_ae.items():
        s = set()
        for ae, bel in aes.items():
            if _belief_strength(bel) >= 1.0 and _belief_mean(bel) >= AE_PRESENT:
                if soc.get(ae):
                    s.add(soc[ae])
        if s:
            csoc[cid] = s
    base = {sc: sum(1 for c in csoc if sc in csoc[c]) / len(csoc) for sc in socs}
    test = [c for c in csoc if c in geo.compound_fp]

    man_pairs, prior_pairs = [], []      # pooled (kept for reference)
    cov_scores = {}                       # covered-only, for per-SOC AUROC + calib
    covered = 0
    for c in test:
        nbrs = []
        for d in test:
            if d == c or (ctrials[c] & ctrials[d]):
                continue
            s = tanimoto(geo.compound_fp[c], geo.compound_fp[d])
            w = kern(s, BW_C, SIMMIN_C)
            if w > 0:
                nbrs.append((d, w))
        wsum = sum(w for _, w in nbrs)
        has_nb = wsum > 0
        if has_nb:
            covered += 1
        for sc in socs:
            truth = 1 if sc in csoc[c] else 0
            score = (sum(w for d, w in nbrs if sc in csoc[d]) / wsum) if has_nb else base[sc]
            man_pairs.append((score, truth))
            prior_pairs.append((base[sc], truth))
            if has_nb:
                cov_scores[(c, sc)] = (score, truth)
    return man_pairs, prior_pairs, len(test), covered, cov_scores, socs


def target_novel_test(geo):
    soc = soc_map(geo)
    socs = sorted(set(soc.values()))
    # target -> set(SOC) truth from target_associated_ae (PT-tier)
    tsoc = {}
    for tid, aes in geo.target_ae.items():
        s = set()
        for ae, bel in aes.items():
            if ae.startswith("AE:soc:"):
                continue
            if _belief_strength(bel) >= 1.0 and _belief_mean(bel) >= AE_PRESENT:
                if soc.get(ae):
                    s.add(soc[ae])
        if s and geo.target_pathways.get(tid):
            tsoc[tid] = s
    if not tsoc:
        return [], [], 0, 0, {}, socs
    base = {sc: sum(1 for t in tsoc if sc in tsoc[t]) / len(tsoc) for sc in socs}
    test = list(tsoc)
    man_pairs, prior_pairs = [], []
    cov_scores = {}
    covered = 0
    for t in test:
        nbrs = []
        for u in test:
            if u == t:
                continue
            s = jaccard(geo.target_pathways[t], geo.target_pathways[u])
            w = kern(s, BW_T, SIMMIN_T)
            if w > 0:
                nbrs.append((u, w))
        wsum = sum(w for _, w in nbrs)
        has_nb = wsum > 0
        if has_nb:
            covered += 1
        for sc in socs:
            truth = 1 if sc in tsoc[t] else 0
            score = (sum(w for u, w in nbrs if sc in tsoc[u]) / wsum) if has_nb else base[sc]
            man_pairs.append((score, truth))
            prior_pairs.append((base[sc], truth))
            if has_nb:
                cov_scores[(t, sc)] = (score, truth)
    return man_pairs, prior_pairs, len(test), covered, cov_scores, socs


def invariance(geo):
    """within-target target_associated_ae posterior SD (should be ~0.048)."""
    import statistics
    sds = []
    for tid, aes in geo.target_ae.items():
        means = [_belief_mean(b) for ae, b in aes.items()
                 if not ae.startswith("AE:soc:") and _belief_strength(b) >= 1.0]
        if len(means) >= 2:
            sds.append(statistics.pstdev(means))
    return statistics.mean(sds) if sds else float("nan"), len(sds)


def _report(name, man, prior, n, cov, cov_scores, socs):
    print(f"\n=== {name} novel-entity test (n={n}, served by ≥1 trial-disjoint neighbor={cov}) ===")
    psoc, nsoc = per_soc_auroc(cov_scores, socs)
    print(f"  per-SOC AUROC (covered entities)  manifold = {psoc:.3f}   prior = 0.500   "
          f"lift = {psoc-0.5:+.3f}   [{nsoc} SOCs scored]")
    print(f"  (pooled cross-SOC AUROC — base-rate-trapped, shown for context: "
          f"manifold {auroc(man):.3f} / prior {auroc(prior):.3f})")
    cov_pairs = list(cov_scores.values())
    print("  manifold calibration on covered entities (pred bin -> observed):")
    for lo, hi, k, obs in calibration(cov_pairs):
        print(f"    [{lo:.1f},{hi:.1f})  n={k:5d}  observed={obs:.3f}")


if __name__ == "__main__":
    geo = load_geometry("data/exports/multi_500_annotated.json")
    cm, cp, cn_, ccov, ccs, csocs = compound_novel_test(geo)
    _report("COMPOUND (structure)", cm, cp, cn_, ccov, ccs, csocs)
    tm, tp, tn_, tcov, tcs, tsocs = target_novel_test(geo)
    _report("TARGET (pathway)", tm, tp, tn_, tcov, tcs, tsocs)
    sd, k = invariance(geo)
    print(f"\n=== invariance: within-target target_associated_ae SD = {sd:.3f} (n={k} targets) ===")
