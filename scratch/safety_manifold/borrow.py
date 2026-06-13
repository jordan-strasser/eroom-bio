"""Phase 2 — cross-node borrowing layer (Nadaraya-Watson over the manifolds).

An AE belief for entity e is its exact-id evidence (full weight, the anchor)
PLUS a kernel-weighted sum of its manifold-neighbors' evidence:

    alpha(e,a) = alpha_own(e,a) + Sum_{e' != e} w(e,e') * (alpha'(e',a) - 1)
    beta (e,a) = beta_own (e,a) + Sum_{e' != e} w(e,e') * (beta' (e',a) - 1)

The (-1) strips each neighbor's Beta(1,1) prior so we borrow *evidence*, not
prior mass. Direction is preserved (a neighbor that strongly has the AE pushes
alpha; one that strongly lacks it pushes beta). The own term dominates; neighbors
fill the tail. Kernel:

    w(e,e') = exp((sim - 1) / bandwidth)   if sim >= sim_min   else 0

The sim_min floor is principled: Phase 1 showed the AE-sharing signal lives
above ~0.3 Tanimoto / ~0.1 pathway-Jaccard, so we refuse to borrow from the
misaligned mass below it (that is exactly where the text field leaked).

`exclude_self=True` drops the entity's own evidence entirely — the held-out
novel-entity mode for the Phase-4 leakage test.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from geometry import tanimoto, jaccard


def _kernel(sim: float, bandwidth: float, sim_min: float) -> float:
    if sim < sim_min:
        return 0.0
    return math.exp((sim - 1.0) / bandwidth)


@dataclass
class BorrowedBelief:
    alpha: float
    beta: float
    own_mass: float          # exact-id evidence pseudo-counts (alpha+beta-2 from self)
    borrowed_mass: float     # kernel-weighted neighbor pseudo-counts added
    contributors: list       # [(neighbor_id, weight, neighbor_mean)]

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def strength(self) -> float:
        return max(0.0, (self.alpha + self.beta) - 2.0)


class SafetyManifold:
    # Tuned defaults (Phase 2 sweep, recorded in SAFETY_MANIFOLD_RESULTS.md):
    #   compound bw=0.4 / sim_min=0.25 — ECFP4 Tanimoto >= 0.25 is the
    #     "structurally related" floor; the alignment lift (Phase 1) is
    #     monotone and permutation-significant above it, and the exp kernel
    #     softly discounts the weaker neighbors within the admitted band.
    #   target   bw=0.4 / sim_min=0.05 — pathway-Jaccard >= 0.05 == "shares
    #     >=1 Reactome/GO pathway" for typical size-5 sets, the right floor
    #     for on-target class effects.
    def __init__(self, geo, *,
                 bw_compound=0.4, simmin_compound=0.25,
                 bw_target=0.4, simmin_target=0.05):
        self.geo = geo
        self.bw_c = bw_compound
        self.smin_c = simmin_compound
        self.bw_t = bw_target
        self.smin_t = simmin_target
        self._cnb = {}   # cid -> [(cid', w)]
        self._tnb = {}   # tid -> [(tid', w)]

    # --- neighbor lists (cached) -----------------------------------------
    def compound_neighbors(self, cid):
        if cid in self._cnb:
            return self._cnb[cid]
        geo = self.geo
        fp = geo.compound_fp.get(cid)
        out = []
        if fp is not None:
            for did, fpd in geo.compound_fp.items():
                if did == cid:
                    continue
                s = tanimoto(fp, fpd)
                w = _kernel(s, self.bw_c, self.smin_c)
                if w > 0:
                    out.append((did, w, s))
        self._cnb[cid] = out
        return out

    def target_neighbors(self, tid):
        if tid in self._tnb:
            return self._tnb[tid]
        geo = self.geo
        pw = geo.target_pathways.get(tid)
        out = []
        if pw:
            for uid, pwd in geo.target_pathways.items():
                if uid == tid or not pwd:
                    continue
                s = jaccard(pw, pwd)
                w = _kernel(s, self.bw_t, self.smin_t)
                if w > 0:
                    out.append((uid, w, s))
        self._tnb[tid] = out
        return out

    # --- borrowed beliefs -------------------------------------------------
    @staticmethod
    def _ab(belief):
        return belief.get("alpha", 1.0), belief.get("beta", 1.0)

    def borrowed_causes_ae(self, cid, ae_id, *, exclude_self=False) -> BorrowedBelief:
        geo = self.geo
        a, b = 1.0, 1.0
        own_mass = 0.0
        if not exclude_self:
            bel = geo.causes_ae.get(cid, {}).get(ae_id)
            if bel:
                aa, bb = self._ab(bel)
                a, b = aa, bb
                own_mass = max(0.0, aa + bb - 2.0)
        borrowed = 0.0
        contributors = []
        for did, w, s in self.compound_neighbors(cid):
            bel = geo.causes_ae.get(did, {}).get(ae_id)
            if not bel:
                continue
            aa, bb = self._ab(bel)
            da, db = max(0.0, aa - 1.0), max(0.0, bb - 1.0)
            if da + db <= 0:
                continue
            a += w * da
            b += w * db
            borrowed += w * (da + db)
            contributors.append((did, w, aa / (aa + bb)))
        return BorrowedBelief(a, b, own_mass, borrowed, contributors)

    def borrowed_target_ae(self, tid, ae_id, *, exclude_self=False) -> BorrowedBelief:
        geo = self.geo
        a, b = 1.0, 1.0
        own_mass = 0.0
        if not exclude_self:
            bel = geo.target_ae.get(tid, {}).get(ae_id)
            if bel:
                aa, bb = self._ab(bel)
                a, b = aa, bb
                own_mass = max(0.0, aa + bb - 2.0)
        borrowed = 0.0
        contributors = []
        for uid, w, s in self.target_neighbors(tid):
            bel = geo.target_ae.get(uid, {}).get(ae_id)
            if not bel:
                continue
            aa, bb = self._ab(bel)
            da, db = max(0.0, aa - 1.0), max(0.0, bb - 1.0)
            if da + db <= 0:
                continue
            a += w * da
            b += w * db
            borrowed += w * (da + db)
            contributors.append((uid, w, aa / (aa + bb)))
        return BorrowedBelief(a, b, own_mass, borrowed, contributors)

    # --- effective reuse on an entity's whole AE profile ------------------
    def compound_profile_reuse(self, cid):
        """(own_mass, borrowed_mass) summed over the compound's AE profile."""
        own = brc = 0.0
        for ae_id in self.geo.causes_ae.get(cid, {}):
            bb = self.borrowed_causes_ae(cid, ae_id)
            own += bb.own_mass
            brc += bb.borrowed_mass
        return own, brc

    def target_profile_reuse(self, tid):
        own = brc = 0.0
        for ae_id in self.geo.target_ae.get(tid, {}):
            if ae_id.startswith("AE:soc:"):
                continue
            bb = self.borrowed_target_ae(tid, ae_id)
            own += bb.own_mass
            brc += bb.borrowed_mass
        return own, brc
