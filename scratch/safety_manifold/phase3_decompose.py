"""Phase 3 — on-target vs off-target AE decomposition (the differentiator).

For an observed AE `a` on compound `c` that binds target `t`, attribute it to:
  ON-TARGET    recurs across STRUCTURALLY DIVERSE compounds sharing target t
               (mechanism-intrinsic; not escapable by scaffold change)
  OFF-TARGET   recurs across STRUCTURE-NEIGHBORS of c hitting DIFFERENT targets
               (chemistry-specific; escapable by scaffold change)
  IDIOSYNCRATIC neither class explains it — specific to this compound

Two fixes that make the routing real:
  (1) SOC roll-up. Sibling compounds report toxicity at DISJOINT PT terms
      (colitis / ulcerative_colitis / microscopic_colitis), so PT-exact
      prevalence is ~0. We match at the MedDRA SOC parent (the same fix the
      target_associated_ae propagation uses).
  (2) Background correction. Oncology toxicity is dominated by ubiquitous SOCs
      (GI, blood). We score the LIFT over the global SOC base rate, so only
      class-SPECIFIC enrichment routes on/off-target; ubiquitous tox stays low.

Discriminator = the contrast (on_lift vs off_lift) plus structural diversity of
the on-target carriers. Leakage-safe: c's own evidence never enters its scores.
Noisy-OR over sources (EM derivation §3) gives the calibrated liability.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

sys.path.insert(0, "scratch/safety_manifold")
from geometry import load_geometry, tanimoto, _belief_mean, _belief_strength  # noqa: E402

MIN_EV = 1.0
SIMMIN = 0.25
AE_PRESENT = 0.55
DIVERSE = 0.40
LIFT_MIN = 0.10        # class enrichment over base rate to count as a class effect
CONTRAST = 1.3         # how much one lift must beat the other to route cleanly


def _soc_map(geo):
    out = {}
    for nid, n in geo.nodes.items():
        if n.get("node_type") == "AdverseEventNode" and not nid.startswith("AE:soc:"):
            soc = (n.get("soc_id") or "").strip()
            if soc:
                out[nid] = soc
    return out


@dataclass
class Decomp:
    compound: str
    target: str
    ae: str
    soc: str
    own_mean: float
    on_prev: float
    off_prev: float
    base: float
    on_lift: float
    off_lift: float
    on_n: int
    on_diversity: float
    off_n: int
    p_on: float
    p_off: float
    p_idio: float
    tag: str
    on_contributors: list
    off_contributors: list

    @property
    def noisy_or(self):
        return 1 - (1 - self.p_on) * (1 - self.p_off) * (1 - self.p_idio)

    @property
    def responsibility(self):
        tot = self.p_on + self.p_off + self.p_idio
        if tot <= 0:
            return (0.0, 0.0, 0.0)
        return (self.p_on / tot, self.p_off / tot, self.p_idio / tot)


class Decomposer:
    def __init__(self, geo):
        self.geo = geo
        self.soc_of = _soc_map(geo)
        # compound -> set(soc it shows with belief>=AE_PRESENT)
        self.compound_socs = {}
        for cid, aes in geo.causes_ae.items():
            s = set()
            for ae, bel in aes.items():
                if _belief_strength(bel) >= MIN_EV and _belief_mean(bel) >= AE_PRESENT:
                    soc = self.soc_of.get(ae)
                    if soc:
                        s.add(soc)
            self.compound_socs[cid] = s
        # global base rate per SOC over profiled compounds
        profiled = [c for c, s in self.compound_socs.items() if s]
        self.n_profiled = len(profiled)
        base = {}
        for c in profiled:
            for soc in self.compound_socs[c]:
                base[soc] = base.get(soc, 0) + 1
        self.base = {soc: n / self.n_profiled for soc, n in base.items()}

    def decompose(self, cid, tid, ae) -> Decomp:
        geo = self.geo
        soc = self.soc_of.get(ae, "")
        base = self.base.get(soc, 0.0)
        own_bel = geo.causes_ae.get(cid, {}).get(ae)
        own = _belief_mean(own_bel) if own_bel else 0.0

        # ON-TARGET: same-target siblings carrying this SOC (exclude c).
        # Prevalence conditions on PROFILED siblings (those we have any safety
        # data for) — dividing by all binders deflates it toward 0 for coarse
        # targets (DNA has 53 binders, most unprofiled).
        siblings = [s for s in geo.target_compounds.get(tid, ()) if s != cid]
        prof_sib = [s for s in siblings if self.compound_socs.get(s)]
        on_contrib = [s for s in prof_sib if soc in self.compound_socs.get(s, set())]
        on_prev = (len(on_contrib) / len(prof_sib)) if prof_sib else 0.0
        carriers_fp = [c for c in on_contrib if c in geo.compound_fp]
        if cid in geo.compound_fp:
            carriers_fp = [cid] + carriers_fp
        div = self._diversity(carriers_fp)

        # OFF-TARGET: structure-neighbors NOT binding t, carrying this SOC.
        # Denominator = PROFILED structure-neighbors only.
        off_contrib = []
        wsum = wpos = 0.0
        if cid in geo.compound_fp:
            for d, fpd in geo.compound_fp.items():
                if d == cid or tid in geo.binds.get(d, set()):
                    continue
                if not self.compound_socs.get(d):
                    continue  # unprofiled neighbor — no safety data
                sim = tanimoto(geo.compound_fp[cid], fpd)
                if sim < SIMMIN:
                    continue
                wsum += sim
                if soc in self.compound_socs.get(d, set()):
                    wpos += sim
                    off_contrib.append((d, sim))
        off_prev = (wpos / wsum) if wsum > 0 else 0.0

        on_lift = on_prev - base
        off_lift = off_prev - base

        # Source probabilities (calibrated liability): lift above base, floored at 0.
        p_on = max(0.0, on_lift)
        p_off = max(0.0, off_lift)
        p_idio = max(0.0, own - base - max(p_on, p_off)) if own >= AE_PRESENT else 0.0

        on_ok = on_lift >= LIFT_MIN and len(on_contrib) >= 2
        off_ok = off_lift >= LIFT_MIN and len(off_contrib) >= 2
        diverse = div >= DIVERSE
        if on_ok and off_ok:
            if on_lift >= off_lift * CONTRAST:
                tag = "on-target" if diverse else "on-target(low-div)"
            elif off_lift >= on_lift * CONTRAST:
                tag = "off-target"
            else:
                tag = "mixed"
        elif on_ok:
            tag = "on-target" if diverse else "on-target(low-div)"
        elif off_ok:
            tag = "off-target"
        elif own >= AE_PRESENT and base >= 0.30:
            tag = "baseline"          # ubiquitous tox (≥30% of compounds: blood,
            #                           GI) — not differentially attributable
        elif own >= AE_PRESENT:
            tag = "idiosyncratic"
        else:
            tag = "weak"

        return Decomp(cid, tid, ae, soc, own, on_prev, off_prev, base,
                      on_lift, off_lift, len(on_contrib), div, len(off_contrib),
                      p_on, p_off, p_idio, tag, on_contrib[:6], off_contrib[:6])

    def _diversity(self, cids):
        geo = self.geo
        fps = [geo.compound_fp[c] for c in cids if c in geo.compound_fp]
        if len(fps) < 2:
            return 0.0
        ds = []
        for i in range(len(fps)):
            for j in range(i + 1, len(fps)):
                ds.append(1 - tanimoto(fps[i], fps[j]))
        return sum(ds) / len(ds) if ds else 0.0

    def all_triples(self):
        out = []
        for cid, aes in self.geo.causes_ae.items():
            tids = self.geo.binds.get(cid, set())
            if not tids:
                continue
            for ae, bel in aes.items():
                if _belief_strength(bel) < MIN_EV or _belief_mean(bel) < AE_PRESENT:
                    continue
                for tid in tids:
                    out.append(self.decompose(cid, tid, ae))
        return out


if __name__ == "__main__":
    geo = load_geometry("data/exports/multi_500_annotated.json")
    dec = Decomposer(geo)
    triples = dec.all_triples()
    import collections
    tags = collections.Counter(d.tag for d in triples)
    print(f"decomposed {len(triples)} (compound,target,AE) triples over {dec.n_profiled} profiled compounds")
    for t, c in tags.most_common():
        print(f"  {t:20s} {c} ({100*c/len(triples):.0f}%)")

    nm = geo.nodes
    g = lambda t: nm[t].get("gene_symbol") or nm[t].get("name")
    cn = lambda c: nm[c].get("name", c)

    for tag in ("on-target", "off-target", "on-target(low-div)", "idiosyncratic"):
        rows = [d for d in triples if d.tag == tag]
        print(f"\n=== {tag} ({len(rows)}) ===")
        seen = set()
        for d in rows:
            k = (cn(d.compound), g(d.target), d.soc)
            if k in seen:
                continue
            seen.add(k)
            print(f"  {cn(d.compound)[:16]:16s} {str(g(d.target))[:10]:10s} {d.ae[:24]:24s} "
                  f"soc={d.soc[:22]:22s} on_lift={d.on_lift:+.2f}(n={d.on_n},div={d.on_diversity:.2f}) "
                  f"off_lift={d.off_lift:+.2f}(n={d.off_n})")
            if len(seen) >= 12:
                break
