"""Phase 3 validation — does the decomposition route KNOWN cases correctly?

Pulls specific (gene, SOC, expected-tag) class effects and reports how the
decomposer routes them, with full provenance (which neighbor compounds/targets
contributed, and their structural diversity). Then emits a per-program AE
liability profile — the surface a customer pays for.
"""
from __future__ import annotations

import sys

sys.path.insert(0, "scratch/safety_manifold")
from geometry import load_geometry  # noqa: E402
from phase3_decompose import Decomposer  # noqa: E402


KNOWN_CASES = [
    # (gene_symbol, soc_id substring, AE substring, expected, note)
    ("EGFR", "skin", "rash", "on-target",
     "EGFR-inhibitor rash — canonical on-target class effect"),
    ("INSR", "endocrine", "hypoglyc", "on-target",
     "insulin hypoglycemia — textbook on-target mechanism effect"),
    ("HMGCR", "musculoskeletal", "", "on-target",
     "statin myopathy (corpus caught fractures, not myopathy — see note)"),
    ("TUBB", "nervous", "neuropathy", "on-target",
     "tubulin-inhibitor peripheral neuropathy (taxane+vinca, diverse scaffolds)"),
    ("GLP1R", "gastrointestinal", "", "on-target",
     "GLP-1 agonist GI — on-target incretin effect"),
    ("TNF", "infections", "", "on-target",
     "anti-TNF infection — on-target immunosuppression"),
]


def gene_targets(geo, gene):
    return [t for t, n in geo.nodes.items()
            if n.get("node_type") == "TargetNode" and n.get("gene_symbol") == gene]


def run_known(geo, dec):
    nm = geo.nodes
    cn = lambda c: nm[c].get("name", c)
    print("=" * 78)
    print("KNOWN-CASE VALIDATION")
    print("=" * 78)
    for gene, soc_sub, ae_sub, expected, note in KNOWN_CASES:
        tids = gene_targets(geo, gene)
        hits = []
        for tid in tids:
            for cid in geo.target_compounds.get(tid, ()):
                for ae in geo.causes_ae.get(cid, {}):
                    soc = dec.soc_of.get(ae, "")
                    if soc_sub in soc and (not ae_sub or ae_sub in ae.lower()):
                        d = dec.decompose(cid, tid, ae)
                        if d.own_mean >= 0.55:
                            hits.append(d)
        print(f"\n[{gene} → {soc_sub}] expected {expected}")
        print(f"   {note}")
        if not hits:
            print("   (no qualifying observed AE in corpus)")
            continue
        # best (highest on_lift) representative
        hits.sort(key=lambda d: max(d.on_lift, d.off_lift), reverse=True)
        tags = {}
        for d in hits:
            tags[d.tag] = tags.get(d.tag, 0) + 1
        d = hits[0]
        carriers = ", ".join(cn(c) for c in d.on_contributors[:4])
        verdict = "PASS" if d.tag.startswith("on-target") else "miss"
        print(f"   routed: {dict(tags)}  →  representative={d.tag}  [{verdict}]")
        print(f"   {cn(d.compound)} / {gene} / {d.ae}: "
              f"on_lift={d.on_lift:+.2f} (n={d.on_n}, diversity={d.on_diversity:.2f}) "
              f"off_lift={d.off_lift:+.2f}")
        print(f"   on-target carriers (diverse scaffolds sharing target): {carriers}")


def liability_profile(geo, dec, compound_name):
    """Per-program AE liability profile: each AE tagged + calibrated + provenance."""
    nm = geo.nodes
    cn = lambda c: nm[c].get("name", c)
    g = lambda t: nm[t].get("gene_symbol") or nm[t].get("name")
    cid = None
    for nid, n in nm.items():
        if n.get("node_type") == "InterventionNode" and \
           compound_name.lower() in (n.get("name") or "").lower():
            cid = nid
            break
    if cid is None:
        print(f"compound {compound_name!r} not found")
        return
    tids = list(geo.binds.get(cid, set()))
    print("\n" + "=" * 78)
    print(f"AE LIABILITY PROFILE — {cn(cid)}  (targets: {', '.join(g(t) for t in tids)})")
    print("=" * 78)
    rows = []
    for tid in tids:
        for ae in geo.causes_ae.get(cid, {}):
            d = dec.decompose(cid, tid, ae)
            if d.own_mean >= 0.55:
                rows.append(d)
    # dedupe by AE, keep the most explanatory target
    best = {}
    for d in rows:
        if d.ae not in best or max(d.p_on, d.p_off) > max(best[d.ae].p_on, best[d.ae].p_off):
            best[d.ae] = d
    order = {"on-target": 0, "on-target(low-div)": 1, "off-target": 2,
             "mixed": 3, "idiosyncratic": 4, "baseline": 5, "weak": 6}
    for d in sorted(best.values(), key=lambda d: (order.get(d.tag, 9), -d.noisy_or)):
        r_on, r_off, r_id = d.responsibility
        prov = ""
        if d.tag.startswith("on-target"):
            prov = f"shared w/ {d.on_n} {g(d.target)} compounds (div {d.on_diversity:.2f})"
        elif d.tag == "off-target":
            prov = f"shared w/ {d.off_n} structure-neighbors (other targets)"
        print(f"  {d.ae.replace('AE:',''):26s} P={d.noisy_or:.2f}  {d.tag:18s} "
              f"[on {r_on:.0%}/off {r_off:.0%}/idio {r_id:.0%}]  {prov}")


if __name__ == "__main__":
    geo = load_geometry("data/exports/multi_500_annotated.json")
    dec = Decomposer(geo)
    run_known(geo, dec)
    for name in ("gefitinib", "carboplatin", "Insulin glargine"):
        liability_profile(geo, dec, name)
