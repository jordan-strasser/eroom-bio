"""Audit how Compound and Target nodes are merged — and tie it to the surprising
`affects` (compound->target) field value (scalar 0.716 -> field 0.561).

Three questions:
  1. UNDER-merge: same drug / same gene split across multiple nodes (dups).
  2. OVER-merge / instability: a single affects edge whose anchors sit at MULTIPLE
     (s,t) coordinates — i.e. the compound or target description varies across
     trials on the "same" edge (it should be ONE point for node-desc anchors).
  3. The affects field anomaly: per affects edge, scalar mean vs field E[p], and
     whether the anchors are coincident (they should be, for node-desc).

No mutation. Run:
  python -m scripts.compound_target_merge_audit \
    --graph data/exports/multi_500_annotated.json \
    --field /Users/jordanstrasser/.eroom/private/multi_500_annotated_belief_field.json
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict

from src.graph.store import GraphStore
from src.graph.models import EdgeType
from src.inference.belief_field import expected_p
from src.prediction.field_prediction import load_edge_fields


def _g(node, key, default=None):
    if isinstance(node, dict):
        return node.get(key, default)
    return getattr(node, key, default)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--field", default="")
    a = ap.parse_args()

    store = GraphStore()
    store.import_snapshot(a.graph)

    # ── node-type census ───────────────────────────────────────────────
    type_counts = Counter()
    comps, targs = [], []
    for nid, nd in store._graph.nodes(data=True):  # noqa: SLF001
        nt = _g(nd, "node_type") or "?"
        type_counts[nt] += 1
        if nt in ("compound", "CompoundNode"):
            comps.append((nid, nd))
        elif nt in ("target", "TargetNode"):
            targs.append((nid, nd))
    print("node-type census:", dict(type_counts))
    if comps:
        cid, cnd = comps[0]
        print(f"\nsample compound node keys: {sorted((cnd if isinstance(cnd,dict) else cnd.__dict__).keys())}")
        print(f"  id={cid} fields: name={_g(cnd,'name')!r} chembl_id={_g(cnd,'chembl_id')!r}")
    if targs:
        tid, tnd = targs[0]
        print(f"sample target node keys: {sorted((tnd if isinstance(tnd,dict) else tnd.__dict__).keys())}")
        print(f"  id={tid} fields: name={_g(tnd,'name')!r} gene_symbol={_g(tnd,'gene_symbol')!r}")

    # ── 1. UNDER-merge: dups ───────────────────────────────────────────
    print(f"\n=== COMPOUND under-merge (same chembl_id / same name, >1 node) ===")
    by_chembl = defaultdict(list)
    by_cname = defaultdict(list)
    for nid, nd in comps:
        ch = _g(nd, "chembl_id")
        if ch:
            by_chembl[ch].append(nid)
        nm = (_g(nd, "name") or "").strip().lower()
        if nm:
            by_cname[nm].append(nid)
    chembl_dups = {k: v for k, v in by_chembl.items() if len(v) > 1}
    name_dups = {k: v for k, v in by_cname.items() if len(v) > 1}
    print(f"compound nodes: {len(comps)}; with chembl_id: {sum(1 for _,nd in comps if _g(nd,'chembl_id'))}")
    print(f"  chembl_id shared by >1 node (under-merge): {len(chembl_dups)}")
    for k, v in list(chembl_dups.items())[:8]:
        print(f"    {k}: {v}")
    print(f"  name shared by >1 node: {len(name_dups)}")
    for k, v in list(name_dups.items())[:5]:
        print(f"    {k!r}: {v}")

    print(f"\n=== TARGET under-merge (same gene_symbol, >1 node) ===")
    by_gene = defaultdict(list)
    no_gene = 0
    for nid, nd in targs:
        gs = (_g(nd, "gene_symbol") or "").strip().upper()
        if gs:
            by_gene[gs].append(nid)
        else:
            no_gene += 1
    gene_dups = {k: v for k, v in by_gene.items() if len(v) > 1}
    print(f"target nodes: {len(targs)}; with gene_symbol: {len(targs)-no_gene}; no gene: {no_gene}")
    print(f"  gene_symbol shared by >1 node (under-merge): {len(gene_dups)}")
    for k, v in list(gene_dups.items())[:8]:
        print(f"    {k}: {v}")

    # ── 2 & 3. affects edge field anomaly ──────────────────────────────
    if not a.field:
        return
    fm = load_edge_fields(a.field)
    aff = {(s, t): f for (s, t, et), f in fm.items() if et == "affects"}
    print(f"\n=== AFFECTS edges: scalar mean vs field E[p], anchor coincidence ===")
    print(f"affects edges in field: {len(aff)}")
    multi_coord = 0          # edges whose anchors sit at >1 distinct (s,t)
    big_gap = []             # |scalar - field| large
    rows = 0
    for (s, t), f in aff.items():
        if len(f.anchors) < 2:
            continue
        rows += 1
        coords = {(tuple(round(x,4) for x in anc.s), tuple(round(x,4) for x in anc.t)) for anc in f.anchors}
        if len(coords) > 1:
            multi_coord += 1
        # scalar mean from the graph edge belief
        try:
            sb = store.get_edge_belief(s, t, EdgeType.AFFECTS)
            sm = sb.alpha / (sb.alpha + sb.beta)
        except KeyError:
            continue
        # field E[p] at the anchors' own coordinate (first anchor)
        fp = expected_p(f, f.anchors[0].s, f.anchors[0].t)
        if abs(sm - fp) > 0.10:
            big_gap.append((s, t, sm, fp, len(f.anchors), len(coords)))
    print(f"  multi-anchor affects edges: {rows}")
    print(f"  of those, anchors at >1 DISTINCT coordinate (NOT coincident!): {multi_coord} "
          f"({100*multi_coord/max(rows,1):.0f}%)")
    print(f"  edges with |scalar - field| > 0.10: {len(big_gap)}")
    print(f"\n  worst gaps (src -> tgt: scalar vs field, #anchors, #coords):")
    for s, t, sm, fp, na, nc in sorted(big_gap, key=lambda r: -abs(r[2]-r[3]))[:12]:
        sn = _g(store.get_node(s), "name") if _has(store, s) else s
        tn = _g(store.get_node(t), "name") if _has(store, t) else t
        print(f"    {str(sn)[:24]:24s} -> {str(tn)[:18]:18s}  scalar {sm:.3f} field {fp:.3f}  "
              f"({na} anchors, {nc} coords)")


def _has(store, nid):
    try:
        store.get_node(nid)
        return True
    except KeyError:
        return False


if __name__ == "__main__":
    main()
