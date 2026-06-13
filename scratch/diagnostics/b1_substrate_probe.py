"""B1 substrate probe — Step-1 effective-reuse (does the field lift it?) +
per-target AE SD (Step-1 metric 3) + Step-3 real BioLORD geometry_alignment.

Read-only on the field-materialized annotated snapshot.

  Part A (Step 1, metric 1): biology-NODE discrete reuse (#distinct host trials)
    vs the field's within-edge effective sample size. The field is a PER-EDGE
    surface (query() sums only one edge's anchors), so a biology node's field
    effective reuse = its within-edge anchor count = its discrete edge reuse —
    the field localizes existing evidence but does NOT pool across nodes, so it
    cannot lift a singleton biology node off reuse ≈ 1. Measured, not assumed.

  Part B (Step 1, metric 3): within-target AE posterior SD (P9, baseline 0.048).
    AE edges carry no field (not in materialize_belief_field.EDGE_SPECS) and the
    safety layer keys on target identity (already merged), so B1's biology
    re-representation leaves it invariant — reported for completeness.

  Part C (Step 3): real geometry_alignment. For biology nodes whose incident
    backbone edges have ≥2 host trials (the few with an estimable reliability),
    does BioLORD-cosine proximity predict reliability similarity? Two proxies:
    (i) Pearson corr(cosine_sim, 1−|Δr|) over node pairs; (ii) leave-one-out
    kNN reliability prediction corr(pred, actual). Compare to Step-2's α_cross.

Usage:
  .venv/bin/python -m scratch.diagnostics.b1_substrate_probe \
      data/exports/multi_500_annotated.json
"""
from __future__ import annotations

import sys
from collections import defaultdict

import numpy as np

import scratch.diagnostics._common as C
from src.graph.models import EdgeBeliefState
from src.inference.belief_field import BeliefField, query as field_query

BIO_INCIDENT = {"mechanism_affects": "tgt",   # mechanism -> biology (biology=target)
                "biology_drives": "src",       # biology -> indication
                "reflects_biology": "src"}     # biology -> endpoint


def _node_type(g, nid):
    try:
        return g._graph.nodes[nid].get("node_type", "")
    except Exception:  # noqa: BLE001
        return ""


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "data/exports/multi_500_annotated.json"
    g = C.load_graph(path)
    print(f"graph: {path}")

    # ---- node discrete reuse (#distinct host trials per node) -------------
    node_trials: dict[str, set] = defaultdict(set)
    for nct, ts in g.trial_subgraphs.items():
        seen = set()
        for c in ts.chains:
            for f in ("compound_id", "target_id", "mechanism_id", "biology_id",
                      "indication_id", "endpoint_id", "subgroup_population_id"):
                v = getattr(c, f, None)
                if v and v != "UNKNOWN":
                    seen.add(v)
        for nid in seen:
            node_trials[nid].add(nct)

    bio_nodes = [n for n in g._graph.nodes if _node_type(g, n) == "BiologyNode"]
    disc = np.array([len(node_trials.get(n, ())) for n in bio_nodes], dtype=float)

    print("\n===== Part A — biology-node EFFECTIVE REUSE (discrete vs field) =====")
    print(f"biology nodes: {len(bio_nodes)}")
    print(f"  discrete reuse (#host trials): median={np.median(disc):.2f} "
          f"mean={disc.mean():.2f} p90={np.percentile(disc,90):.1f} "
          f"max={disc.max():.0f}")
    print(f"  singletons (reuse≤1): {100*np.mean(disc<=1):.0f}%   "
          f"reuse≥8: {100*np.mean(disc>=8):.1f}%")

    # field effective sample size per biology-incident edge: query at each
    # anchor's own (s,t) and sum kernel weights (the field's effective N at that
    # point). Per node = max over incident edges (best-case borrowing).
    node_field_eff: dict[str, float] = defaultdict(float)
    edge_eff_list, edge_anchor_trials = [], []
    for u, v, key, data in g._graph.edges(data=True, keys=True):
        side = BIO_INCIDENT.get(key)
        if side is None:
            continue
        bio_id = v if side == "tgt" else u
        bf = (data.get("belief") or {}).get("belief_field")
        if not bf or not bf.get("anchors"):
            continue
        # vectors table: this snapshot dedups anchor (s,t) into _belief_field_vectors
        field = BeliefField.from_dict(bf, _VECTORS)
        if not field.anchors:
            continue
        ncts = {a.nct for a in field.anchors if a.nct}
        # effective sample size at each anchor's own coordinate = Σ_j kernel
        # weight, vectorized: w_ij = exp((cosS_ij + cosT_ij − 2)/bw). Unit-norm
        # the anchor vectors so the cosine is a plain dot product. Cap the matrix
        # on hub edges (subsample) to keep it O(cap²).
        anchors = field.anchors
        if len(anchors) > 400:
            idx = np.random.default_rng(0).choice(len(anchors), 400, replace=False)
            anchors = [field.anchors[i] for i in idx]
        S = np.array([a.s for a in anchors], dtype=float)
        T = np.array([a.t for a in anchors], dtype=float)
        S /= np.clip(np.linalg.norm(S, axis=1, keepdims=True), 1e-9, None)
        T /= np.clip(np.linalg.norm(T, axis=1, keepdims=True), 1e-9, None)
        cosS = S @ S.T
        cosT = T @ T.T
        Wm = np.exp((cosS + cosT - 2.0) / field.bandwidth)
        eff = float(Wm.sum(axis=1).mean())  # mean over anchors of Σ_j w_ij (incl self)
        node_field_eff[bio_id] = max(node_field_eff[bio_id], eff)
        edge_eff_list.append(eff)
        edge_anchor_trials.append(len(ncts))

    if edge_eff_list:
        ee = np.array(edge_eff_list)
        et = np.array(edge_anchor_trials, dtype=float)
        print(f"\n  biology-incident field edges: {len(ee)}")
        print(f"  per-edge field eff sample-size (Σ kernel wt incl self): "
              f"median={np.median(ee):.2f} mean={ee.mean():.2f} max={ee.max():.1f}")
        print(f"  per-edge distinct host trials (anchors' ncts):          "
              f"median={np.median(et):.2f} mean={et.mean():.2f} max={et.max():.0f}")
        print(f"  → field eff ÷ trial count (lift from within-edge borrowing): "
              f"mean {ee.mean()/max(et.mean(),1e-9):.2f}×")
        fe = np.array([node_field_eff[n] for n in bio_nodes if n in node_field_eff])
        if fe.size:
            print(f"  biology-node field eff reuse: median={np.median(fe):.2f} "
                  f"%≥8={100*np.mean(fe>=8):.1f}%  (vs discrete %≥8="
                  f"{100*np.mean(disc>=8):.1f}%)")

    # ---- Part B — within-target AE posterior SD (P9) ----------------------
    print("\n===== Part B — within-target AE SD (P9, baseline 0.048) =====")
    tgt_assoc = defaultdict(list)
    for u, v, key, b in C.iter_edges(g):
        if b is None or b.evidence_strength <= 0:
            continue
        if key == "target_associated_ae":
            tgt_assoc[u].append(b.expected_probability)
    sds = [np.std(vs) for vs in tgt_assoc.values() if len(vs) >= 3]
    if sds:
        print(f"  within-target AE SD (targets w/≥3 AE edges): "
              f"mean={np.mean(sds):.3f} (n_targets={len(sds)})")
    print("  (AE edges carry NO field + key on target identity → B1 biology "
          "re-representation leaves this invariant)")

    # ---- Part C — real BioLORD geometry_alignment -------------------------
    print("\n===== Part C — real BioLORD geometry_alignment (Step 3) =====")
    from src.graph.biolord_embeddings import embed_text, cosine_similarity
    # biology node reliability = mean E[p] over its incident backbone edges with
    # ≥2 host trials (estimable, not prior-dominated)
    rel: dict[str, list[float]] = defaultdict(list)
    for u, v, key, b in C.iter_edges(g):
        side = BIO_INCIDENT.get(key)
        if side is None or b is None:
            continue
        if len(set(C.trial_evidence_ncts(b))) < 2:
            continue
        bio_id = v if side == "tgt" else u
        rel[bio_id].append(b.expected_probability)
    qualified = {n: float(np.mean(vs)) for n, vs in rel.items()}
    print(f"  biology nodes with an estimable (≥2-trial) edge: {len(qualified)}")
    descs = {}
    for n in qualified:
        try:
            nd = g.get_node(n)
        except KeyError:
            continue
        d = (nd.get("description") or nd.get("name") or "").strip()
        if d:
            descs[n] = d
    nodes = [n for n in qualified if n in descs]
    if len(nodes) >= 8:
        embs = {n: embed_text(descs[n]) for n in nodes}
        rs = np.array([qualified[n] for n in nodes])
        # (i) pairwise corr(cosine, 1-|Δr|)
        cos_p, rel_p = [], []
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                cos_p.append(cosine_similarity(embs[nodes[i]], embs[nodes[j]]))
                rel_p.append(1.0 - abs(rs[i] - rs[j]))
        cos_p, rel_p = np.array(cos_p), np.array(rel_p)
        pair_corr = float(np.corrcoef(cos_p, rel_p)[0, 1]) if cos_p.std() > 0 else float("nan")
        # (ii) LOO kNN reliability prediction
        k = min(5, len(nodes) - 1)
        preds = []
        for i in range(len(nodes)):
            sims = np.array([cosine_similarity(embs[nodes[i]], embs[nodes[j]])
                             if j != i else -1 for j in range(len(nodes))])
            nn = np.argsort(-sims)[:k]
            preds.append(float(np.mean(rs[nn])))
        preds = np.array(preds)
        knn_corr = float(np.corrcoef(preds, rs)[0, 1]) if preds.std() > 0 and rs.std() > 0 else float("nan")
        print(f"  (i)  pairwise corr(BioLORD cosine, 1−|Δreliability|) = {pair_corr:+.3f} "
              f"(n_pairs={len(cos_p)})")
        print(f"  (ii) LOO {k}-NN reliability prediction corr(pred, actual) = {knn_corr:+.3f}")
        print(f"  reliability spread: mean={rs.mean():.3f} sd={rs.std():.3f}")
        print("  → this corr ≈ real geometry_alignment; compare to Step-2 α_cross")
    else:
        print(f"  too few estimable biology nodes ({len(nodes)}) for an alignment estimate")
    return 0


# load the snapshot's anchor-vector dedup table once (module global for the edge loop)
_VECTORS = None
if __name__ == "__main__":
    import json
    _p = sys.argv[1] if len(sys.argv) > 1 else "data/exports/multi_500_annotated.json"
    _VECTORS = json.load(open(_p)).get("_belief_field_vectors")
    raise SystemExit(main())
