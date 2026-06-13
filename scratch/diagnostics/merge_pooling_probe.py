"""MERGE_POOLING_MAP probe — node/edge reuse + scalar-vs-field + effective reuse.

Read-only on the field-materialized annotated snapshot
(data/exports/multi_500_annotated.json carries belief_field + the
_belief_field_vectors dedup table). Produces every number for MERGE_POOLING_MAP.md
(P1 node reuse, P2 edge reuse, P4 scalar-vs-field agreement, P5 effective reuse,
P6 per-branch / per-target).

  .venv/bin/python -m scratch.diagnostics.merge_pooling_probe
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np

import scratch.diagnostics._common as C
from src.inference.belief_field import BeliefField

PATH = sys.argv[1] if len(sys.argv) > 1 else "data/exports/multi_500_annotated.json"
VECTORS = json.load(open(PATH)).get("_belief_field_vectors")

CHAIN_FIELDS = ("compound_id", "target_id", "mechanism_id", "biology_id",
                "indication_id", "endpoint_id", "subgroup_population_id")
# chain attr -> node_type, for node reuse
FIELD_TYPE = {
    "compound_id": "InterventionNode", "target_id": "TargetNode",
    "mechanism_id": "MechanismNode", "biology_id": "BiologyNode",
    "indication_id": "IndicationNode", "endpoint_id": "EndpointNode",
    "subgroup_population_id": "PopulationNode",
}
BACKBONE = {"affects", "modulates_via", "mechanism_affects", "biology_drives",
            "reflects_biology", "endpoint_captures", "responds_differently"}


def hist(a):
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return "n=0"
    return (f"n={a.size} median={np.median(a):.2f} mean={a.mean():.2f} "
            f"p90={np.percentile(a,90):.1f} max={a.max():.0f} "
            f"%==0={100*np.mean(a==0):.0f} %==1={100*np.mean(a==1):.0f} "
            f"%>=2={100*np.mean(a>=2):.0f} %>=8={100*np.mean(a>=8):.1f}")


def edge_eff(field: BeliefField, *, cross_trial_only: bool):
    """Field effective sample size for ONE edge (it sums only its own anchors).

    For each anchor as the query point q, Σ_j w(q, a_j) over the edge's anchors.
    cross_trial_only: when q is from trial T, sum only anchors NOT from T (the
    cross-trial borrowing the field manufactures, dropping self-fanout). Returns
    the mean over query points + distinct host-trial count.
    """
    A = field.anchors
    if not A:
        return 0.0, 0
    if len(A) > 400:
        idx = np.random.default_rng(0).choice(len(A), 400, replace=False)
        A = [A[i] for i in idx]
    S = np.array([a.s for a in A], dtype=float)
    T = np.array([a.t for a in A], dtype=float)
    S /= np.clip(np.linalg.norm(S, axis=1, keepdims=True), 1e-9, None)
    T /= np.clip(np.linalg.norm(T, axis=1, keepdims=True), 1e-9, None)
    W = np.exp((S @ S.T + T @ T.T - 2.0) / field.bandwidth)
    ncts = [a.nct for a in A]
    if cross_trial_only:
        same = np.array([[ (ncts[i] is not None and ncts[i] == ncts[j])
                           for j in range(len(A))] for i in range(len(A))])
        W = np.where(same, 0.0, W)
    eff = float(W.sum(axis=1).mean())
    return eff, len({n for n in ncts if n})


def main():
    g = C.load_graph(PATH)
    print(f"graph: {PATH}\n")

    # ===== P1 — node reuse per type =====
    node_trials = defaultdict(set)
    node_type_of = {}
    for nct, ts in g.trial_subgraphs.items():
        seen = set()
        for c in ts.chains:
            for f in CHAIN_FIELDS:
                v = getattr(c, f, None)
                if v and v != "UNKNOWN":
                    seen.add(v); node_type_of[v] = FIELD_TYPE[f]
        for nid in seen:
            node_trials[nid].add(nct)

    # all nodes of each type in the graph (incl. reuse-0 unused ones)
    type_nodes = defaultdict(list)
    for nid in g._graph.nodes:
        nt = C.node_type(g, nid)
        type_nodes[nt].append(nid)

    print("===== P1 — node reuse (# distinct host trials) per node type =====")
    for nt in ("InterventionNode", "TargetNode", "MechanismNode", "BiologyNode",
               "EndpointNode", "IndicationNode", "PopulationNode"):
        reuse = [len(node_trials.get(n, ())) for n in type_nodes.get(nt, [])]
        print(f"  {nt:18s} total={len(reuse):4d}  {hist(reuse)}")

    # AdverseEventNode reuse = distinct host trials via causes_ae / target_associated_ae
    ae_trials = defaultdict(set)
    for u, v, key, b in C.iter_edges(g):
        if key in ("causes_ae", "target_associated_ae") and b is not None:
            for n in C.trial_evidence_ncts(b):
                ae_trials[v].add(n)
    ae_nodes = type_nodes.get("AdverseEventNode", [])
    ae_reuse = [len(ae_trials.get(n, ())) for n in ae_nodes]
    print(f"  {'AdverseEventNode':18s} total={len(ae_reuse):4d}  {hist(ae_reuse)}  (via AE edges)")

    # ===== P2 — edge reuse per class =====
    print("\n===== P2 — edge reuse (# distinct host NCTs) per edge class =====")
    cls_reuse = defaultdict(list)
    bytype_reuse = defaultdict(list)
    for u, v, key, b in C.iter_edges(g):
        if b is None:
            continue
        cls = C.EDGE_CLASS.get(key, "?")
        if cls == "STRUCTURAL" or cls == "structural":
            continue
        r = len(set(C.trial_evidence_ncts(b)))
        cls_reuse[cls].append(r)
        bytype_reuse[key].append(r)
    for cls in ("efficacy", "measurement", "safety", "modulation"):
        a = np.asarray(cls_reuse.get(cls, []), float)
        if a.size:
            print(f"  {cls:12s} edges={a.size:5d}  mean trials/edge={a.mean():.2f}  "
                  f"reuse0={100*np.mean(a==0):.0f}% reuse1={100*np.mean(a==1):.0f}% "
                  f"reuse>=2={100*np.mean(a>=2):.0f}%  max={a.max():.0f}")
    print("  -- by edge type --")
    for key in ("affects", "modulates_via", "mechanism_affects", "biology_drives",
                "reflects_biology", "endpoint_captures", "responds_differently",
                "causes_ae", "target_associated_ae", "modulates_efficacy_of"):
        a = np.asarray(bytype_reuse.get(key, []), float)
        if a.size:
            print(f"    {key:22s} edges={a.size:5d} mean={a.mean():.2f} "
                  f"%>=2={100*np.mean(a>=2):.0f}% max={a.max():.0f}")

    # ===== P4/P5 — scalar vs field, effective reuse (backbone edges w/ field) =====
    print("\n===== P4 — scalar E[p] vs field E[p] (backbone edges, >=2 host trials) =====")
    rows = []  # (key, scalar, field_at_anchors, n_trials, eff, eff_cross)
    eff_by_cls = defaultdict(list)         # class -> [eff cross-trial]
    int_by_cls = defaultdict(list)         # class -> [integer reuse]
    bio_node_eff = defaultdict(float)
    bio_node_int = {}
    BIO_SIDE = {"mechanism_affects": "t", "biology_drives": "s", "reflects_biology": "s"}
    for u, v, key, data in g._graph.edges(data=True, keys=True):
        if key not in BACKBONE:
            continue
        bf = (data.get("belief") or {}).get("belief_field")
        belief = data.get("belief") or {}
        a0, b0 = float(belief.get("alpha", 1.0)), float(belief.get("beta", 1.0))
        if not bf or not bf.get("anchors"):
            continue
        field = BeliefField.from_dict(bf, VECTORS)
        if not field.anchors:
            continue
        scalar = a0 / (a0 + b0)
        # field E[p] averaged over the edge's own anchor coordinates
        fps = [field_expected(field, anc.s, anc.t) for anc in field.anchors[:200]]
        field_ep = float(np.mean(fps))
        eff, ntr = edge_eff(field, cross_trial_only=False)
        eff_x, _ = edge_eff(field, cross_trial_only=True)
        cls = C.EDGE_CLASS.get(key, "?")
        int_by_cls[cls].append(ntr)
        eff_by_cls[cls].append(eff_x)
        if ntr >= 2:
            rows.append((f"{u}--{key}-->{v}", scalar, field_ep, ntr, eff, eff_x))
        side = BIO_SIDE.get(key)
        if side:
            bio = v if side == "t" else u
            bio_node_eff[bio] = max(bio_node_eff[bio], eff)
            bio_node_int[bio] = max(bio_node_int.get(bio, 0), ntr)

    if rows:
        sc = np.array([r[1] for r in rows]); fl = np.array([r[2] for r in rows])
        corr = float(np.corrcoef(sc, fl)[0, 1])
        print(f"  edges compared: {len(rows)}   corr(scalar,field)={corr:+.3f}  "
              f"mean|Δ|={np.mean(np.abs(sc-fl)):.3f}  "
              f"field moves >0.05 from scalar on {100*np.mean(np.abs(sc-fl)>0.05):.0f}% of edges")
        print("  sample (first 20, ≥2 host trials):")
        print(f"    {'edge':52s} {'scalar':>7s} {'field':>7s} {'ntr':>4s} {'eff':>6s} {'effX':>6s}")
        for e, scl, fld, ntr, eff, effx in rows[:20]:
            print(f"    {e[:52]:52s} {scl:7.3f} {fld:7.3f} {ntr:4d} {eff:6.2f} {effx:6.2f}")

    # ===== P5 — effective reuse multiplier + %≥8 =====
    print("\n===== P5 — integer reuse vs field EFFECTIVE reuse =====")
    for cls in ("efficacy", "measurement"):
        ii = np.asarray(int_by_cls.get(cls, []), float)
        ee = np.asarray(eff_by_cls.get(cls, []), float)
        if ii.size:
            mult = np.median(ee[ii > 0] / np.clip(ii[ii > 0], 1e-9, None)) if (ii > 0).any() else float("nan")
            print(f"  {cls:12s} edges={ii.size}: integer %>=8={100*np.mean(ii>=8):.1f}%  "
                  f"field-effX %>=8={100*np.mean(ee>=8):.1f}%  median(effX/int)={mult:.2f}x")
    # biology NODE level
    bn = list(bio_node_int)
    ii = np.array([bio_node_int[n] for n in bn], float)
    ee = np.array([bio_node_eff[n] for n in bn], float)
    if bn:
        print(f"  biology NODES (max over incident edges) n={len(bn)}: "
              f"integer %>=8={100*np.mean(ii>=8):.1f}%  field-eff %>=8={100*np.mean(ee>=8):.1f}%")

    # ===== P6 — per-target reuse + within-target AE SD =====
    print("\n===== P6 — per-target reuse + within-target AE SD =====")
    tgt_edge_trials = defaultdict(set)   # target -> distinct trials over its anchored edges
    tgt_edges = defaultdict(int)
    for u, v, key, b in C.iter_edges(g):
        if b is None or b.evidence_strength <= 0:
            continue
        # target participates as: affects(target=v), modulates_via(target=u), target_associated_ae(target=u)
        tgt = None
        if key == "affects":
            tgt = v
        elif key in ("modulates_via", "target_associated_ae"):
            tgt = u
        if tgt is not None and C.node_type(g, tgt) == "TargetNode":
            tgt_edges[tgt] += 1
            for n in C.trial_evidence_ncts(b):
                tgt_edge_trials[tgt].add(n)
    dense = {t: len(s) for t, s in tgt_edge_trials.items() if tgt_edges[t] >= 3}
    if dense:
        a = np.array(list(dense.values()), float)
        print(f"  targets w/≥3 anchored edges: {len(dense)}  distinct-trial reuse: "
              f"median={np.median(a):.1f} mean={a.mean():.1f} max={a.max():.0f}")
    tgt_assoc = defaultdict(list)
    for u, v, key, b in C.iter_edges(g):
        if b is None or b.evidence_strength <= 0:
            continue
        if key == "target_associated_ae":
            tgt_assoc[u].append(b.expected_probability)
    sds = [np.std(vs) for vs in tgt_assoc.values() if len(vs) >= 3]
    eff_sds = [np.std(vs) for vs in [v for k, v in C_affects(g).items()] if len(vs) >= 3]
    if sds:
        print(f"  within-target AE E[p] SD (targets w/≥3 AE edges): "
              f"mean={np.mean(sds):.3f} (n={len(sds)})  [P9 baseline 0.048]")
    return 0


def field_expected(field, s, t):
    a, b = field.fallback_prior()
    import math
    for anc in field.anchors:
        # cosine via dot on (near) unit vectors
        def cos(x, y):
            dot = sum(p * q for p, q in zip(x, y))
            nx = sum(p * p for p in x) ** 0.5
            ny = sum(q * q for q in y) ** 0.5
            return dot / (nx * ny) if nx and ny else 0.0
        w = math.exp((cos(s, anc.s) + cos(t, anc.t) - 2.0) / field.bandwidth)
        a += w * anc.alpha; b += w * anc.beta
    return a / (a + b)


def C_affects(g):
    """within-target affects E[p] groups (for an efficacy contrast to AE SD)."""
    d = defaultdict(list)
    for u, v, key, b in C.iter_edges(g):
        if key == "affects" and b is not None and b.evidence_strength > 0:
            d[v].append(b.expected_probability)
    return d


if __name__ == "__main__":
    raise SystemExit(main())
