"""P6 (reuse) + P8 (posteriors) + P3 (merge/cycles) + P9.1/9.2 (AE by class).

Read-only over the annotated graph. No predictions, no LLM.
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict

import numpy as np
import networkx as nx

import scratch.diagnostics._common as C

G = sys.argv[1] if len(sys.argv) > 1 else "data/exports/multi_500_annotated.json"
g = C.load_graph(G)
st = g.stats()
print(f"graph: {G}")
print(f"nodes={st['node_count']} edges={st['edge_count']} trials={len(g.trial_subgraphs)} "
      f"total_evidence={st['total_evidence']}")

# Gather per-edge facts.
edges = []  # (src,tgt,key,class,belief)
for u, v, key, b in C.iter_edges(g):
    cls = C.EDGE_CLASS.get(key, "?")
    edges.append((u, v, key, cls, b))

# ── Node-id / merge scheme investigation (P3 / transfer substrate) ──────────
print("\n========== NODE-ID / MERGE SCHEME ==========")
mech_scoped = mech_total = 0
bio_total = bio_hash = 0
for n in g._graph.nodes:
    nt = g._graph.nodes[n].get("node_type")
    if nt == "MechanismNode":
        mech_total += 1
        if "#NCT" in n:
            mech_scoped += 1
    if nt == "BiologyNode":
        bio_total += 1
        if str(n).startswith("bio:"):
            bio_hash += 1
print(f"MechanismNodes: {mech_total}  | with '#NCT' (trial-scoped): {mech_scoped} "
      f"({100*mech_scoped/max(mech_total,1):.0f}%)")
print(f"BiologyNodes:   {bio_total}  | 'bio:<hash>' ids: {bio_hash}")
# How many distinct trials does each MERGED mechanism->biology edge pool?
# (degree of cross-trial sharing on the efficacy hub)

# ── P6: edge reuse (distinct host trials per edge) + node reuse ─────────────
print("\n========== P6 — REUSE HISTOGRAMS ==========")
# node reuse: how many trial subgraphs reference each node
node_trials: dict[str, set] = defaultdict(set)
for nct, ts in g.trial_subgraphs.items():
    seen = set()
    if ts.parent_population_id:
        seen.add(ts.parent_population_id)
    for arm in ts.arms:
        seen.add(arm.regimen_compound_id)
        seen.update(arm.compound_ids)
    for c in ts.chains:
        for f in ("compound_id", "target_id", "mechanism_id", "biology_id",
                  "indication_id", "endpoint_id", "subgroup_population_id"):
            val = getattr(c, f, None)
            if val and val != "UNKNOWN":
                seen.add(val)
    for nid in seen:
        node_trials[nid].add(nct)


def hist(counts: list[int]) -> str:
    c = Counter()
    for n in counts:
        if n <= 0:
            c["0"] += 1
        elif n == 1:
            c["1"] += 1
        elif n == 2:
            c["2"] += 1
        elif n <= 4:
            c["3-4"] += 1
        else:
            c[">=5"] += 1
    tot = max(sum(c.values()), 1)
    parts = []
    for k in ("0", "1", "2", "3-4", ">=5"):
        parts.append(f"{k}:{c[k]} ({100*c[k]/tot:.0f}%)")
    return "  ".join(parts)


# node reuse by node type (only nodes that participate in chains)
by_nt = defaultdict(list)
for n in g._graph.nodes:
    nt = g._graph.nodes[n].get("node_type", "?")
    by_nt[nt].append(len(node_trials.get(n, ())))
print("node reuse (#distinct trials referencing the node), by node type:")
for nt in ("InterventionNode", "TargetNode", "MechanismNode", "BiologyNode",
           "IndicationNode", "EndpointNode", "PopulationNode"):
    vals = by_nt.get(nt, [])
    if vals:
        print(f"  {nt:18} n={len(vals):4}  reuse {hist(vals)}  median={int(np.median(vals))}")

# edge reuse: distinct host NCTs per edge, by class
print("\nedge reuse (#distinct host NCT trials whose outcome touched the edge), by class:")
cls_edge_trials = defaultdict(list)
cls_singleton = defaultdict(int)
for u, v, key, cls, b in edges:
    if b is None:
        continue
    ncts = set(C.trial_evidence_ncts(b))
    cls_edge_trials[cls].append(len(ncts))
for cls in (C.EFFICACY, C.MEASUREMENT, C.SAFETY, C.MODULATION):
    vals = cls_edge_trials.get(cls, [])
    if vals:
        print(f"  {cls:12} n_edges={len(vals):5}  {hist(vals)}  "
              f"mean_trials/edge={np.mean(vals):.2f}")

# ── P8: posterior-mean distribution by class (discrimination check) ─────────
print("\n========== P8 — POSTERIOR-MEAN DISTRIBUTION BY CLASS ==========")
print("(only edges with evidence_strength>0 — the ones prediction actually uses)")
for cls in (C.EFFICACY, C.MEASUREMENT, C.SAFETY, C.MODULATION):
    ep = [b.expected_probability for (u, v, k, c, b) in edges
          if c == cls and b is not None and b.evidence_strength > 0]
    if ep:
        ep = np.array(ep)
        print(f"  {cls:12} n={len(ep):5}  mean={ep.mean():.3f} sd={ep.std():.3f} "
              f"min={ep.min():.3f} p25={np.percentile(ep,25):.3f} "
              f"med={np.median(ep):.3f} p75={np.percentile(ep,75):.3f} max={ep.max():.3f}")
# global base rate of trial success (binary, mechanistic) for reference
labels = []
for nct in g.trial_subgraphs:
    ext, cls_j = C.load_annotation(nct)
    if ext is None:
        continue
    from scripts.eval_holdout_compose import _resolve_label
    lab = _resolve_label(ext, cls_j)
    if lab in ("success", "failure"):
        labels.append(1 if lab == "success" else 0)
print(f"  [ref] mechanistic binary base success rate over in-graph trials: "
      f"{np.mean(labels):.3f}  (n={len(labels)}: succ={sum(labels)}, fail={len(labels)-sum(labels)})")

# ── empty-edge census (#5 — empty edges drag/contribute nothing) ────────────
print("\n========== EMPTY-EDGE CENSUS (Beta(1,1), dropped by predictor) ==========")
for key in ("affects", "modulates_via", "mechanism_affects", "biology_drives",
            "reflects_biology", "endpoint_captures", "responds_differently",
            "causes_ae", "target_associated_ae", "modulates_efficacy_of"):
    tot = [e for e in edges if e[2] == key]
    empty = [e for e in tot if e[4] is None or e[4].evidence_strength <= 0]
    print(f"  {key:22} total={len(tot):5} empty={len(empty):5} "
          f"({100*len(empty)/max(len(tot),1):.0f}%) class={C.EDGE_CLASS.get(key)}")

# ── P3: cycles — the biology/endpoint/indication triangle ───────────────────
print("\n========== P3 — CYCLES (triangle) ==========")
belief_types = {"affects", "modulates_via", "mechanism_affects", "biology_drives",
                "reflects_biology", "endpoint_captures", "responds_differently",
                "causes_ae", "target_associated_ae", "modulates_efficacy_of"}
H = nx.MultiDiGraph()
for u, v, key, cls, b in edges:
    if key in belief_types:
        H.add_edge(u, v, key=key)
# directed cycles among the belief edges
sccs = [c for c in nx.strongly_connected_components(H) if len(c) > 1]
print(f"belief-edge subgraph: {H.number_of_nodes()} nodes {H.number_of_edges()} edges")
print(f"strongly-connected components with >1 node (directed cycles): {len(sccs)}")
# specifically: biology->endpoint (reflects_biology), endpoint->indication
# (endpoint_captures), biology->indication (biology_drives). Is there a 2-cycle
# or back-edge forming a loop biology<->indication via endpoint? It's a DAG
# fan-in (b->ep->ind and b->ind), not a cycle, UNLESS an endpoint also drives
# biology. Count triangles where all three edges exist.
tri = 0
for b_node in [n for n in g._graph.nodes
               if g._graph.nodes[n].get("node_type") == "BiologyNode"]:
    for _, ep, k1 in [(u, v, k) for u, v, k in H.out_edges(b_node, keys=True)
                      if k == "reflects_biology"]:
        for _, ind, k2 in [(u, v, k) for u, v, k in H.out_edges(ep, keys=True)
                           if k == "endpoint_captures"]:
            if H.has_edge(b_node, ind, key="biology_drives"):
                tri += 1
print(f"biology→endpoint→indication + biology→indication 'triangles' (shared-node "
      f"fan-in, double-counted by a flat product): {tri}")

# ── P9.1/9.2: AE edges by class + within-target consistency ─────────────────
print("\n========== P9 — SAFETY EDGE TRANSFER ==========")
# 9.1 reuse already above. 9.2: within-target consistency.
# target_associated_ae: target -> AE. For targets in >=3 trials, is the AE
# posterior consistent? Compare its variance to the efficacy-spine edges on the
# same target (affects into target, modulates_via out of target).
# Build: for each target node, the set of its target_associated_ae E[p] and the
# affects-edge E[p] from compounds.
tgt_assoc = defaultdict(list)   # target -> [E[p] of its target_associated_ae]
tgt_affects = defaultdict(list)  # target -> [E[p] of affects edges into it]
tgt_modvia = defaultdict(list)
for u, v, key, cls, b in edges:
    if b is None or b.evidence_strength <= 0:
        continue
    if key == "target_associated_ae":
        tgt_assoc[u].append(b.expected_probability)
    elif key == "affects":
        tgt_affects[v].append(b.expected_probability)
    elif key == "modulates_via":
        tgt_modvia[u].append(b.expected_probability)
# within-target spread: report mean within-target sd for AE vs efficacy edges
def mean_within_sd(d, min_k=3):
    sds = [np.std(vs) for vs in d.values() if len(vs) >= min_k]
    return (np.mean(sds), len(sds)) if sds else (float("nan"), 0)
ae_sd, ae_n = mean_within_sd(tgt_assoc)
af_sd, af_n = mean_within_sd(tgt_affects)
mv_sd, mv_n = mean_within_sd(tgt_modvia)
print(f"within-target posterior spread (mean SD across targets w/ >=3 edges):")
print(f"  target_associated_ae : SD={ae_sd:.3f}  (n_targets={ae_n})")
print(f"  affects (into target): SD={af_sd:.3f}  (n_targets={af_n})")
print(f"  modulates_via (out)  : SD={mv_sd:.3f}  (n_targets={mv_n})")
# also: how many distinct compounds share a target_associated_ae (transfer breadth)
print(f"\ncauses_ae total edges: {sum(1 for e in edges if e[2]=='causes_ae')}; "
      f"target_associated_ae: {sum(1 for e in edges if e[2]=='target_associated_ae')}")
