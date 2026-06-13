"""P3 — context heterogeneity on the highest-degree merged efficacy edges.

For the 20 efficacy/measurement edges with the most distinct host trials, report
how many distinct INDICATIONS feed them and the success-rate spread across those
host trials — a merged edge pooling incompatible contexts under one Beta.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

import scratch.diagnostics._common as C
from scripts.eval_holdout_compose import _resolve_label

g = C.load_graph("data/exports/multi_500_annotated.json")

# trial -> binary label + indication(s)
label = {}
trial_inds = {}
for nct, ts in g.trial_subgraphs.items():
    ext, cls = C.load_annotation(nct)
    if ext is not None:
        lab = _resolve_label(ext, cls)
        label[nct] = 1 if lab == "success" else 0 if lab == "failure" else None
    inds = set(c.indication_id for c in ts.chains
               if c.indication_id and c.indication_id != "UNKNOWN")
    trial_inds[nct] = inds

rows = []
for u, v, key, b in C.iter_edges(g):
    if b is None or C.EDGE_CLASS.get(key) not in (C.EFFICACY, C.MEASUREMENT):
        continue
    hosts = set(C.trial_evidence_ncts(b))
    if len(hosts) < 3:
        continue
    inds = set()
    labs = []
    for n in hosts:
        inds |= trial_inds.get(n, set())
        if label.get(n) is not None:
            labs.append(label[n])
    succ_rate = np.mean(labs) if labs else float("nan")
    rows.append((len(hosts), len(inds), succ_rate, key, u, v, len(labs)))

rows.sort(reverse=True)
print(f"{'#hosts':>7}{'#indic':>7}{'succ_rate':>10}{'n_lab':>6}  edge")
print("-" * 80)
for nh, ni, sr, key, u, v, nl in rows[:20]:
    nm_u = (g._graph.nodes[u].get('name') or u)[:20]
    print(f"{nh:>7}{ni:>7}{sr:>10.2f}{nl:>6}  {key} {nm_u}->{v[:24]}")

# summary: among multi-host edges, how spread are the host indications/outcomes?
multi = [r for r in rows]
n_multi_ind = sum(1 for r in multi if r[1] >= 2)
print(f"\nedges with >=3 hosts: {len(multi)}")
print(f"  of these, span >=2 distinct indications: {n_multi_ind} "
      f"({100*n_multi_ind/max(len(multi),1):.0f}%)")
mixed = [r for r in multi if r[6] >= 3 and 0.2 < r[2] < 0.8]
print(f"  with host success rate in (0.2,0.8) = genuinely mixed outcomes pooled "
      f"under one Beta: {len(mixed)} ({100*len(mixed)/max(len(multi),1):.0f}%)")
