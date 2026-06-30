"""Cross-indication learning census — across ALL node + edge types, not just biology.

`find_biology_bridges` only sees `biology_drives` fan-out (a biology driving ≥2
indications) — but each biology_drives/endpoint_captures edge terminates at ONE
indication, so that's the rarest form of transfer. The real pooling happens
UPSTREAM, on edges that don't terminate at an indication (`modulates_via`,
`mechanism_affects`, `affects`, `causes_ae`): their Beta belief co-accumulates
evidence from trials in many diseases. This census measures transfer everywhere.

Two views:
  * NODE-level: a node is a cross-indication hub if chains from ≥2 (canonical)
    indications reference it. MechanismNode/TargetNode dominate.
  * EDGE-level: an edge whose evidence records trace to ≥2 indications — the
    actual belief co-update that a prediction rides. `modulates_via` leads.

Run:  python -m scripts.cross_indication_census --graph data/exports/<area>_annotated.json
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict

from src.graph.store import GraphStore
from src.prediction.bridges import (
    build_nct_indication_index,
    canonical_indication,
    therapeutic_area,
)

_NCT = re.compile(r"^NCT\d{8}$")
_CHAIN_ROLES = ("compound_id", "target_id", "mechanism_id", "biology_id",
                "endpoint_id", "subgroup_population_id")
_NODE_ORDER = ("InterventionNode", "TargetNode", "MechanismNode", "BiologyNode",
               "EndpointNode", "PopulationNode")


def node_census(store: GraphStore) -> dict[str, dict]:
    """Per node type: nodes referenced by chains from ≥2 canonical indications
    (shared) and ≥2 therapeutic areas (cross-area), with top examples."""
    gr = store._graph  # noqa: SLF001
    node_inds: dict[str, set[str]] = defaultdict(set)
    for ts in store.trial_subgraphs.values():
        for ch in ts.chains:
            if not ch.indication_id or ch.indication_id == "UNKNOWN":
                continue
            ind = canonical_indication(ch.indication_id)
            for a in _CHAIN_ROLES:
                v = getattr(ch, a, None)
                if v and v != "UNKNOWN":
                    node_inds[v].add(ind)
    out: dict[str, dict] = defaultdict(
        lambda: {"touched": 0, "shared": 0, "cross_area": 0, "examples": []}
    )
    for nid, inds in node_inds.items():
        if nid not in gr:
            continue
        t = gr.nodes[nid].get("node_type")
        if not t:
            continue
        d = out[t]
        d["touched"] += 1
        if len(inds) >= 2:
            d["shared"] += 1
            n_areas = len({therapeutic_area(i) for i in inds})
            if n_areas >= 2:
                d["cross_area"] += 1
            d["examples"].append(
                (gr.nodes[nid].get("name", nid), len(inds), n_areas))
    return out


def edge_census(store: GraphStore) -> dict[str, list[int]]:
    """Per edge type: [edges_with_trial_evidence, cross_indication, cross_area]
    where cross_* counts edges whose evidence records trace to ≥2 indications /
    ≥2 areas. biology_drives & endpoint_captures are ~0 by construction (each
    edge terminates at a single indication — use node_census/find_biology_bridges
    for those)."""
    gr = store._graph  # noqa: SLF001
    idx = build_nct_indication_index(store)
    ec: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for _u, _v, k, data in gr.edges(keys=True, data=True):
        b = data.get("belief") or {}
        cinds: set[str] = set()
        for r in (b.get("evidence") or []):
            s = r.get("source_id", "")
            if _NCT.match(s):
                cinds |= {canonical_indication(i) for i in idx.get(s, set())}
        if not cinds:
            continue
        ec[k][0] += 1
        if len(cinds) >= 2:
            ec[k][1] += 1
            if len({therapeutic_area(i) for i in cinds}) >= 2:
                ec[k][2] += 1
    return ec


def print_census(store: GraphStore) -> None:
    nc = node_census(store)
    print("\nNODE-LEVEL — nodes shared across ≥2 indications (via chains)")
    print(f"  {'node type':16s} {'touched':>8s} {'shared':>8s} {'cross-area':>11s}")
    for t in _NODE_ORDER:
        d = nc.get(t)
        if not d:
            continue
        print(f"  {t:16s} {d['touched']:>8d} {d['shared']:>8d} {d['cross_area']:>11d}")
    for t in ("MechanismNode", "TargetNode"):
        d = nc.get(t)
        if not d:
            continue
        print(f"  top {t} by #indications:")
        for nm, ni, na in sorted(d["examples"], key=lambda x: -x[1])[:6]:
            print(f"     {nm[:40]:40s} {ni} ind / {na} areas")

    ec = edge_census(store)
    print("\nEDGE-LEVEL — belief co-updated by ≥2 indications")
    print(f"  {'edge type':24s} {'w/trial-ev':>10s} {'cross-ind':>10s} {'cross-area':>11s}")
    for et, (tot, ci, ca) in sorted(ec.items(), key=lambda x: -x[1][1]):
        print(f"  {et:24s} {tot:>10d} {ci:>10d} {ca:>11d}")
    print("\n  note: biology_drives/endpoint_captures ≈0 here by construction "
          "(single-indication per edge); use find_biology_bridges for biology fan-out.")
    print("  caveat: mechanism counts include generic over-shared mechanisms "
          "(e.g. 'signal transduction') — the holdout ablation is the predictive test, "
          "this census is the substrate.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    a = ap.parse_args()
    store = GraphStore()
    store.import_snapshot(a.graph)
    print(f"cross-indication census: {a.graph}")
    print_census(store)


if __name__ == "__main__":
    main()
