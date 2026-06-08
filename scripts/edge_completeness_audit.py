"""Systematic per-node edge-completeness audit.

For every node in a built graph, check the backbone edges that *should* touch it
(in and out) against the causal-chain contract, and report the gaps. This is the
topology-integrity counterpart to /graphguard (which checks edge *correctness*);
this checks edge *completeness* — "does every node carry the edges its node type
is supposed to have to participate in a causal hypothesis chain?"

The contract (the chain backbone + the disease-end triangle):

    Intervention --affects--> Target --modulates_via--> Mechanism
        --mechanism_affects--> Biology --biology_drives--> Indication
    Biology --reflects_biology--> Endpoint --endpoint_captures--> Indication
    Population --responds_differently--> Indication

Run:
    python -m scripts.edge_completeness_audit --graph data/exports/multi_500_annotated.json
    python -m scripts.edge_completeness_audit --graph <snap> --type EndpointNode --list
"""
from __future__ import annotations

import argparse
from collections import defaultdict

from src.graph.store import GraphStore

# Per node type: the backbone edge types that SHOULD be present (in / out) for a
# node that participates in a causal-hypothesis chain. "any_in"/"any_out" means
# "at least one of these". TrialNode is degree-0 by design (a container) and is
# excluded. BiomarkerNode participates via population features, not the backbone,
# so it has no contract here (reported as informational only).
CONTRACT: dict[str, dict[str, list[str]]] = {
    "InterventionNode": {"out": ["affects"]},
    "TargetNode": {"in": ["affects"], "out": ["modulates_via"]},
    "MechanismNode": {"in": ["modulates_via"], "out": ["mechanism_affects"]},
    "BiologyNode": {"in": ["mechanism_affects"], "out": ["biology_drives", "reflects_biology"]},
    "EndpointNode": {"in": ["reflects_biology"], "out": ["endpoint_captures"]},
    "IndicationNode": {"any_in": ["biology_drives"]},
    "PopulationNode": {"out": ["responds_differently"]},
    "AdverseEventNode": {"any_in": ["causes_ae", "target_associated_ae"]},
}

# Combo interventions hold the chain via composed_of constituents + inherited
# affects; a node that is ONLY a composed_of source still must have affects.
ROLE_ATTRS = (
    "compound_id", "target_id", "mechanism_id",
    "biology_id", "indication_id", "endpoint_id", "subgroup_population_id",
)
_UNKNOWN = "UNKNOWN"


def _edge_types(g: GraphStore, node: str, direction: str) -> set[str]:
    """Set of edge-type strings on a node's in/out edges (the multigraph key)."""
    gr = g._graph  # noqa: SLF001
    edges = gr.out_edges(node, keys=True) if direction == "out" else gr.in_edges(node, keys=True)
    return {k for *_, k in edges}


def _chain_node_ids(g: GraphStore) -> set[str]:
    """Every node id referenced by any trial's chains (+ parent populations)."""
    ids: set[str] = set()
    for ts in g.trial_subgraphs.values():
        if getattr(ts, "parent_population_id", None):
            ids.add(ts.parent_population_id)
        for ch in ts.chains:
            for a in ROLE_ATTRS:
                v = getattr(ch, a, None)
                if v and v != _UNKNOWN:
                    ids.add(v)
    return ids


def audit(path: str, only_type: str | None, do_list: bool) -> None:
    g = GraphStore()
    g.import_snapshot(path)
    gr = g._graph  # noqa: SLF001
    in_chain = _chain_node_ids(g)

    # node_type -> {"total", "complete", "orphan"(not in any chain),
    #               missing[edge] -> [node_ids]}
    stats: dict[str, dict] = defaultdict(
        lambda: {"total": 0, "complete": 0, "orphan": [], "missing": defaultdict(list)}
    )
    # triangle drill-down for indications
    ind_detail = {"no_biology_drives": [], "no_endpoint_captures": [],
                  "no_responds_differently": [], "fully_isolated": [],
                  "only_via_subtype_parent": []}

    for nid, attrs in gr.nodes(data=True):
        ntype = attrs.get("node_type", "?")
        if ntype == "TrialNode":
            continue
        st = stats[ntype]
        st["total"] += 1
        outs = _edge_types(g, nid, "out")
        ins = _edge_types(g, nid, "in")

        # orphan = not referenced by any chain (ghost/noise node)
        is_orphan = nid not in in_chain and ntype not in {"AdverseEventNode", "BiomarkerNode"}
        if is_orphan:
            st["orphan"].append(nid)

        contract = CONTRACT.get(ntype)
        if contract is None:
            continue
        node_gaps: list[str] = []
        for req in contract.get("out", []):
            if req not in outs:
                st["missing"][f"out:{req}"].append(nid)
                node_gaps.append(f"out:{req}")
        for req in contract.get("in", []):
            if req not in ins:
                st["missing"][f"in:{req}"].append(nid)
                node_gaps.append(f"in:{req}")
        if contract.get("any_in"):
            if not any(r in ins for r in contract["any_in"]):
                lab = "in:" + "|".join(contract["any_in"])
                st["missing"][lab].append(nid)
                node_gaps.append(lab)
        if contract.get("any_out"):
            if not any(r in outs for r in contract["any_out"]):
                lab = "out:" + "|".join(contract["any_out"])
                st["missing"][lab].append(nid)
                node_gaps.append(lab)
        if not node_gaps:
            st["complete"] += 1

        if ntype == "IndicationNode":
            has_bd = "biology_drives" in ins
            has_ec = "endpoint_captures" in ins
            has_rd = "responds_differently" in ins
            if not has_bd:
                ind_detail["no_biology_drives"].append(nid)
                # does a subtype parent carry biology_drives?
                parents = [t for _, t, k in gr.out_edges(nid, keys=True) if k == "subtype_of"]
                if any("biology_drives" in _edge_types(g, p, "in") for p in parents):
                    ind_detail["only_via_subtype_parent"].append(nid)
            if not has_ec:
                ind_detail["no_endpoint_captures"].append(nid)
            if not has_rd:
                ind_detail["no_responds_differently"].append(nid)
            if not (has_bd or has_ec or has_rd):
                ind_detail["fully_isolated"].append(nid)

    # ── report ──
    print(f"\nEDGE-COMPLETENESS AUDIT  ({path})")
    print(f"nodes={gr.number_of_nodes()}  edges={gr.number_of_edges()}  "
          f"chains touch {len(in_chain)} node ids\n")
    order = ["InterventionNode", "TargetNode", "MechanismNode", "BiologyNode",
             "EndpointNode", "IndicationNode", "PopulationNode",
             "AdverseEventNode", "BiomarkerNode"]
    for ntype in order:
        if ntype not in stats or (only_type and ntype != only_type):
            continue
        st = stats[ntype]
        contracted = ntype in CONTRACT
        comp = st["complete"] if contracted else "-"
        print(f"■ {ntype:18s} total={st['total']:4d}  complete={comp}  "
              f"orphan(not in chain)={len(st['orphan'])}")
        for edge, ids in sorted(st["missing"].items(), key=lambda kv: -len(kv[1])):
            print(f"      missing {edge:32s} {len(ids):4d}  e.g. {', '.join(ids[:4])}")
        if do_list and only_type == ntype:
            for edge, ids in sorted(st["missing"].items()):
                print(f"\n  --- {edge} ({len(ids)}) ---")
                for i in ids:
                    print(f"      {i}")

    print("\nINDICATION TRIANGLE DETAIL")
    for k, v in ind_detail.items():
        ex = ", ".join(v[:6])
        print(f"   {k:26s} {len(v):4d}   {ex}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--type", default=None, help="restrict report to one node type")
    ap.add_argument("--list", action="store_true", help="list every gap node id for --type")
    a = ap.parse_args()
    audit(a.graph, a.type, a.list)
