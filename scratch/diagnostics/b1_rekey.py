"""B1 re-key: stamp GO-BP ontology_id on mapped biology nodes in an initial graph,
then run the biology-only Tier-1 merge (the same node_merge projection the build uses
for mechanism/Reactome). Output = a re-keyed initial, identical to the input except
that biology nodes sharing a GO-BP term are now one node. Re-attribute downstream.

Usage: EROOM_BIO_ONTOLOGY=1 [EROOM_BIO_ONTOLOGY_GATE=0.60] python -m scratch.diagnostics.b1_rekey IN OUT
"""
import sys

from src.graph import biology_ontology as bo
from src.graph.node_merge import MergeConfig, assemble
from src.graph.store import GraphStore


def main():
    init, out = sys.argv[1], sys.argv[2]
    g = GraphStore()
    g.import_snapshot(init)

    if not bo.enabled():
        print("EROOM_BIO_ONTOLOGY not set — refusing to re-key (would be a no-op).")
        sys.exit(2)

    before = len(g.get_nodes_by_type("BiologyNode"))
    mapped = 0
    for n in list(g.get_nodes_by_type("BiologyNode")):
        desc = n.get("description") or n.get("name") or ""
        oid = bo.ontology_id_for(desc)
        if oid:
            g._graph.nodes[n["id"]]["ontology_id"] = oid  # noqa: SLF001
            mapped += 1

    cfg = MergeConfig(
        node_types=("BiologyNode",),
        enable_id=True, enable_chembl=False, enable_name_id=False,
        enable_sapbert=False, enable_biolord=False, enable_box=False,
    )
    rep = assemble(g, cfg)
    after = len(g.get_nodes_by_type("BiologyNode"))
    print(f"gate={bo.gate()} mapped={mapped}/{before} biology {before}->{after} "
          f"(merged {before-after}) chains_updated={rep.chains_updated}")
    g.export_snapshot(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
