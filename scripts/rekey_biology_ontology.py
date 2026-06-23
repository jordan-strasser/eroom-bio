"""Re-key biology nodes by curated GO-BP ontology id, then merge same-GO biology.

This is the tracked, documented version of the step that produced the frontier
``*_rekeyed_initial`` graph (previously the untracked ``scratch/diagnostics/
b1_rekey.py``). It takes a populated *initial* snapshot, stamps the GO-BP
``ontology_id`` on each biology node whose description maps above the gate
(``CONFIG.bio_ontology_gate``), then runs the **biology-only Tier-1 (id) merge**
— the same node_merge projection the build uses for mechanism/Reactome — so
biology nodes sharing a GO-BP term collapse to one node. Re-attribute downstream.

  python -m scripts.rekey_biology_ontology IN.json OUT.json

PROVENANCE / RELATIONSHIP TO THE NATIVE BUILD
---------------------------------------------
The frontier graph was built with biology ontology keying OFF at populate time
(so biology got content-hash ``bio:<sha1>`` ids) and then re-keyed here — which
MERGES by a GO ``ontology_id`` attribute but KEEPS the sha1 ids. With the frontier
config now baked in ``src/config.py`` (``bio_ontology=True``), a fresh
``build_graph`` instead GO-keys biology *natively* at populate time, producing
``bio:GO:`` ids directly — a structurally DIFFERENT (and not yet AUROC-validated)
biology layer. This script remains the way to reproduce the EXACT frontier
snapshot from its tracked ``*_initial.json`` base.
"""

from __future__ import annotations

import argparse

from src.config import CONFIG
from src.graph import biology_ontology as bo
from src.graph.node_merge import MergeConfig, assemble
from src.graph.store import GraphStore


def rekey_biology_ontology(initial_path: str, out_path: str) -> None:
    g = GraphStore()
    g.import_snapshot(initial_path)

    before = len(g.get_nodes_by_type("BiologyNode"))
    mapped = 0
    for n in list(g.get_nodes_by_type("BiologyNode")):
        desc = n.get("description") or n.get("name") or ""
        oid = bo.ontology_id_for(desc)
        if oid:
            g._graph.nodes[n["id"]]["ontology_id"] = oid  # noqa: SLF001
            mapped += 1

    # Biology-only Tier-1 (id / ontology_id) merge — geometric tiers OFF, so this
    # collapses ONLY exact same-GO biology (not BioLORD neighbours).
    cfg = MergeConfig(
        node_types=("BiologyNode",),
        enable_id=True, enable_chembl=False, enable_name_id=False,
        enable_sapbert=False, enable_biolord=False, enable_box=False,
    )
    rep = assemble(g, cfg)
    after = len(g.get_nodes_by_type("BiologyNode"))
    print(
        f"gate={CONFIG.bio_ontology_gate} mapped={mapped}/{before} "
        f"biology {before}->{after} (merged {before - after}) "
        f"chains_updated={rep.chains_updated}"
    )
    g.export_snapshot(out_path)
    print(f"wrote {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("initial", help="populated initial.json to re-key")
    ap.add_argument("out", help="output rekeyed_initial.json")
    args = ap.parse_args()
    rekey_biology_ontology(args.initial, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
