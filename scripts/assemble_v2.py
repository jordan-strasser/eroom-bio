"""v2 post-build geometric assembly: fit boxes on ALL chain node types, resolve
box-geometry is-a hierarchy, and materialize the (s,t) belief field.

Runs after `build_graph` (which now produces descriptions + name_id on every
node). The optional `--merge` step applies the tunable node_merge projection
first (chains-first geometric/id consolidation). Boxes + field are PRIVATE
artifacts (under EROOM_PRIVATE_ROOT); the is-a (SUBTYPE_OF) edges are public
structure written back to the snapshot.

Usage:
    EROOM_PRIVATE_ROOT=/path python -m scripts.assemble_v2 --graph data/exports/mi_v2_annotated.json
    ... --merge biolord:0.85   # also run the geometric merge projection
"""

from __future__ import annotations

import argparse

from src.boundary import private_root
from src.graph.box_embeddings import apply_boxes_to_graph, fit_graph_boxes, save_boxes
from src.graph.models import EdgeType
from src.graph.node_merge import MergeConfig, assemble, resolve_hierarchy
from src.graph.store import GraphStore

ALL_CHAIN_TYPES = (
    "InterventionNode", "TargetNode", "MechanismNode", "BiologyNode",
    "EndpointNode", "IndicationNode", "PopulationNode",
)


def _subtype_count(g: GraphStore) -> int:
    return sum(1 for *_, k in g._graph.edges(keys=True)  # noqa: SLF001
               if k == EdgeType.SUBTYPE_OF.value)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--annotations", default="data/annotations")
    ap.add_argument("--merge", default=None,
                    help="Run node_merge projection first, e.g. 'biolord:0.85' or 'id'.")
    args = ap.parse_args()

    g = GraphStore()
    g.import_snapshot(args.graph)
    print(f"loaded {g._graph.number_of_nodes()} nodes, {len(g.trial_subgraphs)} trials")  # noqa: SLF001

    if args.merge:
        embed_fn = None
        cfg = MergeConfig(node_types=ALL_CHAIN_TYPES, enable_id=True,
                          enable_name_id=True, enable_sapbert=False, enable_biolord=False)
        if args.merge.startswith("biolord:"):
            from src.graph.biolord_embeddings import embed_text as embed_fn  # noqa
            cfg.enable_biolord = True
            cfg.biolord_threshold = float(args.merge.split(":", 1)[1])
        rep = assemble(g, cfg, embed_fn=embed_fn)
        print(f"merge: {rep.nodes_before} -> {rep.nodes_after} nodes, by_type={rep.by_type}")

    print("fitting boxes on ALL 7 chain node types…")
    boxes = fit_graph_boxes(g, node_types=ALL_CHAIN_TYPES, annotations_dir=args.annotations)
    apply_boxes_to_graph(g, boxes)
    print(f"  fit {len(boxes)} boxes")

    before = _subtype_count(g)
    added = resolve_hierarchy(g, boxes, node_types=ALL_CHAIN_TYPES)
    print(f"  box-geometry is-a edges: SUBTYPE_OF {before} -> {_subtype_count(g)} (added {added})")

    # public structure (is-a edges) back to the snapshot; boxes are private
    g.export_snapshot(args.graph)
    root = private_root(create=True)
    from pathlib import Path
    save_boxes(boxes, root / "manifold1_boxes.json")
    g.export_private_snapshot(str(root / (Path(args.graph).stem + "_with_boxes.json")))
    print(f"  boxes + private snapshot -> {root}")
    print("now run: python -m scripts.materialize_belief_field --graph", args.graph)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
