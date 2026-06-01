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


def assemble_geometry(
    graph_path: str, *, annotations_dir: str = "data/annotations",
    merge: str | None = None,
) -> dict:
    """Fit boxes on all 7 chain node types, resolve the box-geometry is-a
    hierarchy (public SUBTYPE_OF edges written back to ``graph_path``), and save
    boxes + a private box snapshot under ``EROOM_PRIVATE_ROOT``. The optional
    ``merge`` ("id" or "biolord:<thr>") runs the node_merge projection first.
    Returns ``{nodes, trials, boxes, subtype_before, subtype_added,
    private_root, merged?}``.

    Reused by ``build_graph --assemble`` (geometry half; the field half is
    ``materialize_belief_field.materialize_field``)."""
    from pathlib import Path

    g = GraphStore()
    g.import_snapshot(graph_path)
    result: dict = {
        "nodes": g._graph.number_of_nodes(),  # noqa: SLF001
        "trials": len(g.trial_subgraphs),
    }

    if merge:
        embed_fn = None
        cfg = MergeConfig(node_types=ALL_CHAIN_TYPES, enable_id=True,
                          enable_name_id=True, enable_sapbert=False, enable_biolord=False)
        if merge.startswith("biolord:"):
            from src.graph.biolord_embeddings import embed_text as embed_fn  # noqa
            cfg.enable_biolord = True
            cfg.biolord_threshold = float(merge.split(":", 1)[1])
        rep = assemble(g, cfg, embed_fn=embed_fn)
        result["merged"] = {
            "before": rep.nodes_before, "after": rep.nodes_after, "by_type": rep.by_type,
        }

    boxes = fit_graph_boxes(g, node_types=ALL_CHAIN_TYPES, annotations_dir=annotations_dir)
    apply_boxes_to_graph(g, boxes)
    before = _subtype_count(g)
    added = resolve_hierarchy(g, boxes, node_types=ALL_CHAIN_TYPES)  # dict[type, count]
    after = _subtype_count(g)

    # public structure (is-a edges) back to the snapshot; boxes are private
    g.export_snapshot(graph_path)
    root = private_root(create=True)
    save_boxes(boxes, root / "manifold1_boxes.json")
    g.export_private_snapshot(str(root / (Path(graph_path).stem + "_with_boxes.json")))
    result.update({
        "boxes": len(boxes), "subtype_before": before, "subtype_after": after,
        "subtype_added": sum(added.values()), "subtype_added_by_type": added,
        "private_root": str(root),
    })
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--annotations", default="data/annotations")
    ap.add_argument("--merge", default=None,
                    help="Run node_merge projection first, e.g. 'biolord:0.85' or 'id'.")
    args = ap.parse_args()

    r = assemble_geometry(args.graph, annotations_dir=args.annotations, merge=args.merge)
    print(f"loaded {r['nodes']} nodes, {r['trials']} trials")
    if "merged" in r:
        m = r["merged"]
        print(f"merge: {m['before']} -> {m['after']} nodes, by_type={m['by_type']}")
    print(f"fit {r['boxes']} boxes; box-geometry is-a edges: SUBTYPE_OF "
          f"{r['subtype_before']} -> {r['subtype_after']} "
          f"(added {r['subtype_added']}, by_type={r['subtype_added_by_type']})")
    print(f"  boxes + private snapshot -> {r['private_root']}")
    print("now run: python -m scripts.materialize_belief_field --graph", args.graph)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
