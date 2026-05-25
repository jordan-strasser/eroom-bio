"""Fit manifold-1 box embeddings for a snapshot and store them PRIVATE (A.2).

Computes BioLORD centers for the graph's Population / Biology / Mechanism nodes
(using the A.0 descriptions, falling back to names), fits boxes with the
population-hierarchy supervision (parent ⊇ child by construction), attaches them
to nodes as the private ``box_min`` / ``box_max`` fields (auto-stripped from
public snapshots), and saves the trained boxes + a full private snapshot under
``EROOM_PRIVATE_ROOT`` (default ``~/.eroom/private``, which refuses to resolve
inside the repo).

Usage:
    python -m scripts.fit_boxes --graph data/exports/multi_indication_52_annotated.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.boundary import private_root
from src.graph.box_embeddings import (
    apply_boxes_to_graph,
    biology_parent_child_pairs,
    fit_graph_boxes,
    population_parent_child_pairs,
    relation,
    save_boxes,
)
from src.graph.store import GraphStore


def main() -> int:
    ap = argparse.ArgumentParser(description="Fit manifold-1 boxes (A.2).")
    ap.add_argument(
        "--graph", default="data/exports/multi_indication_52_annotated.json"
    )
    args = ap.parse_args()

    g = GraphStore()
    g.import_snapshot(args.graph)
    pop_pairs = population_parent_child_pairs(g)
    print(f"population-hierarchy supervision: {len(pop_pairs)} parent→child pairs")
    print("biology-hierarchy supervision (Reactome ancestors; first run hits network)…")
    bio_pairs = biology_parent_child_pairs(g)
    print(f"  {len(bio_pairs)} biology parent→child pairs")

    print("fitting boxes (per-node description SET → region + supervision)…")
    boxes = fit_graph_boxes(g, annotations_dir="data/annotations")
    print(f"  fit {len(boxes)} boxes")
    # biology is-a sanity: a parent pathway should test parent_of its child
    for parent, child in bio_pairs[:3]:
        if parent in boxes and child in boxes:
            print(f"  relation({parent}, {child}) = {relation(boxes[parent], boxes[child])}")

    applied = apply_boxes_to_graph(g, boxes)
    print(f"  attached box_min/box_max to {applied} nodes (private — stripped from public)")

    root = private_root(create=True)
    box_path = save_boxes(boxes, root / "manifold1_boxes.json")
    print(f"  trained boxes  -> {box_path}")
    priv_snap = root / (Path(args.graph).stem + "_with_boxes.json")
    g.export_private_snapshot(str(priv_snap))
    print(f"  private snapshot (with boxes) -> {priv_snap}")

    # Sanity: a parent population should test as parent_of its sub-region.
    checked = 0
    for parent, child in pop_pairs:
        if parent in boxes and child in boxes:
            rel = relation(boxes[parent], boxes[child])
            print(f"  relation({parent}, {child}) = {rel}")
            checked += 1
            if checked >= 3:
                break
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
