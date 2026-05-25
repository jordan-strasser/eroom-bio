"""Build the ground-up (chains-first) graph from a top-down snapshot and compare.

Explodes the snapshot's chains into trial-scoped node instances, reassembles via
the tiered node-merge, and prints the bottom-up vs top-down comparison. Tier-1
only (default) is the faithfulness check (should reconstruct the top-down concept
set); add --biolord to see what geometry consolidates beyond exact-id dedup.

Usage:
    python -m scripts.build_groundup --graph data/exports/multi_indication_52_annotated.json
    python -m scripts.build_groundup --graph ... --biolord 0.85
"""

from __future__ import annotations

import argparse

from src.graph.node_merge import MergeConfig
from src.graph.populate_groundup import build_groundup
from src.graph.store import GraphStore


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graph", default="data/exports/multi_indication_52_annotated.json")
    ap.add_argument("--biolord", type=float, default=None,
                    help="Enable Tier-3 BioLORD cosine merge at this threshold (e.g. 0.85).")
    args = ap.parse_args()

    td = GraphStore()
    td.import_snapshot(args.graph)

    embed_fn = None
    if args.biolord is not None:
        from src.graph.biolord_embeddings import embed_text as embed_fn  # noqa
        config = MergeConfig(
            node_types=("MechanismNode", "BiologyNode", "PopulationNode"),
            enable_id=True, enable_name_id=False, enable_sapbert=False,
            enable_biolord=True, biolord_threshold=args.biolord,
        )
    else:
        config = None  # Tier-1 only (faithfulness)

    print(f"top-down graph: {td._graph.number_of_nodes()} nodes, "  # noqa: SLF001
          f"{len(td.trial_subgraphs)} trial subgraphs")
    print(f"merge config: {'Tier-1 only' if config is None else f'Tier-1 + BioLORD@{args.biolord}'}\n")
    _, cmp = build_groundup(td, config=config, embed_fn=embed_fn)
    print(cmp.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
