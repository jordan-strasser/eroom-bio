"""Migrate an existing graph snapshot to use INN names instead of dev codes.

Walks the snapshot, applies `src/graph/compound_codenames.CODENAME_TO_INN`,
and rewrites every InterventionNode whose id matches a code form to use
the canonical INN form. Updates:

  * The InterventionNode itself: id renamed, name swapped, the old
    codename added to aliases (so the historical text is preserved).
  * Every edge whose source/target is the old codename id: source /
    target updated to the new INN id.
  * The `trial_subgraphs` sidecar's `compound_id` / `regimen_compound_id`
    fields — searches arms[] and chains[] for the old codename and
    swaps them.
  * `_entity_index` doesn't live in the snapshot; rebuilt at GraphStore
    import time.

If the INN already exists as a compound node, the codename node's
edges and trial-subgraph references are MERGED into the existing INN
node (with deduplication on edge keys + alias union). If the INN
doesn't exist yet, the codename node is renamed in place.

Usage:
    python -m scripts.canonicalize_codenames \
        --in data/exports/multi_indication_50_annotated.json \
        --out data/exports/multi_indication_50_annotated.json   # in-place
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from src.graph.compound_codenames import CODENAME_TO_INN


def _migrate(snap: dict[str, Any]) -> dict[str, Any]:
    g = snap["graph"]
    nodes_by_id: dict[str, dict] = {n["id"]: n for n in g["nodes"]}
    # Build the rename plan: every existing compound id that maps to a
    # different INN. Skip self-mapped entries.
    rename: dict[str, str] = {}
    for nid, node in nodes_by_id.items():
        if node.get("node_type") != "InterventionNode":
            continue
        target = CODENAME_TO_INN.get(nid)
        if target and target != nid:
            rename[nid] = target

    if not rename:
        print("  no codename-→-INN renames needed")
        return snap

    print(f"  applying {len(rename)} renames:")
    for old, new in rename.items():
        print(f"    {old} → {new}")

    # Step 1: update / merge nodes.
    new_nodes_by_id: dict[str, dict] = {}
    for nid, node in nodes_by_id.items():
        new_id = rename.get(nid, nid)
        if new_id in new_nodes_by_id:
            # An INN already exists; merge the codename's aliases into it.
            existing = new_nodes_by_id[new_id]
            existing_aliases = list(existing.get("aliases") or [])
            new_aliases = [node.get("name", nid)] + list(node.get("aliases") or [])
            seen_aliases = set(existing_aliases)
            for a in new_aliases:
                if a and a not in seen_aliases:
                    existing_aliases.append(a)
                    seen_aliases.add(a)
            existing["aliases"] = existing_aliases
            # Preserve chembl_id if codename had one and INN didn't.
            if node.get("chembl_id") and not existing.get("chembl_id"):
                existing["chembl_id"] = node["chembl_id"]
        elif new_id != nid:
            # Rename in place: id changes, name becomes INN, old name +
            # any old aliases preserved on the aliases list.
            renamed = copy.deepcopy(node)
            renamed["id"] = new_id
            old_name = renamed.get("name", nid)
            renamed["name"] = new_id  # INN slug as display name fallback
            old_aliases = list(renamed.get("aliases") or [])
            new_aliases: list[str] = []
            seen_aliases = set()
            for candidate in [old_name] + old_aliases:
                if candidate and candidate not in seen_aliases:
                    new_aliases.append(candidate)
                    seen_aliases.add(candidate)
            renamed["aliases"] = new_aliases
            new_nodes_by_id[new_id] = renamed
        else:
            new_nodes_by_id[new_id] = node
    g["nodes"] = list(new_nodes_by_id.values())

    # Step 2: update edges that referenced old ids.
    n_edges_changed = 0
    seen_keys: set[tuple] = set()
    deduped_edges: list[dict] = []
    for edge in g["edges"]:
        src = edge.get("source")
        tgt = edge.get("target")
        new_src = rename.get(src, src)
        new_tgt = rename.get(tgt, tgt)
        if (new_src != src) or (new_tgt != tgt):
            edge["source"] = new_src
            edge["target"] = new_tgt
            n_edges_changed += 1
        # Deduplicate exact-match edge triples after rename — the same
        # (src, tgt, key) might now appear under both INN and codename.
        key = (edge["source"], edge["target"], edge.get("key") or edge.get("edge_type"))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped_edges.append(edge)
    g["edges"] = deduped_edges
    print(f"  edges: {n_edges_changed} re-pointed, {len(g['edges'])} after dedup")

    # Step 3: update trial_subgraphs sidecar (arms + chains).
    n_ts_fields_changed = 0
    for tid, ts in snap.get("trial_subgraphs", {}).items():
        for arm in ts.get("arms", []):
            old_rcid = arm.get("regimen_compound_id")
            if old_rcid in rename:
                arm["regimen_compound_id"] = rename[old_rcid]
                n_ts_fields_changed += 1
            cids = arm.get("compound_ids") or []
            new_cids = [rename.get(c, c) for c in cids]
            # dedup
            seen_c: set[str] = set()
            uniq_cids = []
            for c in new_cids:
                if c and c not in seen_c:
                    seen_c.add(c)
                    uniq_cids.append(c)
            if uniq_cids != cids:
                arm["compound_ids"] = uniq_cids
                n_ts_fields_changed += 1
        for chain in ts.get("chains", []):
            old_cid = chain.get("compound_id")
            if old_cid in rename:
                chain["compound_id"] = rename[old_cid]
                n_ts_fields_changed += 1
    print(f"  trial-subgraph compound_id fields changed: {n_ts_fields_changed}")

    return snap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="input", required=True, type=Path)
    parser.add_argument("--out", dest="output", required=True, type=Path)
    args = parser.parse_args()

    print(f"Loading {args.input}…")
    snap = json.loads(args.input.read_text())
    snap = _migrate(snap)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(snap, indent=2, sort_keys=False))
    print(f"  wrote {args.output}")


if __name__ == "__main__":
    main()
