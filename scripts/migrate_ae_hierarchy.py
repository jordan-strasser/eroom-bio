"""Migrate a pre-round-28 snapshot to include MedDRA hierarchy on AE nodes.

Round 28 added ``hlt_id`` / ``hlgt_id`` / ``soc_id`` / ``soc_name``
fields to ``AdverseEventNode`` and SOC-tier ``target_associated_ae``
edges. Snapshots built before round 28 still load (the fields default
to empty strings), but they don't carry parent ids on their AE nodes
and SOC-tier propagation has never run. This script fills both gaps
in place against an existing snapshot:

1. For every AdverseEventNode in the snapshot, look up MedDRA parents
   via the curated hierarchy file (``data/external/meddra_hierarchy.json``)
   and write ``soc_id`` / ``soc_name`` / ``hlt_id`` / ``hlgt_id`` onto
   the node (only fields that are currently empty).
2. For every (compound, AE) pair with a non-zero ``causes_ae`` belief,
   re-run ``propagate_to_target_associated_ae`` with ``roll_up_tier="soc"``.
   That creates the new ``AE:soc:<soc_id>`` SOC-tier nodes and emits
   ``target_associated_ae`` edges keyed on them.

The script is idempotent — running it twice produces the same output
as running it once. AE nodes that already carry hierarchy fields are
left alone; SOC-tier ``target_associated_ae`` edges are rebuilt from
scratch (the propagation routine is idempotent).

Usage:
  .venv/bin/python scripts/migrate_ae_hierarchy.py \\
    --in  data/exports/multi_indication_30_annotated.json \\
    --out data/exports/multi_indication_30_v028_annotated.json
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import json

from src.annotation.meddra import MeddraCache, ae_node_id
from src.annotation.meddra_hierarchy import MedDRAHierarchy
from src.graph.models import EdgeType, MechanismCategory
from src.graph.store import GraphStore
from src.inference.ae_propagation import propagate_to_target_associated_ae
from src.ingestion.lincs import _category_to_direction


_AE_HIERARCHY_FIELDS = ("hlt_id", "hlgt_id", "soc_id", "soc_name")


def _backfill_ae_serious_from_extractions(
    graph: GraphStore,
    annotations_dir: Path = Path("data/annotations"),
) -> Counter:
    """Round-29 backfill: walk per-trial extraction files, OR-merge the
    StructuredAE.serious flag onto each PT-level AE node.

    Pre-round-29 snapshots have no ``serious`` field on AE nodes —
    extraction has the flag (CT.gov populates it on ~90% of AEs) but
    ``_ensure_ae_node`` discarded it. Backfilling lets the safety-
    penalty math pick up the SAE signal without a fresh rebuild.
    Idempotent: re-running OR-merges the same bools into the same state.
    """
    stats: Counter = Counter()
    cache = MeddraCache()
    g = graph._graph  # noqa: SLF001
    if not annotations_dir.exists():
        stats["extraction_files_seen"] = 0
        return stats
    extraction_paths = sorted(annotations_dir.glob("*_extraction.json"))
    stats["extraction_files_seen"] = len(extraction_paths)
    for p in extraction_paths:
        try:
            data = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        # AEs live under results.adverse_events (StructuredAE list).
        results = data.get("results") or {}
        for ae in results.get("adverse_events", []) or []:
            if not isinstance(ae, dict):
                continue
            if not ae.get("serious"):
                continue
            term = (ae.get("term") or "").strip()
            if not term:
                continue
            cached = cache.get(term)
            if not cached or not cached.get("preferred_term"):
                continue
            node_id = ae_node_id(cached["preferred_term"])
            if node_id not in g.nodes:
                continue
            if not g.nodes[node_id].get("serious"):
                g.nodes[node_id]["serious"] = True
                stats["ae_nodes_marked_serious"] += 1
            stats["serious_flags_seen"] += 1
    return stats


def _backfill_ae_node_hierarchy(
    graph: GraphStore, hierarchy: MedDRAHierarchy,
) -> Counter:
    """Backfill round-28 hierarchy fields on every AdverseEventNode.

    Returns a Counter with these keys:
      - ``ae_nodes_seen``: total AE nodes inspected
      - ``ae_nodes_backfilled``: had at least one new field written
      - ``ae_nodes_unchanged``: already complete, no write
      - ``ae_nodes_no_soc_resolved``: hierarchy couldn't resolve a SOC
        (PT slug missing + LLM-emitted system_organ_class also missing
        or non-canonical)
    """
    stats: Counter = Counter()
    g = graph._graph  # noqa: SLF001
    for node_id in list(g.nodes):
        node = g.nodes[node_id]
        if not str(node_id).startswith("AE:"):
            continue
        stats["ae_nodes_seen"] += 1
        # SOC-tier AE nodes get created later in propagation; skip them.
        # Their hierarchy fields are populated at creation time.
        if str(node_id).startswith("AE:soc:"):
            continue
        fallback_soc = (node.get("system_organ_class") or "").strip()
        parents = hierarchy.parents_for_pt(
            node_id, fallback_soc_name=fallback_soc,
        )
        wrote_any = False
        for field in _AE_HIERARCHY_FIELDS:
            value = parents.get(field, "")
            if field == "soc_name" and not value and fallback_soc:
                # Preserve the free-text SOC even when the hierarchy
                # can't canonicalize it — the renderer still needs
                # something to show.
                value = fallback_soc
            if value and not node.get(field):
                node[field] = value
                wrote_any = True
        if wrote_any:
            stats["ae_nodes_backfilled"] += 1
        else:
            stats["ae_nodes_unchanged"] += 1
        if not parents.get("soc_id"):
            stats["ae_nodes_no_soc_resolved"] += 1
    return stats


def _rebuild_soc_propagation(graph: GraphStore) -> Counter:
    """Re-run SOC-tier propagation for every (compound, AE) causes_ae pair.

    Returns counters for compound-AE pairs visited and SOC-tier edges
    newly produced.
    """
    stats: Counter = Counter()
    pairs: list[tuple[str, str]] = []
    g = graph._graph  # noqa: SLF001
    for src, tgt, key in g.edges(keys=True):
        if key != EdgeType.CAUSES_AE.value:
            continue
        pairs.append((src, tgt))
    stats["causes_ae_pairs"] = len(pairs)

    for compound_id, ae_id in pairs:
        updates = propagate_to_target_associated_ae(
            graph, compound_id, ae_id, roll_up_tier="soc",
        )
        for u in updates:
            if u.ae_id.startswith("AE:soc:"):
                stats["soc_tier_edges_emitted"] += 1
            else:
                stats["pt_tier_edges_emitted"] += 1
    return stats


def _backfill_mechanism_direction(graph: GraphStore) -> Counter:
    """Round-28: populate MechanismNode.direction on any node missing it.

    New builds set ``direction`` at MechanismNode creation time. Existing
    snapshots loaded with the new model schema have ``direction=None``
    by default; this fills the field via the canonical lookup keyed on
    MechanismCategory.
    """
    stats: Counter = Counter()
    g = graph._graph  # noqa: SLF001
    for node_id in list(g.nodes):
        node = g.nodes[node_id]
        if not isinstance(node_id, str):
            continue
        # MechanismNodes have mechanism_type — easier than passing the
        # type name through here.
        if "mechanism_type" not in node:
            continue
        stats["mechanism_nodes_seen"] += 1
        if node.get("direction"):
            continue
        try:
            cat = MechanismCategory(node_id)
        except ValueError:
            # Non-canonical id — skip rather than misclassify.
            continue
        direction = _category_to_direction(cat)
        if direction:
            node["direction"] = direction
            stats["mechanism_nodes_backfilled"] += 1
    return stats


def migrate_snapshot(in_path: Path, out_path: Path) -> Counter:
    graph = GraphStore()
    graph.import_snapshot(str(in_path))
    hierarchy = MedDRAHierarchy.load_default()

    ae_stats = _backfill_ae_node_hierarchy(graph, hierarchy)
    serious_stats = _backfill_ae_serious_from_extractions(graph)
    mech_stats = _backfill_mechanism_direction(graph)
    prop_stats = _rebuild_soc_propagation(graph)

    graph.export_snapshot(str(out_path))

    stats = Counter()
    stats.update(ae_stats)
    stats.update(serious_stats)
    stats.update(mech_stats)
    stats.update(prop_stats)
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--in", dest="in_path", type=Path, required=True,
        help="Pre-round-28 snapshot JSON to read.",
    )
    parser.add_argument(
        "--out", dest="out_path", type=Path, required=True,
        help="Migrated round-28 snapshot JSON to write.",
    )
    args = parser.parse_args()

    if not args.in_path.exists():
        print(f"missing input: {args.in_path}")
        return 1
    stats = migrate_snapshot(args.in_path, args.out_path)
    print(f"Migrated {args.in_path} → {args.out_path}\n")
    print("AE-node backfill:")
    for key in (
        "ae_nodes_seen", "ae_nodes_backfilled",
        "ae_nodes_unchanged", "ae_nodes_no_soc_resolved",
    ):
        print(f"  {key}: {stats.get(key, 0)}")
    print("\nRound-29 `serious` backfill (from extraction JSONs):")
    for key in (
        "extraction_files_seen", "serious_flags_seen",
        "ae_nodes_marked_serious",
    ):
        print(f"  {key}: {stats.get(key, 0)}")
    print("\nMechanism direction backfill:")
    for key in ("mechanism_nodes_seen", "mechanism_nodes_backfilled"):
        print(f"  {key}: {stats.get(key, 0)}")
    print("\nSOC-tier propagation:")
    for key in (
        "causes_ae_pairs", "pt_tier_edges_emitted", "soc_tier_edges_emitted",
    ):
        print(f"  {key}: {stats.get(key, 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
