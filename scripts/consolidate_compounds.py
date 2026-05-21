"""Consolidate duplicate compound nodes by stable_id (ChEMBL).

Round-21 Phase C step 2. After `backfill_sapbert.py` adds `stable_id`
and `embedding` to every compound node, this script finds groups of
compounds sharing the same `stable_id` and merges them into a single
canonical node.

For each duplicate group:
  1. Pick the canonical node — the one with the most total evidence
     across its edges (tiebreak: shortest id, alphabetical).
  2. For each duplicate compound:
     a. Rewrite outgoing edges (src=duplicate → src=canonical).
        Edges that now collide on (src, tgt, type) get their evidence
        records unioned and the Beta belief replayed from a (1, 1)
        prior so the merged belief reflects the combined evidence
        WITHOUT double-counting.
     b. Rewrite incoming edges (tgt=duplicate → tgt=canonical) the
        same way.
     c. Union aliases onto the canonical node.
     d. Update every TrialSubgraph that references the duplicate id —
        arm.compound_ids, arm.regimen_compound_id, chain.compound_id.
        Synthesized combo regimens whose constituents change get
        re-slugged.
     e. Delete the duplicate node.

Embedding-based merging (cosine ≥ 0.80 with chembl-non-conflict gate)
is deliberately NOT done here in this first cut — the chembl_id path
handles the round 19 smoke case (fluorouracil / 5_fluorouracil, both
CHEMBL185). Embedding-based consolidation can be a Phase C.5
followup.

Idempotent — running it twice is a no-op once duplicates are merged.

Usage:
  .venv/bin/python -m scripts.consolidate_compounds \\
      --in data/exports/multi_indication_50_embedded.json \\
      --out data/exports/multi_indication_50_consolidated.json
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console

from src.graph.models import (
    EdgeBeliefState,
    EdgeType,
    EvidenceRecord,
    TrialArm,
    TrialSubgraph,
)
from src.graph.store import GraphStore
from src.inference.beliefs import (
    SupportBucket,
    apply_virtual_evidence,
    effective_n_for_evidence,
    p_obs_for_bucket,
)

logger = logging.getLogger(__name__)
console = Console()


def _group_by_stable_id(graph: GraphStore) -> dict[str, list[str]]:
    """Return chembl_id → list of compound_ids. Only includes groups
    with >1 member (no need to enumerate singletons)."""
    by_chembl: dict[str, list[str]] = defaultdict(list)
    for node in graph.get_nodes_by_type("InterventionNode"):
        chembl = node.get("stable_id")
        if not chembl:
            continue
        by_chembl[chembl].append(node["id"])
    return {k: v for k, v in by_chembl.items() if len(v) > 1}


def _evidence_sum_for_node(graph: GraphStore, node_id: str) -> float:
    """Sum of evidence_strength across all edges incident to ``node_id``.
    Used to pick the canonical node in a duplicate group — the one
    with the most cumulative evidence wins.

    `evidence_strength` is a derived property (alpha + beta - 2.0), not
    a stored field, so we compute it from the serialized alpha/beta
    rather than looking it up by key.
    """
    def _strength(belief: dict) -> float:
        if not isinstance(belief, dict):
            return 0.0
        alpha = float(belief.get("alpha", 1.0) or 1.0)
        beta = float(belief.get("beta", 1.0) or 1.0)
        return max(0.0, alpha + beta - 2.0)

    total = 0.0
    for _, _, _, data in graph._graph.edges(node_id, data=True, keys=True):  # noqa: SLF001
        total += _strength(data.get("belief") or {})
    for _, _, _, data in graph._graph.in_edges(node_id, data=True, keys=True):  # noqa: SLF001
        total += _strength(data.get("belief") or {})
    return total


def _pick_canonical(graph: GraphStore, group: list[str]) -> str:
    """Pick the canonical id for a duplicate group: max evidence_strength
    sum, tiebreak shortest id then alphabetical."""
    scored = []
    for cid in group:
        ev = _evidence_sum_for_node(graph, cid)
        scored.append((-ev, len(cid), cid))
    scored.sort()
    return scored[0][2]


def _union_aliases(graph: GraphStore, canonical_id: str, duplicate_id: str) -> None:
    """Add the duplicate's name + id + aliases to the canonical's
    aliases list (deduped, order-preserving)."""
    canon_node = graph._graph.nodes[canonical_id]  # noqa: SLF001
    dup_node = graph._graph.nodes[duplicate_id]  # noqa: SLF001
    existing = list(canon_node.get("aliases") or [])
    seen = set(existing)
    candidates = [
        dup_node.get("name"),
        duplicate_id,
        *(dup_node.get("aliases") or []),
    ]
    for alias in candidates:
        if alias and alias not in seen and alias != canonical_id:
            existing.append(alias)
            seen.add(alias)
    canon_node["aliases"] = existing


def _replay_belief_from_evidence(
    evidence_records: list[EvidenceRecord],
) -> EdgeBeliefState:
    """Rebuild a Beta belief by replaying every evidence record through
    a fresh Beta(1, 1) prior. Used after unioning two edges' evidence
    so the merged belief reflects exactly the unique evidence — no
    double-counting from records that appeared on both edges."""
    belief = EdgeBeliefState(alpha=1.0, beta=1.0)
    for ev in evidence_records:
        try:
            bucket = SupportBucket(ev.support)
        except ValueError:
            continue
        n_eff = effective_n_for_evidence(ev.source_type, ev.quality_score)
        p_obs = p_obs_for_bucket(bucket)
        belief = apply_virtual_evidence(belief, n_eff=n_eff, p_obs=p_obs)
    belief.evidence = list(evidence_records)
    return belief


def _evidence_dedup_key(ev: EvidenceRecord) -> tuple:
    """Identify a unique evidence record for dedup during merge. Two
    records with the same source_id + source_type + support + notes
    are treated as duplicates (avoids re-counting evidence that
    appeared on both pre-merge edges)."""
    return (
        ev.source_id,
        ev.source_type.value if hasattr(ev.source_type, "value") else str(ev.source_type),
        ev.support,
        ev.notes or "",
    )


def _merge_edge(
    graph: GraphStore,
    src: str,
    tgt: str,
    edge_type_value: str,
    incoming_data: dict[str, Any],
) -> None:
    """Apply ``incoming_data`` to the (src, tgt, edge_type) edge,
    creating the edge if missing or unioning evidence + replaying
    Beta if it already exists. ``incoming_data`` is the dict view
    from the duplicate's edge (the source we're merging FROM)."""
    incoming_belief_raw = incoming_data.get("belief") or {}
    incoming_belief = (
        EdgeBeliefState.model_validate(incoming_belief_raw)
        if incoming_belief_raw else EdgeBeliefState()
    )
    incoming_metadata = incoming_data.get("metadata") or {}

    if not graph._graph.has_edge(src, tgt, key=edge_type_value):  # noqa: SLF001
        # New edge on canonical — copy through.
        graph._graph.add_edge(
            src, tgt, key=edge_type_value,
            edge_type=edge_type_value,
            belief=incoming_belief.model_dump(mode="json"),
            metadata=dict(incoming_metadata),
        )
        return

    # Both edges exist — union evidence + replay.
    existing_data = graph._graph.edges[src, tgt, edge_type_value]
    existing_belief = EdgeBeliefState.model_validate(
        existing_data.get("belief") or {}
    )
    seen_keys: set[tuple] = set()
    union_evidence: list[EvidenceRecord] = []
    for ev in existing_belief.evidence:
        k = _evidence_dedup_key(ev)
        if k not in seen_keys:
            seen_keys.add(k)
            union_evidence.append(ev)
    for ev in incoming_belief.evidence:
        k = _evidence_dedup_key(ev)
        if k not in seen_keys:
            seen_keys.add(k)
            union_evidence.append(ev)

    merged_belief = _replay_belief_from_evidence(union_evidence)
    existing_data["belief"] = merged_belief.model_dump(mode="json")
    # Merge metadata (existing wins on conflict, but extend lists).
    merged_md = dict(existing_data.get("metadata") or {})
    for k, v in incoming_metadata.items():
        if k not in merged_md:
            merged_md[k] = v
    existing_data["metadata"] = merged_md


def _rewrite_edges(
    graph: GraphStore, canonical_id: str, duplicate_id: str,
) -> dict[str, int]:
    """Move every edge incident to ``duplicate_id`` over to
    ``canonical_id``, merging beliefs on (src, tgt, type) collisions.
    Then delete the duplicate node (no longer has edges)."""
    moved_out = 0
    moved_in = 0
    collisions = 0

    # Snapshot edges first — we can't mutate while iterating.
    out_edges: list[tuple[str, str, str, dict]] = []
    for src, tgt, key, data in graph._graph.edges(  # noqa: SLF001
        duplicate_id, data=True, keys=True,
    ):
        out_edges.append((src, tgt, key, dict(data)))
    in_edges: list[tuple[str, str, str, dict]] = []
    for src, tgt, key, data in graph._graph.in_edges(  # noqa: SLF001
        duplicate_id, data=True, keys=True,
    ):
        in_edges.append((src, tgt, key, dict(data)))

    # Outgoing: src=duplicate → rewrite to src=canonical.
    for src, tgt, key, data in out_edges:
        new_src = canonical_id
        new_tgt = canonical_id if tgt == duplicate_id else tgt
        if new_src == new_tgt:
            # Self-loop on canonical — skip, would be meaningless.
            graph._graph.remove_edge(src, tgt, key=key)  # noqa: SLF001
            continue
        if graph._graph.has_edge(new_src, new_tgt, key=key):  # noqa: SLF001
            collisions += 1
        _merge_edge(graph, new_src, new_tgt, key, data)
        graph._graph.remove_edge(src, tgt, key=key)  # noqa: SLF001
        moved_out += 1

    # Incoming: tgt=duplicate → rewrite to tgt=canonical (and src might
    # also be the duplicate, in which case the self-loop on canonical
    # gets skipped above — out_edges captures it).
    for src, tgt, key, data in in_edges:
        if src == duplicate_id:
            # Already handled in the outgoing pass.
            continue
        new_tgt = canonical_id
        if src == new_tgt:
            graph._graph.remove_edge(src, tgt, key=key)  # noqa: SLF001
            continue
        if graph._graph.has_edge(src, new_tgt, key=key):  # noqa: SLF001
            collisions += 1
        _merge_edge(graph, src, new_tgt, key, data)
        graph._graph.remove_edge(src, tgt, key=key)  # noqa: SLF001
        moved_in += 1

    return {
        "moved_out": moved_out,
        "moved_in": moved_in,
        "collisions": collisions,
    }


def _update_trial_subgraphs(
    graph: GraphStore, canonical_id: str, duplicate_id: str,
) -> int:
    """Update every TrialSubgraph that references ``duplicate_id`` to
    reference ``canonical_id`` instead. Touches arm.compound_ids,
    arm.regimen_compound_id, and chain.compound_id."""
    updated = 0
    for trial_id, ts in list(graph.trial_subgraphs.items()):
        changed = False
        new_arms: list[TrialArm] = []
        for arm in ts.arms:
            new_compound_ids: list[str] = []
            for cid in arm.compound_ids:
                if cid == duplicate_id:
                    new_compound_ids.append(canonical_id)
                    changed = True
                else:
                    new_compound_ids.append(cid)
            # Dedup while preserving order — a merge can collapse two
            # entries to one.
            seen: set[str] = set()
            deduped: list[str] = []
            for cid in new_compound_ids:
                if cid not in seen:
                    seen.add(cid)
                    deduped.append(cid)

            new_regimen = arm.regimen_compound_id
            if new_regimen == duplicate_id:
                new_regimen = canonical_id
                changed = True
            elif "+" in new_regimen:
                # Synthesized combo regimen — re-derive from the
                # (possibly updated) constituent list.
                if any(cid == duplicate_id for cid in arm.compound_ids):
                    new_regimen = "+".join(sorted(deduped)) if len(deduped) > 1 else deduped[0]
                    changed = True

            new_arms.append(arm.model_copy(update={
                "compound_ids": deduped,
                "regimen_compound_id": new_regimen,
            }))

        new_chains = []
        for chain in ts.chains:
            if chain.compound_id == duplicate_id:
                new_chains.append(chain.model_copy(update={
                    "compound_id": canonical_id,
                }))
                changed = True
            elif "+" in chain.compound_id and duplicate_id in chain.compound_id.split("+"):
                # combo regimen contains the duplicate
                parts = [
                    canonical_id if p == duplicate_id else p
                    for p in chain.compound_id.split("+")
                ]
                # Dedup
                seen2: set[str] = set()
                deduped2: list[str] = []
                for p in parts:
                    if p not in seen2:
                        seen2.add(p)
                        deduped2.append(p)
                new_compound_id = (
                    "+".join(sorted(deduped2)) if len(deduped2) > 1 else deduped2[0]
                )
                new_chains.append(chain.model_copy(update={
                    "compound_id": new_compound_id,
                }))
                changed = True
            else:
                new_chains.append(chain)

        if changed:
            graph.trial_subgraphs[trial_id] = ts.model_copy(update={
                "arms": new_arms,
                "chains": new_chains,
            })
            updated += 1
    return updated


def consolidate_by_chembl(graph: GraphStore) -> dict[str, Any]:
    """Find duplicate-by-chembl_id groups and merge each onto a single
    canonical node. Returns a summary dict."""
    groups = _group_by_stable_id(graph)
    summary: dict[str, Any] = {
        "duplicate_groups": len(groups),
        "compounds_consolidated": 0,
        "edges_moved": 0,
        "edge_collisions_resolved": 0,
        "trial_subgraphs_updated": 0,
        "details": [],
    }
    for chembl, group in sorted(groups.items()):
        canonical = _pick_canonical(graph, group)
        duplicates = [c for c in group if c != canonical]
        group_summary = {
            "chembl_id": chembl,
            "canonical": canonical,
            "duplicates": duplicates,
        }
        for dup in duplicates:
            _union_aliases(graph, canonical, dup)
            edge_stats = _rewrite_edges(graph, canonical, dup)
            ts_updated = _update_trial_subgraphs(graph, canonical, dup)
            graph._graph.remove_node(dup)  # noqa: SLF001
            summary["compounds_consolidated"] += 1
            summary["edges_moved"] += (
                edge_stats["moved_out"] + edge_stats["moved_in"]
            )
            summary["edge_collisions_resolved"] += edge_stats["collisions"]
            summary["trial_subgraphs_updated"] += ts_updated
        summary["details"].append(group_summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Round-21 Phase C: consolidate duplicate compounds "
                    "by stable_id (ChEMBL).",
    )
    parser.add_argument("--in", dest="in_path", required=True, type=Path)
    parser.add_argument("--out", dest="out_path", required=True, type=Path)
    args = parser.parse_args()

    graph = GraphStore()
    console.print(f"[bold]Loading snapshot:[/bold] {args.in_path}")
    graph.import_snapshot(str(args.in_path))
    stats_before = graph.stats()
    console.print(
        f"  {stats_before['node_count']} nodes, "
        f"{stats_before['edge_count']} edges, "
        f"{stats_before['node_types'].get('InterventionNode', 0)} compounds"
    )

    console.rule("[bold]Consolidating by stable_id (ChEMBL)")
    summary = consolidate_by_chembl(graph)

    if summary["duplicate_groups"] == 0:
        console.print("[green]No duplicate groups found — nothing to consolidate.[/green]")
    else:
        console.print(
            f"[yellow]Found {summary['duplicate_groups']} duplicate group(s):[/yellow]"
        )
        for d in summary["details"]:
            console.print(
                f"  {d['chembl_id']}: canonical={d['canonical']}  "
                f"duplicates={d['duplicates']}"
            )
        console.print()
        console.print(
            f"  compounds_consolidated: {summary['compounds_consolidated']}"
        )
        console.print(f"  edges_moved: {summary['edges_moved']}")
        console.print(
            f"  edge_collisions_resolved: "
            f"{summary['edge_collisions_resolved']}"
        )
        console.print(
            f"  trial_subgraphs_updated: "
            f"{summary['trial_subgraphs_updated']}"
        )

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    graph.export_snapshot(str(args.out_path))
    stats_after = graph.stats()
    console.rule(f"[bold green]Wrote {args.out_path}")
    console.print(
        f"  {stats_after['node_count']} nodes "
        f"(was {stats_before['node_count']}), "
        f"{stats_after['edge_count']} edges "
        f"(was {stats_before['edge_count']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
