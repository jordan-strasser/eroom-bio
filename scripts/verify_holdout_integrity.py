"""Round-26: verify a snapshot has TRUE holdouts.

A holdout NCT is "truly held out" if:
  1. its trial_subgraph IS present (so chain prediction works)
  2. its NCT id does NOT appear in any edge's evidence list
     (i.e., its outcomes never updated any edge belief)

The applied_attribution_trial_ids set is NOT a sufficient check — the
round-26 --exclude-from-attribution flag adds excluded NCTs to that
set as the idempotency mechanism. The deeper check is whether actual
EvidenceRecords carrying the NCT exist on any edge.

Run:
  .venv/bin/python scripts/verify_holdout_integrity.py \\
    --snapshot data/exports/multi_indication_30_annotated.json \\
    --holdouts NCT01844505,NCT01127633,NCT00112918,NCT00134264,NCT00970359
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.graph.models import EdgeBeliefState
from src.graph.store import GraphStore


def verify(snapshot_path: Path, holdouts: list[str]) -> tuple[bool, dict]:
    holdouts_set = set(holdouts)
    graph = GraphStore()
    graph.import_snapshot(str(snapshot_path))

    stats: dict = {
        "snapshot": str(snapshot_path),
        "total_subgraphs": len(graph.trial_subgraphs),
        "total_edges": 0,
        "holdouts_with_subgraph": [],
        "holdouts_missing_subgraph": [],
        "edges_with_holdout_evidence": [],  # any leak — should be empty
        "applied_attribution_count": len(graph.applied_attribution_trial_ids),
        "holdouts_in_applied_set": sorted(
            holdouts_set & graph.applied_attribution_trial_ids
        ),
    }

    for nct in holdouts:
        if nct in graph.trial_subgraphs:
            stats["holdouts_with_subgraph"].append(nct)
        else:
            stats["holdouts_missing_subgraph"].append(nct)

    g = graph._graph  # noqa: SLF001
    for u, v, key, data in g.edges(keys=True, data=True):
        stats["total_edges"] += 1
        belief_data = data.get("belief")
        if not belief_data:
            continue
        try:
            belief = EdgeBeliefState.model_validate(belief_data)
        except Exception:
            continue
        for rec in belief.evidence:
            if rec.source_id in holdouts_set:
                stats["edges_with_holdout_evidence"].append({
                    "edge": f"{u} --{key}--> {v}",
                    "holdout_nct": rec.source_id,
                    "bucket": rec.support,
                    "source_type": rec.source_type.value,
                })

    leaks = len(stats["edges_with_holdout_evidence"])
    integrity_ok = (
        leaks == 0
        and len(stats["holdouts_with_subgraph"]) == len(holdouts_set)
    )
    return integrity_ok, stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument(
        "--holdouts", default="",
        help="Comma-separated NCT ids to verify as excluded.",
    )
    args = parser.parse_args()
    holdouts = [n.strip() for n in args.holdouts.split(",") if n.strip()]
    if not holdouts:
        print("--holdouts required")
        return 1
    ok, stats = verify(args.snapshot, holdouts)
    print(json.dumps(stats, indent=2))
    print()
    if ok:
        print("✓ HOLDOUT INTEGRITY OK")
        print(f"  - all {len(holdouts)} holdouts have subgraphs (chain "
              f"prediction can run on them)")
        print(f"  - 0 edges contain any holdout NCT in their evidence list")
        return 0
    else:
        print("✗ HOLDOUT INTEGRITY FAILED")
        if stats["holdouts_missing_subgraph"]:
            print(f"  - missing subgraphs: {stats['holdouts_missing_subgraph']}")
        if stats["edges_with_holdout_evidence"]:
            print(f"  - {len(stats['edges_with_holdout_evidence'])} edges "
                  f"contain holdout evidence — leak!")
            for entry in stats["edges_with_holdout_evidence"][:5]:
                print(f"    {entry}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
