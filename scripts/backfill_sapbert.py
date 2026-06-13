"""Backfill stable_id + SapBERT embedding on InterventionNodes.

Round-21 Phase C step 1: walk an existing annotated snapshot, fill in
the new `stable_id` and `embedding` fields on every compound node.

  - stable_id (ChEMBL id) comes from the OT drug lookup, same path
    the populator uses at Step 1.5. Cached at
    `data/cache/ot_drug_targets.json` so reruns are cheap.
  - embedding (768-dim SapBERT vector of the canonical name) is
    computed via the lazy-loaded SentenceTransformer pipeline in
    `src/graph/sapbert_embeddings.py`. Cached at
    `data/cache/sapbert_embeddings.json`.

Idempotent — nodes that already have both fields populated are
skipped. Run before `consolidate_compounds.py` so the merge step has
the data it needs.

Usage:
  .venv/bin/python -m scripts.backfill_sapbert \\
      --in data/exports/multi_indication_50_annotated.json \\
      --out data/exports/multi_indication_50_embedded.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from rich.console import Console

from src.graph.sapbert_embeddings import (
    SapBertUnavailable,
    embed_compound_names,
)
from src.graph.store import GraphStore
from src.ingestion.opentargets import OpenTargetsClient

logger = logging.getLogger(__name__)
console = Console()

OT_CACHE_PATH = Path("data/cache/ot_drug_targets.json")


def _load_ot_cache() -> dict:
    if not OT_CACHE_PATH.exists():
        return {}
    try:
        return json.loads(OT_CACHE_PATH.read_text())
    except json.JSONDecodeError:
        logger.warning("OT cache corrupt; treating as empty")
        return {}


def _save_ot_cache(cache: dict) -> None:
    OT_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    OT_CACHE_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True))


async def backfill_stable_ids(graph: GraphStore) -> dict[str, int]:
    """For every InterventionNode without stable_id, look up chembl_id
    via Open Targets (cached). Returns counts {already, resolved,
    unresolved}."""
    ot = OpenTargetsClient()
    cache = _load_ot_cache()

    already = resolved = unresolved = 0
    compounds = graph.get_nodes_by_type("InterventionNode")
    for node in compounds:
        node_id = node["id"]
        if node.get("stable_id"):
            already += 1
            continue
        # Try the canonical name first, then the slug-id, then any
        # alias in order.
        candidates = [node.get("name"), node_id] + list(
            node.get("aliases") or []
        )
        chembl_id: str | None = None
        for cand in candidates:
            if not cand:
                continue
            cache_key = _normalize_for_ot_cache(cand)
            if cache_key in cache:
                entry = cache[cache_key]
                if isinstance(entry, dict):
                    chembl_id = entry.get("chembl_id")
                    if chembl_id:
                        break
                continue
            try:
                entry = await ot.get_drug_with_targets(cand)
            except KeyError:
                cache[cache_key] = {"chembl_id": None, "targets": []}
                continue
            except Exception as exc:
                logger.debug("OT lookup failed for %r: %s", cand, exc)
                continue
            cache[cache_key] = entry
            chembl_id = entry.get("chembl_id")
            if chembl_id:
                break
        if chembl_id:
            graph._graph.nodes[node_id]["stable_id"] = chembl_id  # noqa: SLF001
            resolved += 1
        else:
            unresolved += 1
    _save_ot_cache(cache)
    return {"already": already, "resolved": resolved, "unresolved": unresolved}


def backfill_embeddings(graph: GraphStore) -> dict[str, int]:
    """For every InterventionNode without an embedding, compute one
    via SapBERT (batched). Returns counts {already, embedded, errors}."""
    compounds = graph.get_nodes_by_type("InterventionNode")
    already = 0
    to_embed: list[tuple[str, str]] = []  # (node_id, name)
    for node in compounds:
        if node.get("embedding"):
            already += 1
            continue
        name = node.get("name")
        if not name:
            continue
        to_embed.append((node["id"], name))

    if not to_embed:
        return {"already": already, "embedded": 0, "errors": 0}

    names = [n for _, n in to_embed]
    try:
        vectors = embed_compound_names(names)
    except SapBertUnavailable as exc:
        logger.error("SapBERT unavailable: %s", exc)
        return {"already": already, "embedded": 0, "errors": len(to_embed)}

    for (node_id, _), vec in zip(to_embed, vectors):
        graph._graph.nodes[node_id]["embedding"] = vec  # noqa: SLF001
    return {"already": already, "embedded": len(to_embed), "errors": 0}


def _normalize_for_ot_cache(name: str) -> str:
    """Mirror the populator's OT cache key convention."""
    return " ".join((name or "").lower().split())


async def main_async(in_path: Path, out_path: Path) -> int:
    graph = GraphStore()
    console.print(f"[bold]Loading snapshot:[/bold] {in_path}")
    graph.import_snapshot(str(in_path))
    stats_before = graph.stats()
    n_compounds = stats_before["node_types"].get("InterventionNode", 0)
    console.print(
        f"  {stats_before['node_count']} nodes, "
        f"{stats_before['edge_count']} edges, "
        f"{n_compounds} compounds, "
        f"{len(graph.trial_subgraphs)} trial subgraphs"
    )

    console.rule("[bold]Step 1: stable_id (ChEMBL) backfill")
    s1 = await backfill_stable_ids(graph)
    console.print(
        f"  already_set={s1['already']}  resolved={s1['resolved']}  "
        f"unresolved={s1['unresolved']}"
    )

    console.rule("[bold]Step 2: SapBERT embedding backfill")
    s2 = backfill_embeddings(graph)
    if s2["errors"]:
        console.print(
            f"  [red]SapBERT failed for {s2['errors']} compounds[/red] — "
            f"install the sapbert extra (`pip install -e .[sapbert]`) "
            f"and retry. Snapshot will NOT be written."
        )
        return 1
    console.print(
        f"  already_embedded={s2['already']}  embedded_now={s2['embedded']}"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    graph.export_snapshot(str(out_path))
    console.rule(f"[bold green]Wrote {out_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Round-21 Phase C: backfill stable_id + SapBERT "
                    "embedding on InterventionNodes in an existing "
                    "annotated snapshot.",
    )
    parser.add_argument("--in", dest="in_path", required=True, type=Path)
    parser.add_argument("--out", dest="out_path", required=True, type=Path)
    args = parser.parse_args()
    return asyncio.run(main_async(args.in_path, args.out_path))


if __name__ == "__main__":
    raise SystemExit(main())
