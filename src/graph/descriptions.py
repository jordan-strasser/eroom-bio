"""Node description + name_id generation (the Q4 "every node gets a description
and a canonical name_id" piece, run inside the build).

Today only Mechanism/Biology/Population get rich descriptions (from extractions,
via ``attach_node_descriptions_from_extractions``). Target/Compound/Indication/
Endpoint have none — so their BioLORD embeddings/boxes/(s,t) fall back to the bare
name. This fills that gap: a succinct, precise NL ``description`` for every node
(Haiku, cached), plus a normalized ``name_id`` (the Tier-2a merge key; SapBERT
*cosine* on the name is the Tier-2b signal in ``node_merge``).

LLM-first for uniform quality matching the A.0 style; a DB-first source
(OpenTargets/HGNC/MeSH) is a drop-in refinement per type (see ``_DB_HINTS``).
Descriptions are cached in ``data/cache/node_descriptions.json`` so rebuilds are
cheap and reproducible.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import TYPE_CHECKING

from src.graph.populate import INFERENCE_MODEL, JSONCache

if TYPE_CHECKING:
    import anthropic

    from src.graph.store import GraphStore

# All chain node types. Mechanism/Biology/Population get rich descriptions from
# extractions first (attach_node_descriptions_from_extractions, skip-existing);
# this fills every remaining gap — incl. Target/Compound/Indication/Endpoint
# (0% before) and any DB-prior Mechanism/Biology/Population a chain never described.
DEFAULT_DESC_TYPES = (
    "InterventionNode", "TargetNode", "IndicationNode", "EndpointNode",
    "MechanismNode", "BiologyNode", "PopulationNode",
)
DESC_CACHE = Path("data/cache/node_descriptions.json")

# Per-type framing for the description prompt (keeps them on-layer + precise).
_TYPE_FRAMING = {
    "InterventionNode": "a therapeutic compound/intervention",
    "TargetNode": "a molecular target (gene/protein)",
    "IndicationNode": "a disease indication",
    "EndpointNode": "a clinical-trial endpoint",
    "MechanismNode": "a mechanism of action",
    "BiologyNode": "a biological process/pathway",
    "PopulationNode": "a patient population/subgroup",
}


def _node_context(node: dict) -> str:
    """Light, factual context to ground the description (no invention)."""
    bits = [f"name: {node.get('name')}", f"id: {node.get('id')}"]
    if node.get("gene_symbol"):
        bits.append(f"gene symbol: {node['gene_symbol']}")
    if node.get("aliases"):
        bits.append(f"aliases: {', '.join(node['aliases'][:4])}")
    return "; ".join(b for b in bits if b)


def _prompt(node_type: str, node: dict) -> str:
    framing = _TYPE_FRAMING.get(node_type, "a knowledge-graph node")
    return (
        f"Write ONE concise, precise sentence describing {framing} for a clinical "
        f"trial knowledge graph. Be specific and factual; no hedging, no "
        f"'this node represents'. Just the description.\n\n{_node_context(node)}"
    )


def normalize_name_id(name: str) -> str:
    """Canonical Tier-2a key: lowercase, collapse punctuation/space. Equal
    surface forms across nodes collapse; SapBERT cosine (node_merge Tier-2b)
    handles synonymy that this exact key misses."""
    s = (name or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return re.sub(r"\s+", " ", s)


def assign_name_ids(graph: "GraphStore", *, node_types: tuple[str, ...] | None = None) -> int:
    """Set a normalized ``name_id`` (top-level graph attr) on every node — the
    Tier-2a exact-match merge key. Idempotent. Returns count set."""
    sel = set(node_types) if node_types else None
    n = 0
    for nid in graph._graph.nodes:  # noqa: SLF001
        node = graph._graph.nodes[nid]  # noqa: SLF001
        if sel and node.get("node_type") not in sel:
            continue
        nm = normalize_name_id(node.get("name") or nid)
        if nm:
            node["name_id"] = nm
            n += 1
    return n


async def generate_node_descriptions(
    graph: "GraphStore",
    client: "anthropic.AsyncAnthropic",
    *,
    node_types: tuple[str, ...] = DEFAULT_DESC_TYPES,
    concurrency: int = 5,
    cache_path: Path = DESC_CACHE,
    overwrite: bool = False,
) -> int:
    """Fill ``description`` on nodes of ``node_types`` that lack one (Haiku,
    cached). Returns the number of descriptions set. Idempotent: a node that
    already has a non-empty description is skipped unless ``overwrite``."""
    from src.annotation.extractor import _call_messages_with_backoff  # noqa: SLF001

    cache = JSONCache(cache_path)
    sel = set(node_types)
    targets: list[tuple[str, str, dict]] = []
    for nid in graph._graph.nodes:  # noqa: SLF001
        node = graph._graph.nodes[nid]  # noqa: SLF001
        nt = node.get("node_type")
        if nt not in sel:
            continue
        if not overwrite and (node.get("description") or "").strip():
            continue
        targets.append((nid, nt, node))

    sem = asyncio.Semaphore(concurrency)
    set_count = 0

    async def _one(nid: str, nt: str, node: dict) -> None:
        nonlocal set_count
        key = f"{nt}:{nid}"
        desc = cache.get(key)
        if desc is None:
            async with sem:
                resp = await _call_messages_with_backoff(
                    client, model=INFERENCE_MODEL, max_tokens=120, temperature=0,
                    messages=[{"role": "user", "content": _prompt(nt, node)}],
                )
            desc = (resp.content[0].text or "").strip()
            cache.set(key, desc)
        if desc:
            node["description"] = desc
            set_count += 1

    await asyncio.gather(*(_one(nid, nt, node) for nid, nt, node in targets))
    return set_count
