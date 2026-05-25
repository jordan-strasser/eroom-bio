"""Generalized geometric + id node merge (Phase-2 assembly).

`biology_merge` collapses fragmented BiologyNodes via the Reactome↔GO crosswalk.
This generalizes that lossless union-and-replay to **every chain node type**, so
the chains-first build (independent per-trial nodes → assembled graph) has one
re-runnable assembly pass.

Merge runs in **tiers** per node type, highest-precision first, over a single
union-find:

  1. **id / ontology** — equal ``ontology_id`` (or, lacking one, equal node id),
     plus the biology Reactome↔GO crosswalk. Authoritative; subsumes the old
     exact-id dedup.
  2. **SapBERT ``name_id``** — equal canonical surface name. Entity-linking, so it
     separates true distinct entities (PD-1 vs CTLA4) that BioLORD cosine cannot.
  3. **geometry (tunable)** — BioLORD cosine ≥ ``MergeConfig.biolord_threshold``
     (default 0.85) and/or box ``relation == "merge"``.

**Reversibility by projection.** This is a pure function of (nodes, ``MergeConfig``)
— it records ``merged_from`` provenance and never invents evidence, so re-running
on the immutable Phase-1 chains at a tighter threshold *re-separates*
over-merged nodes. That's why a deliberately loose Tier-3 default is safe.

Lossless: evidence records are unioned (dedup by source+timestamp) and the Beta
replayed; ``belief_field`` anchors are concatenated (each carries its own (s,t),
so they stay separable); boxes bound-union. PUBLIC code; populated weights are
private per the open-methods boundary.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from pydantic import BaseModel

from src.graph.biology_merge import _replay_belief
from src.graph.models import EdgeType, EvidenceRecord, GraphEdge

if TYPE_CHECKING:
    from src.graph.box_embeddings import Box
    from src.graph.store import GraphStore

logger = logging.getLogger(__name__)

# Every node-id field on a CausalChain that a merge may need to rewrite.
CHAIN_ID_FIELDS = (
    "compound_id", "target_id", "mechanism_id", "biology_id",
    "indication_id", "endpoint_id", "subgroup_population_id",
)

DEFAULT_NODE_TYPES = (
    "InterventionNode", "TargetNode", "MechanismNode", "BiologyNode",
    "EndpointNode", "IndicationNode", "PopulationNode",
)


def _env_threshold(default: float) -> float:
    raw = os.environ.get("EROOM_MERGE_COSINE", "").strip()
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


@dataclass
class MergeConfig:
    """Tiered-merge knobs. ``biolord_threshold`` is overridable at runtime via
    ``EROOM_MERGE_COSINE``; Tier-3 stays loose-but-tunable by design."""

    node_types: tuple[str, ...] = DEFAULT_NODE_TYPES
    enable_id: bool = True            # Tier 1: ontology id / crosswalk
    enable_name_id: bool = True       # Tier 2a: exact SapBERT-canonical name_id (if populated)
    enable_sapbert: bool = False      # Tier 2b: SapBERT cosine on node NAME (entity-linking)
    enable_biolord: bool = True       # Tier 3: BioLORD cosine on DESCRIPTION (semantic)
    enable_box: bool = False          # Tier 3: box relation == "merge"
    biolord_threshold: float = field(default_factory=lambda: _env_threshold(0.85))
    # SapBERT separates true distinct entities (PD-1 0.67 vs synonym ~0.75), so a
    # high threshold is the precision tier BioLORD lacks. Tunable per the eval.
    sapbert_threshold: float = 0.80


class NodeMergeReport(BaseModel):
    """Summary of one assembly sweep."""

    nodes_before: int
    nodes_after: int
    classes_merged: int
    nodes_removed: list[str]
    winner_by_loser: dict[str, str]
    chains_updated: int
    by_type: dict[str, int] = {}  # node_type -> nodes_removed for that type


# ── merge keys ───────────────────────────────────────────────────────────────


def _ontology_key(node: dict) -> str:
    """The Tier-1 identity: explicit ``ontology_id`` if present (chains-first
    trial-scoped nodes carry it), else the node's own id (today's graph, where
    the id already *is* the ontology id)."""
    meta = node.get("metadata") or {}
    return meta.get("ontology_id") or node.get("ontology_id") or node["id"]


def _name_key(node: dict) -> str | None:
    meta = node.get("metadata") or {}
    return node.get("name_id") or meta.get("name_id")


def _node_text(node: dict) -> str:
    return (node.get("description") or node.get("name") or "").strip()


# ── union-find ───────────────────────────────────────────────────────────────


class _UF:
    def __init__(self, items: list[str]):
        self.p = {i: i for i in items}

    def find(self, x: str) -> str:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)  # deterministic root

    def classes(self) -> list[set[str]]:
        groups: dict[str, set[str]] = {}
        for i in self.p:
            groups.setdefault(self.find(i), set()).add(i)
        return [c for c in groups.values() if len(c) >= 2]


def _classes_for_type(
    nodes: dict[str, dict],
    config: MergeConfig,
    *,
    embed_fn: Callable[[str], list[float]] | None,
    sapbert_embed_fn: Callable[[str], list[float]] | None,
    boxes: dict[str, "Box"] | None,
    crosswalk: dict[str, str] | None,
) -> list[set[str]]:
    """Build merge classes for one node type by unioning all enabled tiers."""
    ids = list(nodes)
    uf = _UF(ids)

    if config.enable_id:
        by_key: dict[str, list[str]] = {}
        for nid in ids:
            by_key.setdefault(_ontology_key(nodes[nid]), []).append(nid)
        for group in by_key.values():
            for other in group[1:]:
                uf.union(group[0], other)
        if crosswalk:  # Reactome↔GO (biology): shared GO term ⇒ same class
            go_to_ids: dict[str, list[str]] = {}
            for nid in ids:
                go = crosswalk.get(nid)
                if go:
                    go_to_ids.setdefault(go, []).append(nid)
                if nid in crosswalk.values():  # the GO node itself is present
                    go_to_ids.setdefault(nid, []).append(nid)
            for group in go_to_ids.values():
                for other in group[1:]:
                    uf.union(group[0], other)

    if config.enable_name_id:
        by_name: dict[str, list[str]] = {}
        for nid in ids:
            key = _name_key(nodes[nid])
            if key:
                by_name.setdefault(key, []).append(nid)
        for group in by_name.values():
            for other in group[1:]:
                uf.union(group[0], other)

    if config.enable_sapbert and sapbert_embed_fn is not None:
        from src.graph.biolord_embeddings import cosine_similarity
        named = [(nid, (nodes[nid].get("name") or "").strip()) for nid in ids]
        named = [(i, n) for i, n in named if n]
        svecs = {i: sapbert_embed_fn(n) for i, n in named}
        skeys = [i for i, _ in named]
        for a in range(len(skeys)):
            for b in range(a + 1, len(skeys)):
                if cosine_similarity(svecs[skeys[a]], svecs[skeys[b]]) >= config.sapbert_threshold:
                    uf.union(skeys[a], skeys[b])

    if config.enable_biolord and embed_fn is not None:
        from src.graph.biolord_embeddings import cosine_similarity
        texts = [(nid, _node_text(nodes[nid])) for nid in ids]
        texts = [(i, t) for i, t in texts if t]
        vecs = {i: embed_fn(t) for i, t in texts}
        keys = [i for i, _ in texts]
        for a in range(len(keys)):
            for b in range(a + 1, len(keys)):
                if cosine_similarity(vecs[keys[a]], vecs[keys[b]]) >= config.biolord_threshold:
                    uf.union(keys[a], keys[b])

    if config.enable_box and boxes:
        from src.graph.box_embeddings import relation
        boxed = [i for i in ids if i in boxes]
        for a in range(len(boxed)):
            for b in range(a + 1, len(boxed)):
                if relation(boxes[boxed[a]], boxes[boxed[b]]) == "merge":
                    uf.union(boxed[a], boxed[b])

    return uf.classes()


# ── winner selection + lossless merge ────────────────────────────────────────


def _pick_winner(class_: set[str], graph: "GraphStore") -> str:
    """Highest ``metadata.primary_score`` wins; prefer a node that has an
    explicit ontology id; tie-break on sorted id for determinism."""
    def key(nid: str):
        try:
            node = graph.get_node(nid)
        except KeyError:
            return (1.0, 1, nid)
        meta = node.get("metadata") or {}
        has_ont = 0 if (meta.get("ontology_id") or node.get("ontology_id")) else 1
        return (-float(meta.get("primary_score") or 0.0), has_ont, nid)

    return min(class_, key=key)


def _merge_belief_data(existing: dict | None, incoming: dict | None) -> dict:
    """Union evidence (dedup by source+timestamp), replay the Beta, and
    concatenate ``belief_field`` anchors (each keeps its own (s,t) ⇒ separable),
    re-centering the field marginal on the replayed scalar."""
    def records(b: dict | None) -> list[EvidenceRecord]:
        if not b:
            return []
        return [EvidenceRecord.model_validate(r) for r in (b.get("evidence") or [])]

    combined: dict[tuple[str, str], EvidenceRecord] = {}
    for r in records(existing) + records(incoming):
        k = (r.source_id, r.timestamp.isoformat() if r.timestamp else "")
        combined.setdefault(k, r)
    deduped = list(combined.values())
    replayed = _replay_belief(deduped)
    replayed.evidence = deduped
    out = replayed.model_dump(mode="json")

    bf_e = (existing or {}).get("belief_field")
    bf_i = (incoming or {}).get("belief_field")
    if bf_e or bf_i:
        anchors = list((bf_e or {}).get("anchors", [])) + list((bf_i or {}).get("anchors", []))
        base = bf_e or bf_i or {}
        out["belief_field"] = {
            "bandwidth": base.get("bandwidth", 0.25),
            "marginal_alpha": replayed.alpha,
            "marginal_beta": replayed.beta,
            "fallback_strength": base.get("fallback_strength", 2.0),
            "anchors": anchors,
        }
    return out


def _redirect_edges(graph: "GraphStore", winner_id: str, loser_id: str) -> None:
    """Move every edge on ``loser_id`` to ``winner_id``; on collision, union
    belief data losslessly. Edge keys (= edge_type) are preserved."""
    g = graph._graph  # noqa: SLF001
    for u, _v, key, data in list(g.in_edges(loser_id, data=True, keys=True)):
        if g.has_edge(u, winner_id, key=key):
            ed = g.edges[u, winner_id, key]
            ed["belief"] = _merge_belief_data(ed.get("belief"), data.get("belief"))
        else:
            g.add_edge(u, winner_id, key=key, **data)
    for _u, v, key, data in list(g.out_edges(loser_id, data=True, keys=True)):
        if g.has_edge(winner_id, v, key=key):
            ed = g.edges[winner_id, v, key]
            ed["belief"] = _merge_belief_data(ed.get("belief"), data.get("belief"))
        else:
            g.add_edge(winner_id, v, key=key, **data)


def _merge_node_data(graph: "GraphStore", winner_id: str, loser_id: str) -> None:
    """Fold loser node fields into winner: union ``pathway_ids``, record
    ``merged_from`` provenance (reversibility), keep a description/name_id if the
    winner lacks one."""
    g = graph._graph  # noqa: SLF001
    w, l = g.nodes[winner_id], g.nodes[loser_id]

    seen = set(w.get("pathway_ids") or [])
    pids = list(w.get("pathway_ids") or [])
    for pid in (l.get("pathway_ids") or []):
        if pid not in seen:
            pids.append(pid)
            seen.add(pid)
    if pids:
        w["pathway_ids"] = pids

    meta = dict(w.get("metadata") or {})
    merged_from = set(meta.get("merged_from") or [])
    merged_from.add(loser_id)
    meta["merged_from"] = sorted(merged_from)
    w["metadata"] = meta

    for fld in ("description", "name_id"):
        if not (w.get(fld) or "").strip() and (l.get(fld) or "").strip():
            w[fld] = l[fld]


def _rewrite_chain_ids(graph: "GraphStore", remap: dict[str, str]) -> int:
    """Rewrite every CHAIN_ID_FIELD on every chain through ``remap`` (loser→winner).
    Returns the number of field rewrites."""
    if not remap:
        return 0
    rewrites = 0
    for trial_id, ts in list(graph.trial_subgraphs.items()):
        new_chains = []
        changed = False
        for c in ts.chains:
            updates = {}
            for fld in CHAIN_ID_FIELDS:
                cur = getattr(c, fld, None)
                if cur in remap:
                    updates[fld] = remap[cur]
                    rewrites += 1
            if updates:
                changed = True
                new_chains.append(c.model_copy(update=updates))
            else:
                new_chains.append(c)
        if changed:
            graph.trial_subgraphs[trial_id] = ts.model_copy(update={"chains": new_chains})
    return rewrites


def resolve_hierarchy(
    graph: "GraphStore",
    boxes: dict[str, "Box"],
    *,
    node_types: tuple[str, ...] = DEFAULT_NODE_TYPES,
    hi: float | None = None,
    lo: float | None = None,
    source: str = "box_geometry",
) -> dict[str, int]:
    """Emit ``SUBTYPE_OF`` (is-a) edges from box containment — the *hierarchy*
    half of "merge by geometry + hierarchy" that `assemble` (the *merge* half)
    doesn't cover.

    For same-type node pairs with fitted boxes, ``box_embeddings.relation``
    classifies ``parent_of`` (a ⊇ b) / ``child_of`` (b ⊇ a); we add the
    corresponding ``child --SUBTYPE_OF--> parent`` edge (specific→general, the
    direction the indication taxonomy already uses). ``merge`` is left to
    `assemble`; ``sibling``/``unrelated`` add nothing. Idempotent: existing or
    reverse-direction SUBTYPE_OF edges are skipped (no 2-cycles). Returns the
    count of edges added per node type.

    Run **after** `assemble` and box-fitting (boxes must reflect the merged
    nodes). This makes the box relation load-bearing instead of diagnostic.
    """
    from src.graph.box_embeddings import relation

    kw = {}
    if hi is not None:
        kw["hi"] = hi
    if lo is not None:
        kw["lo"] = lo

    added: dict[str, int] = {}
    g = graph._graph  # noqa: SLF001
    key = EdgeType.SUBTYPE_OF.value
    for nt in node_types:
        ids = [n["id"] for n in graph.get_nodes_by_type(nt) if n["id"] in boxes]
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                rel = relation(boxes[a], boxes[b], **kw)
                if rel == "parent_of":      # a ⊇ b  →  b is-a a
                    child, parent = b, a
                elif rel == "child_of":     # b ⊇ a  →  a is-a b
                    child, parent = a, b
                else:
                    continue
                if g.has_edge(child, parent, key=key) or g.has_edge(parent, child, key=key):
                    continue
                graph.add_edge(GraphEdge(
                    source_id=child, target_id=parent,
                    edge_type=EdgeType.SUBTYPE_OF, metadata={"source": source},
                ))
                added[nt] = added.get(nt, 0) + 1
    return added


def assemble(
    graph: "GraphStore",
    config: MergeConfig | None = None,
    *,
    embed_fn: Callable[[str], list[float]] | None = None,
    sapbert_embed_fn: Callable[[str], list[float]] | None = None,
    boxes: dict[str, "Box"] | None = None,
    crosswalk: dict[str, str] | None = None,
) -> NodeMergeReport:
    """Merge equivalent nodes per ``config`` across all configured types.

    Idempotent and re-runnable. ``embed_fn`` (BioLORD, description) and
    ``sapbert_embed_fn`` (SapBERT, name) default to their lazy loaders when the
    matching tier is enabled; inject stubs in tests. ``crosswalk`` is the
    Reactome↔GO map for biology (optional).
    """
    config = config or MergeConfig()
    if config.enable_biolord and embed_fn is None:
        from src.graph.biolord_embeddings import embed_text as embed_fn  # noqa
    if config.enable_sapbert and sapbert_embed_fn is None:
        from src.graph.sapbert_embeddings import embed_compound_name as sapbert_embed_fn  # noqa

    before = graph._graph.number_of_nodes()  # noqa: SLF001
    nodes_removed: list[str] = []
    winner_by_loser: dict[str, str] = {}
    by_type: dict[str, int] = {}
    classes_merged = 0
    remap: dict[str, str] = {}

    for node_type in config.node_types:
        nodes = {n["id"]: n for n in graph.get_nodes_by_type(node_type)}
        if len(nodes) < 2:
            continue
        classes = _classes_for_type(
            nodes, config, embed_fn=embed_fn, sapbert_embed_fn=sapbert_embed_fn,
            boxes=boxes, crosswalk=crosswalk,
        )
        for class_ in classes:
            classes_merged += 1
            winner = _pick_winner(class_, graph)
            for loser in sorted(class_ - {winner}):
                _merge_node_data(graph, winner, loser)
                _redirect_edges(graph, winner, loser)
                graph._graph.remove_node(loser)  # noqa: SLF001
                nodes_removed.append(loser)
                winner_by_loser[loser] = winner
                remap[loser] = winner
                by_type[node_type] = by_type.get(node_type, 0) + 1

    # Collapse transitive remaps (loser→winner where winner was itself merged).
    def resolve(x: str) -> str:
        while x in remap and remap[x] != x and remap[x] in remap:
            x = remap[x]
        return remap.get(x, x)

    final_remap = {k: resolve(k) for k in remap}
    chains_updated = _rewrite_chain_ids(graph, final_remap)

    return NodeMergeReport(
        nodes_before=before,
        nodes_after=before - len(nodes_removed),
        classes_merged=classes_merged,
        nodes_removed=nodes_removed,
        winner_by_loser=winner_by_loser,
        chains_updated=chains_updated,
        by_type=by_type,
    )
