"""Bottom-up (chains-first) graph build — the production Stage-3 builder.

Inverts the top-down ``populate.py`` (``PopulationPipeline.populate_oncology``):
instead of resolving every trial's nodes into a SHARED store (sharing implicitly
by canonical-id match at ``add_node`` time), this builds each trial's subgraph in
ISOLATION (trial-scoped ``{id}#{nct}`` ids), then reassembles the population with
an explicit, re-runnable ``node_merge.assemble`` projection. Sharing becomes a
decision the merge makes, not a build-time commitment — so ingestion is append-
only and any merge (id OR geometric tier) is re-tunable without a rebuild.

    Phase 1  per-trial resolve + build (isolated)   -> list[trial-scoped GraphStore]
    Phase 2  union + node_merge.assemble (projection) -> merged GraphStore

============================ STATUS: SCAFFOLD (WIP) ============================
- Phase 2 (assemble) is DONE + proven: node_merge reconstructs the n=52 top-down
  graph exactly (367 instances -> 226 concepts == top-down 226, 0 lost). See
  populate_groundup (the explode->merge faithfulness harness).
- Phase 1 here REUSES PopulationPipeline per-trial. The open risk being validated
  is cross-trial canonicalization: codename->INN sharing and one-IndicationNode-
  per-disease happen at RESOLVE time in top-down, but must move to MERGE time here
  (each trial resolves alone). The n=10 -> n=52 faithfulness audit (node/edge
  counts + per-edge beliefs vs the top-down build) is the bar before the
  populate.py -> populate_topdown.py rename.
- TODO before production: (a) carry edge beliefs/priors through the namespace +
  merge (explode_to_chains_first drops them — fine for the structural harness, not
  for a real build); (b) validate per-trial populate doesn't trip populate_oncology
  batch assumptions (diagnostic filter, codename resolution operate over a list);
  (c) wire build_graph --bottom-up; (d) the n=52 audit.

Leave populate.py untouched until this passes the audit (the user's de-risking
plan: new file now, rename later).
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from src.graph.node_merge import DEFAULT_NODE_TYPES, MergeConfig, assemble
from src.graph.populate_groundup import ROLE_ATTRS, _UNKNOWN
from src.graph.store import GraphStore

logger = logging.getLogger(__name__)


def _namespace_graph(g_t: GraphStore, nct: str) -> GraphStore:
    """Rename every node id to ``{id}#{nct}``, preserving edges (WITH their
    Beta beliefs / evidence), chains, and recording ``ontology_id`` (the original
    id) as the Tier-1 merge key.

    Unlike ``populate_groundup.explode_to_chains_first`` (structural-only — it
    rebuilds bare backbone edges for the faithfulness *comparison*), this keeps
    every edge's ``belief`` so the merged graph is attribution/prediction-ready —
    the merge then unions beliefs via ``node_merge._merge_belief_data``.
    """
    out = GraphStore()

    def scoped(nid: str) -> str:
        return f"{nid}#{nct}"

    for nid in g_t._graph.nodes:  # noqa: SLF001
        sid = scoped(nid)
        attrs = dict(g_t._graph.nodes[nid])  # noqa: SLF001
        out._graph.add_node(sid, **attrs)  # noqa: SLF001
        n = out._graph.nodes[sid]  # noqa: SLF001
        n["ontology_id"] = nid          # original (canonical) id = merge key
        n["from_trial"] = nct
        if "id" in n:
            n["id"] = sid
    for s, t, key, data in g_t._graph.edges(keys=True, data=True):  # noqa: SLF001
        out._graph.add_edge(scoped(s), scoped(t), key=key, **dict(data))  # noqa: SLF001
    for ts in g_t.trial_subgraphs.values():
        new_chains = []
        for ch in ts.chains:
            upd = {a: scoped(getattr(ch, a)) for a in ROLE_ATTRS
                   if getattr(ch, a, None) and getattr(ch, a) != _UNKNOWN}
            new_chains.append(ch.model_copy(update=upd))
        ppid = ts.parent_population_id
        out.set_trial_subgraph(ts.model_copy(update={
            "chains": new_chains,
            "parent_population_id": scoped(ppid) if ppid else ppid,
        }))
    return out


def _union_into(dst: GraphStore, src: GraphStore) -> None:
    """Copy all nodes, edges, and trial subgraphs from ``src`` into ``dst``.

    Safe because Phase-1 ids are trial-scoped (``{id}#{nct}``) — no cross-trial
    collisions until the merge deliberately collapses them.
    """
    for nid in src._graph.nodes:  # noqa: SLF001
        if nid not in dst._graph:  # noqa: SLF001
            attrs = dict(src._graph.nodes[nid])  # noqa: SLF001
            dst._graph.add_node(nid, **attrs)  # noqa: SLF001
    for s, t, key, data in src._graph.edges(keys=True, data=True):  # noqa: SLF001
        if not dst._graph.has_edge(s, t, key=key):  # noqa: SLF001
            dst._graph.add_edge(s, t, key=key, **dict(data))  # noqa: SLF001
    for nct, ts in src.trial_subgraphs.items():
        dst.set_trial_subgraph(ts)


def _canonicalize_ids(g: GraphStore) -> int:
    """Rename each surviving merged node from its winner-instance id
    (``{id}#{nct}``) back to its canonical ``ontology_id``, rewriting edges +
    chains. After the merge every ontology_id has exactly one survivor, so the
    mapping is injective — the result is a normal canonical-id graph the
    annotation/prediction pipeline can consume (OT/MedDRA lookups assume canonical
    ids). Returns the number of nodes renamed."""
    import networkx as nx

    mapping: dict[str, str] = {}
    for nid in list(g._graph.nodes):  # noqa: SLF001
        oid = g._graph.nodes[nid].get("ontology_id")  # noqa: SLF001
        if oid and oid != nid and oid not in g._graph and oid not in mapping.values():  # noqa: SLF001
            mapping[nid] = oid
    if not mapping:
        return 0
    nx.relabel_nodes(g._graph, mapping, copy=False)  # noqa: SLF001 — renames nodes + edges
    for nid in g._graph.nodes:  # noqa: SLF001
        if "id" in g._graph.nodes[nid]:  # noqa: SLF001
            g._graph.nodes[nid]["id"] = nid  # noqa: SLF001
    for ts in list(g.trial_subgraphs.values()):
        new_chains = []
        for ch in ts.chains:
            upd = {a: mapping[v] for a in ROLE_ATTRS
                   if (v := getattr(ch, a, None)) and v in mapping}
            new_chains.append(ch.model_copy(update=upd) if upd else ch)
        ppid = ts.parent_population_id
        g.set_trial_subgraph(ts.model_copy(update={
            "chains": new_chains,
            "parent_population_id": mapping.get(ppid, ppid),
        }))
    return len(mapping)


async def build_bottomup(
    trials: list[Any],
    anthropic_client: Any,
    *,
    condition: str = "cancer",
    merge_config: MergeConfig | None = None,
    embed_fn: Callable[[str], list[float]] | None = None,
) -> GraphStore:
    """Phase 1 (per-trial isolated resolve+build) -> Phase 2 (union + assemble).

    Reuses ``PopulationPipeline`` per trial so all resolution logic
    (OT/ChEMBL/Reactome/GO/LLM mechanism + endpoint) is shared with top-down,
    then namespaces each trial's chains to ``{id}#{nct}`` and merges. Returns the
    assembled chains-first graph.

    SCAFFOLD: structural only (no belief carry-through yet — see module TODO).
    """
    from src.graph.populate import PopulationPipeline

    merged = GraphStore()
    for trial in trials:
        # Phase 1: resolve + build THIS trial alone into its own store.
        g_t = GraphStore()
        await PopulationPipeline(g_t, anthropic_client=anthropic_client).populate_oncology(
            condition=condition, trials=[trial],
        )
        # Namespace its full graph to {id}#{nct} (preserving edge beliefs) so the
        # union can't collide across trials and the merge has beliefs to fold.
        scoped = _namespace_graph(g_t, getattr(trial, "nct_id", ""))
        _union_into(merged, scoped)
        logger.info("bottom-up Phase 1: built %s (%d scoped nodes)",
                    getattr(trial, "nct_id", "?"), scoped._graph.number_of_nodes())  # noqa: SLF001

    # Phase 2: the re-runnable projection. Tier-1 (id) by default; pass a config
    # with enable_biolord to also fold the geometric tier. Merge all canonical-id
    # node types — incl. AdverseEvent/Biomarker, which are shared across trials
    # (one MedDRA PT = one node) but absent from the chain-only DEFAULT_NODE_TYPES.
    # TrialNode is deliberately excluded (each trial is inherently unique).
    cfg = merge_config or MergeConfig(
        node_types=DEFAULT_NODE_TYPES + ("AdverseEventNode", "BiomarkerNode"),
        enable_id=True, enable_name_id=False, enable_sapbert=False, enable_biolord=False,
    )
    report = assemble(merged, cfg, embed_fn=embed_fn)
    renamed = _canonicalize_ids(merged)
    logger.info("bottom-up Phase 2: assembled -> %d nodes (by_type=%s, canonicalized %d ids)",
                merged._graph.number_of_nodes(), report.by_type, renamed)  # noqa: SLF001
    return merged
