"""Bottom-up (chains-first) graph build — the production Stage-3 builder.

Inverts the top-down ``populate.py`` (``PopulationPipeline.populate_trials``):
instead of resolving every trial's nodes into a SHARED store (sharing implicitly
by canonical-id match at ``add_node`` time), this builds each trial's subgraph in
ISOLATION (trial-scoped ``{id}#{nct}`` ids), then reassembles the population with
an explicit, re-runnable ``node_merge.assemble`` projection. Sharing becomes a
decision the merge makes, not a build-time commitment — so ingestion is append-
only and any merge (id OR geometric tier) is re-tunable without a rebuild.

    Phase 1  per-trial resolve + build (isolated)   -> list[trial-scoped GraphStore]
    Phase 2  union + node_merge.assemble (projection) -> merged GraphStore

======================= STATUS: WORKING BUILD MODE (WIP) =======================
Runs END-TO-END via ``build_graph --bottom-up``: on n=10 it produces an annotated
graph with 100% chain coverage (8/8 trials full) through the full
build->extract->classify->attribute pipeline. Validated faithful to top-down on
n=10:
  - chain concepts 61 == 61 (0 missing, 0 splits);
  - belief coverage 205/258 vs top-down 203/257 (populate stage, like-for-like);
  - same keep/drop decisions (8/10 — the 2 drops are non-therapeutic, a PET tracer
    + a no-drug-arm study, which top-down also drops);
  - edge beliefs preserved through namespace+merge (``_namespace_graph``) and ids
    canonicalized post-merge (``_canonicalize_ids``) so the graph is usable.

REMAINING before it can REPLACE populate.py: (a) the n=52 faithfulness audit (scale
the n=10 check); (b) a holdout P(success) parity check vs the top-down graph;
(c) the populate.py -> populate_topdown.py rename. Leave populate.py untouched
until then (the de-risking plan: new file now, rename later).
"""

from __future__ import annotations

import logging
from pathlib import Path
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


def _prune_orphan_biology(g: GraphStore) -> int:
    """Remove BiologyNodes no chain references — ghost pathway nodes left by
    dropped/extra trials whose chains never survived into a trial_subgraph (e.g.
    target-gene→Reactome biology created during populate for a trial later
    dropped). Keeps the biology layer to what the surviving chains actually use."""
    referenced = {
        ch.biology_id for ts in g.trial_subgraphs.values()
        for ch in ts.chains if getattr(ch, "biology_id", None)
    }
    orphans = [n["id"] for n in g.get_nodes_by_type("BiologyNode")
               if n["id"] not in referenced]
    for nid in orphans:
        g._graph.remove_node(nid)  # noqa: SLF001
    return len(orphans)


async def build_bottomup(
    trials: list[Any],
    anthropic_client: Any,
    *,
    condition: str = "cancer",
    annotations_dir: str | None = None,
    merge_config: MergeConfig | None = None,
    embed_fn: Callable[[str], list[float]] | None = None,
    premerge_dump_path: str | None = None,
    base_graph: GraphStore | None = None,
) -> GraphStore:
    """Phase 1 (per-trial isolated resolve+build) -> Phase 2 (union + assemble).

    Reuses ``PopulationPipeline`` per trial so all resolution logic
    (OT/ChEMBL/Reactome/GO/LLM mechanism + endpoint) is shared with top-down,
    then namespaces each trial's chains to ``{id}#{nct}`` and merges. Returns the
    assembled chains-first graph.

    **Incremental append** (``base_graph`` given — the 200K seam): start from the
    already-merged base graph instead of an empty store, build Phase 1 ONLY for the
    passed ``trials`` (the new ones), union them in, and re-run the Phase-2 merge
    **restricted to new-involving pairs** (``new_ids``) — so existing↔existing nodes
    aren't re-scored and the base isn't reprocessed. The new trials' structural
    nodes merge into the base by exact id/chembl; their biology/mechanism merge by
    the geometric tier against the base.
    """
    from src.graph.populate import (
        PopulationPipeline,
        deorphan_nonchain_endpoints,
        fan_biology_drives_to_coconditions,
    )

    incremental = base_graph is not None
    merged = base_graph if incremental else GraphStore()
    new_ids: set[str] = set()
    endpoint_deorphans = 0
    coconditions = 0
    for trial in trials:
        # Phase 1: resolve + build THIS trial alone into its own store.
        g_t = GraphStore()
        await PopulationPipeline(g_t, anthropic_client=anthropic_client).populate_trials(
            condition=condition, trials=[trial], annotations_dir=annotations_dir,
        )
        # Per-trial completeness passes — MUST run here (pre-merge), where the
        # trial's FULL endpoint + condition membership is intact; the merge
        # collapses it to chain-only and these can't be reconstructed after.
        endpoint_deorphans += deorphan_nonchain_endpoints(g_t)
        coconditions += fan_biology_drives_to_coconditions(g_t)
        # Namespace its full graph to {id}#{nct} (preserving edge beliefs) so the
        # union can't collide across trials and the merge has beliefs to fold.
        scoped = _namespace_graph(g_t, getattr(trial, "nct_id", ""))
        _union_into(merged, scoped)
        new_ids |= set(scoped._graph.nodes)  # noqa: SLF001 — the just-added trial-scoped ids
        logger.info("bottom-up Phase 1: built %s (%d scoped nodes)",
                    getattr(trial, "nct_id", "?"), scoped._graph.number_of_nodes())  # noqa: SLF001

    # Phase 2: the re-runnable projection. Tier-1 (id) for every type, PLUS the
    # Tier-3 BioLORD geometric merge gated to BiologyNode only. Biology takes its
    # identity from the extracted process description (populate.py
    # _populate_trial_biology), so two trials' biology pools when the
    # descriptions are semantically close even if their ontology ids differ or
    # one didn't resolve — this is what recovers the cross-trial evidence the
    # old per-trial slug fragmented. BioLORD is scoped OUT of mechanism/target
    # (it over-merges siblings there; SapBERT is the right tier — Phase C).
    if premerge_dump_path:
        # Additive: dump the Phase-1 union (PRE-assemble) so the chain
        # visualizer's before-block is faithful — each chain still references its
        # OWN per-trial node instances; the merge destroys that mapping, so it
        # can't be reconstructed afterwards. No effect on the build itself.
        # Public snapshot (ids/names/descriptions/chains) — enough for the viz
        # before-block; private export would trip the open-methods boundary guard
        # (it must live under the private root, not data/exports/).
        merged.export_snapshot(premerge_dump_path)
        logger.info("bottom-up: pre-merge snapshot -> %s", premerge_dump_path)

    # Merge all canonical-id node types — incl. AdverseEvent/Biomarker, shared
    # across trials (one MedDRA PT = one node) but absent from the chain-only
    # DEFAULT_NODE_TYPES. TrialNode is deliberately excluded (each is unique).
    cfg = merge_config or MergeConfig(
        node_types=DEFAULT_NODE_TYPES + ("AdverseEventNode", "BiomarkerNode"),
        enable_id=True, enable_name_id=False,
        # Merge-tier redesign (T4): the embedding model matches what the node IS.
        # BioLORD (definition/DESCRIPTION embedder) for the description nodes —
        # Biology (process phrases). SapBERT (biomedical ENTITY-LINKER / UMLS
        # synonym normalization) for the clinical ENTITY nodes — Indication,
        # Population, Endpoint (standardizes "NSCLC"≡"non-small-cell lung cancer"
        # while keeping breast≠ovarian, which BioLORD would conflate). VERIFIED
        # working + safe at threshold 0.80 by scripts/instrument_entity_merge.py
        # (T5): 0 synonym misses, siblings correctly separated.
        #
        # AdverseEventNode is DELIBERATELY NOT in the SapBERT tier (T5 verdict).
        # AE nodes are created during attribution (attributor._main), AFTER this
        # merge pass, so the tier was INERT on fresh builds — and a re-merge
        # (incremental append / assemble) WOULD then over-merge: SapBERT name
        # cosine can't separate true AE synonyms ("Cardiac"/"Myocardial ischaemia"
        # 0.94, SHOULD merge) from clinically-distinct siblings ("Alanine"/
        # "Aspartate aminotransferase increased" 0.90; "Lymphopenia"/"Leukopenia"
        # 0.89, must NOT) at any single threshold. AE canonicalization is MedDRA's
        # job (ae_node_id PT normalization); genuine synonym misses belong in a
        # MedDRA synonym table, not geometric name-merge.
        #
        # MechanismNode is INTENTIONALLY in NEITHER geometric tier — it merges by
        # ID only (Option 2). A mechanism node IS a curated Reactome pathway the
        # target gene drives (id = R-HSA / GO stable_id), fanned out one chain per
        # pathway in populate._populate_trial_mechanisms. Shared pathways across
        # targets (EGFR+BRAF+MEK → the SAME RAF/MAP-cascade stable_id) collapse
        # DETERMINISTICALLY at the Tier-1 id merge — that shared edge is the
        # cross-trial credit-assignment substrate. SapBERT-on-pathway-NAME
        # over-merged 65-72% (T3) and BioLORD-on-name has the same failure mode;
        # the id IS the canonical key, so no embedding merge is needed or wanted.
        enable_sapbert=True,
        sapbert_node_types=(
            "IndicationNode", "PopulationNode", "EndpointNode",
        ),
        # BiologyNode merges by ID ONLY — its curated GO-BP id (bio:GO:<acc>),
        # exactly the MechanismNode rationale: the ontology id IS the canonical
        # key, so no geometric embedding merge is needed. Biology is GO-keyed at
        # populate (_biology_id_from_description), so the Tier-1 id merge pools it
        # cheaply; the former BioLORD-on-DESCRIPTION tier ground through ~6k
        # content-hash biology nodes (O(n^2)) and was the build's slow step.
        enable_biolord=False,
    )
    # Incremental: restrict the O(n²) geometric tiers to pairs touching a just-added
    # node (existing↔existing are already merged) — O(new × total), not O(total²).
    #
    # Wire the BATCH encoders so each geometric tier embeds ALL its node
    # descriptions in one ``model.encode(list)`` + one cache round-trip, instead of
    # one model call per node. The per-node-only path here made the n=500 bottom-up
    # merge take HOURS (the standalone ``assemble_v2`` already batched via
    # ``embed_texts``; this DEFAULT build path silently fell back to per-node).
    from src.graph.biolord_embeddings import embed_texts as _biolord_batch
    from src.graph.sapbert_embeddings import embed_compound_names as _sapbert_batch
    # SapBERT (sentence-transformers) is an optional extra. When it isn't
    # installed, skip the SapBERT entity-linking merge tier and fall back to the
    # id/name_id merges — the documented graceful degradation (see
    # SapBertUnavailable) — instead of raising out of the whole build.
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        from dataclasses import replace
        logger.warning(
            "sentence-transformers not installed; skipping the SapBERT merge "
            "tier and falling back to id/name_id merges (install the `sapbert` "
            "extra to enable name canonicalization)."
        )
        cfg = replace(cfg, enable_sapbert=False)
        _sapbert_batch = None
    report = assemble(merged, cfg, embed_fn=embed_fn,
                      batch_embed_fn=_biolord_batch,
                      batch_sapbert_embed_fn=_sapbert_batch,
                      new_ids=new_ids if incremental else None)
    # Prune orphan biology BEFORE canonicalizing ids. Option 2 puts MechanismNode
    # ids in the R-HSA / GO namespace that the BiologyNode Reactome FALLBACK also
    # uses, so a transient orphan biology pathway node can occupy the canonical id
    # and block the mechanism node's scope-strip in ``_canonicalize_ids`` (its
    # ``oid not in g._graph`` guard) — the mechanism then stays ``R-HSA-…#NCT``
    # forever, defeating the deterministic cross-trial id-merge. Pruning is a pure
    # chain-reference check (scoping-consistent, canonicalize-independent), so
    # running it first frees those ids with no other effect.
    pruned_bio = _prune_orphan_biology(merged)
    renamed = _canonicalize_ids(merged)
    logger.info("bottom-up Phase 2: assembled -> %d nodes (by_type=%s, canonicalized %d ids, "
                "pruned %d orphan biology%s)",
                merged._graph.number_of_nodes(), report.by_type, renamed, pruned_bio,  # noqa: SLF001
                f", incremental +{len(new_ids)} new scoped ids" if incremental else "")

    # Complete the backbone: seed the biology→endpoint REFLECTS_BIOLOGY edge for
    # every chain (post-merge, so it uses final node ids; pre-attribution, so the
    # attributor can condition it). The per-role populate methods build every
    # other backbone edge but never this one — see ensure_reflects_biology_edges.
    from src.graph.populate import (
        ensure_reflects_biology_edges,
        populate_chain_descriptions,
        prune_disconnected_noise,
        prune_orphan_targets,
    )
    rb_added = ensure_reflects_biology_edges(merged)
    if rb_added:
        logger.info("bottom-up Phase 2: seeded %d reflects_biology edges", rb_added)
    if endpoint_deorphans or coconditions:
        logger.info("bottom-up Phase 2: de-orphaned %d non-chain endpoints, "
                    "fanned %d co-condition biology_drives (pre-merge)",
                    endpoint_deorphans, coconditions)
    # Topology hygiene: drop gene-family over-resolution (TargetNodes with
    # affects-in but no mechanism, referenced by no chain) then any node left
    # fully disconnected. Run AFTER all edge-creation passes.
    pruned_t = prune_orphan_targets(merged)
    pruned_n = prune_disconnected_noise(merged)
    if pruned_t or pruned_n:
        logger.info("bottom-up Phase 2: pruned %d orphan targets, %d disconnected nodes",
                    pruned_t, pruned_n)
    # Stamp per-drug descriptions onto chains (graph-native edge-view provenance).
    if annotations_dir:
        cd = populate_chain_descriptions(merged, Path(annotations_dir))
        if cd:
            logger.info("bottom-up Phase 2: stamped descriptions on %d chains", cd)
    return merged
