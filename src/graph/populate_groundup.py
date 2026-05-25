"""Ground-up (chains-first) graph assembly — the Q2 bottom-up build, parallel to
the top-down ``PopulationPipeline.populate_oncology``.

Top-down builds nodes straight into a shared store, sharing by exact-id match at
``add_node`` time (overlap created implicitly). Ground-up inverts that: each
trial's chains are exploded into **independent trial-scoped node instances**
(``{ontology_id}#{nct}``), each carrying its ``ontology_id`` + a SapBERT
``name_id``, then the whole population is reassembled by the explicit tiered
merge (`node_merge.assemble`). No node is shared until merge decides it — so the
merge is a re-runnable projection, not a build-time commitment.

This v1 reuses the top-down build's already-resolved chains as the source of
per-trial node references (so it needs no re-resolution / API). It exists to
**compare** the bottom-up assembly against the top-down graph: does merge-by-id
faithfully reconstruct the shared graph (Tier-1 ≡ exact-id dedup), and what does
geometry/`name_id` consolidate that top-down's string match missed? A future v2
swaps the source for genuine per-trial resolution.

Beliefs/attribution are out of scope here (P(success) is downstream) — this
compares node + chain-backbone *structure*.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from src.graph.models import EdgeType, GraphEdge
from src.graph.node_merge import MergeConfig, assemble
from src.graph.store import GraphStore

# Backbone edges along a causal chain: (edge_type, source role-attr, target role-attr).
CHAIN_BACKBONE = [
    (EdgeType.AFFECTS, "compound_id", "target_id"),
    (EdgeType.MODULATES_VIA, "target_id", "mechanism_id"),
    (EdgeType.MECHANISM_AFFECTS, "mechanism_id", "biology_id"),
    (EdgeType.BIOLOGY_DRIVES, "biology_id", "indication_id"),
    (EdgeType.REFLECTS_BIOLOGY, "biology_id", "endpoint_id"),
    (EdgeType.ENDPOINT_CAPTURES, "endpoint_id", "indication_id"),
    (EdgeType.RESPONDS_DIFFERENTLY, "subgroup_population_id", "indication_id"),
]
ROLE_ATTRS = (
    "compound_id", "target_id", "mechanism_id", "biology_id",
    "indication_id", "endpoint_id", "subgroup_population_id",
)
_UNKNOWN = "UNKNOWN"


def _scoped(ont_id: str, nct: str) -> str:
    return f"{ont_id}#{nct}"


@dataclass
class BuildComparison:
    topdown_concepts: int          # distinct chain-referenced nodes (top-down)
    topdown_backbone_edges: int
    groundup_instances: int        # trial-scoped nodes before merge
    groundup_concepts: int         # distinct surviving concepts after merge
    groundup_backbone_edges: int
    consolidated: int              # concepts merge collapsed beyond exact-id
    merge_by_type: dict[str, int]

    def summary(self) -> str:
        return (
            f"top-down chain concepts={self.topdown_concepts}, backbone edges="
            f"{self.topdown_backbone_edges}\n"
            f"ground-up instances(pre-merge)={self.groundup_instances} -> "
            f"concepts(post-merge)={self.groundup_concepts}, backbone edges="
            f"{self.groundup_backbone_edges}\n"
            f"merge consolidated {self.consolidated} concept(s) beyond exact-id "
            f"dedup, by type: {self.merge_by_type}"
        )


def explode_to_chains_first(
    topdown: GraphStore,
    *,
    sapbert_fn: Callable[[str], list[float]] | None = None,
    gen_name_id: bool = True,
) -> GraphStore:
    """Build a trial-scoped, no-overlap graph from the top-down graph's chains.

    Each chain role becomes a ``{ontology_id}#{nct}`` node instance copying the
    canonical node's name/type/description and recording ``ontology_id`` (the
    Tier-1 merge key) + a SapBERT ``name_id`` (Tier-2). Backbone edges are
    recreated between the trial-scoped instances.
    """
    from src.graph.models import (
        BiologyNode, EndpointNode, IndicationNode, InterventionNode,
        MechanismNode, PopulationNode, TargetNode,
    )
    type_ctor = {
        "InterventionNode": InterventionNode, "TargetNode": TargetNode,
        "MechanismNode": MechanismNode, "BiologyNode": BiologyNode,
        "EndpointNode": EndpointNode, "IndicationNode": IndicationNode,
        "PopulationNode": PopulationNode,
    }
    if gen_name_id and sapbert_fn is None:
        from src.graph.sapbert_embeddings import embed_compound_name as sapbert_fn  # noqa

    gu = GraphStore()
    seen: set[str] = set()
    name_id_cache: dict[str, str] = {}

    def _name_id_for(name: str) -> str | None:
        # Cheap deterministic canonical key: SapBERT is used at MERGE time
        # (cosine); here we just stash the normalized surface form so equal
        # canonical names collapse in Tier-2a. (SapBERT-cosine handled in merge.)
        if not name:
            return None
        if name not in name_id_cache:
            name_id_cache[name] = name.strip().lower()
        return name_id_cache[name]

    from src.graph.models import TrialSubgraph

    for nct, ts in topdown.trial_subgraphs.items():
        scoped_chains = []
        for ch in ts.chains:
            for attr in ROLE_ATTRS:
                ont_id = getattr(ch, attr, None)
                if not ont_id or ont_id == _UNKNOWN:
                    continue
                sid = _scoped(ont_id, nct)
                if sid in seen:
                    continue
                seen.add(sid)
                try:
                    src = topdown.get_node(ont_id)
                except KeyError:
                    continue
                nt = src.get("node_type")
                ctor = type_ctor.get(nt)
                if ctor is None:
                    continue
                # copy ALL stored model fields (keeps required type fields like
                # mechanism_type / gene_symbol), then override the id.
                data = {k: v for k, v in src.items() if k in ctor.model_fields}
                data["id"] = sid
                node = ctor(**data)
                gu.add_node(node)
                # ontology_id / name_id live as TOP-LEVEL graph attrs (not every
                # node model has a metadata field); node_merge reads them there.
                attrs = gu._graph.nodes[sid]  # noqa: SLF001
                attrs["ontology_id"] = ont_id
                attrs["from_trial"] = nct
                if gen_name_id:
                    nid = _name_id_for(node.name)
                    if nid:
                        attrs["name_id"] = nid
            # backbone edges (trial-scoped)
            for et, sattr, tattr in CHAIN_BACKBONE:
                s, t = getattr(ch, sattr, None), getattr(ch, tattr, None)
                if not s or not t or s == _UNKNOWN or t == _UNKNOWN:
                    continue
                ss, tt = _scoped(s, nct), _scoped(t, nct)
                if ss in gu._graph and tt in gu._graph and not gu._graph.has_edge(  # noqa: SLF001
                    ss, tt, key=et.value,
                ):
                    gu.add_edge(GraphEdge(source_id=ss, target_id=tt, edge_type=et))
            # trial-scoped chain (ids namespaced; UNKNOWNs left as-is)
            updates = {}
            for attr in ROLE_ATTRS:
                v = getattr(ch, attr, None)
                if v and v != _UNKNOWN:
                    updates[attr] = _scoped(v, nct)
            scoped_chains.append(ch.model_copy(update=updates))
        parent = ts.parent_population_id
        gu.set_trial_subgraph(TrialSubgraph(
            trial_id=nct,
            parent_population_id=_scoped(parent, nct) if parent else f"pop#{nct}",
            chains=scoped_chains,
        ))
    return gu


def _concept_ids(graph: GraphStore) -> set[str]:
    """Distinct surviving concepts = each node's ontology_id (or its id)."""
    out: set[str] = set()
    for nid in graph._graph.nodes:  # noqa: SLF001
        node = graph._graph.nodes[nid]
        meta = node.get("metadata") or {}
        out.add(node.get("ontology_id") or meta.get("ontology_id") or nid)
    return out


def _topdown_chain_concepts(topdown: GraphStore) -> set[str]:
    out: set[str] = set()
    for ts in topdown.trial_subgraphs.values():
        for ch in ts.chains:
            for attr in ROLE_ATTRS:
                v = getattr(ch, attr, None)
                if v and v != _UNKNOWN:
                    out.add(v)
    return out


def _distinct_chain_edges(graph: GraphStore) -> int:
    """Distinct (src, tgt, edge_type) backbone triples implied by the chains —
    chain-derived on both sides so top-down and ground-up are comparable (the
    raw graph also carries non-chain prior edges, which v1 doesn't explode)."""
    seen: set[tuple[str, str, str]] = set()
    for ts in graph.trial_subgraphs.values():
        for ch in ts.chains:
            for et, sa, ta in CHAIN_BACKBONE:
                s, t = getattr(ch, sa, None), getattr(ch, ta, None)
                if s and t and s != _UNKNOWN and t != _UNKNOWN:
                    seen.add((s, t, et.value))
    return len(seen)


def build_groundup(
    topdown: GraphStore,
    *,
    config: MergeConfig | None = None,
    embed_fn: Callable[[str], list[float]] | None = None,
    sapbert_fn: Callable[[str], list[float]] | None = None,
) -> tuple[GraphStore, BuildComparison]:
    """Explode the top-down chains to trial-scoped instances, reassemble via
    ``node_merge``, and compare to the top-down graph. Returns (ground-up graph,
    comparison). ``config`` defaults to Tier-1-only (faithfulness check)."""
    config = config or MergeConfig(
        enable_id=True, enable_name_id=False, enable_sapbert=False,
        enable_biolord=False,
    )
    gu = explode_to_chains_first(topdown, sapbert_fn=sapbert_fn)
    instances = gu._graph.number_of_nodes()  # noqa: SLF001
    td_concepts = _topdown_chain_concepts(topdown)
    report = assemble(gu, config, embed_fn=embed_fn, sapbert_embed_fn=sapbert_fn)
    gu_concepts = _concept_ids(gu)
    return gu, BuildComparison(
        topdown_concepts=len(td_concepts),
        topdown_backbone_edges=_distinct_chain_edges(topdown),
        groundup_instances=instances,
        groundup_concepts=len(gu_concepts),
        groundup_backbone_edges=_distinct_chain_edges(gu),
        consolidated=len(td_concepts) - len(gu_concepts),
        merge_by_type=report.by_type,
    )
