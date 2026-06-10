"""Biology-node fragmentation merge.

Today the same biology can land on different BiologyNode ids depending on
where the chain happened to source from:
  - VEGFA's "Signaling by VEGF" might end up as ``R-HSA-194138`` (Reactome
    primary for one chain) or as ``GO:0048010`` (Reactome's curator-chosen
    GO term for the same pathway, picked by another chain).
  - PI3K/AKT biology is fragmented across multiple Reactome pathways (one
    per signaling step) that all crosswalk to the same GO BP term.

These fragments don't pool evidence — a trial that hits one of them
doesn't strengthen beliefs on its semantic-twin. This module collapses
them at populate-time and offers a callable sweep to clean up existing
graphs.

Equivalence source: Reactome's ``goBiologicalProcess`` curator annotation
on each pathway. One GO accession per Reactome pathway (or none) — clean
many-to-one mapping that, transitively, lets us cluster:

  R-HSA-X ─crosswalk──→ GO:Y  ←──crosswalk─ R-HSA-Z   ⟹  {R-HSA-X, R-HSA-Z, GO:Y}

Within each equivalence class we keep the BiologyNode whose
``metadata["primary_score"]`` is highest — that's the chain that picked
the *most contextually-relevant* label, which is a better stand-in for
"the right one to keep" than name-length or recency. Tie-break by
sorted stable-id order so the result is deterministic across rebuilds.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from pydantic import BaseModel

from src.graph.models import EvidenceRecord, EdgeBeliefState
from src.inference.beliefs import (
    BUCKET_TO_P_OBS,
    SupportBucket,
    applied_weights,
    apply_virtual_evidence,
    effective_n_for_evidence,
)

if TYPE_CHECKING:
    from src.graph.store import GraphStore
    from src.ingestion.lincs import LINCSClient

logger = logging.getLogger(__name__)


class MergeReport(BaseModel):
    """Summary of a single merge sweep."""
    biology_nodes_before: int
    biology_nodes_after: int
    classes_merged: int
    nodes_removed: list[str]
    winner_by_loser: dict[str, str]  # loser_id → winner_id mapping
    chains_updated: int

    @property
    def nodes_collapsed(self) -> int:
        return len(self.nodes_removed)


async def fetch_reactome_to_go_crosswalk(
    reactome_ids: list[str],
    lincs_client: "LINCSClient",
) -> dict[str, str]:
    """Resolve Reactome stIds to their curator-annotated GO BP accessions.

    Returns a dict ``{R-HSA-...: GO:...}``. Pathways with no goBiologicalProcess
    (Reactome's crosswalk slot is sometimes empty) are simply absent from the
    returned dict — they can't merge against anything.
    """
    crosswalk: dict[str, str] = {}
    for stable_id in reactome_ids:
        try:
            go_acc = await lincs_client.get_pathway_go_term(stable_id)
        except Exception:
            logger.debug(
                "Reactome→GO lookup failed for %s", stable_id, exc_info=True,
            )
            continue
        if go_acc:
            crosswalk[stable_id] = go_acc
    return crosswalk


def find_equivalence_classes(
    biology_ids: list[str],
    crosswalk: dict[str, str],
) -> list[set[str]]:
    """Cluster BiologyNode ids into equivalence classes via the crosswalk.

    Two ids are equivalent iff they share a GO accession through the crosswalk:
      - A Reactome id R is equivalent to ``crosswalk[R]`` (the GO term Reactome
        annotates this pathway with) — BUT only when that GO id is also in
        the graph as its own BiologyNode.
      - Two Reactome ids that crosswalk to the same GO accession are equivalent
        to each other, regardless of whether the GO accession itself is in the
        graph. The GO term is the "shared canonical ancestor" — even if no
        chain materialized it as a node, both Reactome pathways describe the
        same underlying biology and should pool evidence.
      - A GO id that *is* in the graph as a node is equivalent to all Reactome
        ids that crosswalk to it, transitively pulling them into one class.

    Returns the classes with ≥ 2 members (singletons aren't actionable).
    """
    ids_in_graph = set(biology_ids)
    # union-find over node ids
    parent: dict[str, str] = {nid: nid for nid in ids_in_graph}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            # Pick the lexicographically smaller as root for determinism.
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    # Reverse-index: GO accession → list of in-graph Reactome ids that map to it.
    # Used to find both pairwise Reactome equivalence and Reactome↔GO equivalence.
    go_to_reactomes: dict[str, list[str]] = {}
    for reactome_id, go_id in crosswalk.items():
        if reactome_id in ids_in_graph:
            go_to_reactomes.setdefault(go_id, []).append(reactome_id)

    for go_id, reactome_ids in go_to_reactomes.items():
        # Union all Reactome pathways sharing this GO ancestor — even when
        # the GO node itself isn't in the graph. This is the Reactome-Reactome
        # fragmentation case (e.g. R-HSA-174048, R-HSA-174178, R-HSA-174184
        # all crosswalk to GO:0031145).
        for r in reactome_ids[1:]:
            union(reactome_ids[0], r)
        # If the GO node *is* in the graph, fold it into the same class.
        if go_id in ids_in_graph:
            union(reactome_ids[0], go_id)

    classes: dict[str, set[str]] = {}
    for nid in ids_in_graph:
        root = find(nid)
        classes.setdefault(root, set()).add(nid)

    return [c for c in classes.values() if len(c) >= 2]


def pick_winner(
    class_: set[str],
    graph: "GraphStore",
) -> str:
    """Pick the merge winner from an equivalence class.

    Higher ``metadata["primary_score"]`` wins (Option D). Ties break on
    sorted stable-id order for determinism. A node missing the field is
    treated as score 0.0 — older nodes that predate the score-tracking
    addition lose to any chain that did record a score.
    """
    def score(node_id: str) -> float:
        try:
            meta = graph.get_node(node_id).get("metadata") or {}
        except KeyError:
            return -1.0
        return float(meta.get("primary_score") or 0.0)

    # Sort key: highest score first; on ties, lexicographically smaller id first.
    return min(class_, key=lambda n: (-score(n), n))


def _evidence_dedup_key(r: EvidenceRecord):
    """Content identity for merge dedup. Replicated idempotent DATABASE facts — the
    SAME OT/ChEMBL fact stamped once per trial-scoped copy, each with its own
    ``datetime.now()`` — collapse to ONE, keyed by ``(source_id, support)``; per-arm
    trial-outcome records stay distinct via ``arm_id``. Replaces the old
    ``(source_id, timestamp)`` key, which OVER-COUNTED database facts (timestamp
    varies across copies → not deduped) and could wrongly merge two arms sharing a
    timestamp."""
    arm = (r.context or {}).get("arm_id")
    return (r.source_id, r.support, arm)


def _replay_belief(records: list[EvidenceRecord]) -> EdgeBeliefState:
    """Re-derive a Beta(α,β) by replaying records from prior(1,1), each at the EXACT
    weight it applied (``applied_weights``) — NOT a nominal recompute, which drops the
    explaining-away split and re-counts replicated facts at full weight."""
    belief = EdgeBeliefState(alpha=1.0, beta=1.0)
    for r in sorted(records, key=lambda x: x.timestamp):
        try:
            n_eff, p_obs = applied_weights(r)
        except (KeyError, ValueError):
            continue
        belief = apply_virtual_evidence(belief, n_eff=n_eff, p_obs=p_obs)
    return belief


def _merge_belief_data(
    existing_belief: dict | None,
    incoming_belief: dict | None,
) -> dict:
    """Union evidence lists (dedup by source_id+timestamp) and replay."""
    def _records(b: dict | None) -> list[EvidenceRecord]:
        if not b:
            return []
        return [EvidenceRecord.model_validate(r) for r in (b.get("evidence") or [])]

    combined: dict = {}
    for r in _records(existing_belief) + _records(incoming_belief):
        # Content dedup (see _evidence_dedup_key): collapse replicated idempotent
        # database facts to one; keep per-arm trial records distinct.
        combined.setdefault(_evidence_dedup_key(r), r)
    deduped = list(combined.values())
    replayed = _replay_belief(deduped)
    replayed.evidence = deduped
    return replayed.model_dump(mode="json")


def _merge_metadata(winner_meta: dict, loser_id: str, loser_node: dict) -> dict:
    """Fold loser node metadata into winner's metadata. Idempotent."""
    merged = dict(winner_meta) if winner_meta else {}

    # Track that this loser was merged in.
    merged_from = set(merged.get("merged_from") or [])
    merged_from.add(loser_id)
    merged["merged_from"] = sorted(merged_from)

    # Extend pathway_ids on the winner with the loser's pathway_ids.
    # Both lists are id-only; display names live in alternate_pathway_names.
    winner_pathway_ids = set()  # filled by caller after we update node data

    # Combine alternate_pathway_names (dedup by id, keep first name).
    winner_alts = dict(merged.get("alternate_pathway_names") or {})
    loser_meta = loser_node.get("metadata") or {}
    loser_alts = loser_meta.get("alternate_pathway_names") or {}
    for sid, name in loser_alts.items():
        winner_alts.setdefault(sid, name)
    # Also fold the loser's own name in as an alternate, so the audit trail
    # is complete: future readers can see "this winner subsumed loser X
    # which was named Y."
    winner_alts.setdefault(loser_id, loser_node.get("name") or loser_id)
    merged["alternate_pathway_names"] = winner_alts

    # Combine reactome_alternatives (only present when the chain picked GO
    # over a Reactome set).
    winner_rx = dict(merged.get("reactome_alternatives") or {})
    loser_rx = loser_meta.get("reactome_alternatives") or {}
    for sid, name in loser_rx.items():
        winner_rx.setdefault(sid, name)
    if winner_rx:
        merged["reactome_alternatives"] = winner_rx

    return merged


def _merge_node_data(
    graph: "GraphStore",
    winner_id: str,
    loser_id: str,
) -> None:
    """In-place merge of loser node's data fields into winner's."""
    winner_node = graph._graph.nodes[winner_id]  # noqa: SLF001
    loser_node = graph._graph.nodes[loser_id]  # noqa: SLF001

    # Union pathway_ids.
    winner_ids = list(winner_node.get("pathway_ids") or [])
    loser_ids = list(loser_node.get("pathway_ids") or [])
    seen = set(winner_ids)
    for pid in loser_ids:
        if pid not in seen:
            winner_ids.append(pid)
            seen.add(pid)
    winner_node["pathway_ids"] = winner_ids

    # Merge metadata.
    winner_node["metadata"] = _merge_metadata(
        winner_node.get("metadata") or {}, loser_id, loser_node,
    )


def _redirect_edges(
    graph: "GraphStore",
    winner_id: str,
    loser_id: str,
) -> None:
    """Move every edge touching loser_id to point at winner_id instead.

    Edge keys (= edge_type values) are preserved. When the redirect collides
    with an existing edge on the winner with the same direction + edge type,
    evidence records are unioned and beliefs replayed so no signal is lost.
    """
    g = graph._graph  # noqa: SLF001
    # Snapshot first — we'll mutate g.
    incoming = list(g.in_edges(loser_id, data=True, keys=True))
    outgoing = list(g.out_edges(loser_id, data=True, keys=True))

    for u, _v, key, data in incoming:
        target = winner_id
        if g.has_edge(u, target, key=key):
            # Collision — merge belief data into existing.
            existing_data = g.edges[u, target, key]
            existing_data["belief"] = _merge_belief_data(
                existing_data.get("belief"), data.get("belief"),
            )
        else:
            g.add_edge(u, target, key=key, **data)

    for _u, v, key, data in outgoing:
        source = winner_id
        if g.has_edge(source, v, key=key):
            existing_data = g.edges[source, v, key]
            existing_data["belief"] = _merge_belief_data(
                existing_data.get("belief"), data.get("belief"),
            )
        else:
            g.add_edge(source, v, key=key, **data)


def _update_chain_references(
    graph: "GraphStore",
    loser_id: str,
    winner_id: str,
) -> int:
    """Rewrite ``biology_id`` on every chain that referenced the loser."""
    rewrites = 0
    for trial_id, ts in list(graph.trial_subgraphs.items()):
        if not any(c.biology_id == loser_id for c in ts.chains):
            continue
        new_chains = [
            (
                c.model_copy(update={"biology_id": winner_id})
                if c.biology_id == loser_id else c
            )
            for c in ts.chains
        ]
        graph.trial_subgraphs[trial_id] = ts.model_copy(
            update={"chains": new_chains}
        )
        rewrites += sum(1 for c in ts.chains if c.biology_id == loser_id)
    return rewrites


def embedding_merge_enabled() -> bool:
    """A.4 feature flag (``EROOM_EMBEDDING_MERGE``). Default off — the build
    path stays crosswalk-only until the embedding merger is threshold-validated
    on the gold set (A.5)."""
    return os.environ.get("EROOM_EMBEDDING_MERGE", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def embedding_merge_pairs(
    graph: "GraphStore",
    *,
    embed_fn=None,
    threshold: float = 0.95,
) -> list[tuple[str, str]]:
    """Biology-node pairs the embedding merger judges equivalent — semantic
    twins the Reactome↔GO crosswalk missed (e.g. a slug node vs its Reactome
    pathway, or two pathways with no shared GO term but near-identical
    descriptions).

    Cosine ≥ ``threshold`` on BioLORD embeddings of each node's A.0 description
    (falling back to its name). PUBLIC code (open methods); the merge DECISION
    is flag-gated (``EROOM_EMBEDDING_MERGE``, default off).

    **Gold-set validation (2026-05-24, scripts/eval_merge_threshold.py):** raw
    cosine is a WEAK merge discriminator — best F1 only ~0.53 because `sibling`
    (mean 0.63, max 0.98) and `parent_of`/`child_of` (max ~0.99) pairs often
    score as high as true `merge` pairs (mean 0.95). At the old 0.92 default,
    precision was ~0.35 (would over-merge distinct biology, irreversibly). The
    default is raised to 0.95 (precision-leaning) and the flag stays OFF; the
    recommended upgrade is to gate the decision on the manifold-1 **box
    relation** (`box_embeddings.relation == "merge"`), which uses learned
    containment to separate merge from is-a/sibling — needs a graph-node-pair
    eval to validate before enabling. ``embed_fn`` is injectable for tests.
    """
    bio = [
        (n["id"], (n.get("description") or n.get("name") or "").strip())
        for n in graph.get_nodes_by_type("BiologyNode")
    ]
    bio = [(i, t) for i, t in bio if t]
    if len(bio) < 2:
        return []
    if embed_fn is None:
        from src.graph.biolord_embeddings import embed_text as embed_fn  # noqa
    from src.graph.biolord_embeddings import cosine_similarity

    vecs = {i: embed_fn(t) for i, t in bio}
    ids = [i for i, _ in bio]
    pairs: list[tuple[str, str]] = []
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            if cosine_similarity(vecs[ids[a]], vecs[ids[b]]) >= threshold:
                pairs.append((ids[a], ids[b]))
    return pairs


def box_merge_pairs(
    graph: "GraphStore",
    boxes: dict,
    *,
    node_type: str = "BiologyNode",
) -> list[tuple[str, str]]:
    """Biology-node pairs the manifold-1 **box relation** judges `merge`.

    Where ``embedding_merge_pairs`` thresholds raw cosine (which the gold set
    showed conflates merge with sibling/parent — F1 ~0.53), this uses the
    trained box geometry: ``relation == "merge"`` requires the two boxes to be
    near-coincident (mutually contained). A parent/child pair — one box
    contained in the other — is classified `parent_of`/`child_of` and is NOT
    merged, which is the precision raw cosine lacked. ``boxes`` maps node_id ->
    ``box_embeddings.Box`` (load from the private ``manifold1_boxes.json``).
    Only nodes that have a fitted box participate.

    **Validation (52-graph, 2026-05-24): still NOT safe to enable.** This does
    respect hierarchy (0 Reactome ancestor-pairs merged, unlike cosine), but it
    still over-merges biologically-distinct pairs whose *descriptions* are
    near-identical — e.g. "Co-inhibition by PD-1" vs "Co-inhibition by CTLA4"
    (distinct checkpoints), and templated slug nodes ("antimetabolite biology"
    vs "interleukin signaling"). Embedding-of-description can't see the
    gene-level distinction (PD-1 != CTLA4), so no box-size calibration fixes it.
    Conclusion: the ontology crosswalk remains the more reliable merger; an
    embedding merger needs a gene/target-aware signal (or a node-pair gold set
    to set a high-precision operating point) before it's worth enabling. The
    flag (``EROOM_EMBEDDING_MERGE``) stays OFF.
    """
    from src.graph.box_embeddings import relation

    ids = [
        n["id"]
        for n in graph.get_nodes_by_type(node_type)
        if n["id"] in boxes
    ]
    pairs: list[tuple[str, str]] = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if relation(boxes[ids[i]], boxes[ids[j]]) == "merge":
                pairs.append((ids[i], ids[j]))
    return pairs


def augment_classes_with_pairs(
    classes: list[set[str]], pairs: list[tuple[str, str]],
) -> list[set[str]]:
    """Union crosswalk equivalence classes that an embedding-merge pair bridges.

    Pairs between two nodes the crosswalk left as singletons still merge them.
    Returns only classes with ≥2 members (singletons are no-op merges).
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a: str, b: str) -> None:
        parent[find(a)] = find(b)

    for cls in classes:
        members = list(cls)
        for m in members[1:]:
            union(members[0], m)
    for a, b in pairs:
        union(a, b)

    groups: dict[str, set[str]] = {}
    for node in list(parent):
        groups.setdefault(find(node), set()).add(node)
    return [g for g in groups.values() if len(g) >= 2]


async def merge_equivalent_biology_nodes(
    graph: "GraphStore",
    lincs_client: "LINCSClient",
) -> MergeReport:
    """Sweep the graph for equivalent BiologyNodes and merge each class.

    Designed to be idempotent: running twice on the same graph is a no-op
    on the second pass (no equivalent pairs remain). Used both at the end
    of populate (to clean up new builds automatically) and as a callable
    on already-built graphs.

    Returns a ``MergeReport`` so callers can log or surface the changes.
    """
    bio_nodes = [
        n["id"] for n in graph.get_nodes_by_type("BiologyNode")
    ]
    before = len(bio_nodes)

    reactome_ids = [b for b in bio_nodes if b.startswith("R-")]
    crosswalk = await fetch_reactome_to_go_crosswalk(reactome_ids, lincs_client)
    classes = find_equivalence_classes(bio_nodes, crosswalk)

    # A.4 (flag-gated, EROOM_EMBEDDING_MERGE): catch semantic equivalence the
    # ontology crosswalk missed, via BioLORD cosine on node descriptions. Off
    # by default so the crosswalk-only build path is unchanged until validated.
    if embedding_merge_enabled():
        emb_pairs = embedding_merge_pairs(graph)
        if emb_pairs:
            classes = augment_classes_with_pairs(classes, emb_pairs)
            logger.info("A.4 embedding merger added %d pairs", len(emb_pairs))

    nodes_removed: list[str] = []
    winner_by_loser: dict[str, str] = {}
    chains_updated_total = 0

    for class_ in classes:
        winner = pick_winner(class_, graph)
        losers = sorted(class_ - {winner})
        for loser in losers:
            _merge_node_data(graph, winner, loser)
            _redirect_edges(graph, winner, loser)
            chains_updated_total += _update_chain_references(
                graph, loser, winner,
            )
            graph._graph.remove_node(loser)  # noqa: SLF001
            nodes_removed.append(loser)
            winner_by_loser[loser] = winner

    after = before - len(nodes_removed)
    return MergeReport(
        biology_nodes_before=before,
        biology_nodes_after=after,
        classes_merged=len(classes),
        nodes_removed=nodes_removed,
        winner_by_loser=winner_by_loser,
        chains_updated=chains_updated_total,
    )
