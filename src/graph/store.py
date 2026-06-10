"""NetworkX-backed knowledge graph store."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx
from pydantic import BaseModel

from src.graph.models import (
    EdgeBeliefState,
    EdgeType,
    EvidenceRecord,
    EvidenceType,
    GraphEdge,
    TrialSubgraph,
)
from src.inference.beliefs import (
    EVIDENCE_TYPE_N_EFF,
    SupportBucket,
    apply_virtual_evidence,
    effective_n_for_evidence,
    p_obs_for_bucket,
    redundancy_factor,
)
from src.boundary import (
    assert_public_safe,
    require_under_private_root,
    strip_private,
)
from src.inference.belief_field import (
    BeliefField,
    apply_virtual_evidence_local,
    field_enabled,
    index_anchor_vectors,
    rehydrate_anchor_vectors,
)


class GraphStore:
    """In-process knowledge graph backed by NetworkX MultiDiGraph.

    Trial chains live in ``trial_subgraphs`` (sidecar) rather than as
    additional nodes/edges, because a chain is a *measurement* (this
    arm in this subgroup produced this outcome) rather than an entity
    other things relate to. The TrialNode in the graph is just an
    anchor; rich per-(arm × subgroup) data is reached via the sidecar
    keyed by trial id.
    """

    def __init__(self) -> None:
        self._graph = nx.MultiDiGraph()
        self.trial_subgraphs: dict[str, TrialSubgraph] = {}
        # Round-19: trials whose attribution has already been applied to
        # this graph. The attributor consults this set before applying a
        # trial's updates so re-running attribution on an existing
        # snapshot doesn't double-count evidence on already-touched
        # edges. Persisted in the snapshot via export/import_snapshot.
        self.applied_attribution_trial_ids: set[str] = set()

    # ── CRUD: nodes ──────────────────────────────────────────────────────

    def add_node(self, node: BaseModel) -> None:
        node_id = node.id  # type: ignore[attr-defined]
        data = node.model_dump(mode="json")
        node_type = type(node).__name__
        self._graph.add_node(node_id, node_type=node_type, **data)

    def get_node(self, node_id: str) -> dict[str, Any]:
        if node_id not in self._graph:
            raise KeyError(f"Node '{node_id}' not found")
        return dict(self._graph.nodes[node_id])

    # ── CRUD: edges ──────────────────────────────────────────────────────

    def add_edge(self, edge: GraphEdge) -> None:
        self._graph.add_edge(
            edge.source_id,
            edge.target_id,
            key=edge.edge_type.value,
            edge_type=edge.edge_type.value,
            belief=edge.belief.model_dump(mode="json"),
            metadata=edge.metadata,
        )

    def _get_edge_data(
        self, src_id: str, tgt_id: str, edge_type: EdgeType
    ) -> dict[str, Any]:
        key = edge_type.value
        if not self._graph.has_edge(src_id, tgt_id, key=key):
            raise KeyError(
                f"Edge '{src_id}' -> '{tgt_id}' ({key}) not found"
            )
        return self._graph.edges[src_id, tgt_id, key]

    def get_edge_belief(
        self, src_id: str, tgt_id: str, edge_type: EdgeType
    ) -> EdgeBeliefState:
        data = self._get_edge_data(src_id, tgt_id, edge_type)
        return EdgeBeliefState.model_validate(data["belief"])

    def get_edge_belief_conditioned(
        self,
        src_id: str,
        tgt_id: str,
        edge_type: EdgeType,
        relevant_tissues: set[str] | None = None,
        off_tissue_weight: float = 0.3,
    ) -> EdgeBeliefState:
        """Belief recomputed from evidence records, downweighting any record
        whose ``context["tissue"]`` is not in ``relevant_tissues``.

        Records with no ``tissue`` context are treated as fully relevant
        (context-free evidence applies regardless of indication). Records
        whose tissue *is* tagged but doesn't match get weighted by
        ``off_tissue_weight`` rather than dropped entirely—they're still
        partial evidence that the mechanism perturbs the biology, just in a
        different cellular context.

        When ``relevant_tissues`` is None or empty, behaves identically to
        ``get_edge_belief`` (no conditioning).
        """
        data = self._get_edge_data(src_id, tgt_id, edge_type)
        stored = EdgeBeliefState.model_validate(data["belief"])
        if not relevant_tissues:
            return stored
        if not stored.evidence:
            return stored

        # Replay every record through the conjugate update, scaling each
        # one's effective sample size by tissue match. Off-tissue evidence
        # still counts (the mechanism likely perturbs the same biology in
        # a different cellular context) but at reduced N_eff. Records
        # without a tissue tag are context-free and apply at full weight.
        belief = EdgeBeliefState(alpha=1.0, beta=1.0)
        cluster_seen: dict[str, int] = {}
        for ev in stored.evidence:
            tissue = (ev.context or {}).get("tissue")
            if tissue is None or tissue in relevant_tissues:
                tissue_weight = 1.0
            else:
                tissue_weight = off_tissue_weight
            eff_key = ev.cluster_key or ev.source_id
            prior_same = cluster_seen.get(eff_key, 0)
            n_eff = effective_n_for_evidence(
                ev.source_type, ev.quality_score,
                n_obs=ev.n_obs, edge_type=edge_type.value,
            ) * tissue_weight * redundancy_factor(prior_same)
            cluster_seen[eff_key] = prior_same + 1
            p_obs = p_obs_for_bucket(SupportBucket(ev.support))
            belief = apply_virtual_evidence(belief, n_eff=n_eff, p_obs=p_obs)
        return EdgeBeliefState(
            alpha=belief.alpha, beta=belief.beta, evidence=stored.evidence
        )

    def update_edge_belief(
        self,
        src_id: str,
        tgt_id: str,
        edge_type: EdgeType,
        evidence: EvidenceRecord,
        *,
        n_eff_override: float | None = None,
        p_obs_override: float | None = None,
    ) -> EdgeBeliefState:
        """Beta-Binomial conjugate update from one evidence record.

        See ``src/inference/beliefs.py`` for the full recipe. This method
        owns the I/O (reads the edge belief, computes the update, writes
        it back, and appends the record to the replay log); the math
        itself lives in ``apply_virtual_evidence``.

        ``n_eff_override`` / ``p_obs_override`` (outcome-conditioning
        attributor): when supplied, the caller has already computed the
        precise virtual sample size (and/or the implied p_obs) for this
        record — e.g. the per-edge explaining-away split of one failed
        trial's evidence across its chain — so the standard
        ``effective_n_for_evidence`` × redundancy derivation is bypassed.
        The EvidenceRecord is STILL appended to the replay log (with the
        override stashed in ``context`` so the update stays replayable),
        only the (n_eff, p_obs) used for THIS conjugate step changes.
        """
        data = self._get_edge_data(src_id, tgt_id, edge_type)
        belief = EdgeBeliefState.model_validate(data["belief"])

        if n_eff_override is not None:
            n_eff = n_eff_override
        else:
            # Independence/redundancy: existing records on this edge that
            # share the incoming one's correlation cluster (explicit
            # cluster_key, else the source/study id) make it a
            # non-independent observation.
            eff_key = evidence.cluster_key or evidence.source_id
            prior_same = sum(
                1 for e in belief.evidence
                if (e.cluster_key or e.source_id) == eff_key
            )
            n_eff = effective_n_for_evidence(
                evidence.source_type, evidence.quality_score,
                n_obs=evidence.n_obs, edge_type=edge_type.value,
            ) * redundancy_factor(prior_same)
        if p_obs_override is not None:
            if not 0.0 <= p_obs_override <= 1.0:
                raise ValueError(
                    f"p_obs_override must be in [0, 1], got {p_obs_override!r}"
                )
            p_obs = p_obs_override
        else:
            p_obs = p_obs_for_bucket(SupportBucket(evidence.support))
        belief = apply_virtual_evidence(belief, n_eff=n_eff, p_obs=p_obs)

        # A.3 (flag-gated, EROOM_BELIEF_FIELD): also localize this evidence on
        # the per-region belief field when it carries (s, t) embeddings. The
        # scalar update above is unchanged — default behavior and public
        # predictions are byte-identical until the flag is on. The field is the
        # private "edge weights" (stripped from public snapshots).
        if (
            field_enabled()
            and evidence.source_embedding is not None
            and evidence.target_embedding is not None
        ):
            bf = BeliefField.from_dict(belief.belief_field or {})
            apply_virtual_evidence_local(
                bf,
                s=evidence.source_embedding,
                t=evidence.target_embedding,
                n_eff=n_eff,
                p_obs=p_obs,
            )
            belief.belief_field = bf.to_dict()

        # Persist the EXACT weights this record contributed (incl. the
        # explaining-away override and the redundancy discount above) so any
        # replay — the (s,t) field materializer, the LOO self-exclusion —
        # reconstructs this scalar update faithfully instead of recomputing a
        # nominal n_eff that ignores the split. See EvidenceRecord.applied_n_eff.
        evidence.applied_n_eff = n_eff
        evidence.applied_p_obs = p_obs

        belief.evidence.append(evidence)
        data["belief"] = belief.model_dump(mode="json")
        return belief

    # ── Query operations ─────────────────────────────────────────────────

    def find_paths(
        self, src_id: str, tgt_id: str, max_length: int = 6
    ) -> list[list[str]]:
        try:
            return list(
                nx.all_simple_paths(
                    self._graph, src_id, tgt_id, cutoff=max_length
                )
            )
        except nx.NodeNotFound:
            return []

    def get_trial_subgraph(self, trial: TrialSubgraph) -> nx.MultiDiGraph:
        """Return the union of nodes touched by any chain in the trial.

        Aggregates each chain's compound/target/mechanism/biology/
        indication/endpoint plus its subgroup population, plus the
        parent enrollment population.
        """
        node_ids: set[str] = {trial.parent_population_id}
        for arm in trial.arms:
            node_ids.add(arm.regimen_compound_id)
            node_ids.update(arm.compound_ids)
        for chain in trial.chains:
            node_ids.update([
                chain.target_id,
                chain.mechanism_id,
                chain.biology_id,
                chain.indication_id,
                chain.endpoint_id,
                chain.subgroup_population_id,
            ])
        present = [n for n in node_ids if n in self._graph]
        return self._graph.subgraph(present).copy()

    # ── Trial subgraph sidecar ───────────────────────────────────────────

    def set_trial_subgraph(self, trial: TrialSubgraph) -> None:
        """Persist (or overwrite) a trial's chain set."""
        self.trial_subgraphs[trial.trial_id] = trial

    def get_trial_subgraph_by_id(self, trial_id: str) -> TrialSubgraph:
        if trial_id not in self.trial_subgraphs:
            raise KeyError(f"No trial subgraph for '{trial_id}'")
        return self.trial_subgraphs[trial_id]

    def get_neighboring_edges(
        self, node_id: str, edge_types: list[EdgeType] | None = None
    ) -> list[dict[str, Any]]:
        if node_id not in self._graph:
            raise KeyError(f"Node '{node_id}' not found")
        results: list[dict[str, Any]] = []
        type_vals = {et.value for et in edge_types} if edge_types else None
        for u, v, key, data in self._graph.edges(node_id, data=True, keys=True):
            if type_vals is None or key in type_vals:
                results.append(
                    {"source_id": u, "target_id": v, "edge_type": key, **data}
                )
        return results

    def get_nodes_by_type(self, node_type: str) -> list[dict[str, Any]]:
        return [
            {"id": node_id, **data}
            for node_id, data in self._graph.nodes(data=True)
            if data.get("node_type") == node_type
        ]

    def get_edges_by_type(self, edge_type: EdgeType) -> list[dict[str, Any]]:
        return [
            {"source_id": u, "target_id": v, **data}
            for u, v, key, data in self._graph.edges(data=True, keys=True)
            if key == edge_type.value
        ]

    # ── Persistence ──────────────────────────────────────────────────────

    def _build_snapshot_payload(self) -> dict[str, Any]:
        """Full serializable payload (public + any private values present).

        Format:
            {
              "graph": <node_link_data>,
              "trial_subgraphs": {trial_id: TrialSubgraph.model_dump()},
              "applied_attribution_trial_ids": [...]
            }
        """
        graph_data = nx.node_link_data(self._graph)
        trials_data = {
            tid: ts.model_dump(mode="json")
            for tid, ts in self.trial_subgraphs.items()
        }
        return {
            "graph": graph_data,
            "trial_subgraphs": trials_data,
            "applied_attribution_trial_ids": sorted(
                self.applied_attribution_trial_ids
            ),
        }

    def export_snapshot(self, filepath: str) -> None:
        """Serialize the **public** projection of the graph to one JSON file.

        Private values (fine-tuned embeddings, trained boxes, per-region
        belief fields — anything matching ``src/boundary.py``'s convention)
        are stripped before writing, so the committed ``data/exports/``
        artifact is clean by construction even when the in-memory graph holds
        them during a combined build. An ``assert_public_safe`` self-check
        then turns any field that slips the convention into a loud failure
        rather than a silent leak. See ``src/boundary.py`` for the contract.

        The scalar ``Beta(alpha, beta)`` edge belief, evidence provenance, and
        trial subgraphs are public and pass through unchanged — the public
        snapshot stays exactly as marketed today.

        Old-format snapshots (bare node_link_data) remain readable by
        ``import_snapshot`` for backwards compatibility.
        """
        public_payload = strip_private(self._build_snapshot_payload())
        # The manifold-2 belief field is now PUBLIC (open-core predictor). Dedup
        # its anchor (s,t) BioLORD vectors into a shared ``_belief_field_vectors``
        # table — the same compaction the private snapshot used — since the same
        # description embedding recurs across thousands of anchors. ``index_anchor_
        # vectors`` rewrites only fresh belief dicts, so the live graph is untouched.
        graph_data = public_payload.get("graph", {})
        links = graph_data.get("links") or graph_data.get("edges") or []
        table = index_anchor_vectors(links)
        if table:
            public_payload["_belief_field_vectors"] = table
        assert_public_safe(public_payload, source=filepath)
        # Compact when a field is present: its vector table dominates, and at scale
        # ``indent=2`` put every float on its own line (~330 MB of whitespace at
        # n=252). Fieldless snapshots stay pretty for human readability.
        if table:
            Path(filepath).write_text(
                json.dumps(public_payload, separators=(",", ":"), default=str)
            )
        else:
            Path(filepath).write_text(
                json.dumps(public_payload, indent=2, default=str)
            )

    def export_private_snapshot(self, filepath: str) -> None:
        """Serialize the **full** payload (public + private values) for the
        enterprise tree.

        Refuses any ``filepath`` outside ``boundary.private_root()`` so a
        private snapshot can never be written into the tracked ``data/exports/``
        directory by a slipped argument. Use this for the per-region belief
        field and any other manifold-2/3 artifacts.
        """
        target = require_under_private_root(filepath)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Compact (no indent, tight separators): the belief-field snapshot is
        # machine-read only, and at scale it's dominated by 768-dim anchor
        # vectors — with ``indent=2`` every float got its own indented line
        # (~330 MB of pure whitespace at n=252). json.loads reads this fine via
        # both load paths (store.import_snapshot + field_prediction.load_edge_fields).
        #
        # Then dedup those vectors into a shared ``_belief_field_vectors`` table:
        # anchor (s, t) are BioLORD embeddings of node/arm DESCRIPTIONS, which
        # recur across thousands of anchors, so the unique-vector table is far
        # smaller than the raw anchor count. ``index_anchor_vectors`` rewrites
        # only fresh copies of the touched link beliefs, so the live in-memory
        # graph (whose nested objects node_link_data shares) is untouched.
        payload = self._build_snapshot_payload()
        graph_data = payload.get("graph", {})
        links = graph_data.get("links") or graph_data.get("edges") or []
        table = index_anchor_vectors(links)
        if table:
            payload["_belief_field_vectors"] = table
        target.write_text(
            json.dumps(payload, default=str, separators=(",", ":"))
        )

    def import_snapshot(self, filepath: str) -> None:
        raw = json.loads(Path(filepath).read_text())
        # New format has the wrapper; old format is bare node_link_data.
        if isinstance(raw, dict) and "graph" in raw and "trial_subgraphs" in raw:
            # Private snapshots index anchor (s, t) into a shared dedup table
            # (export_private_snapshot). Resolve indices back to shared list
            # refs in the just-parsed payload — so the in-memory graph carries
            # real vectors (and repeated vectors are one object, not N copies).
            # No-op for public/old snapshots (no table → inline vectors stay).
            table = raw.get("_belief_field_vectors")
            graph_blob = raw["graph"]
            rehydrate_anchor_vectors(
                graph_blob.get("links") or graph_blob.get("edges") or [], table
            )
            self._graph = nx.node_link_graph(
                raw["graph"], directed=True, multigraph=True
            )
            self.trial_subgraphs = {
                tid: TrialSubgraph.model_validate(blob)
                for tid, blob in raw.get("trial_subgraphs", {}).items()
            }
            # Round-19: legacy snapshots written before this field existed
            # are treated as "all trial_subgraphs already attributed". This
            # is safe because the build pipeline only writes a trial
            # subgraph into the annotated snapshot after attribution has
            # run over it — there's no state where a subgraph exists but
            # its updates haven't been folded into edge beliefs. Without
            # the backfill, the first incremental --add-trials run against
            # a pre-round-19 snapshot would re-attribute every existing
            # trial and double-count every edge.
            if "applied_attribution_trial_ids" in raw:
                self.applied_attribution_trial_ids = set(
                    raw["applied_attribution_trial_ids"] or []
                )
            else:
                self.applied_attribution_trial_ids = set(
                    self.trial_subgraphs.keys()
                )
        else:
            self._graph = nx.node_link_graph(raw, directed=True, multigraph=True)
            self.trial_subgraphs = {}
            self.applied_attribution_trial_ids = set()

    # ── Stats ────────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        node_types: dict[str, int] = {}
        for _, data in self._graph.nodes(data=True):
            nt = data.get("node_type", "unknown")
            node_types[nt] = node_types.get(nt, 0) + 1

        edge_types: dict[str, int] = {}
        total_evidence = 0
        high_conflict: list[dict[str, str]] = []
        for u, v, key, data in self._graph.edges(data=True, keys=True):
            edge_types[key] = edge_types.get(key, 0) + 1
            belief_data = data.get("belief", {})
            evidence_list = belief_data.get("evidence", [])
            total_evidence += len(evidence_list)
            belief = EdgeBeliefState.model_validate(belief_data) if belief_data else None
            if belief and belief.conflict_score > 5.0:
                high_conflict.append(
                    {"source_id": u, "target_id": v, "edge_type": key}
                )

        return {
            "node_count": self._graph.number_of_nodes(),
            "edge_count": self._graph.number_of_edges(),
            "node_types": node_types,
            "edge_types": edge_types,
            "total_evidence": total_evidence,
            "high_conflict_edges": high_conflict,
        }
