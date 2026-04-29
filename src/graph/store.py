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
    EvidenceDirection,
    EvidenceRecord,
    EvidenceType,
    GraphEdge,
    TrialSubgraph,
)

EVIDENCE_TYPE_WEIGHTS: dict[EvidenceType, float] = {
    EvidenceType.CLINICAL_PHASE3: 5.0,
    EvidenceType.CLINICAL_PHASE2: 3.0,
    EvidenceType.CLINICAL_PHASE1: 1.5,
    EvidenceType.GENETIC_MR: 4.0,
    EvidenceType.GENETIC_GWAS: 2.5,
    EvidenceType.PRECLINICAL_IN_VIVO: 1.5,
    EvidenceType.PRECLINICAL_IN_VITRO: 1.0,
    EvidenceType.COMPUTATIONAL: 0.5,
    EvidenceType.LITERATURE: 0.3,
}


class GraphStore:
    """In-process knowledge graph backed by NetworkX MultiDiGraph."""

    def __init__(self) -> None:
        self._graph = nx.MultiDiGraph()

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

    def update_edge_belief(
        self,
        src_id: str,
        tgt_id: str,
        edge_type: EdgeType,
        evidence: EvidenceRecord,
    ) -> EdgeBeliefState:
        data = self._get_edge_data(src_id, tgt_id, edge_type)
        belief = EdgeBeliefState.model_validate(data["belief"])

        weight = EVIDENCE_TYPE_WEIGHTS[evidence.source_type]
        delta = weight * evidence.quality_score * evidence.magnitude

        if evidence.direction == EvidenceDirection.SUPPORTING:
            belief = belief.model_copy(
                update={"alpha": belief.alpha + delta}
            )
        elif evidence.direction == EvidenceDirection.CONTRADICTING:
            belief = belief.model_copy(
                update={"beta": belief.beta + delta}
            )
        else:  # ambiguous
            belief = belief.model_copy(
                update={
                    "alpha": belief.alpha + delta * 0.3,
                    "beta": belief.beta + delta * 0.3,
                }
            )

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
        node_ids = [
            trial.compound_id,
            trial.target_id,
            trial.mechanism_id,
            trial.biology_id,
            trial.indication_id,
            trial.endpoint_id,
            trial.population_id,
        ]
        present = [n for n in node_ids if n in self._graph]
        return self._graph.subgraph(present).copy()

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

    def export_snapshot(self, filepath: str) -> None:
        data = nx.node_link_data(self._graph)
        Path(filepath).write_text(json.dumps(data, indent=2, default=str))

    def import_snapshot(self, filepath: str) -> None:
        raw = json.loads(Path(filepath).read_text())
        self._graph = nx.node_link_graph(raw, directed=True, multigraph=True)

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
