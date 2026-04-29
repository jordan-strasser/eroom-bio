"""Damped belief propagation across graph neighbors."""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from src.graph.models import (
    EdgeBeliefState,
    EdgeType,
    EvidenceDirection,
    EvidenceRecord,
    EvidenceType,
)
from src.graph.store import GraphStore

logger = logging.getLogger(__name__)

# ── Compatible edge type groups ─────────────────────────────────────────
# Edges in the same group can propagate to each other when they share a node.

_EDGE_TYPE_GROUPS: dict[str, set[EdgeType]] = {
    "binding": {EdgeType.BINDS_TO, EdgeType.MODULATES_VIA},
    "mechanism": {EdgeType.MODULATES_VIA, EdgeType.MECHANISM_AFFECTS},
    "biology": {EdgeType.MECHANISM_AFFECTS, EdgeType.BIOLOGY_DRIVES, EdgeType.REFLECTS_BIOLOGY},
    "endpoint": {EdgeType.REFLECTS_BIOLOGY, EdgeType.ENDPOINT_CAPTURES},
    "population": {EdgeType.RESPONDS_DIFFERENTLY, EdgeType.BIOLOGY_DRIVES},
}

# Precompute: for each edge type, which other types are compatible
_COMPATIBLE_TYPES: dict[EdgeType, set[EdgeType]] = {}
for _group in _EDGE_TYPE_GROUPS.values():
    for _et in _group:
        _COMPATIBLE_TYPES.setdefault(_et, set()).update(_group)
# Remove self from compatible set
for _et in _COMPATIBLE_TYPES:
    _COMPATIBLE_TYPES[_et].discard(_et)


# ── Output model ────────────────────────────────────────────────────────


class PropagatedUpdate(BaseModel):
    """Record of a propagated belief change."""

    source_id: str
    target_id: str
    edge_type: EdgeType
    hop: int
    damped_magnitude: float
    pre_belief: EdgeBeliefState
    post_belief: EdgeBeliefState

    @property
    def probability_change(self) -> float:
        return (
            self.post_belief.expected_probability
            - self.pre_belief.expected_probability
        )


# ── Propagation ─────────────────────────────────────────────────────────


def propagate(
    graph: GraphStore,
    source_id: str,
    target_id: str,
    edge_type: EdgeType,
    direction: EvidenceDirection,
    magnitude: float,
    max_hops: int = 2,
    damping: float = 0.3,
    min_magnitude: float = 0.01,
) -> list[PropagatedUpdate]:
    """Walk outward from an updated edge, applying damped updates to compatible neighbors.

    At each hop, magnitude is multiplied by damping. Stops when damped
    magnitude falls below min_magnitude or max_hops is reached.
    """
    updates: list[PropagatedUpdate] = []
    # Track visited edges to avoid cycles: (src, tgt, edge_type_value)
    visited: set[tuple[str, str, str]] = {(source_id, target_id, edge_type.value)}

    # BFS queue: (node_to_explore, current_hop, current_magnitude, propagated_direction, from_edge_type)
    queue: deque[tuple[str, int, float, EvidenceDirection, EdgeType]] = deque()

    # Seed: explore both endpoints of the updated edge
    queue.append((source_id, 1, magnitude * damping, direction, edge_type))
    queue.append((target_id, 1, magnitude * damping, direction, edge_type))

    while queue:
        node_id, hop, current_mag, current_dir, from_type = queue.popleft()

        if current_mag < min_magnitude or hop > max_hops:
            continue

        compatible = _COMPATIBLE_TYPES.get(from_type, set())
        if not compatible:
            continue

        # Find all edges touching this node (both directions)
        neighbors = _get_all_edges_at_node(graph, node_id)

        for nbr in neighbors:
            nbr_src = nbr["source_id"]
            nbr_tgt = nbr["target_id"]
            nbr_type_str = nbr["edge_type"]

            try:
                nbr_edge_type = EdgeType(nbr_type_str)
            except ValueError:
                continue

            if nbr_edge_type not in compatible:
                continue

            edge_key = (nbr_src, nbr_tgt, nbr_type_str)
            if edge_key in visited:
                continue
            visited.add(edge_key)

            # Apply damped update
            try:
                pre_belief = graph.get_edge_belief(nbr_src, nbr_tgt, nbr_edge_type)
            except KeyError:
                continue

            evidence = EvidenceRecord(
                source_id=f"propagation_hop{hop}",
                source_type=EvidenceType.COMPUTATIONAL,
                quality_score=current_mag,
                direction=current_dir,
                magnitude=current_mag,
                timestamp=datetime.now(timezone.utc),
                notes=f"Propagated from {source_id}->{target_id} ({edge_type.value}), hop {hop}, damping {damping}",
            )

            post_belief = graph.update_edge_belief(
                nbr_src, nbr_tgt, nbr_edge_type, evidence
            )

            updates.append(PropagatedUpdate(
                source_id=nbr_src,
                target_id=nbr_tgt,
                edge_type=nbr_edge_type,
                hop=hop,
                damped_magnitude=current_mag,
                pre_belief=pre_belief,
                post_belief=post_belief,
            ))

            # Continue BFS from the other end of this edge
            next_node = nbr_tgt if nbr_src == node_id else nbr_src
            next_mag = current_mag * damping
            if next_mag >= min_magnitude and hop + 1 <= max_hops:
                queue.append((next_node, hop + 1, next_mag, current_dir, nbr_edge_type))

    return updates


def _get_all_edges_at_node(
    graph: GraphStore, node_id: str
) -> list[dict[str, Any]]:
    """Get all edges (incoming + outgoing) touching a node."""
    results: list[dict[str, Any]] = []
    g = graph._graph  # Access underlying NetworkX graph

    # Outgoing
    if node_id in g:
        for u, v, key, data in g.out_edges(node_id, data=True, keys=True):
            results.append({"source_id": u, "target_id": v, "edge_type": key, **data})

        # Incoming
        for u, v, key, data in g.in_edges(node_id, data=True, keys=True):
            results.append({"source_id": u, "target_id": v, "edge_type": key, **data})

    return results
