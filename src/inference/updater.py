"""Evidence updater: processes trial outcomes into graph edge updates."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from src.annotation.taxonomy import (
    FAILURE_MODE_RULES,
    FailureClassification,
    FailureMode,
)
from src.graph.models import (
    EdgeBeliefState,
    EdgeType,
    EvidenceDirection,
    EvidenceRecord,
    EvidenceType,
    TrialSubgraph,
)
from src.graph.store import GraphStore

logger = logging.getLogger(__name__)

# ── Weight tables ───────────────────────────────────────────────────────

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

QUALITY_WEIGHTS: dict[str, float] = {
    "high": 1.0,
    "medium": 0.6,
    "low": 0.3,
    "unknown": 0.5,
}

_PHASE_TO_EVIDENCE: dict[str, EvidenceType] = {
    "1": EvidenceType.CLINICAL_PHASE1,
    "early_1": EvidenceType.CLINICAL_PHASE1,
    "2": EvidenceType.CLINICAL_PHASE2,
    "2/3": EvidenceType.CLINICAL_PHASE3,
    "3": EvidenceType.CLINICAL_PHASE3,
    "4": EvidenceType.CLINICAL_PHASE3,
}

# Map edge_type to (source_field, target_field) on TrialSubgraph
_EDGE_TYPE_TO_SUBGRAPH_FIELDS: dict[str, tuple[str, str]] = {
    "binds_to": ("compound_id", "target_id"),
    "modulates_via": ("target_id", "mechanism_id"),
    "mechanism_affects": ("mechanism_id", "biology_id"),
    "biology_drives": ("biology_id", "indication_id"),
    "reflects_biology": ("biology_id", "endpoint_id"),
    "endpoint_captures": ("endpoint_id", "indication_id"),
    "responds_differently": ("population_id", "indication_id"),
}


# ── Output model ────────────────────────────────────────────────────────


class EdgeUpdate(BaseModel):
    """Record of a single edge belief update."""

    source_id: str
    target_id: str
    edge_type: EdgeType
    direction: EvidenceDirection
    magnitude: float
    weight: float
    pre_belief: EdgeBeliefState
    post_belief: EdgeBeliefState

    @property
    def probability_change(self) -> float:
        return (
            self.post_belief.expected_probability
            - self.pre_belief.expected_probability
        )


# ── Updater ──────────────────────────────────────────────���──────────────


class EvidenceUpdater:
    """Processes trial outcomes into weighted edge belief updates."""

    def __init__(self, graph: GraphStore) -> None:
        self.graph = graph

    def process_trial_outcome(
        self,
        trial: TrialSubgraph,
        classification: FailureClassification,
    ) -> list[EdgeUpdate]:
        """Translate a classified trial into concrete edge updates on the graph."""
        raw = getattr(classification, "_raw", {})
        raw_edges = raw.get("edges_to_update", [])
        rule = FAILURE_MODE_RULES.get(classification.primary_failure_mode)
        evidence_type = _PHASE_TO_EVIDENCE.get(trial.phase, EvidenceType.LITERATURE)
        ev_weight = EVIDENCE_TYPE_WEIGHTS[evidence_type]

        updates: list[EdgeUpdate] = []

        for item in raw_edges:
            edge_type_str = item.get("edge_type", "")
            try:
                edge_type = EdgeType(edge_type_str)
            except ValueError:
                logger.warning("Unknown edge type '%s', skipping", edge_type_str)
                continue

            # Resolve nodes from trial subgraph
            mapping = _EDGE_TYPE_TO_SUBGRAPH_FIELDS.get(edge_type_str)
            if not mapping:
                continue
            src_field, tgt_field = mapping
            src_id = getattr(trial, src_field, None)
            tgt_id = getattr(trial, tgt_field, None)
            if not src_id or not tgt_id or src_id == "UNKNOWN" or tgt_id == "UNKNOWN":
                continue

            # Parse direction and magnitude
            direction_str = item.get("direction", "neutral")
            raw_magnitude = item.get("magnitude", 0.5)

            if direction_str == "strengthen":
                ev_direction = EvidenceDirection.SUPPORTING
            elif direction_str == "weaken":
                ev_direction = EvidenceDirection.CONTRADICTING
            else:
                ev_direction = EvidenceDirection.AMBIGUOUS

            # Cross-check with taxonomy rules
            if rule:
                if ev_direction == EvidenceDirection.CONTRADICTING and edge_type in rule.edges_to_strengthen:
                    ev_direction = EvidenceDirection.AMBIGUOUS
                elif ev_direction == EvidenceDirection.SUPPORTING and edge_type in rule.edges_to_weaken:
                    ev_direction = EvidenceDirection.AMBIGUOUS

            # Compute effective weight
            quality_str = item.get("quality", "unknown")
            quality_mult = QUALITY_WEIGHTS.get(quality_str, QUALITY_WEIGHTS["unknown"])
            effective_magnitude = raw_magnitude * classification.confidence
            effective_weight = ev_weight * quality_mult

            # Build evidence record
            evidence = EvidenceRecord(
                source_id=trial.trial_id,
                source_type=evidence_type,
                quality_score=min(classification.confidence, 1.0),
                direction=ev_direction,
                magnitude=effective_magnitude * effective_weight,
                timestamp=datetime.now(timezone.utc),
                notes=item.get("reasoning", ""),
            )

            # Get pre-update belief
            try:
                pre_belief = self.graph.get_edge_belief(src_id, tgt_id, edge_type)
            except KeyError:
                logger.debug(
                    "Edge %s -> %s (%s) not in graph, skipping",
                    src_id, tgt_id, edge_type_str,
                )
                continue

            # Apply update
            post_belief = self.graph.update_edge_belief(
                src_id, tgt_id, edge_type, evidence
            )

            updates.append(EdgeUpdate(
                source_id=src_id,
                target_id=tgt_id,
                edge_type=edge_type,
                direction=ev_direction,
                magnitude=effective_magnitude,
                weight=effective_weight,
                pre_belief=pre_belief,
                post_belief=post_belief,
            ))

        return updates
