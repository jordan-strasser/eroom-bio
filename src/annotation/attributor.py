"""Translate failure classifications into graph edge updates."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src.graph.models import (
    EdgeBeliefState,
    EdgeType,
    EvidenceDirection,
    EvidenceRecord,
    EvidenceType,
    TrialSubgraph,
)
from src.graph.store import GraphStore
from src.annotation.taxonomy import (
    FAILURE_MODE_RULES,
    FailureClassification,
    FailureMode,
)

logger = logging.getLogger(__name__)

_ANNOTATIONS_DIR = Path("data/annotations")

# Map trial phase string to EvidenceType
_PHASE_TO_EVIDENCE: dict[str, EvidenceType] = {
    "1": EvidenceType.CLINICAL_PHASE1,
    "early_1": EvidenceType.CLINICAL_PHASE1,
    "2": EvidenceType.CLINICAL_PHASE2,
    "2/3": EvidenceType.CLINICAL_PHASE3,
    "3": EvidenceType.CLINICAL_PHASE3,
    "4": EvidenceType.CLINICAL_PHASE3,
}

# Map edge_type string to the (source_field, target_field) on TrialSubgraph
_EDGE_TYPE_TO_SUBGRAPH_FIELDS: dict[str, tuple[str, str]] = {
    "binds_to": ("compound_id", "target_id"),
    "modulates_via": ("target_id", "mechanism_id"),
    "mechanism_affects": ("mechanism_id", "biology_id"),
    "biology_drives": ("biology_id", "indication_id"),
    "reflects_biology": ("biology_id", "endpoint_id"),
    "endpoint_captures": ("endpoint_id", "indication_id"),
    "responds_differently": ("population_id", "indication_id"),
}


# ── Output model ─────────────────────────────────────────────────────────


class AppliedEdgeUpdate(BaseModel):
    """A single edge update that was applied to the graph."""

    source_id: str
    target_id: str
    edge_type: EdgeType
    evidence: EvidenceRecord
    pre_update_belief: EdgeBeliefState
    post_update_belief: EdgeBeliefState

    @property
    def probability_change(self) -> float:
        return (
            self.post_update_belief.expected_probability
            - self.pre_update_belief.expected_probability
        )


# ── Attributor ───────────────────────────────────────────────────────────


class Attributor:
    def __init__(self, graph: GraphStore) -> None:
        self.graph = graph

    def attribute(
        self,
        classification: FailureClassification,
        trial: TrialSubgraph,
    ) -> list[AppliedEdgeUpdate]:
        """Translate a classification into concrete edge updates."""
        raw = getattr(classification, "_raw", {})
        raw_edges = raw.get("edges_to_update", [])
        rule = FAILURE_MODE_RULES.get(classification.primary_failure_mode)
        phase = trial.phase
        evidence_type = _PHASE_TO_EVIDENCE.get(phase, EvidenceType.LITERATURE)

        updates: list[AppliedEdgeUpdate] = []

        for item in raw_edges:
            edge_type_str = item.get("edge_type", "")
            try:
                edge_type = EdgeType(edge_type_str)
            except ValueError:
                logger.warning("Unknown edge type '%s', skipping", edge_type_str)
                continue

            direction_str = item.get("direction", "neutral")
            magnitude = item.get("magnitude", 0.5)

            # Resolve entity names to node IDs via trial subgraph
            src_id, tgt_id = self._resolve_edge_nodes(
                edge_type, trial, item
            )
            if not src_id or not tgt_id:
                logger.debug(
                    "Could not resolve nodes for %s edge, skipping", edge_type_str
                )
                continue

            # Map direction to EvidenceDirection
            if direction_str == "strengthen":
                ev_direction = EvidenceDirection.SUPPORTING
            elif direction_str == "weaken":
                ev_direction = EvidenceDirection.CONTRADICTING
            else:
                ev_direction = EvidenceDirection.AMBIGUOUS

            # Cross-check with taxonomy rule
            if rule:
                if ev_direction == EvidenceDirection.CONTRADICTING and edge_type in rule.edges_to_strengthen:
                    logger.debug(
                        "Classifier says weaken %s but taxonomy says strengthen — using ambiguous",
                        edge_type_str,
                    )
                    ev_direction = EvidenceDirection.AMBIGUOUS
                elif ev_direction == EvidenceDirection.SUPPORTING and edge_type in rule.edges_to_weaken:
                    logger.debug(
                        "Classifier says strengthen %s but taxonomy says weaken — using ambiguous",
                        edge_type_str,
                    )
                    ev_direction = EvidenceDirection.AMBIGUOUS

            evidence = EvidenceRecord(
                source_id=trial.trial_id,
                source_type=evidence_type,
                quality_score=min(classification.confidence, 1.0),
                direction=ev_direction,
                magnitude=magnitude * classification.confidence,
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

            updates.append(AppliedEdgeUpdate(
                source_id=src_id,
                target_id=tgt_id,
                edge_type=edge_type,
                evidence=evidence,
                pre_update_belief=pre_belief,
                post_update_belief=post_belief,
            ))

        return updates

    def apply_updates(
        self, updates: list[AppliedEdgeUpdate]
    ) -> dict[str, Any]:
        """Summarize applied updates (already applied during attribute())."""
        if not updates:
            return {"edges_updated": 0, "largest_changes": []}

        sorted_updates = sorted(
            updates, key=lambda u: abs(u.probability_change), reverse=True
        )

        largest = [
            {
                "edge": f"{u.source_id} -> {u.target_id} ({u.edge_type.value})",
                "pre": round(u.pre_update_belief.expected_probability, 4),
                "post": round(u.post_update_belief.expected_probability, 4),
                "change": round(u.probability_change, 4),
            }
            for u in sorted_updates[:5]
        ]

        return {
            "edges_updated": len(updates),
            "largest_changes": largest,
        }

    def _resolve_edge_nodes(
        self,
        edge_type: EdgeType,
        trial: TrialSubgraph,
        item: dict[str, Any],
    ) -> tuple[str | None, str | None]:
        """Resolve source/target entity names to graph node IDs."""
        # Open Targets seeds biology_drives as target_id → indication_id.
        # The subgraph resolver stashes those coordinates in metadata so we
        # can update the real OT edge here instead of a phantom one keyed
        # on the trial's (BIO/MECH-labelled) biology_id.
        if edge_type == EdgeType.BIOLOGY_DRIVES:
            ot_coords = trial.metadata.get("ot_biology_drives")
            if ot_coords:
                return ot_coords.get("source_id"), ot_coords.get("target_id")

        mapping = _EDGE_TYPE_TO_SUBGRAPH_FIELDS.get(edge_type.value)
        if not mapping:
            return None, None

        src_field, tgt_field = mapping
        src_id = getattr(trial, src_field, None)
        tgt_id = getattr(trial, tgt_field, None)

        # Skip UNKNOWN placeholders
        if src_id == "UNKNOWN" or tgt_id == "UNKNOWN":
            return None, None

        return src_id, tgt_id


# ── CLI ──────────────────────────────────────────────────────────────────


def _load_classifications(annotations_dir: Path) -> list[tuple[dict, dict]]:
    """Load paired extraction + classification JSONs."""
    pairs = []
    for clf_path in sorted(annotations_dir.glob("*_classification.json")):
        nct_id = clf_path.stem.replace("_classification", "")
        ext_path = annotations_dir / f"{nct_id}_extraction.json"
        if ext_path.exists():
            clf_data = json.loads(clf_path.read_text())
            ext_data = json.loads(ext_path.read_text())
            pairs.append((ext_data, clf_data))
    return pairs


def _main(annotations_dir: str, graph_path: str, output_path: str) -> None:
    from rich.console import Console

    from src.annotation.taxonomy import TrialExtraction
    from src.graph.models import TrialOutcome

    console = Console()

    # Load graph
    graph = GraphStore()
    graph_file = Path(graph_path)
    if graph_file.exists():
        console.print(f"[bold]Loading graph from {graph_path}...[/bold]")
        graph.import_snapshot(graph_path)
        stats = graph.stats()
        console.print(f"  Loaded: {stats['node_count']} nodes, {stats['edge_count']} edges")
    else:
        console.print(f"[yellow]Graph file not found: {graph_path}[/yellow]")
        return

    attributor = Attributor(graph)
    pairs = _load_classifications(Path(annotations_dir))
    console.print(f"\n[bold]Found {len(pairs)} annotated trials[/bold]")

    total_updates: list[AppliedEdgeUpdate] = []

    for ext_data, clf_data in pairs:
        trial_id = clf_data.get("nct_id", ext_data.get("nct_id", "unknown"))

        # Build a minimal TrialSubgraph — use graph node lookups where possible
        # For now, use the extraction data to find nodes
        hypothesis = ext_data.get("therapeutic_hypothesis", {})
        compound_name = hypothesis.get("compound", "")
        target_name = hypothesis.get("claimed_target", "")

        # Try to find matching nodes
        compound_id = _find_node_by_name(graph, compound_name, "CompoundNode") or "UNKNOWN"
        target_id = _find_node_by_name(graph, target_name, "TargetNode") or "UNKNOWN"
        indication_id = _find_node_by_name(
            graph,
            ext_data.get("therapeutic_hypothesis", {}).get("target_population", ""),
            "IndicationNode",
        ) or "UNKNOWN"

        trial = TrialSubgraph(
            trial_id=trial_id,
            compound_id=compound_id,
            target_id=target_id,
            mechanism_id="UNKNOWN",
            biology_id="UNKNOWN",
            indication_id=indication_id,
            endpoint_id="UNKNOWN",
            population_id="UNKNOWN",
            outcome=TrialOutcome.UNKNOWN,
            phase=ext_data.get("therapeutic_hypothesis", {}).get("phase", "3") or "3",
        )

        # Build classification
        modes = clf_data.get("failure_modes", [])
        primary_mode = FailureMode.INSUFFICIENT_INFORMATION
        if modes:
            sorted_modes = sorted(modes, key=lambda m: m.get("confidence", 0), reverse=True)
            try:
                primary_mode = FailureMode(sorted_modes[0]["mode"])
            except (ValueError, KeyError):
                pass

        classification = FailureClassification(
            trial_id=trial_id,
            primary_failure_mode=primary_mode,
            confidence=clf_data.get("confidence_overall", 0.5),
            reasoning=clf_data.get("reasoning", ""),
        )
        classification._raw = clf_data  # type: ignore[attr-defined]

        updates = attributor.attribute(classification, trial)
        total_updates.extend(updates)
        if updates:
            console.print(f"  {trial_id}: {len(updates)} edge updates")

    # Summary
    summary = attributor.apply_updates(total_updates)
    console.print(f"\n[bold green]Processed {len(pairs)} trials. Updated {summary['edges_updated']} edges.[/bold green]")

    if summary["largest_changes"]:
        console.print("[bold]Largest changes:[/bold]")
        for change in summary["largest_changes"]:
            direction = "+" if change["change"] > 0 else ""
            console.print(
                f"  {change['edge']}: {change['pre']:.4f} → {change['post']:.4f} "
                f"({direction}{change['change']:.4f})"
            )

    # Save updated graph
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    graph.export_snapshot(output_path)
    console.print(f"\n[bold]Saved annotated graph to {output_path}[/bold]")


def _find_node_by_name(
    graph: GraphStore, name: str, node_type: str
) -> str | None:
    """Find a node ID by name substring match."""
    if not name:
        return None
    name_lower = name.lower()
    for node in graph.get_nodes_by_type(node_type):
        node_name = node.get("name", "").lower()
        if node_name and (name_lower in node_name or node_name in name_lower):
            return node.get("id")
    return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Apply failure attributions to knowledge graph")
    parser.add_argument("--input", default="data/annotations/", help="Annotations directory")
    parser.add_argument("--graph", default="data/exports/oncology_initial.json", help="Input graph snapshot")
    parser.add_argument("--output", default="data/exports/oncology_annotated.json", help="Output graph snapshot")
    args = parser.parse_args()

    _main(args.input, args.graph, args.output)
