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
    CausalChain,
    EdgeBeliefState,
    EdgeType,
    EvidenceDirection,
    EvidenceRecord,
    EvidenceType,
    TrialArm,
    TrialSubgraph,
)
from src.graph.store import GraphStore
from src.inference.beliefs import SupportBucket, bucket_to_direction
from src.annotation.taxonomy import (
    FAILURE_MODE_RULES,
    FailureClassification,
    FailureMode,
)

logger = logging.getLogger(__name__)

_ANNOTATIONS_DIR = Path("data/annotations")
# Misrouted-update audit log — written when a classifier-emitted edge update
# can't be matched to any chain in the trial subgraph. The expected use is
# vocab/extraction-prompt review, not silent drop.
_UNROUTED_LOG_PATH = Path("data/dev/unrouted_attribution_updates.jsonl")

# Map trial phase string to EvidenceType
_PHASE_TO_EVIDENCE: dict[str, EvidenceType] = {
    "1": EvidenceType.CLINICAL_PHASE1,
    "early_1": EvidenceType.CLINICAL_PHASE1,
    "2": EvidenceType.CLINICAL_PHASE2,
    "2/3": EvidenceType.CLINICAL_PHASE3,
    "3": EvidenceType.CLINICAL_PHASE3,
    "4": EvidenceType.CLINICAL_PHASE3,
}

# Node-type pairs each edge type connects (source_type, target_type). Used to
# constrain free-text entity-name → canonical-id resolution to plausible
# node types.
_EDGE_TYPE_TO_NODE_TYPES: dict[EdgeType, tuple[str, str]] = {
    EdgeType.BINDS_TO:             ("CompoundNode", "TargetNode"),
    EdgeType.MODULATES_VIA:        ("TargetNode", "MechanismNode"),
    EdgeType.MECHANISM_AFFECTS:    ("MechanismNode", "BiologyNode"),
    EdgeType.BIOLOGY_DRIVES:       ("BiologyNode", "IndicationNode"),
    EdgeType.REFLECTS_BIOLOGY:     ("BiologyNode", "EndpointNode"),
    EdgeType.ENDPOINT_CAPTURES:    ("EndpointNode", "IndicationNode"),
    EdgeType.RESPONDS_DIFFERENTLY: ("PopulationNode", "IndicationNode"),
}


# Sentinel used in CausalChain fields when a graph id wasn't yet resolved
# (e.g. by populate.py before extraction filled in the biology id).
_UNKNOWN_PLACEHOLDER = "UNKNOWN"


# ── Routing helpers ──────────────────────────────────────────────────────


_NON_ALNUM_RE = __import__("re").compile(r"[^a-z0-9]+")


def _norm_name(text: str) -> str:
    """Lowercase, strip non-alphanumerics. PD-1 / PD1 / pd_1 → 'pd1'."""
    return _NON_ALNUM_RE.sub("", (text or "").lower())


class _NameIndex:
    """Case-insensitive, punctuation-insensitive name → node-id index.

    Built once per ``attribute()`` call. ``matches`` returns True for an
    exact normalized match or a substring containment (length-gated to
    avoid 1-2 char noise). Normalization strips dashes and spaces so
    "CTLA-4" / "CTLA4" / "ctla 4" all collapse to "ctla4".
    """
    def __init__(self) -> None:
        # node_id -> list of normalized names
        self._names_by_id: dict[str, list[str]] = {}

    def add(self, node_type: str, node_id: str, names: list[str]) -> None:
        normed: list[str] = []
        for name in names:
            n = _norm_name(name)
            if n:
                normed.append(n)
        # Always include the id itself as a fallback name to match against
        # — some classifier emissions reuse the canonical id directly.
        normed.append(_norm_name(node_id))
        self._names_by_id.setdefault(node_id, []).extend(normed)

    def matches(self, node_id: str, query: str) -> bool:
        q = _norm_name(query)
        if not q:
            return False
        names = self._names_by_id.get(node_id, [])
        for name in names:
            if name == q:
                return True
        if len(q) < 3:
            return False
        for name in names:
            if len(name) < 3:
                continue
            if q in name or name in q:
                return True
        return False


def _build_name_index(
    graph: GraphStore, *, node_types: set[str]
) -> _NameIndex:
    """Build a name → node-id index for lookup at attribution time.

    For TargetNodes, also seed the index with HGNC aliases of the
    canonical gene_symbol (so a classifier emitting "PD-1" routes to
    the same node as its HUGO canonical "PDCD1"). HGNC lookup is
    best-effort — when the resolver isn't loaded the index falls back
    to name + gene_symbol only.
    """
    from src.graph.hgnc_resolver import (
        _ALIAS_TO_CANONICAL,
        canonical_symbol,
        is_loaded as hgnc_loaded,
    )

    idx = _NameIndex()

    # Reverse the HGNC dict once: canonical → list[alias]. Cheap because
    # HGNC has ~50k canonicals and we only iterate Targets here.
    canonical_to_aliases: dict[str, list[str]] = {}
    if hgnc_loaded() and _ALIAS_TO_CANONICAL is not None:
        for alias_norm, canonical in _ALIAS_TO_CANONICAL.items():
            canonical_to_aliases.setdefault(canonical, []).append(alias_norm)

    for node_type in node_types:
        for node in graph.get_nodes_by_type(node_type):
            names = [node.get("name", "")]
            if node_type == "TargetNode":
                gs = node.get("gene_symbol", "") or ""
                if gs:
                    names.append(gs)
                # Alias expansion: if gene_symbol resolves through HGNC,
                # add every known alias so any classifier-emitted variant
                # ("PD-1", "PDL1", "B7-H1") matches the same TargetNode.
                if gs:
                    canonical = canonical_symbol(gs) or gs.upper()
                    for alias_norm in canonical_to_aliases.get(canonical, []):
                        names.append(alias_norm)
            idx.add(node_type, node["id"], names)
    return idx


def _chain_edges_for_type(
    chain: CausalChain,
    arm: TrialArm,
    edge_type: EdgeType,
) -> list[tuple[str, str]]:
    """All (source_id, target_id) candidates this chain implies for the edge type.

    binds_to gets one candidate per constituent compound on the arm — that
    way the classifier-emitted ``Ipilimumab → CTLA-4`` update routes to the
    ipi mono chain (or the combo chain's ipi side), never to the nivo→PD-1
    pair.
    """
    if edge_type == EdgeType.BINDS_TO:
        return [(cid, chain.target_id) for cid in arm.compound_ids]
    if edge_type == EdgeType.MODULATES_VIA:
        return [(chain.target_id, chain.mechanism_id)]
    if edge_type == EdgeType.MECHANISM_AFFECTS:
        return [(chain.mechanism_id, chain.biology_id)]
    if edge_type == EdgeType.BIOLOGY_DRIVES:
        return [(chain.biology_id, chain.indication_id)]
    if edge_type == EdgeType.REFLECTS_BIOLOGY:
        return [(chain.biology_id, chain.endpoint_id)]
    if edge_type == EdgeType.ENDPOINT_CAPTURES:
        return [(chain.endpoint_id, chain.indication_id)]
    if edge_type == EdgeType.RESPONDS_DIFFERENTLY:
        return [(chain.subgroup_population_id, chain.indication_id)]
    return []


def _score_pair_against_names(
    src_id: str,
    tgt_id: str,
    src_name: str,
    tgt_name: str,
    name_index: _NameIndex,
) -> int:
    """Higher = better fit. Both sides must match for the pair to win.

    Score 2: both source and target match the classifier-emitted names.
    Score 1: one side matches, other side has no name to check (empty
             classifier emission). Falls back rather than dropping.
    Score 0: at least one side has a name that *doesn't* match — reject.
    """
    src_match = name_index.matches(src_id, src_name) if src_name else None
    tgt_match = name_index.matches(tgt_id, tgt_name) if tgt_name else None

    if src_match is False or tgt_match is False:
        return 0

    score = 0
    if src_match:
        score += 1
    if tgt_match:
        score += 1
    # Both names empty → unmatched (caller treats as no-route).
    return score


def _log_unrouted(trial_id: str, item: dict[str, Any], *, reason: str) -> None:
    _UNROUTED_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "trial_id": trial_id,
        "edge_type": item.get("edge_type"),
        "source_entity": item.get("source_entity"),
        "target_entity": item.get("target_entity"),
        "support": item.get("support"),
        "reason": reason,
        "logged_at": datetime.now(timezone.utc).isoformat(),
    }
    with _UNROUTED_LOG_PATH.open("a") as fh:
        fh.write(json.dumps(record) + "\n")


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
        """Translate a classification into concrete edge updates.

        Each classifier-emitted edge update is routed to the specific chain
        whose canonical ids match the classifier's free-text source/target
        entity names — preventing the misrouting bug where (e.g.)
        Ipilimumab→CTLA-4 evidence lands on the nivolumab→PD-1 edge in a
        combo trial. Updates that don't match any chain are logged to
        ``data/dev/unrouted_attribution_updates.jsonl`` rather than
        silently misapplied.
        """
        raw = getattr(classification, "_raw", {})
        raw_edges = raw.get("edges_to_update", [])
        rule = FAILURE_MODE_RULES.get(classification.primary_failure_mode)
        phase = trial.phase
        evidence_type = _PHASE_TO_EVIDENCE.get(phase, EvidenceType.LITERATURE)

        # Pre-compute name-resolution helpers from the graph's nodes (one
        # pass per node type used here).
        name_index = _build_name_index(
            self.graph,
            node_types={
                "CompoundNode", "TargetNode", "MechanismNode",
                "BiologyNode", "EndpointNode", "IndicationNode",
                "PopulationNode",
            },
        )
        arm_by_id = {arm.arm_id: arm for arm in trial.arms}

        updates: list[AppliedEdgeUpdate] = []

        for item in raw_edges:
            edge_type_str = item.get("edge_type", "")
            try:
                edge_type = EdgeType(edge_type_str)
            except ValueError:
                logger.warning("Unknown edge type '%s', skipping", edge_type_str)
                continue
            if edge_type == EdgeType.COMPOSED_OF:
                # Structural edges aren't classifier-modulable.
                continue

            support_str = item.get("support", "ambiguous")
            try:
                bucket = SupportBucket(support_str)
            except ValueError:
                logger.warning(
                    "Unknown support bucket %r, defaulting to ambiguous", support_str,
                )
                bucket = SupportBucket.AMBIGUOUS

            src_id, tgt_id = self._route_to_chain_edge(
                edge_type, trial, arm_by_id, name_index, item
            )
            if not src_id or not tgt_id:
                _log_unrouted(trial.trial_id, item, reason="no_chain_match")
                continue

            # Cross-check with taxonomy rule. If the classifier picked a
            # bucket whose coarse direction disagrees with the taxonomy's
            # expectation for this failure mode, downgrade to AMBIGUOUS —
            # the conjugate update then contributes only neutral pseudocounts.
            ev_direction = bucket_to_direction(bucket)
            if rule:
                if ev_direction == EvidenceDirection.CONTRADICTING and edge_type in rule.edges_to_strengthen:
                    logger.debug(
                        "Classifier says contradict %s but taxonomy says strengthen — using ambiguous",
                        edge_type_str,
                    )
                    bucket = SupportBucket.AMBIGUOUS
                elif ev_direction == EvidenceDirection.SUPPORTING and edge_type in rule.edges_to_weaken:
                    logger.debug(
                        "Classifier says support %s but taxonomy says weaken — using ambiguous",
                        edge_type_str,
                    )
                    bucket = SupportBucket.AMBIGUOUS

            evidence = EvidenceRecord(
                source_id=trial.trial_id,
                source_type=evidence_type,
                support=bucket.value,
                quality_score=min(classification.confidence, 1.0),
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

    def _route_to_chain_edge(
        self,
        edge_type: EdgeType,
        trial: TrialSubgraph,
        arm_by_id: dict[str, TrialArm],
        name_index: "_NameIndex",
        item: dict[str, Any],
    ) -> tuple[str | None, str | None]:
        """Pick the chain whose canonical ids best match the classifier's
        free-text source/target entity names, then return that chain's
        (source_id, target_id) for the given edge type.

        Open Targets-seeded biology_drives gets a special pass-through:
        OT writes a single (target_id, indication_id) edge during populate
        and the trial-time biology id may be a different node, so we
        update the OT-keyed edge instead.
        """
        if edge_type == EdgeType.BIOLOGY_DRIVES:
            ot_coords = trial.metadata.get("ot_biology_drives")
            if ot_coords:
                return ot_coords.get("source_id"), ot_coords.get("target_id")

        src_name = (item.get("source_entity") or "").strip()
        tgt_name = (item.get("target_entity") or "").strip()

        best: tuple[str, str] | None = None
        best_score = -1
        for chain in trial.chains:
            arm = arm_by_id.get(chain.arm_id)
            if arm is None:
                continue
            for src_id, tgt_id in _chain_edges_for_type(chain, arm, edge_type):
                if src_id == _UNKNOWN_PLACEHOLDER or tgt_id == _UNKNOWN_PLACEHOLDER:
                    continue
                score = _score_pair_against_names(
                    src_id, tgt_id, src_name, tgt_name, name_index,
                )
                if score > best_score:
                    best_score = score
                    best = (src_id, tgt_id)

        if best is None or best_score <= 0:
            return None, None
        return best


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

        # The trial subgraph (with arms + chains) must already exist in the
        # graph sidecar — produced by populate.build_trial_subgraphs and
        # extended by add_subgroup_chains during the extraction pipeline.
        try:
            trial = graph.get_trial_subgraph_by_id(trial_id)
        except KeyError:
            console.print(
                f"  [yellow]Skipped {trial_id}:[/yellow] no trial_subgraph in sidecar"
            )
            continue

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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Apply failure attributions to knowledge graph")
    parser.add_argument("--input", default="data/annotations/", help="Annotations directory")
    parser.add_argument("--graph", default="data/exports/oncology_initial.json", help="Input graph snapshot")
    parser.add_argument("--output", default="data/exports/oncology_annotated.json", help="Output graph snapshot")
    args = parser.parse_args()

    _main(args.input, args.graph, args.output)
