"""Compositional path prediction: P(success) ≈ product of edge beliefs along the causal chain."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field
from scipy import stats as sp_stats

from src.graph.models import (
    EdgeBeliefState,
    EdgeType,
    TrialOutcome,
    TrialSubgraph,
)
from src.graph.store import GraphStore

logger = logging.getLogger(__name__)

# The canonical causal chain edges in order
_CAUSAL_CHAIN: list[tuple[str, str, EdgeType]] = [
    ("compound_id", "target_id", EdgeType.BINDS_TO),
    ("target_id", "mechanism_id", EdgeType.MODULATES_VIA),
    ("mechanism_id", "biology_id", EdgeType.MECHANISM_AFFECTS),
    ("biology_id", "indication_id", EdgeType.BIOLOGY_DRIVES),
]

# Auxiliary edges that contribute to the prediction
_AUXILIARY_EDGES: list[tuple[str, str, EdgeType]] = [
    ("biology_id", "endpoint_id", EdgeType.REFLECTS_BIOLOGY),
    ("endpoint_id", "indication_id", EdgeType.ENDPOINT_CAPTURES),
    ("population_id", "indication_id", EdgeType.RESPONDS_DIFFERENTLY),
]

_DEFAULT_BELIEF = EdgeBeliefState(alpha=1.0, beta=1.0)


# ── Models ──────────────────────────────────────────────────────────────


class EdgeContribution(BaseModel):
    """One edge's contribution to the prediction."""

    source_id: str
    target_id: str
    edge_type: EdgeType
    belief: EdgeBeliefState
    sampled_mean: float
    bottleneck_score: float = Field(
        description="1 - belief.mean; higher = weaker link"
    )


class PredictionResult(BaseModel):
    """Full prediction for a trial hypothesis."""

    trial_hypothesis: str
    overall_probability: float
    ci_lower: float
    ci_upper: float
    edge_contributions: list[EdgeContribution]
    weakest_link: EdgeContribution | None
    n_samples: int


# ── Engine ──────────────────────────────────────────────────────────────


class PredictionEngine:
    def __init__(self, graph: GraphStore) -> None:
        self.graph = graph

    def predict(
        self, trial: TrialSubgraph, n_samples: int = 10_000
    ) -> PredictionResult:
        """Compositional prediction via Monte Carlo sampling along the causal chain."""
        # 1. Collect edges and their beliefs
        edges = self._collect_edges(trial)

        # 2. Sample from each Beta and multiply
        rng = np.random.default_rng()
        samples = np.ones(n_samples)
        edge_samples: list[np.ndarray] = []

        for _src, _tgt, _etype, belief in edges:
            s = rng.beta(belief.alpha, belief.beta, size=n_samples)
            samples *= s
            edge_samples.append(s)

        # 3. Compute statistics
        overall_prob = float(np.mean(samples))
        ci_lower = float(np.percentile(samples, 2.5))
        ci_upper = float(np.percentile(samples, 97.5))

        # 4. Build edge contributions
        contributions: list[EdgeContribution] = []
        for i, (src, tgt, etype, belief) in enumerate(edges):
            sampled_mean = float(np.mean(edge_samples[i]))
            contributions.append(EdgeContribution(
                source_id=src,
                target_id=tgt,
                edge_type=etype,
                belief=belief,
                sampled_mean=sampled_mean,
                bottleneck_score=1.0 - belief.expected_probability,
            ))

        # 5. Identify weakest link
        weakest = max(contributions, key=lambda c: c.bottleneck_score) if contributions else None

        hypothesis = (
            f"{trial.compound_id} -> {trial.target_id} -> "
            f"{trial.mechanism_id} -> {trial.biology_id} -> "
            f"{trial.indication_id}"
        )

        return PredictionResult(
            trial_hypothesis=hypothesis,
            overall_probability=overall_prob,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            edge_contributions=contributions,
            weakest_link=weakest,
            n_samples=n_samples,
        )

    def compare_hypotheses(
        self, trials: list[TrialSubgraph], n_samples: int = 10_000
    ) -> list[PredictionResult]:
        """Predict and rank multiple hypotheses by probability."""
        results = [self.predict(trial, n_samples=n_samples) for trial in trials]
        results.sort(key=lambda r: r.overall_probability, reverse=True)
        return results

    def suggest_improvements(self, result: PredictionResult) -> list[str]:
        """Suggest which edges to strengthen based on bottleneck analysis."""
        suggestions: list[str] = []
        if not result.edge_contributions:
            return ["No edges in the causal chain to evaluate."]

        # Sort by bottleneck score (highest = most room for improvement)
        ranked = sorted(
            result.edge_contributions,
            key=lambda c: c.bottleneck_score,
            reverse=True,
        )

        for ec in ranked:
            if ec.bottleneck_score < 0.2:
                break  # Already strong enough
            belief_str = f"Beta({ec.belief.alpha:.1f}, {ec.belief.beta:.1f})"
            p = ec.belief.expected_probability
            if ec.belief.evidence_strength < 2.0:
                suggestions.append(
                    f"[DATA GAP] {ec.edge_type.value} ({ec.source_id} → {ec.target_id}): "
                    f"P={p:.2f} {belief_str} — insufficient evidence. "
                    f"Need direct experimental validation."
                )
            elif ec.bottleneck_score > 0.5:
                suggestions.append(
                    f"[WEAK LINK] {ec.edge_type.value} ({ec.source_id} → {ec.target_id}): "
                    f"P={p:.2f} {belief_str} — evidence contradicts this link. "
                    f"Consider alternative targets or mechanisms."
                )
            else:
                suggestions.append(
                    f"[MODERATE] {ec.edge_type.value} ({ec.source_id} → {ec.target_id}): "
                    f"P={p:.2f} {belief_str} — could be strengthened with "
                    f"additional supporting evidence."
                )

        if not suggestions:
            suggestions.append(
                f"All edges strong. Overall P={result.overall_probability:.3f}. "
                f"Consider expanding to new indications."
            )

        return suggestions

    def _collect_edges(
        self, trial: TrialSubgraph
    ) -> list[tuple[str, str, EdgeType, EdgeBeliefState]]:
        """Collect belief states for all edges in the causal chain."""
        edges: list[tuple[str, str, EdgeType, EdgeBeliefState]] = []

        for src_field, tgt_field, edge_type in _CAUSAL_CHAIN + _AUXILIARY_EDGES:
            src_id = getattr(trial, src_field)
            tgt_id = getattr(trial, tgt_field)

            if src_id == "UNKNOWN" or tgt_id == "UNKNOWN":
                continue

            try:
                belief = self.graph.get_edge_belief(src_id, tgt_id, edge_type)
            except KeyError:
                belief = _DEFAULT_BELIEF

            edges.append((src_id, tgt_id, edge_type, belief))

        return edges


# ── CLI ─────────────────────────────────────────────────────────────────


def _find_node_by_name(
    graph: GraphStore, name: str, node_type: str
) -> str | None:
    if not name:
        return None
    name_lower = name.lower()
    for node in graph.get_nodes_by_type(node_type):
        node_name = node.get("name", "").lower()
        if node_name and (name_lower in node_name or node_name in name_lower):
            return node.get("id")
    return None


def _main(
    graph_path: str,
    compound: str,
    target: str,
    indication: str,
    endpoint: str | None,
) -> None:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()

    # Load graph
    graph = GraphStore()
    graph_file = Path(graph_path)
    if not graph_file.exists():
        console.print(f"[red]Graph file not found: {graph_path}[/red]")
        return
    console.print(f"[bold]Loading graph from {graph_path}...[/bold]")
    graph.import_snapshot(graph_path)
    stats = graph.stats()
    console.print(
        f"  Loaded: {stats['node_count']} nodes, {stats['edge_count']} edges"
    )

    # Resolve names to IDs
    compound_id = _find_node_by_name(graph, compound, "CompoundNode") or "UNKNOWN"
    target_id = _find_node_by_name(graph, target, "TargetNode") or "UNKNOWN"
    indication_id = _find_node_by_name(graph, indication, "IndicationNode") or "UNKNOWN"
    endpoint_id = _find_node_by_name(graph, endpoint or "", "EndpointNode") or "UNKNOWN"

    if compound_id == "UNKNOWN":
        console.print(f"[red]Compound '{compound}' not found in graph[/red]")
        return
    if target_id == "UNKNOWN":
        console.print(f"[red]Target '{target}' not found in graph[/red]")
        return

    console.print(f"\n  Compound: {compound} → {compound_id}")
    console.print(f"  Target: {target} → {target_id}")
    console.print(f"  Indication: {indication} → {indication_id}")
    console.print(f"  Endpoint: {endpoint or 'any'} → {endpoint_id}")

    trial = TrialSubgraph(
        trial_id="prediction_query",
        compound_id=compound_id,
        target_id=target_id,
        mechanism_id="UNKNOWN",
        biology_id="UNKNOWN",
        indication_id=indication_id,
        endpoint_id=endpoint_id,
        population_id="UNKNOWN",
        outcome=TrialOutcome.UNKNOWN,
        phase="3",
    )

    engine = PredictionEngine(graph)
    result = engine.predict(trial)

    # Display results
    console.print(Panel(
        f"[bold]P(success) = {result.overall_probability:.3f}[/bold]\n"
        f"95% CI: [{result.ci_lower:.3f}, {result.ci_upper:.3f}]\n"
        f"Samples: {result.n_samples:,}",
        title="Prediction",
    ))

    if result.edge_contributions:
        table = Table(title="Edge Contributions")
        table.add_column("Edge Type")
        table.add_column("Source → Target")
        table.add_column("P(edge)")
        table.add_column("Bottleneck")
        table.add_column("Evidence")

        for ec in result.edge_contributions:
            is_weakest = ec == result.weakest_link
            style = "bold red" if is_weakest else ""
            table.add_row(
                ec.edge_type.value,
                f"{ec.source_id} → {ec.target_id}",
                f"{ec.belief.expected_probability:.3f}",
                f"{ec.bottleneck_score:.3f}" + (" ← WEAKEST" if is_weakest else ""),
                f"Beta({ec.belief.alpha:.1f}, {ec.belief.beta:.1f})",
                style=style,
            )
        console.print(table)

    suggestions = engine.suggest_improvements(result)
    if suggestions:
        console.print("\n[bold]Suggestions:[/bold]")
        for s in suggestions:
            console.print(f"  {s}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Predict trial success probability"
    )
    parser.add_argument(
        "--graph",
        default="data/exports/oncology_annotated.json",
        help="Graph snapshot path",
    )
    parser.add_argument("--compound", required=True, help="Compound name")
    parser.add_argument("--target", required=True, help="Target name")
    parser.add_argument("--indication", required=True, help="Indication name")
    parser.add_argument("--endpoint", default=None, help="Endpoint name")
    args = parser.parse_args()

    _main(args.graph, args.compound, args.target, args.indication, args.endpoint)
