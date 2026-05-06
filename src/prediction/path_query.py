"""Compositional path prediction: P(success) ≈ product of edge beliefs along the causal chain."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from pydantic import BaseModel, Field
from scipy import stats as sp_stats

from src.graph.models import (
    CausalChain,
    EdgeBeliefState,
    EdgeType,
    TrialOutcome,
)
from src.graph.store import GraphStore

logger = logging.getLogger(__name__)

# The canonical causal chain edges in order. Each entry maps a pair of
# CausalChain field names to its edge type.
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
    ("subgroup_population_id", "indication_id", EdgeType.RESPONDS_DIFFERENTLY),
]

_DEFAULT_BELIEF = EdgeBeliefState(alpha=1.0, beta=1.0)

_TRUST_FULL_AT = 10.0  # evidence_strength at which trust_weight saturates to 1.0
_LOG_FLOOR = 1e-12  # clip per-sample probabilities before taking log


def _trust_weight(belief: EdgeBeliefState) -> float:
    """Map an edge's evidence_strength to a [0, 1] trust weight.

    Beta(1,1) (no evidence) → 0; Beta(11,1) or stronger → 1.0.
    """
    return min(1.0, max(0.0, belief.evidence_strength) / _TRUST_FULL_AT)


def _aggregate_samples(
    edge_samples: list[np.ndarray],
    weights: list[float],
) -> np.ndarray:
    """Combine per-edge sample arrays via trust-weighted geometric mean."""
    if not edge_samples:
        return np.array([])
    n_samples = edge_samples[0].shape[0]
    sum_w = float(sum(weights))
    log_sum = np.zeros(n_samples)
    if sum_w <= 0.0:
        # No evidence anywhere — back off to unweighted geomean so we
        # don't blow up; conceptually "everyone abstains, take the mean
        # of the priors."
        for s in edge_samples:
            log_sum += np.log(np.clip(s, _LOG_FLOOR, 1.0))
        return np.exp(log_sum / len(edge_samples))
    for s, w in zip(edge_samples, weights):
        if w <= 0.0:
            continue
        log_sum += w * np.log(np.clip(s, _LOG_FLOOR, 1.0))
    return np.exp(log_sum / sum_w)


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
        self,
        chain: CausalChain,
        n_samples: int = 10_000,
    ) -> PredictionResult:
        """Compositional prediction via Monte Carlo sampling along one causal chain.

        Aggregation: trust-weighted geometric mean. Edges with no evidence
        beyond the prior contribute little; edges with substantial evidence
        dominate. Trial-level prediction (across multiple arms × subgroups)
        is the caller's responsibility — predict each chain and aggregate
        as appropriate (e.g. per arm, per subgroup, or trial-wide).
        """
        # 1. Collect edges and their beliefs
        edges = self._collect_edges(chain)

        # 2. Sample from each edge's Beta and compute per-edge trust weights
        rng = np.random.default_rng()
        edge_samples: list[np.ndarray] = []
        trust_weights: list[float] = []
        for _src, _tgt, _etype, belief in edges:
            edge_samples.append(rng.beta(belief.alpha, belief.beta, size=n_samples))
            trust_weights.append(_trust_weight(belief))

        # 3. Aggregate samples via trust-weighted geometric mean
        samples = _aggregate_samples(edge_samples, trust_weights)

        # 4. Compute statistics
        if samples.size:
            overall_prob = float(np.mean(samples))
            ci_lower = float(np.percentile(samples, 2.5))
            ci_upper = float(np.percentile(samples, 97.5))
        else:
            overall_prob, ci_lower, ci_upper = 0.5, 0.0, 1.0

        # 5. Build edge contributions. Bottleneck score is now weighted by
        #    trust: an uncontradicted Beta(1,1) edge is not a "weak link",
        #    just unobserved.
        contributions: list[EdgeContribution] = []
        for i, (src, tgt, etype, belief) in enumerate(edges):
            sampled_mean = float(np.mean(edge_samples[i])) if edge_samples[i].size else belief.expected_probability
            bottleneck = (1.0 - belief.expected_probability) * trust_weights[i]
            contributions.append(EdgeContribution(
                source_id=src,
                target_id=tgt,
                edge_type=etype,
                belief=belief,
                sampled_mean=sampled_mean,
                bottleneck_score=bottleneck,
            ))

        # 6. Identify weakest link. Tiebreak on raw (1 - mean) so all-zero
        #    bottleneck scores still produce a sensible pick.
        weakest = (
            max(
                contributions,
                key=lambda c: (
                    c.bottleneck_score,
                    1.0 - c.belief.expected_probability,
                ),
            )
            if contributions
            else None
        )

        hypothesis = (
            f"{chain.compound_id} -> {chain.target_id} -> "
            f"{chain.mechanism_id} -> {chain.biology_id} -> "
            f"{chain.indication_id}"
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
        self,
        chains: list[CausalChain],
        n_samples: int = 10_000,
    ) -> list[PredictionResult]:
        """Predict and rank multiple chains by probability."""
        results = [self.predict(chain, n_samples=n_samples) for chain in chains]
        results.sort(key=lambda r: r.overall_probability, reverse=True)
        return results

    def suggest_improvements(self, result: PredictionResult) -> list[str]:
        """Suggest which edges to strengthen based on evidence + bottleneck analysis.

        Three categories, evaluated independently per edge:
          - DATA GAP: evidence_strength < 2.0 (priors only, regardless of mean).
          - WEAK LINK: evidence_strength ≥ 2.0 and bottleneck_score > 0.5.
          - MODERATE: evidence_strength ≥ 2.0 and 0.2 ≤ bottleneck_score ≤ 0.5.

        Under weighted-geomean prediction, low-trust edges have bottleneck_score
        near 0, so DATA GAP must be detected from evidence_strength rather than
        from the bottleneck ranking.
        """
        suggestions: list[str] = []
        if not result.edge_contributions:
            return ["No edges in the causal chain to evaluate."]

        ranked = sorted(
            result.edge_contributions,
            key=lambda c: (
                c.belief.evidence_strength < 2.0,  # data gaps last
                -c.bottleneck_score,                 # then by bottleneck desc
            ),
        )

        for ec in ranked:
            belief_str = f"Beta({ec.belief.alpha:.1f}, {ec.belief.beta:.1f})"
            p = ec.belief.expected_probability
            if ec.belief.evidence_strength < 2.0:
                suggestions.append(
                    f"[DATA GAP] {ec.edge_type.value} ({ec.source_id} → {ec.target_id}): "
                    f"P={p:.2f} {belief_str} — insufficient evidence. "
                    f"Need direct experimental validation."
                )
                continue
            if ec.bottleneck_score > 0.5:
                suggestions.append(
                    f"[WEAK LINK] {ec.edge_type.value} ({ec.source_id} → {ec.target_id}): "
                    f"P={p:.2f} {belief_str} — evidence contradicts this link. "
                    f"Consider alternative targets or mechanisms."
                )
            elif ec.bottleneck_score >= 0.2:
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
        self, chain: CausalChain
    ) -> list[tuple[str, str, EdgeType, EdgeBeliefState]]:
        """Collect belief states for all edges in the causal chain.

        For ``mechanism_affects`` specifically, retrieves a belief that has
        been conditioned on the indication's relevant tissues — so cell-line
        evidence from the wrong tissue (e.g. a melanoma signature when the
        trial is in NSCLC) gets downweighted rather than counted equally.
        Other edge types are context-free at retrieval.
        """
        edges: list[tuple[str, str, EdgeType, EdgeBeliefState]] = []
        relevant_tissues = self._tissues_for_chain(chain)

        for src_field, tgt_field, edge_type in _CAUSAL_CHAIN + _AUXILIARY_EDGES:
            src_id = getattr(chain, src_field)
            tgt_id = getattr(chain, tgt_field)

            if src_id == "UNKNOWN" or tgt_id == "UNKNOWN":
                continue

            try:
                if edge_type == EdgeType.MECHANISM_AFFECTS and relevant_tissues:
                    belief = self.graph.get_edge_belief_conditioned(
                        src_id, tgt_id, edge_type, relevant_tissues
                    )
                else:
                    belief = self.graph.get_edge_belief(src_id, tgt_id, edge_type)
            except KeyError:
                belief = _DEFAULT_BELIEF

            edges.append((src_id, tgt_id, edge_type, belief))

        return edges

    def _tissues_for_chain(self, chain: CausalChain) -> set[str]:
        """Resolve the chain's indication name to the tissues whose
        cell-line evidence is relevant. Empty set = no conditioning.
        """
        if chain.indication_id == "UNKNOWN":
            return set()
        try:
            ind_node = self.graph.get_node(chain.indication_id)
        except KeyError:
            return set()
        # Lazy import to avoid src.prediction → src.ingestion dependency at
        # module import time.
        from src.ingestion.lincs import tissues_for_indication_name

        return tissues_for_indication_name(ind_node.get("name"))


# ── Stateless query: compound + indication → full chain prediction ─────


def _resolve_target_for_compound(
    graph: GraphStore, compound_id: str
) -> str:
    """Pick the most-supported binds_to target of a compound.

    Tiebreaks on belief.expected_probability * evidence_strength so an edge
    with real evidence beats a Beta(1,1) placeholder. Returns ``"UNKNOWN"``
    if the compound has no binds_to neighbors.
    """
    g = graph._graph
    if compound_id not in g:
        return "UNKNOWN"
    best_id = "UNKNOWN"
    best_score = -1.0
    for _u, v, key, data in g.out_edges(compound_id, data=True, keys=True):
        if key != EdgeType.BINDS_TO.value:
            continue
        belief_data = data.get("belief") or {}
        try:
            belief = EdgeBeliefState.model_validate(belief_data)
        except Exception:  # noqa: BLE001 — defensive against legacy snapshots
            belief = _DEFAULT_BELIEF
        score = belief.expected_probability * (1.0 + belief.evidence_strength)
        if score > best_score:
            best_score = score
            best_id = v
    return best_id


def _resolve_chain_via_topology(
    graph: GraphStore, target_id: str, indication_id: str
) -> tuple[str, str]:
    """Walk simple paths target → indication and label mechanism / biology.

    Mirrors the resolution semantics used by the backtest's subgraph builder:
      - modulates_via       : v is mechanism
      - mechanism_affects   : u is mechanism, v is biology
      - biology_drives      : u is biology
    Picks the path that resolves the most nodes; ties go to the first
    encountered. Returns ("UNKNOWN", "UNKNOWN") if no path exists.

    Falls back to the first modulates_via neighbor of the target when no
    simple path resolves a mechanism — in graphs where target→mechanism
    edges are dead-ends (no mechanism→indication wiring), this is the only
    way to recover the mechanism node.
    """
    g = graph._graph
    if target_id == "UNKNOWN" or indication_id == "UNKNOWN":
        return "UNKNOWN", "UNKNOWN"
    if target_id not in g or indication_id not in g:
        return "UNKNOWN", "UNKNOWN"
    try:
        paths = list(
            nx.all_simple_paths(g, target_id, indication_id, cutoff=3)
        )
    except nx.NodeNotFound:
        paths = []

    best_mech, best_bio = "UNKNOWN", "UNKNOWN"
    best_score = -1
    for path in paths:
        mech, bio = "UNKNOWN", "UNKNOWN"
        for u, v in zip(path[:-1], path[1:]):
            edges_between = g.get_edge_data(u, v) or {}
            for key in edges_between:
                if key == EdgeType.MODULATES_VIA.value and mech == "UNKNOWN":
                    mech = v
                elif key == EdgeType.MECHANISM_AFFECTS.value:
                    if mech == "UNKNOWN":
                        mech = u
                    if bio == "UNKNOWN":
                        bio = v
                elif key == EdgeType.BIOLOGY_DRIVES.value and bio == "UNKNOWN":
                    bio = u
        score = int(mech != "UNKNOWN") + int(bio != "UNKNOWN")
        if score > best_score:
            best_score = score
            best_mech, best_bio = mech, bio

    if best_mech == "UNKNOWN":
        for _, mid, key in g.out_edges(target_id, keys=True):
            if key == EdgeType.MODULATES_VIA.value:
                best_mech = mid
                break

    return best_mech, best_bio


def predict_clinical_hypothesis(
    graph: GraphStore,
    compound_id: str,
    indication_id: str,
    *,
    endpoint_id: str | None = None,
    population_id: str | None = None,
    n_samples: int = 10_000,
) -> PredictionResult:
    """Stateless prediction for a (compound, indication) pair.

    Walks the graph to assemble the full causal chain — target via binds_to,
    then mechanism + biology by walking target→indication paths — and runs
    the standard engine on the resulting subgraph. No in-memory trial cache
    needed: the graph snapshot itself is the source of truth.

    ``endpoint_id`` / ``population_id`` are optional. When omitted, the
    auxiliary edges (``reflects_biology``, ``endpoint_captures``,
    ``responds_differently``) are skipped at engine time — so the prediction
    reflects the causal chain only, not endpoint translatability or
    population responsiveness.

    Raises ``KeyError`` if either node is not in the graph; this is a
    programmer error worth surfacing rather than silently returning a flat
    prediction.
    """
    if compound_id not in graph._graph:
        raise KeyError(f"Compound '{compound_id}' not in graph")
    if indication_id not in graph._graph:
        raise KeyError(f"Indication '{indication_id}' not in graph")

    target_id = _resolve_target_for_compound(graph, compound_id)
    mechanism_id, biology_id = _resolve_chain_via_topology(
        graph, target_id, indication_id
    )

    chain = CausalChain(
        arm_id="hypothesis",
        compound_id=compound_id,
        subgroup_population_id=population_id or "UNKNOWN",
        target_id=target_id,
        mechanism_id=mechanism_id,
        biology_id=biology_id,
        indication_id=indication_id,
        endpoint_id=endpoint_id or "UNKNOWN",
        outcome=TrialOutcome.UNKNOWN,
    )
    engine = PredictionEngine(graph)
    return engine.predict(chain, n_samples=n_samples)


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

    chain = CausalChain(
        arm_id="cli_query",
        compound_id=compound_id,
        subgroup_population_id="UNKNOWN",
        target_id=target_id,
        mechanism_id="UNKNOWN",
        biology_id="UNKNOWN",
        indication_id=indication_id,
        endpoint_id=endpoint_id,
        outcome=TrialOutcome.UNKNOWN,
    )

    engine = PredictionEngine(graph)
    result = engine.predict(chain)

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
