"""Orchestrator that constructs the initial knowledge graph."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any

import anthropic
from rich.console import Console

from src.annotation.extractor import _call_messages_with_backoff
from src.graph.models import (
    CompoundNode,
    EdgeBeliefState,
    EdgeType,
    EndpointNode,
    GraphEdge,
    IndicationNode,
    MechanismNode,
    MechanismType,
    PopulationNode,
    TargetNode,
    TrialOutcome,
    TrialSubgraph,
)
from src.graph.store import GraphStore
from src.ingestion.clinicaltrials import (
    ClinicalTrialsClient,
    TrialRecord,
    map_trial_to_graph_nodes,
)
from src.ingestion.lincs import LINCSClient, populate_lincs_signatures
from src.ingestion.opentargets import (
    OpenTargetsClient,
    populate_target_disease_edges,
)

logger = logging.getLogger(__name__)
console = Console()

# Haiku is fast + cheap for short categorical labels. The structural inferences
# (endpoint type, mechanism, population) are short — Haiku is more than enough.
INFERENCE_MODEL = "claude-haiku-4-5-20251001"


# ── Endpoint classification ──────────────────────────────────────────────


# Beta(α, β) priors per endpoint class. Higher α/β reflects more regulatory
# precedent for translating endpoint movement into disease-level benefit.
ENDPOINT_PRIORS_BY_CLASS: dict[str, tuple[float, float]] = {
    "OS": (3.0, 1.0),
    "DFS": (2.5, 1.0),
    "composite_survival": (2.5, 1.0),
    "PFS": (2.0, 1.0),
    "TTP": (2.0, 1.0),
    "CR": (1.7, 1.0),
    "ORR": (1.5, 1.0),
    "composite_response": (1.5, 1.0),
    "biomarker": (1.2, 1.0),
    "PRO": (1.0, 1.0),
    "other": (1.0, 1.0),
}
ENDPOINT_CLASSES = list(ENDPOINT_PRIORS_BY_CLASS.keys())


class JSONCache:
    """Tiny dict-style cache that flushes to a JSON file on every write.

    Used for structural LLM classifications (endpoint type, mechanism, population)
    so the same input is never paid for twice.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, str] = {}
        if path.exists():
            try:
                self._data = json.loads(path.read_text())
            except json.JSONDecodeError:
                logger.warning("Cache %s unreadable; starting empty", path)
                self._data = {}

    def get(self, key: str) -> str | None:
        return self._data.get(key)

    def set(self, key: str, value: str) -> None:
        self._data[key] = value
        self.path.write_text(json.dumps(self._data, indent=2, sort_keys=True))

    def __len__(self) -> int:
        return len(self._data)


async def classify_endpoint_with_llm(
    client: anthropic.AsyncAnthropic,
    endpoint_name: str,
    cache: JSONCache,
) -> str:
    """Classify a primary endpoint name into one of ENDPOINT_CLASSES."""
    cached = cache.get(endpoint_name)
    if cached is not None:
        return cached
    classes = ", ".join(ENDPOINT_CLASSES)
    user_msg = (
        f"Clinical trial primary endpoint: {endpoint_name!r}\n\n"
        f"Classify into ONE of: {classes}\n\n"
        "Reply with only the category name. No other text."
    )
    response = await _call_messages_with_backoff(
        client,
        model=INFERENCE_MODEL,
        max_tokens=10,
        temperature=0,
        messages=[{"role": "user", "content": user_msg}],
    )
    raw = response.content[0].text.strip()
    classification = raw if raw in ENDPOINT_CLASSES else "other"
    cache.set(endpoint_name, classification)
    return classification


def seed_endpoint_captures_edge(
    graph: GraphStore,
    endpoint_id: str,
    endpoint_name: str,
    indication_id: str,
    classification: str,
) -> bool:
    """Add an endpoint_captures edge with the prior for the given class."""
    prior = ENDPOINT_PRIORS_BY_CLASS.get(classification)
    if prior is None:
        return False
    alpha, beta_val = prior
    graph.add_edge(GraphEdge(
        source_id=endpoint_id,
        target_id=indication_id,
        edge_type=EdgeType.ENDPOINT_CAPTURES,
        belief=EdgeBeliefState(alpha=alpha, beta=beta_val),
        metadata={
            "source": "endpoint_type_prior",
            "endpoint_name": endpoint_name,
            "classification": classification,
        },
    ))
    return True


# ── Mechanism inference ──────────────────────────────────────────────────


def _sanitize_label(text: str, fallback: str = "unknown") -> str:
    """Coerce LLM free-text into a snake_case identifier safe to embed in node IDs."""
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", text.strip()).strip("_")
    return cleaned[:60] or fallback


async def infer_mechanism_for_trial(
    client: anthropic.AsyncAnthropic,
    trial: TrialRecord,
    target_node: dict[str, Any],
    cache: JSONCache,
) -> str:
    """Infer a snake_case mechanism-of-action label for a trial+target pair."""
    cached = cache.get(trial.nct_id)
    if cached is not None:
        return cached
    intervention_text = "; ".join(
        f"{iv.name}: {iv.description}".strip()
        for iv in trial.interventions
        if iv.type == "DRUG" and iv.name
    ) or "unknown"
    target_symbol = target_node.get("gene_symbol") or target_node.get("name") or ""
    user_msg = (
        f"Drug intervention: {intervention_text}\n"
        f"Target gene/protein: {target_symbol}\n\n"
        "Reply with one short snake_case mechanism-of-action label, e.g. "
        "'PD1_blockade', 'kinase_inhibition', 'CD20_depletion', 'HER2_ADC'. "
        "One label only, no other text."
    )
    response = await _call_messages_with_backoff(
        client,
        model=INFERENCE_MODEL,
        max_tokens=20,
        temperature=0,
        messages=[{"role": "user", "content": user_msg}],
    )
    label = _sanitize_label(response.content[0].text, fallback="unknown_mechanism")
    cache.set(trial.nct_id, label)
    return label


# ── Population inference ─────────────────────────────────────────────────


async def infer_population_for_trial(
    client: anthropic.AsyncAnthropic,
    trial: TrialRecord,
    cache: JSONCache,
) -> str:
    """Infer a snake_case patient-population label from title + conditions."""
    cached = cache.get(trial.nct_id)
    if cached is not None:
        return cached
    user_msg = (
        f"Trial title: {trial.title}\n"
        f"Conditions: {', '.join(trial.conditions) or 'unspecified'}\n\n"
        "Reply with one short snake_case patient-population label, e.g. "
        "'pdl1_positive_nsclc', 'her2_positive_breast', 'treatment_naive_aml', "
        "'unselected_advanced_solid'. One label only, no other text."
    )
    response = await _call_messages_with_backoff(
        client,
        model=INFERENCE_MODEL,
        max_tokens=20,
        temperature=0,
        messages=[{"role": "user", "content": user_msg}],
    )
    label = _sanitize_label(response.content[0].text, fallback="unselected")
    cache.set(trial.nct_id, label)
    return label

# Placeholder for unresolvable subgraph fields
_UNKNOWN = "UNKNOWN"


class PopulationPipeline:
    def __init__(self, graph: GraphStore) -> None:
        self.graph = graph
        self._ct_client = ClinicalTrialsClient()
        self._ot_client = OpenTargetsClient()
        # name → node_id index, keyed by (normalized_name, node_type)
        self._entity_index: dict[tuple[str, str], str] = {}

    # ── Entity resolution ────────────────────────────────────────────────

    def resolve_entity(self, name: str, entity_type: str) -> str | None:
        """Normalize a name and look up its node ID.

        Uses exact string matching + lowercasing.
        TODO: fuzzy matching / NER for better resolution.
        """
        key = (_normalize(name), entity_type)
        return self._entity_index.get(key)

    def _index_node(self, node_id: str, name: str, entity_type: str) -> None:
        key = (_normalize(name), entity_type)
        self._entity_index[key] = node_id

    # ── Main pipeline ────────────────────────────────────────────────────

    async def populate_oncology(
        self,
        max_trials: int = 500,
        include_terminated_no_results: bool = True,
    ) -> dict[str, Any]:
        # Step 1: Fetch trials
        console.print(f"[bold]Fetching up to {max_trials} oncology trials...[/bold]")
        trials = await self._ct_client.fetch_oncology_with_results(
            max_results=max_trials
        )
        console.print(f"  Fetched {len(trials)} with-results trials")

        if include_terminated_no_results:
            terminated = await self._ct_client.fetch_oncology_terminated_with_reason(
                max_results=max_trials
            )
            seen = {t.nct_id for t in trials}
            added = [t for t in terminated if t.nct_id not in seen]
            trials.extend(added)
            console.print(
                f"  +{len(added)} terminated/withdrawn trials with why_stopped"
            )

        # Step 2: Extract and add nodes per trial.
        # endpoint_captures seeding now requires an LLM classification per
        # endpoint name — the backtest pipeline does this in its own
        # _populate_graph; canonical populate keeps node creation only.
        console.print("[bold]Extracting graph nodes from trials...[/bold]")
        seen_indications: set[str] = set()
        for trial in trials:
            nodes = map_trial_to_graph_nodes(trial)
            for ind in nodes["indications"]:
                self.graph.add_node(ind)
                self._index_node(ind.id, ind.name, "indication")
                seen_indications.add(ind.name)
            for comp in nodes["compounds"]:
                self.graph.add_node(comp)
                self._index_node(comp.id, comp.name, "compound")
            for ep in nodes["endpoints"]:
                self.graph.add_node(ep)
                self._index_node(ep.id, ep.name, "endpoint")

        stats_after_nodes = self.graph.stats()
        console.print(
            f"  Nodes: {stats_after_nodes['node_count']} "
            f"({stats_after_nodes['node_types']})"
        )

        # Step 3: For each unique indication, fetch Open Targets associations
        console.print(
            f"[bold]Fetching Open Targets associations for "
            f"{len(seen_indications)} indications...[/bold]"
        )
        ot_edges_added = 0
        for ind_name in seen_indications:
            try:
                ot_edges_added += await self._fetch_ot_for_indication(ind_name)
            except Exception:
                logger.debug("Open Targets lookup failed for '%s'", ind_name, exc_info=True)
        console.print(f"  Added {ot_edges_added} biology_drives edges from Open Targets")

        # Step 4: Cross-reference compound→target binds_to edges
        console.print("[bold]Cross-referencing compound-target edges...[/bold]")
        binds_added = self._add_compound_target_edges(trials)
        console.print(f"  Added {binds_added} binds_to edges")

        # Step 5: Build trial subgraphs
        console.print("[bold]Building trial subgraphs...[/bold]")
        subgraphs = self.build_trial_subgraphs(trials)
        console.print(f"  Built {len(subgraphs)} trial subgraphs")

        # Step 6: LINCS L1000 Touchstone signatures.
        # Adds mechanism_affects evidence and materializes BiologyNodes for
        # any Reactome pathway consistently perturbed by a Touchstone
        # compound whose target+mechanism are already in the graph.
        # Skipped silently if CLUE_API_KEY is unset.
        console.print("[bold]Adding LINCS L1000 mechanism→biology evidence...[/bold]")
        try:
            lincs_client = LINCSClient()
            lincs_added = await populate_lincs_signatures(lincs_client, self.graph)
            console.print(f"  Added {lincs_added} LINCS evidence records")
        except RuntimeError as exc:
            console.print(f"  [yellow]Skipped LINCS:[/yellow] {exc}")

        # Summary
        final = self.graph.stats()
        summary = {
            "trials_fetched": len(trials),
            "compounds": final["node_types"].get("CompoundNode", 0),
            "targets": final["node_types"].get("TargetNode", 0),
            "indications": final["node_types"].get("IndicationNode", 0),
            "endpoints": final["node_types"].get("EndpointNode", 0),
            "edges": final["edge_count"],
            "trial_subgraphs": len(subgraphs),
        }
        console.print(
            f"\n[bold green]Graph populated:[/bold green] "
            f"{summary['compounds']} compounds, "
            f"{summary['targets']} targets, "
            f"{summary['indications']} indications, "
            f"{summary['edges']} edges, "
            f"{summary['trial_subgraphs']} trial subgraphs"
        )
        return summary

    async def _fetch_ot_for_indication(self, indication_name: str) -> int:
        """Search Open Targets for a disease by name and populate edges."""
        # Open Targets uses EFO IDs; we search by name to find the ID.
        # For now, use the search endpoint via the target client's _post.
        query = """
        query SearchDisease($name: String!) {
          search(queryString: $name, entityNames: ["disease"], page: {size: 1, index: 0}) {
            hits { id }
          }
        }
        """
        data = await self._ot_client._post(query, {"name": indication_name})
        hits = data["search"]["hits"]
        if not hits:
            return 0
        efo_id = hits[0]["id"]
        return await populate_target_disease_edges(
            self._ot_client, self.graph, efo_id
        )

    def _add_compound_target_edges(self, trials: list[TrialRecord]) -> int:
        """For compounds with known targets in the graph, add binds_to edges."""
        added = 0
        targets_in_graph = {
            n["id"] for n in self.graph.get_nodes_by_type("TargetNode")
        }
        for trial in trials:
            drug_interventions = [
                iv for iv in trial.interventions if iv.type == "DRUG"
            ]
            for iv in drug_interventions:
                comp_id = self.resolve_entity(iv.name, "compound")
                if not comp_id:
                    continue
                # Try to find a target whose name/symbol appears in the
                # trial title or intervention description
                text = f"{trial.title} {iv.description}".lower()
                for target_id in targets_in_graph:
                    try:
                        tdata = self.graph.get_node(target_id)
                    except KeyError:
                        continue
                    symbol = tdata.get("gene_symbol", "").lower()
                    name = tdata.get("name", "").lower()
                    if symbol and len(symbol) >= 3 and symbol in text:
                        self._add_binds_edge(comp_id, target_id)
                        added += 1
                    elif name and len(name) >= 5 and name in text:
                        self._add_binds_edge(comp_id, target_id)
                        added += 1
        return added

    def _add_binds_edge(self, compound_id: str, target_id: str) -> None:
        edge = GraphEdge(
            source_id=compound_id,
            target_id=target_id,
            edge_type=EdgeType.BINDS_TO,
            belief=EdgeBeliefState(alpha=2.0, beta=1.5),
            metadata={"source": "cross_reference", "method": "name_matching"},
        )
        self.graph.add_edge(edge)

    # ── Trial subgraph construction ──────────────────────────────────────

    def build_trial_subgraphs(
        self, trials: list[TrialRecord]
    ) -> list[TrialSubgraph]:
        subgraphs: list[TrialSubgraph] = []
        for trial in trials:
            # Resolve compound (first DRUG intervention)
            compound_id = None
            for iv in trial.interventions:
                if iv.type == "DRUG":
                    compound_id = self.resolve_entity(iv.name, "compound")
                    if compound_id:
                        break

            # Resolve indication (first condition)
            indication_id = None
            for cond in trial.conditions:
                indication_id = self.resolve_entity(cond, "indication")
                if indication_id:
                    break

            # Resolve endpoint (first primary outcome)
            endpoint_id = None
            for om in trial.primary_outcomes:
                endpoint_id = self.resolve_entity(om.measure, "endpoint")
                if endpoint_id:
                    break

            # Only build subgraph if the three core fields resolved
            if not (compound_id and indication_id and endpoint_id):
                continue

            subgraphs.append(
                TrialSubgraph(
                    trial_id=trial.nct_id,
                    compound_id=compound_id,
                    target_id=_UNKNOWN,
                    mechanism_id=_UNKNOWN,
                    biology_id=_UNKNOWN,
                    indication_id=indication_id,
                    endpoint_id=endpoint_id,
                    population_id=_UNKNOWN,
                    outcome=TrialOutcome.UNKNOWN,
                    phase=trial.phase or "unknown",
                    metadata={
                        "title": trial.title,
                        "status": trial.status,
                        "enrollment": trial.enrollment,
                    },
                )
            )
        return subgraphs


def _normalize(name: str) -> str:
    """Lowercase, strip whitespace, collapse runs of spaces."""
    return re.sub(r"\s+", " ", name.strip().lower())


# ── CLI entry point ──────────────────────────────────────────────────────


async def _main(area: str, max_trials: int) -> None:
    from pathlib import Path

    graph = GraphStore()
    pipeline = PopulationPipeline(graph)

    if area == "oncology":
        await pipeline.populate_oncology(max_trials=max_trials)
    else:
        console.print(f"[red]Unknown area: {area}[/red]")
        return

    export_path = Path("data/exports")
    export_path.mkdir(parents=True, exist_ok=True)
    snapshot = str(export_path / f"{area}_initial.json")
    graph.export_snapshot(snapshot)
    console.print(f"[bold]Snapshot saved to {snapshot}[/bold]")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Populate Eroom Bio knowledge graph")
    parser.add_argument("--area", default="oncology", help="Disease area (default: oncology)")
    parser.add_argument("--max-trials", type=int, default=500, help="Max trials to fetch")
    args = parser.parse_args()

    asyncio.run(_main(args.area, args.max_trials))
