"""Trace a trial through the whole pipeline.

For a given NCT id, prints in order:

  1. RAW TEXT          — key fields from the ClinicalTrials.gov record
  2. EXTRACTION JSON   — what the Sonnet extractor produced
  3. CLASSIFICATION JSON — what the failure classifier produced
  4. NODE MAPPING      — canonical graph ids the trial routed to
  5. EDGE UPDATES      — per-edge Beta(α,β) before/after this trial's records
  6. PREDICTION        — P(success) WITH and WITHOUT this trial's evidence

Usage:
    python scripts/inspect_trial.py NCT01844505
    python scripts/inspect_trial.py NCT01844505 NCT00109005 NCT02752074
    python scripts/inspect_trial.py --best 3 --worst 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.graph.models import (
    EdgeBeliefState,
    EvidenceRecord,
    TrialSubgraph,
)
from src.graph.store import GraphStore
from src.inference.beliefs import (
    BUCKET_TO_P_OBS,
    SupportBucket,
    apply_virtual_evidence,
    effective_n_for_evidence,
)
from src.ingestion.clinicaltrials import ClinicalTrialsClient
from src.prediction.path_query import predict_clinical_hypothesis

console = Console()

ANNOTATIONS_DIR = Path("data/annotations")
GRAPH_PATH = Path("data/exports/oncology_annotated.json")
TEMPORAL_TEST_PATH = Path("data/dev/test_ncts_2014.txt")


# ── Section 1: RAW TEXT ─────────────────────────────────────────────────


_MAX_SUBGROUPS_PER_MEASURE = 4
_MAX_ANALYSES_PER_MEASURE = 4


def _format_results_block(results_summary: dict[str, Any] | None) -> str:
    """Compact human-readable rendering of CT.gov's resultsSection.

    Renders PRIMARY outcomes in full; for SECONDARY/OTHER outcomes prints the
    title only (full subgroup × week tables can run thousands of rows on
    Phase 3 trials). Per-measure subgroup and analysis lists are capped.
    """
    if not results_summary:
        return "(no posted results)"
    lines: list[str] = []
    om_module = results_summary.get("outcomeMeasuresModule", {}) or {}
    secondaries: list[str] = []
    for om in om_module.get("outcomeMeasures", []) or []:
        om_type = (om.get("type", "") or "").upper()
        title = om.get("title", "")
        if om_type != "PRIMARY":
            secondaries.append(f"[{om_type}] {title}")
            continue
        lines.append(f"  [{om_type}] {title}")
        if om.get("timeFrame"):
            lines.append(f"    timeFrame: {om['timeFrame']}")
        groups = {g.get("id"): g.get("title", "") for g in om.get("groups", []) or []}
        classes = om.get("classes", []) or []
        for cls in classes[:_MAX_SUBGROUPS_PER_MEASURE]:
            cls_title = cls.get("title", "")
            if cls_title:
                lines.append(f"    subgroup: {cls_title}")
            for cat in cls.get("categories", []) or []:
                for m in cat.get("measurements", []) or []:
                    gname = groups.get(m.get("groupId", ""), m.get("groupId", ""))
                    val = m.get("value", "")
                    lo = m.get("lowerLimit", "")
                    hi = m.get("upperLimit", "")
                    ci = f" (CI {lo}–{hi})" if lo and hi else ""
                    lines.append(f"      {gname}: {val}{ci}")
        if len(classes) > _MAX_SUBGROUPS_PER_MEASURE:
            lines.append(f"    … and {len(classes) - _MAX_SUBGROUPS_PER_MEASURE} more subgroups elided")
        analyses = om.get("analyses", []) or []
        for analysis in analyses[:_MAX_ANALYSES_PER_MEASURE]:
            method = analysis.get("statisticalMethod", "")
            param = analysis.get("paramType", "")
            value = analysis.get("paramValue", "")
            pval = analysis.get("pValue", "")
            lines.append(f"    analysis: {method} — {param}={value}, p={pval}")
        if len(analyses) > _MAX_ANALYSES_PER_MEASURE:
            lines.append(f"    … and {len(analyses) - _MAX_ANALYSES_PER_MEASURE} more analyses elided")
    if secondaries:
        lines.append(f"  secondary/other outcomes ({len(secondaries)} total, titles only):")
        for s in secondaries[:8]:
            lines.append(f"    - {s}")
        if len(secondaries) > 8:
            lines.append(f"    … and {len(secondaries) - 8} more elided")
    ae_mod = results_summary.get("adverseEventsModule", {}) or {}
    serious = ae_mod.get("seriousEvents", []) or []
    if serious:
        lines.append(f"  serious AEs ({len(serious)} reported, first 6):")
        for ev in serious[:6]:
            lines.append(f"    - {ev.get('term', '')}")
    return "\n".join(lines) if lines else "(empty resultsSection)"


async def section_raw_text(nct_id: str) -> None:
    console.rule(f"[bold]1. RAW TEXT — {nct_id}[/bold]")
    ct = ClinicalTrialsClient()
    try:
        trial = await ct.get_study(nct_id)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]CT.gov fetch failed for {nct_id}: {exc}[/red]")
        return
    console.print(f"[bold]Title:[/bold] {trial.title}")
    console.print(f"[bold]Phase:[/bold] {trial.phase or '(none)'}")
    console.print(f"[bold]Status:[/bold] {trial.status}")
    console.print(f"[bold]Sponsor:[/bold] {trial.sponsor}")
    console.print(f"[bold]Enrollment:[/bold] {trial.enrollment}")
    console.print(f"[bold]Conditions:[/bold] {', '.join(trial.conditions) or '(none)'}")
    if trial.why_stopped:
        console.print(f"[bold]Why stopped:[/bold] {trial.why_stopped}")
    console.print("[bold]Interventions:[/bold]")
    for iv in trial.interventions:
        desc = f" — {iv.description}" if iv.description else ""
        console.print(f"  - {iv.name} ({iv.type}){desc}")
    console.print("[bold]Primary outcomes:[/bold]")
    for o in trial.primary_outcomes:
        tf = f" [{o.timeframe}]" if o.timeframe else ""
        console.print(f"  - {o.measure}{tf}")
    if trial.secondary_outcomes:
        console.print("[bold]Secondary outcomes (first 5):[/bold]")
        for o in trial.secondary_outcomes[:5]:
            tf = f" [{o.timeframe}]" if o.timeframe else ""
            console.print(f"  - {o.measure}{tf}")
    console.print("[bold]Results section:[/bold]")
    console.print(_format_results_block(trial.results_summary))


# ── Section 2 + 3: cached annotation JSONs ──────────────────────────────


def section_extraction(nct_id: str) -> dict[str, Any] | None:
    console.rule(f"[bold]2. EXTRACTION JSON — {nct_id}[/bold]")
    path = ANNOTATIONS_DIR / f"{nct_id}_extraction.json"
    if not path.exists():
        console.print(f"[yellow]No cached extraction at {path}[/yellow]")
        return None
    raw = json.loads(path.read_text())
    console.print(json.dumps(raw, indent=2))
    return raw


def section_classification(nct_id: str) -> dict[str, Any] | None:
    console.rule(f"[bold]3. CLASSIFICATION JSON — {nct_id}[/bold]")
    path = ANNOTATIONS_DIR / f"{nct_id}_classification.json"
    if not path.exists():
        console.print(f"[yellow]No cached classification at {path}[/yellow]")
        return None
    raw = json.loads(path.read_text())
    console.print(json.dumps(raw, indent=2))
    return raw


# ── Section 4: NODE MAPPING ─────────────────────────────────────────────


def _describe_target(node_id: str) -> str:
    if node_id.startswith("ENSG"):
        return "Ensembl gene id (Open Targets drug→target lookup, HGNC-canonical)"
    if node_id == "UNKNOWN":
        return "unresolved"
    return "uppercase gene symbol fallback (no OT drug match)"


def _describe_mechanism(node_id: str) -> str:
    if node_id == "UNKNOWN":
        return "unresolved"
    if node_id == "other":
        return "LLM did not map to a MechanismCategory → 'other'"
    return f"LLM inference → MechanismCategory.{node_id.upper()}"


def _describe_biology(node_id: str) -> str:
    if node_id.startswith("R-"):
        return "Reactome stable id (LINCS/OT pathway resolution)"
    if "__" in node_id:
        return "synthetic '{mechanism}__{indication}' slug fallback (no Reactome match)"
    if node_id == "UNKNOWN":
        return "unresolved"
    return "unknown id form"


def _describe_endpoint(node_id: str) -> str:
    if node_id == "UNKNOWN":
        return "unresolved"
    cls = node_id.split("_", 1)[0]
    return f"EndpointClass.{cls} prefix (deterministic match preferred, Haiku LLM fallback)"


def _describe_population(node_id: str) -> str:
    if node_id.endswith("__unselected"):
        return "parent enrollment population (no subgroup stratifier)"
    return "subgroup PopulationNode (composed from extracted SubgroupFeatures)"


def section_node_mapping(graph: GraphStore, nct_id: str) -> TrialSubgraph | None:
    console.rule(f"[bold]4. NODE MAPPING — {nct_id}[/bold]")
    try:
        ts = graph.get_trial_subgraph_by_id(nct_id)
    except KeyError:
        console.print(f"[yellow]No trial_subgraph for {nct_id} in {GRAPH_PATH}[/yellow]")
        return None

    console.print(f"[bold]Phase:[/bold] {ts.phase or '(unknown)'}")
    console.print(f"[bold]Parent population:[/bold] {ts.parent_population_id}")
    console.print(f"[bold]Arms ({len(ts.arms)}):[/bold]")
    for arm in ts.arms:
        combo = " [combo]" if arm.is_combination else ""
        console.print(
            f"  - arm_id={arm.arm_id}\n"
            f"    regimen_compound_id={arm.regimen_compound_id}{combo}\n"
            f"    compound_ids={arm.compound_ids}"
        )

    console.print(f"[bold]Chains ({len(ts.chains)}):[/bold]")

    for i, c in enumerate(ts.chains, 1):
        title = (
            f"Chain {i}  "
            f"(arm={c.arm_id}, population={c.subgroup_population_id})"
        )
        table = Table(title=title, show_header=False, box=None, pad_edge=False)
        table.add_column("field", style="bold")
        table.add_column("id")
        table.add_column("resolution")
        table.add_row("compound_id", c.compound_id, "from arm.regimen_compound_id")
        table.add_row("target_id", c.target_id, _describe_target(c.target_id))
        table.add_row("mechanism_id", c.mechanism_id, _describe_mechanism(c.mechanism_id))
        table.add_row("biology_id", c.biology_id, _describe_biology(c.biology_id))
        table.add_row("indication_id", c.indication_id, "lowercased MeSH-like slug from CT.gov condition")
        table.add_row("endpoint_id", c.endpoint_id, _describe_endpoint(c.endpoint_id))
        table.add_row("population_id", c.subgroup_population_id, _describe_population(c.subgroup_population_id))
        console.print(table)
    return ts


# ── Section 5: EDGE UPDATES ─────────────────────────────────────────────


def _replay(records: list[EvidenceRecord]) -> EdgeBeliefState:
    """Re-derive a Beta(α,β) by replaying records from the implicit prior."""
    belief = EdgeBeliefState(alpha=1.0, beta=1.0)
    for r in sorted(records, key=lambda x: x.timestamp):
        try:
            p_obs = BUCKET_TO_P_OBS[SupportBucket(r.support)]
        except (KeyError, ValueError):
            continue
        n_eff = effective_n_for_evidence(r.source_type, r.quality_score)
        belief = apply_virtual_evidence(belief, n_eff=n_eff, p_obs=p_obs)
    return belief


def section_edge_updates(graph: GraphStore, nct_id: str) -> None:
    console.rule(f"[bold]5. EDGE UPDATES — {nct_id}[/bold]")
    rows: list[tuple[str, str, str, list[EvidenceRecord]]] = []
    for u, v, key, data in graph._graph.edges(data=True, keys=True):
        belief_data = data.get("belief") or {}
        evidence = belief_data.get("evidence") or []
        trial_records = [
            EvidenceRecord.model_validate(r) for r in evidence
            if r.get("source_id") == nct_id
        ]
        if not trial_records:
            continue
        rows.append((u, v, key, trial_records))

    if not rows:
        console.print("[yellow]This trial contributed no evidence records to any edge.[/yellow]")
        return

    console.print(f"[bold]Edges this trial updated: {len(rows)}[/bold]")
    for u, v, etype, trial_records in rows:
        belief_data = graph._graph.edges[u, v, etype]["belief"]
        all_records = [
            EvidenceRecord.model_validate(r) for r in belief_data.get("evidence", [])
        ]
        other_records = [r for r in all_records if r.source_id != nct_id]
        before = _replay(other_records)
        after = _replay(all_records)
        delta = after.expected_probability - before.expected_probability

        n_eff_total = sum(
            effective_n_for_evidence(r.source_type, r.quality_score)
            for r in trial_records
        )
        buckets = ", ".join(r.support for r in trial_records)
        ev_types = ", ".join(sorted({r.source_type.value for r in trial_records}))

        console.print(
            Panel(
                f"[bold]{etype}:[/bold] {u} → {v}\n"
                f"  support buckets: {buckets}\n"
                f"  evidence type(s): {ev_types}\n"
                f"  N_eff applied: {n_eff_total:.2f}\n"
                f"  BEFORE: Beta({before.alpha:.2f}, {before.beta:.2f})  →  P = {before.expected_probability:.4f}\n"
                f"  AFTER:  Beta({after.alpha:.2f}, {after.beta:.2f})  →  P = {after.expected_probability:.4f}\n"
                f"  Δ E[p] = {delta:+.4f}    (strength {before.evidence_strength:.2f} → {after.evidence_strength:.2f})",
                title=None,
                expand=False,
            )
        )
        for r in trial_records:
            note = (r.notes or "").strip().replace("\n", " ")
            if len(note) > 120:
                note = note[:117] + "..."
            console.print(f"    evidence note: {note or '(none)'}")


# ── Section 6: PREDICTION (with / without this trial) ───────────────────


def _strip_trial_evidence(graph: GraphStore, nct_id: str) -> GraphStore:
    """Return a deep-copied graph with this trial's evidence records removed
    from every edge belief, with α/β replayed from the remaining records.
    """
    snap_text = json.dumps({
        "graph": __import__("networkx").node_link_data(graph._graph),
        "trial_subgraphs": {
            tid: ts.model_dump(mode="json")
            for tid, ts in graph.trial_subgraphs.items()
        },
    })
    clone = GraphStore()
    tmp = Path("/tmp") / f"_inspect_strip_{nct_id}.json"
    tmp.write_text(snap_text)
    clone.import_snapshot(str(tmp))
    tmp.unlink(missing_ok=True)

    for u, v, key, data in clone._graph.edges(data=True, keys=True):
        belief_data = data.get("belief") or {}
        evidence = belief_data.get("evidence") or []
        records = [EvidenceRecord.model_validate(r) for r in evidence]
        kept = [r for r in records if r.source_id != nct_id]
        if len(kept) == len(records):
            continue
        replayed = _replay(kept)
        replayed.evidence = kept
        data["belief"] = replayed.model_dump(mode="json")
    return clone


def _pick_treatment_chain(ts: TrialSubgraph):
    """Pick a chain to predict on: skip placebo and UNKNOWN entries."""
    for c in ts.chains:
        cid = (c.compound_id or "").lower()
        if cid and cid != "placebo" and c.compound_id != "UNKNOWN" and c.indication_id != "UNKNOWN":
            return c
    return ts.chains[0] if ts.chains else None


def section_prediction(graph: GraphStore, nct_id: str, ts: TrialSubgraph | None) -> None:
    console.rule(f"[bold]6. PREDICTION — {nct_id}[/bold]")
    if ts is None:
        console.print("[yellow]No trial_subgraph; cannot pick a chain to predict on.[/yellow]")
        return
    chain = _pick_treatment_chain(ts)
    if chain is None:
        console.print("[yellow]No chain to predict on.[/yellow]")
        return
    console.print(
        f"[bold]Hypothesis:[/bold] compound={chain.compound_id}  "
        f"indication={chain.indication_id}  endpoint={chain.endpoint_id}"
    )

    def _run(g: GraphStore) -> str:
        try:
            res = predict_clinical_hypothesis(
                g, chain.compound_id, chain.indication_id,
                endpoint_id=chain.endpoint_id if chain.endpoint_id != "UNKNOWN" else None,
                n_samples=5000,
            )
        except KeyError as e:
            return f"[red]predict KeyError: {e}[/red]"
        weakest = (
            f"{res.weakest_link.edge_type.value}: "
            f"{res.weakest_link.source_id} → {res.weakest_link.target_id}"
            if res.weakest_link else "—"
        )
        return (
            f"P(success) = {res.overall_probability:.4f}  "
            f"[CI {res.ci_lower:.3f}, {res.ci_upper:.3f}]\n"
            f"  weakest link: {weakest}"
        )

    with_msg = _run(graph)
    console.print(Panel(with_msg, title="WITH this trial's updates", expand=False))

    stripped = _strip_trial_evidence(graph, nct_id)
    without_msg = _run(stripped)
    console.print(Panel(without_msg, title="WITHOUT this trial's updates (replayed)", expand=False))


# ── Top-level per-trial driver ──────────────────────────────────────────


async def inspect_one(nct_id: str, graph: GraphStore) -> None:
    console.print()
    console.rule(f"[bold cyan]══  {nct_id}  ══[/bold cyan]")
    await section_raw_text(nct_id)
    section_extraction(nct_id)
    section_classification(nct_id)
    ts = section_node_mapping(graph, nct_id)
    section_edge_updates(graph, nct_id)
    section_prediction(graph, nct_id, ts)


# ── --best / --worst auto-find mode ─────────────────────────────────────


def _load_test_set(path: Path) -> set[str]:
    out: set[str] = set()
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            out.add(line)
    return out


def find_best_worst(
    graph: GraphStore,
    test_ncts: set[str],
    n_best: int,
    n_worst: int,
) -> tuple[list[str], list[str]]:
    """Score each labeled test trial, return (best, worst) NCT id lists.

    A trial is "best" when prediction was correct and confident, i.e. small
    |P(pred) − label|. "Worst" is the largest |P(pred) − label|.
    """
    rows: list[tuple[str, float, int, float]] = []
    for nct in sorted(test_ncts):
        ext_path = ANNOTATIONS_DIR / f"{nct}_extraction.json"
        if not ext_path.exists():
            continue
        ext = json.loads(ext_path.read_text())
        met = (ext.get("results") or {}).get("primary_endpoint_met")
        if met is None:
            continue
        try:
            ts = graph.get_trial_subgraph_by_id(nct)
        except KeyError:
            continue
        chain = _pick_treatment_chain(ts)
        if chain is None:
            continue
        if not chain.compound_id or chain.compound_id == "UNKNOWN" or chain.indication_id == "UNKNOWN":
            continue
        try:
            result = predict_clinical_hypothesis(
                graph, chain.compound_id, chain.indication_id,
                endpoint_id=chain.endpoint_id if chain.endpoint_id != "UNKNOWN" else None,
                n_samples=5000,
            )
        except KeyError:
            continue
        label = 1 if met else 0
        p = result.overall_probability
        rows.append((nct, p, label, abs(p - label)))

    if not rows:
        return [], []

    by_err_asc = sorted(rows, key=lambda r: r[3])
    best = by_err_asc[:n_best]
    worst = list(reversed(by_err_asc[-n_worst:])) if n_worst > 0 else []

    summary = Table(title="Best (smallest |P(pred) − label|)")
    summary.add_column("NCT"); summary.add_column("label", justify="right")
    summary.add_column("P(pred)", justify="right"); summary.add_column("|error|", justify="right")
    for nct, p, label, err in best:
        summary.add_row(nct, str(label), f"{p:.3f}", f"{err:.3f}")
    console.print(summary)

    worst_table = Table(title="Worst (largest |P(pred) − label|)")
    worst_table.add_column("NCT"); worst_table.add_column("label", justify="right")
    worst_table.add_column("P(pred)", justify="right"); worst_table.add_column("|error|", justify="right")
    for nct, p, label, err in worst:
        worst_table.add_row(nct, str(label), f"{p:.3f}", f"{err:.3f}")
    console.print(worst_table)

    return [r[0] for r in best], [r[0] for r in worst]


# ── CLI ─────────────────────────────────────────────────────────────────


async def _async_main(args: argparse.Namespace) -> None:
    if not GRAPH_PATH.exists():
        console.print(f"[red]Graph snapshot not found: {GRAPH_PATH}[/red]")
        return
    graph = GraphStore()
    graph.import_snapshot(str(GRAPH_PATH))
    console.print(
        f"[dim]Loaded graph: {graph.stats()['node_count']} nodes, "
        f"{graph.stats()['edge_count']} edges, "
        f"{len(graph.trial_subgraphs)} trial_subgraphs[/dim]"
    )

    targets: list[str] = list(args.nct_ids)

    if args.best or args.worst:
        if not args.test_set.exists():
            console.print(
                f"[red]Temporal test set not found at {args.test_set}; "
                f"pass --test-set or generate it.[/red]"
            )
            return
        test_ncts = _load_test_set(args.test_set)
        console.rule(
            f"[bold]Auto-selecting best={args.best} worst={args.worst} "
            f"from {len(test_ncts)} test NCTs[/bold]"
        )
        best, worst = find_best_worst(graph, test_ncts, args.best, args.worst)
        targets = best + worst

    if not targets:
        console.print("[yellow]No NCT ids to inspect. Provide IDs or --best/--worst.[/yellow]")
        return

    for nct in targets:
        await inspect_one(nct, graph)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("nct_ids", nargs="*", help="NCT ids to inspect")
    parser.add_argument(
        "--best", type=int, default=0,
        help="Auto-find N trials with smallest |P(pred) − label| on the temporal test set",
    )
    parser.add_argument(
        "--worst", type=int, default=0,
        help="Auto-find N trials with largest |P(pred) − label| on the temporal test set",
    )
    parser.add_argument(
        "--test-set", type=Path, default=TEMPORAL_TEST_PATH,
        help=f"Temporal-split test NCT list (default: {TEMPORAL_TEST_PATH})",
    )
    args = parser.parse_args()
    asyncio.run(_async_main(args))


if __name__ == "__main__":
    main()
