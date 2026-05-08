"""Structured breakdown of a finished build_graph run.

Three analyses, designed to validate a graph snapshot before scaling
the corpus up:

  #1  Coverage audit         — which trials made it into trial_subgraphs,
                                and for those that didn't, what was missing
                                (no arms / no targets / no endpoints / etc.).

  #2  Belief calibration     — per-edge-type evidence_strength + posterior
                                distribution, and the top-N edges that
                                moved most from prior → posterior.

  #1.5 Cross-trial accumulation — for each edge updated by ≥2 distinct
                                trials, replay the trajectory and flag
                                concordance vs. disagreement. This is
                                the system's value proposition: edges
                                that accumulate concordant evidence are
                                learned biology.

Run:
    .venv/bin/python -m scripts.analyze_run \\
        --graph data/exports/oncology_annotated.json \\
        --annotations data/annotations
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from src.graph.models import EdgeBeliefState, EdgeType
from src.graph.store import GraphStore
from src.inference.beliefs import (
    BUCKET_TO_P_OBS,
    EVIDENCE_TYPE_N_EFF,
    SupportBucket,
    apply_virtual_evidence,
    effective_n_for_evidence,
)

console = Console()

_CHAIN_EDGE_TYPES = {
    EdgeType.BINDS_TO,
    EdgeType.MODULATES_VIA,
    EdgeType.MECHANISM_AFFECTS,
    EdgeType.BIOLOGY_DRIVES,
    EdgeType.REFLECTS_BIOLOGY,
    EdgeType.ENDPOINT_CAPTURES,
    EdgeType.RESPONDS_DIFFERENTLY,
}


# ── Coverage audit (#1) ─────────────────────────────────────────────────


def _resolve_compound_in_graph(
    graph: GraphStore, name: str,
) -> str | None:
    """Best-effort lookup of a free-text compound name to a CompoundNode id.

    Mirrors what the population pipeline does at build time: the resolver
    builds a slug (lowercase, alnum) and checks the in-memory index. Here
    we just walk CompoundNodes by name/id, which is enough to decide
    "is this drug in the graph at all".
    """
    if not name:
        return None
    needle = "".join(c for c in name.lower() if c.isalnum())
    for node in graph.get_nodes_by_type("CompoundNode"):
        node_id = node.get("id", "")
        node_name = node.get("name", "")
        for candidate in (node_id, node_name):
            slug = "".join(c for c in candidate.lower() if c.isalnum())
            if slug == needle:
                return node_id
    return None


def _diagnose_skip(
    extraction: dict[str, Any], graph: GraphStore,
) -> str:
    """Bucket why a trial that has an extraction didn't end up in trial_subgraphs."""
    arms = extraction.get("arms") or []
    if not arms:
        return "no_arms_extracted"

    have_compounds = any(a.get("compounds") for a in arms)
    if not have_compounds:
        return "arms_have_no_compounds"

    primary = (extraction.get("therapeutic_hypothesis") or {}).get("primary_endpoint")
    if not primary:
        return "no_primary_endpoint"

    # Did any arm's compound make it into the graph as a CompoundNode?
    # Codenames like GSK2118436 / RO5185426 (the actual dabrafenib /
    # vemurafenib developmental ids) often fail this check because OT
    # ranks the canonical drug name first. Vaccines / cell therapies
    # (ATL001, 6MHP) genuinely have no compound-class entry.
    any_compound_in_graph = False
    any_compound_with_target = False
    for arm in arms:
        for cname in arm.get("compounds") or []:
            cid = _resolve_compound_in_graph(graph, cname)
            if cid is None:
                continue
            any_compound_in_graph = True
            # Does this compound have any binds_to edges?
            for src, tgt, key in graph._graph.out_edges(cid, keys=True):
                if key == EdgeType.BINDS_TO.value:
                    any_compound_with_target = True
                    break
            if any_compound_with_target:
                break
        if any_compound_with_target:
            break

    if not any_compound_in_graph:
        return "compound_not_in_graph"
    if not any_compound_with_target:
        return "compound_has_no_target_binding"

    # Compound + target are both in the graph. Did the endpoint classifier
    # return a known EndpointClass for the primary endpoint? Most pure
    # PD / DLT / safety-only primary endpoints fall through this gate.
    primary_lower = primary.lower()
    pd_signals = (
        "phosphorylation", "expression level", "biomarker",
        "immunogenicity", "cd4", "cd8", "t cell response",
        "skin cancer", "rate of development",
    )
    safety_signals = (
        "safety", "tolerability", "adverse event", "dlt",
        "dose limiting", "maximum tolerated", "mtd", "rp2d",
    )
    if any(s in primary_lower for s in pd_signals):
        return "primary_endpoint_is_pd_readout"
    if any(s in primary_lower for s in safety_signals):
        return "primary_endpoint_is_safety_only"
    return "endpoint_classifier_no_match"


def coverage_audit(
    graph_path: Path, annotations_dir: Path,
) -> tuple[set[str], dict[str, str]]:
    """Bucket trials by whether they made it into trial_subgraphs and why.

    Returns (built_trial_ids, skip_reason_by_trial).
    """
    snap = json.loads(graph_path.read_text())
    built = set(snap.get("trial_subgraphs", {}).keys())

    graph = GraphStore()
    graph.import_snapshot(str(graph_path))

    extractions: dict[str, dict[str, Any]] = {}
    for path in sorted(annotations_dir.glob("*_extraction.json")):
        nct = path.stem.split("_")[0]
        extractions[nct] = json.loads(path.read_text())

    extracted_total = set(extractions.keys())
    skipped = sorted(extracted_total - built)

    skip_reason: dict[str, str] = {}
    for nct in skipped:
        skip_reason[nct] = _diagnose_skip(extractions[nct], graph)

    console.rule("[bold]#1 Coverage audit[/bold]")
    console.print(
        f"Extracted: {len(extracted_total)} | "
        f"Built into trial_subgraphs: {len(built)} | "
        f"Skipped: {len(skipped)}"
    )

    if skipped:
        bucket_table = Table(title="Skip reasons (by bucket)")
        bucket_table.add_column("Bucket")
        bucket_table.add_column("Count", justify="right")
        bucket_table.add_column("Trials")
        bucket_counts: dict[str, list[str]] = defaultdict(list)
        for nct, reason in skip_reason.items():
            bucket_counts[reason].append(nct)
        for reason, ncts in sorted(
            bucket_counts.items(), key=lambda kv: -len(kv[1])
        ):
            bucket_table.add_row(reason, str(len(ncts)), ", ".join(ncts))
        console.print(bucket_table)

        # Per-trial detail
        detail = Table(title="Per-skipped-trial detail")
        detail.add_column("NCT")
        detail.add_column("Arms", justify="right")
        detail.add_column("Compounds (first arm)")
        detail.add_column("Primary endpoint")
        detail.add_column("Skip reason")
        for nct in skipped:
            ex = extractions[nct]
            arms = ex.get("arms") or []
            primary = (ex.get("therapeutic_hypothesis") or {}).get(
                "primary_endpoint", ""
            )
            first_compounds = (
                ", ".join(arms[0].get("compounds", [])[:2]) if arms else "—"
            )
            detail.add_row(
                nct,
                str(len(arms)),
                first_compounds[:35],
                primary[:45] if primary else "—",
                skip_reason[nct],
            )
        console.print(detail)

    return built, skip_reason


# ── Belief calibration (#2) ─────────────────────────────────────────────


def belief_calibration(graph_path: Path, top_n: int = 20) -> None:
    snap = json.loads(graph_path.read_text())
    edges = snap["graph"]["edges"]

    by_type: dict[str, list[EdgeBeliefState]] = defaultdict(list)
    for e in edges:
        belief_data = e.get("belief") or {}
        try:
            belief = EdgeBeliefState.model_validate(belief_data)
        except Exception:
            continue
        et = e.get("edge_type") or e.get("key", "?")
        by_type[et].append(belief)

    console.rule("[bold]#2 Belief calibration[/bold]")

    cal_table = Table(title="Per-edge-type belief distribution")
    cal_table.add_column("Edge type")
    cal_table.add_column("N", justify="right")
    cal_table.add_column("E[p] median", justify="right")
    cal_table.add_column("E[p] p25–p75", justify="right")
    cal_table.add_column("evidence_strength median", justify="right")
    cal_table.add_column("% with evidence (>0)", justify="right")

    for et in sorted(by_type):
        beliefs = by_type[et]
        ps = sorted(b.expected_probability for b in beliefs)
        ss = sorted(b.evidence_strength for b in beliefs)
        if not ps:
            continue
        median_p = statistics.median(ps)
        p25 = ps[len(ps) // 4]
        p75 = ps[(3 * len(ps)) // 4]
        median_s = statistics.median(ss)
        with_ev = 100.0 * sum(1 for s in ss if s > 0) / len(ss)
        cal_table.add_row(
            et, str(len(beliefs)),
            f"{median_p:.3f}", f"{p25:.2f}–{p75:.2f}",
            f"{median_s:.2f}", f"{with_ev:.0f}%",
        )
    console.print(cal_table)

    # Top movers: edges where E[p] moved most away from 0.5 with substantial
    # evidence. Sorts by |E[p] - 0.5| × evidence_strength so a confident +
    # well-supported posterior beats a confident-but-thin one.
    movers: list[tuple[float, dict[str, Any], EdgeBeliefState]] = []
    for e in edges:
        belief_data = e.get("belief") or {}
        try:
            belief = EdgeBeliefState.model_validate(belief_data)
        except Exception:
            continue
        if belief.evidence_strength < 1.0:
            continue
        score = abs(belief.expected_probability - 0.5) * belief.evidence_strength
        movers.append((score, e, belief))
    movers.sort(key=lambda t: -t[0])

    mover_table = Table(title=f"Top {top_n} most-moved edges (|E[p]−0.5| × strength)")
    mover_table.add_column("Edge type")
    mover_table.add_column("Source → Target", overflow="fold")
    mover_table.add_column("E[p]", justify="right")
    mover_table.add_column("Strength (α+β−2)", justify="right")
    mover_table.add_column("Direction")
    for score, e, belief in movers[:top_n]:
        et = e.get("edge_type") or e.get("key", "?")
        direction = "→ true" if belief.expected_probability > 0.5 else "→ false"
        mover_table.add_row(
            et,
            f"{e['source']} → {e['target']}",
            f"{belief.expected_probability:.3f}",
            f"{belief.evidence_strength:.1f}",
            direction,
        )
    console.print(mover_table)


# ── Cross-trial knowledge accumulation (#1.5) ───────────────────────────


_BUCKET_DIR_LABEL = {
    "strong_support":      "++",
    "moderate_support":    "+",
    "weak_support":        "·+",
    "ambiguous":           "~",
    "weak_contradict":     "·−",
    "moderate_contradict": "−",
    "strong_contradict":   "−−",
}


def _classify_concordance(buckets: list[str]) -> str:
    """All-support, all-contradict, or mixed."""
    supports = sum(1 for b in buckets if "support" in b)
    contradicts = sum(1 for b in buckets if "contradict" in b)
    if contradicts == 0 and supports >= 2:
        return "concordant_support"
    if supports == 0 and contradicts >= 2:
        return "concordant_contradict"
    if supports >= 1 and contradicts >= 1:
        return "conflicting"
    return "mostly_ambiguous"


def cross_trial_accumulation(graph_path: Path, top_n: int = 10) -> None:
    snap = json.loads(graph_path.read_text())
    edges = snap["graph"]["edges"]

    multi_trial: list[tuple[int, dict[str, Any], EdgeBeliefState]] = []
    for e in edges:
        belief_data = e.get("belief") or {}
        try:
            belief = EdgeBeliefState.model_validate(belief_data)
        except Exception:
            continue
        # Only count NCT-derived (i.e. trial-derived) evidence records.
        # LINCS/OT/literature use non-NCT source ids and aren't "trials".
        trial_records = [
            r for r in belief.evidence
            if isinstance(r.source_id, str) and r.source_id.startswith("NCT")
        ]
        distinct_trials = {r.source_id for r in trial_records}
        if len(distinct_trials) < 2:
            continue
        multi_trial.append((len(distinct_trials), e, belief))
    multi_trial.sort(key=lambda t: (-t[0], -t[2].evidence_strength))

    console.rule("[bold]#1.5 Cross-trial knowledge accumulation[/bold]")
    console.print(
        f"Edges with ≥2 distinct contributing trials: {len(multi_trial)}"
    )
    if not multi_trial:
        console.print("[yellow]No edges yet have multiple-trial evidence.[/yellow]")
        return

    # Use a wide console for this table — trajectory strings can be long.
    wide_console = Console(width=200)
    summary = Table(
        title=f"Top {top_n} edges by # distinct contributing trials",
        show_lines=True,
    )
    summary.add_column("#", justify="right", no_wrap=True)
    summary.add_column("Edge type", no_wrap=True)
    summary.add_column("Source → Target", overflow="fold")
    summary.add_column("Trials", justify="right", no_wrap=True)
    summary.add_column("Concordance", no_wrap=True)
    summary.add_column("Prior", justify="right", no_wrap=True)
    summary.add_column("Posterior", justify="right", no_wrap=True)
    summary.add_column("Δ", justify="right", no_wrap=True)
    summary.add_column("Trajectory", overflow="fold")

    for i, (n_trials, e, belief) in enumerate(multi_trial[:top_n], 1):
        et = e.get("edge_type") or e.get("key", "?")
        # Replay: start from the implicit prior and apply each trial-derived
        # record in timestamp order. Non-trial evidence (LINCS, OT priors) is
        # folded into a synthetic starting belief so the "prior" we report
        # is what the trial sequence saw on entry, not Beta(1,1).
        all_records = sorted(belief.evidence, key=lambda r: r.timestamp)
        non_trial = [
            r for r in all_records
            if not (isinstance(r.source_id, str) and r.source_id.startswith("NCT"))
        ]
        # Synthetic starting state: Beta(1,1) plus any non-trial evidence
        # applied first (LINCS / OT / literature priors).
        start = EdgeBeliefState(alpha=1.0, beta=1.0)
        for r in non_trial:
            n_eff = effective_n_for_evidence(r.source_type, r.quality_score)
            try:
                p_obs = BUCKET_TO_P_OBS[SupportBucket(r.support)]
            except (KeyError, ValueError):
                continue
            start = apply_virtual_evidence(start, n_eff=n_eff, p_obs=p_obs)
        prior_p = start.expected_probability

        # Now replay trial-derived records, grouping by NCT id (one trial
        # may emit multiple records on the same edge — collapse to one
        # symbolic entry per trial, taking the strongest signed bucket).
        trial_records_in_order: list[tuple[str, str]] = []
        seen: set[str] = set()
        running = start
        traj_buckets: list[str] = []
        for r in all_records:
            if not (isinstance(r.source_id, str) and r.source_id.startswith("NCT")):
                continue
            n_eff = effective_n_for_evidence(r.source_type, r.quality_score)
            try:
                p_obs = BUCKET_TO_P_OBS[SupportBucket(r.support)]
            except (KeyError, ValueError):
                continue
            running = apply_virtual_evidence(running, n_eff=n_eff, p_obs=p_obs)
            if r.source_id not in seen:
                trial_records_in_order.append((r.source_id, r.support))
                seen.add(r.source_id)
                traj_buckets.append(r.support)

        post_p = running.expected_probability
        delta = post_p - prior_p
        concordance = _classify_concordance(traj_buckets)
        trajectory_str = " → ".join(
            f"{nct[-4:]}({_BUCKET_DIR_LABEL.get(b, b)})"
            for nct, b in trial_records_in_order
        )

        summary.add_row(
            str(i), et,
            f"{e['source']} → {e['target']}",
            str(n_trials), concordance,
            f"{prior_p:.3f}", f"{post_p:.3f}",
            f"{delta:+.3f}",
            trajectory_str,
        )

    wide_console.print(summary)
    wide_console.print(
        "\n[dim]Trajectory legend: NCT-suffix(direction). "
        "++ = strong_support, + = moderate_support, ·+ = weak_support, "
        "~ = ambiguous, ·− = weak_contradict, − = moderate_contradict, "
        "−− = strong_contradict.[/dim]"
    )

    # Concordance rollup — across ALL multi-trial edges (not just top N)
    rollup = Counter()
    for _, _, b in multi_trial:
        trial_buckets = [
            r.support for r in b.evidence
            if isinstance(r.source_id, str) and r.source_id.startswith("NCT")
        ]
        rollup[_classify_concordance(trial_buckets)] += 1
    console.print()
    cr = Table(title="Concordance distribution (all multi-trial edges)")
    cr.add_column("Class")
    cr.add_column("Count", justify="right")
    cr.add_column("% of multi-trial edges", justify="right")
    total = sum(rollup.values())
    for cls in (
        "concordant_support", "concordant_contradict",
        "conflicting", "mostly_ambiguous",
    ):
        n = rollup.get(cls, 0)
        pct = 100.0 * n / total if total else 0.0
        cr.add_row(cls, str(n), f"{pct:.0f}%")
    console.print(cr)


# ── Entrypoint ──────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph", type=Path,
        default=Path("data/exports/oncology_annotated.json"),
    )
    parser.add_argument(
        "--annotations", type=Path,
        default=Path("data/annotations"),
    )
    parser.add_argument("--top-movers", type=int, default=20)
    parser.add_argument("--top-multi-trial", type=int, default=10)
    args = parser.parse_args()

    coverage_audit(args.graph, args.annotations)
    belief_calibration(args.graph, top_n=args.top_movers)
    cross_trial_accumulation(args.graph, top_n=args.top_multi_trial)


if __name__ == "__main__":
    main()
