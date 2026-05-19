"""In-sample sanity-check eval: do training trials predict correctly?

Same machinery as eval_holdout_v2 but iterates over the 95 training
trials instead of the 50 holdout. This is the trivial sanity case —
the trained graph SAW these trials' outcomes and updated edges
accordingly, so predictions on them should be meaningfully better than
random. If in-sample AUROC is near 0.5, our resolver / prediction loop
has a bug; if it's high, the OOS confidence-collapse is genuinely an
architectural limitation rather than a code problem.

Differences from eval_holdout_v2:
  - Iterates over `graph.trial_subgraphs.keys()` instead of holdout.
  - Same target-anchored filter (target + indication in training-used,
    ≥5/7 overlap, novel compounds allowed).
  - Same three-bucket labels (binary scoring drops ambiguous).
  - Same metrics: AUROC + binary accuracy + histograms.

Usage:
    python -m scripts.eval_in_sample
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

from scripts.eval_holdout_v2 import (
    ANN_DIR,
    _ascii_hist,
    _auroc,
    _binary_accuracy,
    _load_canonicalization_cache,
    _overlap_count,
    _resolve_label,
    _training_used_nodes,
    resolve_chain,
)
from src.graph.store import GraphStore
from src.ingestion.clinicaltrials import ClinicalTrialsClient
from src.prediction.path_query import predict_clinical_hypothesis

console = Console()

GRAPH_PATH = Path("data/exports/oncology_annotated.json")


async def main_async(args: argparse.Namespace) -> int:
    console.rule("[bold]In-sample sanity eval (trials IN training graph)[/bold]")

    graph = GraphStore()
    graph.import_snapshot(str(args.graph))
    stats = graph.stats()
    console.print(
        f"Trained graph: {stats['node_count']} nodes, "
        f"{stats['edge_count']} edges, "
        f"{len(graph.trial_subgraphs)} training trial subgraphs"
    )

    training_used = _training_used_nodes(graph)

    in_sample = list(graph.trial_subgraphs.keys())
    console.print(f"In-sample NCTs: {len(in_sample)}")
    if args.limit:
        in_sample = in_sample[: args.limit]
        console.print(f"  (limited to {len(in_sample)})")

    console.print("Fetching CT.gov records for in-sample trials...")
    ct = ClinicalTrialsClient()
    sem = asyncio.Semaphore(8)
    conditions_map: dict[str, list[str]] = {}

    async def _fetch(nct: str) -> None:
        async with sem:
            try:
                rec = await ct.get_study(nct)
                conditions_map[nct] = (
                    list(getattr(rec, "conditions", []) or []) if rec else []
                )
            except Exception as exc:  # noqa: BLE001
                conditions_map[nct] = []
                console.print(f"  [warn] {nct} CT.gov fetch failed: {exc}")
    await asyncio.gather(*(_fetch(n) for n in in_sample))
    console.print(
        f"  fetched {sum(1 for v in conditions_map.values() if v)}/{len(in_sample)}"
    )

    canon_cache = _load_canonicalization_cache()

    rows: list[dict] = []
    skip_resolve: list[tuple[str, str]] = []

    for nct in in_sample:
        ext_path = ANN_DIR / f"{nct}_extraction.json"
        cls_path = ANN_DIR / f"{nct}_classification.json"
        if not ext_path.exists():
            skip_resolve.append((nct, "no extraction cached"))
            continue
        try:
            extraction = json.loads(ext_path.read_text())
        except json.JSONDecodeError as exc:
            skip_resolve.append((nct, f"extraction parse: {exc}"))
            continue
        classification = None
        if cls_path.exists():
            try:
                classification = json.loads(cls_path.read_text())
            except json.JSONDecodeError:
                pass
        label = _resolve_label(extraction, classification)
        conditions = conditions_map.get(nct, [])
        chain = resolve_chain(extraction, conditions, graph, canon_cache)
        overlap = _overlap_count(chain, training_used)
        rows.append({
            "nct": nct,
            "label": label,
            "chain": chain,
            "overlap": overlap,
            "predicted": False,
            "skip_reason": None,
            "p_success": None,
            "edge_contribs": None,
            "weakest": None,
            "n_edges_used": None,
        })

    console.print(
        f"[bold]Filter: target + indication MUST be in training-used set, "
        f"AND ≥{args.min_overlap}/7 total chain-node overlap.[/bold]"
    )
    np.random.seed(42)
    for r in rows:
        chain = r["chain"]
        if chain["target_id"] not in training_used:
            r["skip_reason"] = f"target {chain['target_id']!r} not in training-used set"
            continue
        if chain["indication_id"] not in training_used:
            r["skip_reason"] = f"indication {chain['indication_id']!r} not in training-used set"
            continue
        if r["overlap"] < args.min_overlap:
            r["skip_reason"] = f"overlap {r['overlap']}/7 below threshold"
            continue
        kwargs = {}
        for k in ("target_id", "mechanism_id", "biology_id",
                  "endpoint_id", "population_id"):
            v = chain[k]
            if v and v != "UNKNOWN":
                kwargs[k] = v
        try:
            result = predict_clinical_hypothesis(
                graph,
                chain["compound_id"],
                chain["indication_id"],
                n_samples=5000,
                **kwargs,
            )
        except KeyError as exc:
            r["skip_reason"] = f"predict KeyError: {exc}"
            continue
        if not result.edge_contributions:
            r["skip_reason"] = "no evidenced edges in resolved chain"
            continue
        r["predicted"] = True
        r["p_success"] = result.overall_probability
        r["edge_contribs"] = result.edge_contributions
        r["n_edges_used"] = len(result.edge_contributions)
        r["weakest"] = (
            result.weakest_link.edge_type.value
            if result.weakest_link else "—"
        )
        r["compound_novel"] = chain["compound_id"] is None

    _report(rows, skip_resolve, args)
    return 0


def _report(rows, skip_resolve, args):
    console.rule("[bold]Funnel[/bold]")
    n_total = len(rows) + len(skip_resolve)
    n_resolved = len(rows)
    n_overlap = sum(1 for r in rows if r["overlap"] >= args.min_overlap)
    n_predicted = sum(1 for r in rows if r["predicted"])
    n_success = sum(1 for r in rows if r["predicted"] and r["label"] == "success")
    n_failure = sum(1 for r in rows if r["predicted"] and r["label"] == "failure")
    n_ambig = sum(1 for r in rows if r["predicted"] and r["label"] == "ambiguous")

    funnel = Table(show_header=False, box=None)
    funnel.add_column(""); funnel.add_column("n", justify="right")
    funnel.add_row("Total in-sample", str(n_total))
    funnel.add_row("Chain resolved", str(n_resolved))
    funnel.add_row(f"Pass overlap ≥{args.min_overlap}/7", str(n_overlap))
    funnel.add_row("Predicted", str(n_predicted))
    funnel.add_row("  Label = success", str(n_success))
    funnel.add_row("  Label = failure", str(n_failure))
    funnel.add_row("  Label = ambiguous (dropped from binary)", str(n_ambig))
    funnel.add_row(
        "  Label = null/unresolvable",
        str(n_predicted - n_success - n_failure - n_ambig),
    )
    console.print(funnel)

    bin_rows = [
        r for r in rows
        if r["predicted"] and r["label"] in ("success", "failure")
    ]
    probs = [r["p_success"] for r in bin_rows]
    labels = [1 if r["label"] == "success" else 0 for r in bin_rows]

    console.rule("[bold]Metrics: success vs failure (ambiguous dropped)[/bold]")
    if bin_rows:
        auroc = _auroc(probs, labels)
        acc, tp, tn, fp, fn = _binary_accuracy(probs, labels, thr=0.5)
        pos = [p for p, y in zip(probs, labels) if y == 1]
        neg = [p for p, y in zip(probs, labels) if y == 0]
        console.print(f"n = {len(bin_rows)} (success={len(pos)}, failure={len(neg)})")
        console.print(f"AUROC = {auroc:.3f}    (rank-based, threshold-free)")
        console.print(
            f"Binary accuracy @ P≥0.5 = {acc:.3f}    "
            f"(TP={tp}, TN={tn}, FP={fp}, FN={fn})"
        )
        if pos:
            console.print(f"Mean P | success = {np.mean(pos):.3f}")
        if neg:
            console.print(f"Mean P | failure = {np.mean(neg):.3f}")
        if pos:
            console.print("\nP(success) — label=SUCCESS:")
            console.print(_ascii_hist(pos))
        if neg:
            console.print("\nP(success) — label=FAILURE:")
            console.print(_ascii_hist(neg))
    else:
        console.print("No success/failure trials in predicted set.")

    if any(r["predicted"] for r in rows):
        console.rule("[bold]Bottleneck edge-type frequency[/bold]")
        counter = Counter(r["weakest"] for r in rows if r["predicted"])
        for etype, n in counter.most_common():
            console.print(f"  {etype:<22} {n}")

    # Per-failed-trial chain decomposition — top 10 most overconfident
    # failed trials, sorted by P(success) desc.
    failed_rows = [
        r for r in rows if r["predicted"] and r["label"] == "failure"
    ]
    if failed_rows:
        console.rule("[bold]Top 10 most overconfident FAILED trials[/bold]")
        for r in sorted(failed_rows, key=lambda x: -x["p_success"])[:10]:
            c = r["chain"]
            novel = "★ novel compound" if r.get("compound_novel") else ""
            print()
            print(
                f"── {r['nct']}   P(success)={r['p_success']:.3f}   "
                f"(label=FAILURE)   {novel}"
            )
            print(
                f"   chain: {c.get('compound_str') or c.get('compound_id') or 'UNKNOWN'} "
                f"→ {c['target_id']} → {c['mechanism_id']} → {c['biology_id']} "
                f"→ {c['indication_id']}"
            )
            print(
                f"   endpoint={c['endpoint_id']}   population={c['population_id']}"
            )
            for ec in r["edge_contribs"]:
                a = ec.belief.alpha
                b = ec.belief.beta
                ep = ec.belief.expected_probability
                n_eff = ec.belief.evidence_strength
                direction = "UP  ↑" if ep > 0.5 else ("DN  ↓" if ep < 0.5 else "==  •")
                print(
                    f"   [{direction}] {ec.edge_type.value:<22} "
                    f"{ec.source_id} → {ec.target_id}"
                )
                print(
                    f"        Beta({a:.2f}, {b:.2f})  E[p]={ep:.3f}  "
                    f"n_eff={n_eff:.2f}  bottleneck={ec.bottleneck_score:.3f}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=GRAPH_PATH)
    parser.add_argument("--min-overlap", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
