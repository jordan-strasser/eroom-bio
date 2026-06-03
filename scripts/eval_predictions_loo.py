"""Leave-one-out (LOO) prediction eval — proper out-of-sample AUROC.

For each labeled trial T, removes T's evidence records from every edge
in the graph, replays Beta(α,β) on the remaining evidence, then predicts
P(success) for T's headline chain. The trial whose outcome we're
predicting contributes nothing to the prediction → a real out-of-sample
benchmark (vs. `eval_predictions.py` which is in-sample / upper-bound).

Outputs:
  - AUROC, Brier, calibration bins for the LOO predictions
  - Side-by-side comparison with in-sample numbers when both run
  - Worst LOO misses, with per-trial weakest-link annotation

Run:

    python -m scripts.eval_predictions_loo
    python -m scripts.eval_predictions_loo --compare-in-sample

The comparison flag reuses the in-sample prediction path to surface
the LOO-vs-in-sample gap. Real-world generalization sits at the LOO
number; the gap shows how much information leakage was driving the
in-sample score.
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path

import networkx as nx
import numpy as np
from rich.console import Console
from rich.table import Table

from src.graph.models import EdgeBeliefState, EvidenceRecord
from src.graph.store import GraphStore
from src.inference.beliefs import (
    BUCKET_TO_P_OBS,
    SupportBucket,
    apply_virtual_evidence,
    effective_n_for_evidence,
)
from src.prediction.path_query import predict_clinical_hypothesis

console = Console()


# ── metrics (mirrors eval_predictions.py) ────────────────────────────────


def _brier(probs: list[float], labels: list[int]) -> float:
    return float(np.mean([(p - y) ** 2 for p, y in zip(probs, labels)]))


def _auroc(probs: list[float], labels: list[int]) -> float:
    """Mann-Whitney U / DeLong-equivalent AUROC. P(score_pos > score_neg)."""
    pos = [p for p, y in zip(probs, labels) if y == 1]
    neg = [p for p, y in zip(probs, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    correct = 0.0
    for pp in pos:
        for nn in neg:
            if pp > nn:
                correct += 1.0
            elif pp == nn:
                correct += 0.5
    return correct / (len(pos) * len(neg))


def _calibration_bins(probs: list[float], labels: list[int], n_bins: int = 5):
    bins = []
    for i in range(n_bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        in_bin = [
            (p, y) for p, y in zip(probs, labels)
            if (lo <= p < hi) or (i == n_bins - 1 and p == hi)
        ]
        if not in_bin:
            bins.append((lo, hi, 0, math.nan, math.nan))
            continue
        avg_p = sum(p for p, _ in in_bin) / len(in_bin)
        obs = sum(y for _, y in in_bin) / len(in_bin)
        bins.append((lo, hi, len(in_bin), avg_p, obs))
    return bins


# ── LOO evidence stripping ───────────────────────────────────────────────
#
# Mirrors `_strip_trial_evidence` and `_replay` in
# scripts/inspect_trial.py. Inlined here to avoid the cross-script
# import. If this duplicates further, promote to
# src/prediction/eval_utils.py.


def _replay(records: list[EvidenceRecord]) -> EdgeBeliefState:
    """Re-derive a Beta(α,β) by replaying evidence from the implicit prior."""
    belief = EdgeBeliefState(alpha=1.0, beta=1.0)
    for r in sorted(records, key=lambda x: x.timestamp):
        try:
            p_obs = BUCKET_TO_P_OBS[SupportBucket(r.support)]
        except (KeyError, ValueError):
            continue
        n_eff = effective_n_for_evidence(r.source_type, r.quality_score)
        belief = apply_virtual_evidence(belief, n_eff=n_eff, p_obs=p_obs)
    return belief


def _strip_trial_evidence(graph: GraphStore, nct_id: str) -> GraphStore:
    """Deep-copied graph with `nct_id`'s evidence stripped and α/β replayed."""
    snap_text = json.dumps({
        "graph": nx.node_link_data(graph._graph),  # noqa: SLF001
        "trial_subgraphs": {
            tid: ts.model_dump(mode="json")
            for tid, ts in graph.trial_subgraphs.items()
        },
    })
    clone = GraphStore()
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=f"_loo_strip_{nct_id}.json", delete=False,
    ) as fh:
        fh.write(snap_text)
        tmp_path = fh.name
    try:
        clone.import_snapshot(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    for _u, _v, _key, data in clone._graph.edges(data=True, keys=True):  # noqa: SLF001
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


# ── eval driver ──────────────────────────────────────────────────────────


def _resolve_label(
    ext: dict, annotations_dir: Path, nct: str,
    counts: dict[str, int],
) -> int | None:
    """Combine extractor's `primary_endpoint_met` and classifier's
    `trial_outcome` into a single binary label. Mirrors the helper in
    eval_predictions.py — kept inline here to avoid cross-script import.
    """
    met = (ext.get("results") or {}).get("primary_endpoint_met")
    if met is True:
        counts["primary_endpoint_met"] += 1
        return 1
    if met is False:
        counts["primary_endpoint_met"] += 1
        return 0
    cls_path = annotations_dir / f"{nct}_classification.json"
    if not cls_path.exists():
        return None
    try:
        cls = json.loads(cls_path.read_text())
    except json.JSONDecodeError:
        return None
    outcome = (cls.get("trial_outcome") or "").lower()
    if outcome == "success":
        counts["trial_outcome"] += 1
        return 1
    if outcome == "failure":
        counts["trial_outcome"] += 1
        return 0
    return None


def _pick_headline_chain(ts: dict) -> dict | None:
    chains = ts.get("chains") or []
    if not chains:
        return None
    return next(
        (
            c for c in chains
            if (c.get("compound_id") or "").lower() not in ("placebo", "")
        ),
        chains[0],
    )


def _predict_one(graph: GraphStore, chain: dict) -> tuple[float, str, str] | None:
    compound_id = chain.get("compound_id")
    indication_id = chain.get("indication_id")
    if not compound_id or compound_id == "UNKNOWN" or indication_id == "UNKNOWN":
        return None
    try:
        result = predict_clinical_hypothesis(
            graph, compound_id, indication_id,
            endpoint_id=chain.get("endpoint_id") or None,
            n_samples=5000,
        )
    except KeyError:
        return None
    weakest = (
        f"{result.weakest_link.edge_type.value}: "
        f"{result.weakest_link.source_id} → {result.weakest_link.target_id}"
        if result.weakest_link else "—"
    )
    return result.overall_probability, weakest, (
        f"{compound_id} → {indication_id}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph", type=Path,
        default=Path("data/exports/oncology_annotated.json"),
    )
    parser.add_argument(
        "--annotations", type=Path, default=Path("data/annotations"),
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--compare-in-sample", action="store_true",
        help="Also compute in-sample predictions for side-by-side comparison",
    )
    parser.add_argument(
        "--filter-ncts", type=Path, default=None,
        help="Optional NCT-id allowlist file (one per line, # comments ok).",
    )
    args = parser.parse_args()

    keep_ncts: set[str] | None = None
    if args.filter_ncts:
        keep_ncts = set()
        for raw in args.filter_ncts.read_text().splitlines():
            line = raw.split("#", 1)[0].strip()
            if line:
                keep_ncts.add(line)

    graph = GraphStore()
    graph.import_snapshot(str(args.graph))
    snap = json.loads(args.graph.read_text())
    trial_subgraphs = snap.get("trial_subgraphs", {})

    # Labels come from two sources combined (see _resolve_label):
    #   1. extractor's primary_endpoint_met (preferred)
    #   2. classifier's trial_outcome (fallback when (1) is None)
    # Matches the convention in eval_predictions.py.
    labeled: list[tuple[str, int, dict]] = []  # (nct, label, chain)
    skipped: list[tuple[str, str]] = []
    label_source_counts: dict[str, int] = {
        "primary_endpoint_met": 0, "trial_outcome": 0,
    }
    for ext_path in sorted(args.annotations.glob("*_extraction.json")):
        nct = ext_path.stem.split("_")[0]
        if keep_ncts is not None and nct not in keep_ncts:
            continue
        try:
            ext = json.loads(ext_path.read_text())
        except json.JSONDecodeError:
            continue
        label = _resolve_label(ext, args.annotations, nct, label_source_counts)
        if label is None:
            continue
        ts = trial_subgraphs.get(nct)
        if not ts:
            skipped.append((nct, "no trial_subgraph"))
            continue
        chain = _pick_headline_chain(ts)
        if chain is None:
            skipped.append((nct, "no chain"))
            continue
        labeled.append((nct, label, chain))

    console.rule(f"[bold]Labeled trials: {len(labeled)}[/bold]")
    n_pos = sum(y for _, y, _ in labeled)
    console.print(
        f"successes: {n_pos}  |  failures: {len(labeled) - n_pos}  |  "
        f"skipped: {len(skipped)}"
    )
    console.print(
        f"Label source: {label_source_counts['primary_endpoint_met']} from "
        f"primary_endpoint_met, {label_source_counts['trial_outcome']} from "
        f"trial_outcome fallback"
    )

    if len(labeled) < 2 or n_pos == 0 or n_pos == len(labeled):
        console.print("[yellow]Not enough labels to compute AUROC.[/yellow]")
        return

    # In-sample baseline (optional, fast)
    insample_rows: list[dict] = []
    if args.compare_in_sample:
        console.rule("[bold]In-sample predictions[/bold]")
        for nct, label, chain in labeled:
            pred = _predict_one(graph, chain)
            if pred is None:
                continue
            p, weakest, route = pred
            insample_rows.append({
                "nct": nct, "label": label, "p_success": p,
                "weakest": weakest, "route": route,
            })

    # LOO predictions — the real benchmark
    console.rule("[bold]LOO predictions (stripping each trial's evidence)[/bold]")
    loo_rows: list[dict] = []
    for i, (nct, label, chain) in enumerate(labeled, 1):
        console.print(f"  [{i}/{len(labeled)}] {nct}", end="\r")
        stripped = _strip_trial_evidence(graph, nct)
        pred = _predict_one(stripped, chain)
        if pred is None:
            continue
        p, weakest, route = pred
        loo_rows.append({
            "nct": nct, "label": label, "p_success": p,
            "weakest": weakest, "route": route,
        })
    console.print()

    # Report side-by-side
    def _report(rows: list[dict], title: str) -> dict:
        probs = [r["p_success"] for r in rows]
        labels = [r["label"] for r in rows]
        n_pos = sum(labels)
        n_neg = len(labels) - n_pos
        if n_pos == 0 or n_neg == 0:
            return {"auroc": float("nan"), "rows": rows}
        auroc = _auroc(probs, labels)
        brier = _brier(probs, labels)
        base_rate = n_pos / len(labels)
        base_brier = base_rate * (1 - base_rate)
        thr = args.threshold
        tp = sum(1 for p, y in zip(probs, labels) if p >= thr and y == 1)
        fp = sum(1 for p, y in zip(probs, labels) if p >= thr and y == 0)
        fn = sum(1 for p, y in zip(probs, labels) if p < thr and y == 1)
        tn = sum(1 for p, y in zip(probs, labels) if p < thr and y == 0)
        acc = (tp + tn) / len(labels)
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        rec = tp / (tp + fn) if (tp + fn) else float("nan")
        mean_pos = float(np.mean([p for p, y in zip(probs, labels) if y == 1]))
        mean_neg = float(np.mean([p for p, y in zip(probs, labels) if y == 0]))

        console.rule(f"[bold]{title} (n={len(rows)})[/bold]")
        tbl = Table()
        tbl.add_column("metric"); tbl.add_column("value", justify="right")
        tbl.add_column("notes")
        tbl.add_row("AUROC", f"{auroc:.3f}", "1.0=perfect, 0.5=random")
        tbl.add_row(
            "Brier", f"{brier:.3f}",
            f"baseline (predict base rate {base_rate:.2f}): {base_brier:.3f}",
        )
        tbl.add_row(
            f"Accuracy@p≥{thr}", f"{acc:.3f}",
            f"chance: {max(base_rate, 1 - base_rate):.3f}",
        )
        tbl.add_row("Precision (success)", f"{prec:.3f}", "")
        tbl.add_row("Recall (success)", f"{rec:.3f}", "")
        tbl.add_row("Mean P | label=1", f"{mean_pos:.3f}", "want > base rate")
        tbl.add_row("Mean P | label=0", f"{mean_neg:.3f}", "want < base rate")
        console.print(tbl)

        cm = Table(title=f"Confusion @ {thr}")
        cm.add_column(""); cm.add_column("pred=success", justify="right")
        cm.add_column("pred=failure", justify="right")
        cm.add_row("actual=success", str(tp), str(fn))
        cm.add_row("actual=failure", str(fp), str(tn))
        console.print(cm)

        cal = Table(title="Calibration bins")
        cal.add_column("bin"); cal.add_column("n", justify="right")
        cal.add_column("avg P(pred)", justify="right")
        cal.add_column("observed rate", justify="right")
        for lo, hi, n, ap, obs in _calibration_bins(probs, labels, n_bins=5):
            if n == 0:
                cal.add_row(f"[{lo:.1f}, {hi:.1f})", "0", "—", "—")
            else:
                cal.add_row(
                    f"[{lo:.1f}, {hi:.1f})", str(n),
                    f"{ap:.3f}", f"{obs:.3f}",
                )
        console.print(cal)
        return {"auroc": auroc, "brier": brier, "rows": rows}

    loo_summary = _report(loo_rows, "LOO (out-of-sample)")
    if insample_rows:
        in_summary = _report(insample_rows, "In-sample (upper bound)")
        gap = in_summary["auroc"] - loo_summary["auroc"]
        console.rule("[bold]In-sample → LOO gap[/bold]")
        console.print(
            f"AUROC drop: {gap:+.3f}  "
            f"(in-sample {in_summary['auroc']:.3f} → LOO "
            f"{loo_summary['auroc']:.3f})"
        )

    # Worst LOO misses
    rows_sorted = sorted(
        loo_rows, key=lambda r: -abs(r["p_success"] - r["label"]),
    )
    miss = Table(title="Top 10 worst LOO misses (|P − label|)")
    miss.add_column("nct"); miss.add_column("label", justify="right")
    miss.add_column("P(pred)", justify="right")
    miss.add_column("route"); miss.add_column("weakest")
    for r in rows_sorted[:10]:
        miss.add_row(
            r["nct"], str(r["label"]), f"{r['p_success']:.3f}",
            r["route"][:40], r["weakest"][:50],
        )
    console.print(miss)


if __name__ == "__main__":
    main()
