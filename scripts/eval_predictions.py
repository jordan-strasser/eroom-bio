"""Quick in-sample evaluation of predict_clinical_hypothesis.

Labels come from two extractor/classifier signals combined:
  1. ``extraction.results.primary_endpoint_met`` (True/False) — preferred
     when set; the extractor's read of whether the trial hit its primary
     endpoint.
  2. ``classification.trial_outcome`` (success/failure/...) as a fallback
     when (1) is None. The classifier synthesizes this from the
     extraction's results section even when the extractor didn't fill
     the binary field.

Trials whose combined signal is ``partial`` or ``unknown`` are excluded
(genuinely ambiguous outcomes — neither labeling lane resolves them).
Phase I safety-only trials end up here, which is correct: there's no
binary efficacy outcome to predict against.

CAVEAT—IN-SAMPLE: every trial we predict on contributed evidence to the
graph it's being predicted against. Numbers here are an upper bound on
generalization, not a real benchmark. Real OOS validation requires LOO
(eval_predictions_loo.py).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

from src.graph.store import GraphStore
from src.prediction.path_query import predict_clinical_hypothesis

console = Console()


def _brier(probs: list[float], labels: list[int]) -> float:
    return float(np.mean([(p - y) ** 2 for p, y in zip(probs, labels)]))


def _auroc(probs: list[float], labels: list[int]) -> float:
    """Mann-Whitney U / DeLong-equivalent AUROC. Computes P(score_pos > score_neg)
    over all positive/negative pairs, with 0.5 credit for ties."""
    pos = [p for p, y in zip(probs, labels) if y == 1]
    neg = [p for p, y in zip(probs, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    correct = 0.0
    for pp in pos:
        for nn in neg:
            if pp > nn: correct += 1.0
            elif pp == nn: correct += 0.5
    return correct / (len(pos) * len(neg))


def _resolve_label(
    ext: dict, annotations_dir: Path, nct: str,
    counts: dict[str, int],
) -> int | None:
    """Combine extractor's `primary_endpoint_met` and classifier's
    `trial_outcome` into a single binary label. Returns None for trials
    where both signals are ambiguous (partial / unknown / safety-only).

    Counts which label source resolved each trial for downstream
    reporting — useful to see how much we depend on the fallback.
    """
    met = (ext.get("results") or {}).get("primary_endpoint_met")
    if met is True:
        counts["primary_endpoint_met"] += 1
        return 1
    if met is False:
        counts["primary_endpoint_met"] += 1
        return 0
    # Fallback: classifier's trial_outcome
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


def _calibration_bins(probs: list[float], labels: list[int], n_bins: int = 5):
    """Reliability bins: (lower, upper, n, predicted_avg, observed_rate)."""
    bins = []
    for i in range(n_bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        in_bin = [(p, y) for p, y in zip(probs, labels) if (lo <= p < hi) or (i == n_bins - 1 and p == hi)]
        if not in_bin:
            bins.append((lo, hi, 0, math.nan, math.nan))
            continue
        avg_p = sum(p for p, _ in in_bin) / len(in_bin)
        obs = sum(y for _, y in in_bin) / len(in_bin)
        bins.append((lo, hi, len(in_bin), avg_p, obs))
    return bins


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=Path("data/exports/oncology_annotated.json"))
    parser.add_argument("--annotations", type=Path, default=Path("data/annotations"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--filter-ncts", type=Path, default=None,
        help="If set, only evaluate trials whose NCT id appears in this file "
             "(one per line, # comments allowed). Use to score a held-out "
             "test split without rebuilding the eval logic.",
    )
    args = parser.parse_args()

    keep_ncts: set[str] | None = None
    if args.filter_ncts:
        keep_ncts = set()
        for raw in args.filter_ncts.read_text().splitlines():
            line = raw.split("#", 1)[0].strip()
            if line:
                keep_ncts.add(line)
        console.print(
            f"[bold]Filtering eval to {len(keep_ncts)} NCT ids from {args.filter_ncts}[/bold]"
        )

    graph = GraphStore()
    graph.import_snapshot(str(args.graph))
    snap = json.loads(args.graph.read_text())
    trial_subgraphs = snap.get("trial_subgraphs", {})

    rows: list[dict] = []
    skipped: list[tuple[str, str]] = []
    label_source_counts: dict[str, int] = {"primary_endpoint_met": 0, "trial_outcome": 0}
    for ext_path in sorted(args.annotations.glob("*_extraction.json")):
        nct = ext_path.stem.split("_")[0]
        if keep_ncts is not None and nct not in keep_ncts:
            continue
        ext = json.loads(ext_path.read_text())
        label = _resolve_label(ext, args.annotations, nct, label_source_counts)
        if label is None:
            continue
        ts = trial_subgraphs.get(nct)
        if not ts:
            skipped.append((nct, "no trial_subgraph"))
            continue
        chains = ts.get("chains") or []
        if not chains:
            skipped.append((nct, "no chains"))
            continue
        # Use the first treatment chain (skip placebo-only chains by checking
        # compound_id != placebo). For combo arms, regimen_compound_id is the
        # synthesized "A+B" id which is itself a CompoundNode in the graph.
        chain = next(
            (c for c in chains if c.get("compound_id", "").lower() not in ("placebo", "")),
            chains[0],
        )
        compound_id = chain.get("compound_id")
        indication_id = chain.get("indication_id")
        if not compound_id or compound_id == "UNKNOWN" or indication_id == "UNKNOWN":
            skipped.append((nct, f"unknown compound/indication ({compound_id}/{indication_id})"))
            continue
        try:
            result = predict_clinical_hypothesis(
                graph, compound_id, indication_id,
                endpoint_id=chain.get("endpoint_id") or None,
                n_samples=5000,
            )
        except KeyError as e:
            skipped.append((nct, f"predict KeyError: {e}"))
            continue
        rows.append({
            "nct": nct,
            "label": label,
            "p_success": result.overall_probability,
            "ci_lo": result.ci_lower,
            "ci_hi": result.ci_upper,
            "compound": compound_id,
            "indication": indication_id,
            "weakest_edge": (
                f"{result.weakest_link.edge_type.value}: {result.weakest_link.source_id} → {result.weakest_link.target_id}"
                if result.weakest_link else "—"
            ),
        })

    probs = [r["p_success"] for r in rows]
    labels = [r["label"] for r in rows]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos

    console.rule("[bold]In-sample prediction eval[/bold]")
    console.print(f"Labeled trials evaluated: {len(rows)}  |  successes: {n_pos}  |  failures: {n_neg}")
    console.print(
        f"Label source: {label_source_counts['primary_endpoint_met']} from "
        f"primary_endpoint_met, {label_source_counts['trial_outcome']} from "
        f"trial_outcome fallback"
    )
    console.print(f"Skipped: {len(skipped)}")
    for nct, reason in skipped[:10]:
        console.print(f"  {nct}: {reason}")

    if not rows:
        return

    auroc = _auroc(probs, labels)
    brier = _brier(probs, labels)
    base_rate = n_pos / len(labels)
    base_brier = base_rate * (1 - base_rate)

    # Threshold-based confusion matrix
    thr = args.threshold
    tp = sum(1 for p, y in zip(probs, labels) if p >= thr and y == 1)
    fp = sum(1 for p, y in zip(probs, labels) if p >= thr and y == 0)
    fn = sum(1 for p, y in zip(probs, labels) if p < thr and y == 1)
    tn = sum(1 for p, y in zip(probs, labels) if p < thr and y == 0)
    acc = (tp + tn) / len(labels)
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")

    summary = Table(title="Headline metrics")
    summary.add_column("metric"); summary.add_column("value", justify="right"); summary.add_column("notes")
    summary.add_row("AUROC", f"{auroc:.3f}", "1.0=perfect, 0.5=random")
    summary.add_row("Brier score", f"{brier:.3f}", f"baseline (predict base rate {base_rate:.2f}): {base_brier:.3f}")
    summary.add_row(f"Accuracy @ p≥{thr}", f"{acc:.3f}", f"chance: {max(base_rate, 1-base_rate):.3f}")
    summary.add_row("Precision (success)", f"{prec:.3f}", "")
    summary.add_row("Recall (success)", f"{rec:.3f}", "")
    summary.add_row("Mean P(success) | label=1", f"{np.mean([p for p,y in zip(probs,labels) if y==1]):.3f}", "want > base rate")
    summary.add_row("Mean P(success) | label=0", f"{np.mean([p for p,y in zip(probs,labels) if y==0]):.3f}", "want < base rate")
    console.print(summary)

    cm = Table(title=f"Confusion matrix at threshold {thr}")
    cm.add_column(""); cm.add_column("pred=success", justify="right"); cm.add_column("pred=failure", justify="right")
    cm.add_row("actual=success", str(tp), str(fn))
    cm.add_row("actual=failure", str(fp), str(tn))
    console.print(cm)

    cal = Table(title="Calibration bins")
    cal.add_column("bin"); cal.add_column("n", justify="right")
    cal.add_column("avg P(pred)", justify="right"); cal.add_column("observed success rate", justify="right")
    for lo, hi, n, ap, obs in _calibration_bins(probs, labels, n_bins=5):
        if n == 0:
            cal.add_row(f"[{lo:.1f}, {hi:.1f})", "0", "—", "—")
        else:
            cal.add_row(f"[{lo:.1f}, {hi:.1f})", str(n), f"{ap:.3f}", f"{obs:.3f}")
    console.print(cal)

    # Worst misses
    rows_sorted = sorted(rows, key=lambda r: -abs(r["p_success"] - r["label"]))
    miss_table = Table(title="Top 10 worst misses (|P(pred) − label|)")
    miss_table.add_column("nct"); miss_table.add_column("label", justify="right")
    miss_table.add_column("P(pred)", justify="right"); miss_table.add_column("compound → indication")
    miss_table.add_column("weakest")
    for r in rows_sorted[:10]:
        miss_table.add_row(
            r["nct"], str(r["label"]), f"{r['p_success']:.3f}",
            f"{r['compound']} → {r['indication']}",
            r["weakest_edge"][:60],
        )
    console.print(miss_table)


if __name__ == "__main__":
    main()
