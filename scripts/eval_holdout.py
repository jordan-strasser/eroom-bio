"""True out-of-sample evaluation against a held-out melanoma set.

The trained graph (data/exports/oncology_annotated.json) is built from
the first N trials in the melanoma_145 corpus. The remaining trials are
the "holdout" set — they were never part of the training graph and
their evidence isn't routed into the per-edge Beta posteriors.

For each holdout trial we:
  1. fetch from CT.gov (cached) and extract + classify (caches reused
     when present; LLM only fires for fresh trials)
  2. derive a (compound_id, indication_id, endpoint_id) hypothesis
     from the extraction
  3. predict P(success) against the trained graph
  4. compare to the trial's combined-source binary label
     (primary_endpoint_met preferred, trial_outcome fallback —
     same convention as scripts/eval_predictions.py)

This is the cleanest available eroom-bio benchmark: the graph genuinely
has no information about the holdout trial when scoring it. LOO via
evidence-stripping (eval_predictions_loo.py) is a near-OOS proxy that
still leaks via shared edges; this script removes that leakage.

Usage:

    python -m scripts.eval_holdout
    python -m scripts.eval_holdout --max-fresh 0   # only score cached holdout

`--max-fresh` caps how many holdout trials may trigger fresh extraction/
classification LLM calls. Default unlimited; set to 0 to score only
trials with both extraction and classification already cached on disk.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from pathlib import Path

import anthropic
import numpy as np
from rich.console import Console
from rich.table import Table

from src.annotation.classifier import Classifier
from src.annotation.extractor import Extractor
from src.graph.models import normalize_entity
from src.graph.store import GraphStore
from src.ingestion.clinicaltrials import ClinicalTrialsClient
from src.prediction.path_query import predict_clinical_hypothesis

console = Console()


# ── Metrics (mirrors eval_predictions.py) ────────────────────────────────


def _brier(probs: list[float], labels: list[int]) -> float:
    return float(np.mean([(p - y) ** 2 for p, y in zip(probs, labels)]))


def _auroc(probs: list[float], labels: list[int]) -> float:
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
        lo, hi = i / n_bins, (i + 1) / n_bins
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


# ── Label resolution (mirrors eval_predictions.py) ──────────────────────


def _resolve_label(extraction_dict: dict, classification_dict: dict | None) -> int | None:
    """Combine extractor's primary_endpoint_met and classifier's
    trial_outcome. Returns 1 (success), 0 (failure), or None (ambiguous
    / partial / safety-only)."""
    met = (extraction_dict.get("results") or {}).get("primary_endpoint_met")
    if met is True:
        return 1
    if met is False:
        return 0
    if classification_dict is None:
        return None
    outcome = (classification_dict.get("trial_outcome") or "").lower()
    if outcome == "success":
        return 1
    if outcome == "failure":
        return 0
    return None


# ── Holdout derivation ──────────────────────────────────────────────────


def _holdout_ncts(corpus_path: Path, snapshot_path: Path) -> list[str]:
    """Corpus NCTs minus the trials present in the trained snapshot."""
    snap = json.loads(snapshot_path.read_text())
    in_slice = set(snap.get("trial_subgraphs", {}).keys())
    corpus: list[str] = []
    for line in corpus_path.read_text().splitlines():
        clean = line.split("#", 1)[0].strip()
        if clean.startswith("NCT"):
            corpus.append(clean)
    return [n for n in corpus if n not in in_slice]


# ── Hypothesis derivation from a held-out extraction ────────────────────


def _derive_hypothesis(
    extraction_dict: dict, trial_record, graph: GraphStore,
) -> tuple[str, str, str | None] | None:
    """Pick (compound_id, indication_id, endpoint_id) for prediction.

    Compound resolution tries (in order): (1) exact slug match for any
    constituent in arms[].compounds, (2) the trained graph's name index
    via fuzzy name resolution. Returns None when nothing resolves.
    Indication comes from the TRIAL record's conditions (extraction's
    `conditions` field isn't always populated).
    """
    # Compound — collect candidate names from arms (preferred) +
    # therapeutic_hypothesis.compound. Try each against the graph.
    candidate_names: list[str] = []
    for arm in (extraction_dict.get("arms") or []):
        for c in (arm.get("compounds") or []):
            if c and c.lower() != "placebo":
                candidate_names.append(c)
    th = extraction_dict.get("therapeutic_hypothesis") or {}
    if th.get("compound"):
        candidate_names.append(th["compound"])

    compound_id: str | None = None
    for name in candidate_names:
        try:
            slug = normalize_entity(name, "InterventionNode")
        except ValueError:
            continue
        try:
            graph.get_node(slug)
            compound_id = slug
            break
        except KeyError:
            pass
        # Fuzzy: ask the graph's name index for any compound that knows
        # this name as an alias. _entity_index uses normalized lookup.
        key = (_eval_normalize(name), "compound")
        idx = getattr(graph, "_entity_index", None)
        if idx and key in idx:
            compound_id = idx[key]
            break
    if compound_id is None:
        return None

    # Indication — pull from the trial record, not the extraction
    # (which doesn't always carry it).
    conditions = list(getattr(trial_record, "conditions", []) or [])
    if not conditions:
        return None
    indication_id: str | None = None
    for cond in conditions:
        try:
            slug = normalize_entity(cond, "IndicationNode")
        except ValueError:
            continue
        try:
            graph.get_node(slug)
            indication_id = slug
            break
        except KeyError:
            pass
        idx = getattr(graph, "_entity_index", None)
        if idx and (_eval_normalize(cond), "indication") in idx:
            indication_id = idx[(_eval_normalize(cond), "indication")]
            break
    if indication_id is None:
        return None

    return compound_id, indication_id, None


def _eval_normalize(text: str) -> str:
    """Mirror src.graph.populate._normalize without the cross-script import."""
    import re
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


# ── Main driver ─────────────────────────────────────────────────────────


async def _score_holdout(
    ncts: list[str],
    graph: GraphStore,
    extractor: Extractor,
    classifier: Classifier,
    annotations_dir: Path,
    max_fresh: int | None,
) -> tuple[list[dict], list[tuple[str, str]]]:
    """Extract+classify+predict each holdout NCT. Returns (rows, skipped)."""
    rows: list[dict] = []
    skipped: list[tuple[str, str]] = []
    ct = ClinicalTrialsClient()
    fresh = 0

    for nct in ncts:
        ext_path = annotations_dir / f"{nct}_extraction.json"
        cls_path = annotations_dir / f"{nct}_classification.json"

        # Cache check — if neither extraction nor classification exist,
        # this trial needs fresh LLM calls.
        needs_fresh = not (ext_path.exists() and cls_path.exists())
        if needs_fresh and max_fresh is not None and fresh >= max_fresh:
            skipped.append((nct, "would exceed --max-fresh budget"))
            continue

        try:
            trial_record = await ct.get_study(nct)
        except Exception as exc:
            skipped.append((nct, f"CT.gov fetch failed: {exc}"))
            continue
        if trial_record is None:
            skipped.append((nct, "no CT.gov record"))
            continue

        try:
            extraction = await extractor.extract(trial_record)
        except Exception as exc:
            skipped.append((nct, f"extract failed: {exc}"))
            continue
        if needs_fresh and not ext_path.exists():
            fresh += 1

        try:
            classification = await classifier.classify(
                extraction, graph=graph,
            )
        except Exception as exc:
            skipped.append((nct, f"classify failed: {exc}"))
            continue
        if needs_fresh and not cls_path.exists():
            fresh += 1

        # Re-read the on-disk JSON for label resolution (we want the
        # raw extractor / classifier output, not the parsed models).
        try:
            extraction_dict = json.loads(ext_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            skipped.append((nct, "extraction file missing post-extract"))
            continue
        classification_dict = None
        if cls_path.exists():
            try:
                classification_dict = json.loads(cls_path.read_text())
            except json.JSONDecodeError:
                pass

        label = _resolve_label(extraction_dict, classification_dict)
        if label is None:
            skipped.append((nct, "no resolvable label (partial / safety)"))
            continue

        hyp = _derive_hypothesis(extraction_dict, trial_record, graph)
        if hyp is None:
            skipped.append((nct, "compound/indication not in trained graph"))
            continue
        compound_id, indication_id, endpoint_id = hyp

        try:
            result = predict_clinical_hypothesis(
                graph, compound_id, indication_id,
                endpoint_id=endpoint_id, n_samples=5000,
            )
        except KeyError as exc:
            skipped.append((nct, f"predict KeyError: {exc}"))
            continue

        rows.append({
            "nct": nct,
            "label": label,
            "p_success": result.overall_probability,
            "ci_lo": result.ci_lower,
            "ci_hi": result.ci_upper,
            "compound": compound_id,
            "indication": indication_id,
            "weakest": (
                f"{result.weakest_link.edge_type.value}: "
                f"{result.weakest_link.source_id} → "
                f"{result.weakest_link.target_id}"
                if result.weakest_link else "—"
            ),
        })
    return rows, skipped


async def main_async(args: argparse.Namespace) -> int:
    graph = GraphStore()
    graph.import_snapshot(str(args.graph))

    holdout = _holdout_ncts(args.corpus, args.graph)
    console.print(
        f"[bold]Holdout NCTs: {len(holdout)}[/bold] "
        f"(corpus minus trained-snapshot slice)"
    )

    client = anthropic.AsyncAnthropic(timeout=60.0)
    extractor = Extractor(client)
    classifier = Classifier(client)

    rows, skipped = await _score_holdout(
        holdout, graph, extractor, classifier,
        args.annotations, args.max_fresh,
    )

    console.rule("[bold]Holdout eval results[/bold]")
    console.print(
        f"Predicted: {len(rows)}  |  Skipped: {len(skipped)}"
    )
    for nct, reason in skipped[:15]:
        console.print(f"  {nct}: {reason}")

    if not rows:
        console.print("[yellow]No rows to score.[/yellow]")
        return 0

    probs = [r["p_success"] for r in rows]
    labels = [r["label"] for r in rows]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        console.print(
            f"[yellow]Class imbalance: {n_pos} pos, {n_neg} neg — "
            f"AUROC not computable.[/yellow]"
        )
        return 0

    auroc = _auroc(probs, labels)
    brier = _brier(probs, labels)
    base = n_pos / len(labels)

    summary = Table(title=f"Holdout metrics (n={len(rows)})")
    summary.add_column("metric"); summary.add_column("value", justify="right")
    summary.add_column("notes")
    summary.add_row("AUROC", f"{auroc:.3f}", "1.0=perfect, 0.5=random")
    summary.add_row(
        "Brier", f"{brier:.3f}",
        f"base-rate baseline ({base:.2f}): {base*(1-base):.3f}",
    )
    summary.add_row(
        "Mean P | label=1", f"{np.mean([p for p,y in zip(probs,labels) if y==1]):.3f}",
        "want > base rate",
    )
    summary.add_row(
        "Mean P | label=0", f"{np.mean([p for p,y in zip(probs,labels) if y==0]):.3f}",
        "want < base rate",
    )
    console.print(summary)

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

    rows_sorted = sorted(rows, key=lambda r: -abs(r["p_success"] - r["label"]))
    miss = Table(title="Top 10 worst holdout misses")
    miss.add_column("nct"); miss.add_column("label", justify="right")
    miss.add_column("P(pred)", justify="right")
    miss.add_column("compound → indication"); miss.add_column("weakest")
    for r in rows_sorted[:10]:
        miss.add_row(
            r["nct"], str(r["label"]), f"{r['p_success']:.3f}",
            f"{r['compound']} → {r['indication']}",
            r["weakest"][:50],
        )
    console.print(miss)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph", type=Path,
        default=Path("data/exports/oncology_annotated.json"),
        help="Trained graph snapshot.",
    )
    parser.add_argument(
        "--corpus", type=Path,
        default=Path("data/corpora/melanoma_145.txt"),
        help="Frozen NCT-id corpus file.",
    )
    parser.add_argument(
        "--annotations", type=Path, default=Path("data/annotations"),
        help="Annotation cache directory (holdout extracts/classifies "
             "live here too, but the trained graph excludes them since "
             "they're not in its trial_subgraphs).",
    )
    parser.add_argument(
        "--max-fresh", type=int, default=None,
        help="Cap on how many holdout trials may trigger fresh LLM "
             "extraction+classification (cost control). Default: no cap.",
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
