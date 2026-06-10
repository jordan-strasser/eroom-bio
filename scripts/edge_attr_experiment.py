"""TASK 2 — per-edge attribution-math experiment (owner: "SEE how it changes
the graph").

Re-attributes a pre-attribution ``initial.json`` (default the n=100 neff100
harness) under each ``EROOM_EDGE_ATTR`` mode and the ``EROOM_EDGE_EFFECT``
toggle, EFFICACY-ONLY (no AE / no Anthropic client — the backbone math is what
TASK 2 is about), then compares:

  * per-backbone-edge-type belief distribution — MEAN and SPREAD (std). The
    differentiation (spread across compounds/edges) is the thing the asymmetry
    is supposed to shape; T1 found the over-count inflated CONFIDENCE not MEAN,
    so SPREAD is the headline number.
  * in-sample ``min``-over-stated-chains AUROC (the current default
    aggregation). IN-SAMPLE so the absolute is optimistic, but the optimism is
    shared across modes → the RELATIVE ranking is the signal (same logic as
    phasec_aggregation_diagnostic).

Modes (2a): explain_away (default) | symmetric_full | symmetric_uniform |
symmetric_explain.  Effect (2b): EROOM_EDGE_EFFECT on folds effect_size+p_value
into p_obs/n_eff.

    python -m scripts.edge_attr_experiment \
        --initial data/exports/neff100_initial.json \
        --annotations data/annotations --n-samples 1000
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import mean, pstdev

from scripts.eval_holdout_compose import _auroc, _resolve_label
from src.annotation.attributor import (
    Attributor,
    _load_classifications,
)
from src.annotation.extractor import _parse_extraction_response
from src.annotation.taxonomy import FailureClassification, FailureMode
from src.graph.models import EdgeBeliefState, EdgeType
from src.graph.store import GraphStore
from src.prediction.path_query import PredictionEngine

# The conjunctive causal backbone (what _condition_chain_on_outcomes touches).
BACKBONE = [
    EdgeType.AFFECTS,
    EdgeType.MODULATES_VIA,
    EdgeType.MECHANISM_AFFECTS,
    EdgeType.BIOLOGY_DRIVES,
    EdgeType.REFLECTS_BIOLOGY,
    EdgeType.ENDPOINT_CAPTURES,
]

# (mode, effect) runs: 4 modes effect-off (2a) + explain_away effect-on (2b).
RUNS = [
    ("explain_away", False),
    ("symmetric_full", False),
    ("symmetric_uniform", False),
    ("symmetric_explain", False),
    ("explain_away", True),
]


def reattribute_efficacy(graph: GraphStore, annotations_dir: Path) -> None:
    """Apply EFFICACY attribution (backbone + modulation) to a pre-attribution
    graph in place. Mirrors attributor._main minus the AE phase + PubMed enrich
    (neither touches the conjunctive backbone the experiment measures)."""
    attributor = Attributor(graph)
    for ext_data, clf_data in _load_classifications(annotations_dir):
        trial_id = clf_data.get("nct_id", ext_data.get("nct_id", "unknown"))
        if trial_id in graph.applied_attribution_trial_ids:
            continue
        try:
            trial = graph.get_trial_subgraph_by_id(trial_id)
        except KeyError:
            continue
        modes = clf_data.get("failure_modes", [])
        primary_mode = FailureMode.INSUFFICIENT_INFORMATION
        if modes:
            top = sorted(modes, key=lambda m: m.get("confidence", 0), reverse=True)[0]
            try:
                primary_mode = FailureMode(top["mode"])
            except (ValueError, KeyError):
                pass
        op_fail = clf_data.get("operational_failure")
        if not isinstance(op_fail, bool):
            op_fail = None
        classification = FailureClassification(
            trial_id=trial_id,
            primary_failure_mode=primary_mode,
            confidence=clf_data.get("confidence_overall", 0.5),
            reasoning=clf_data.get("reasoning", ""),
            operational_failure=op_fail,
        )
        classification._raw = clf_data  # type: ignore[attr-defined]
        try:
            extraction = _parse_extraction_response(ext_data, trial_id)
        except Exception:  # noqa: BLE001
            extraction = None
        attributor.attribute(classification, trial, extraction)
        graph.applied_attribution_trial_ids.add(trial_id)


def edge_stats(graph: GraphStore) -> dict[str, dict]:
    """Per-backbone-type belief distribution: mean E[p], std E[p], mean
    evidence_strength, count."""
    out: dict[str, dict] = {}
    for et in BACKBONE:
        eps: list[float] = []
        strengths: list[float] = []
        for d in graph.get_edges_by_type(et):
            b = EdgeBeliefState.model_validate(d["belief"])
            eps.append(b.expected_probability)
            strengths.append(b.evidence_strength)
        if not eps:
            out[et.value] = dict(n=0, mean=float("nan"), std=float("nan"),
                                 strength=float("nan"))
            continue
        out[et.value] = dict(
            n=len(eps),
            mean=mean(eps),
            std=pstdev(eps) if len(eps) > 1 else 0.0,
            strength=mean(strengths),
        )
    return out


def _load_label(annotations: Path, nct: str) -> str | None:
    ext_p = annotations / f"{nct}_extraction.json"
    if not ext_p.exists():
        return None
    extraction = json.loads(ext_p.read_text())
    cls_p = annotations / f"{nct}_classification.json"
    classification = json.loads(cls_p.read_text()) if cls_p.exists() else None
    return _resolve_label(extraction, classification)


def insample_auroc(graph: GraphStore, annotations: Path, n_samples: int):
    """min-over-stated-chains AUROC, in-sample (relative ranking is the signal)."""
    engine = PredictionEngine(graph)
    probs: list[float] = []
    labels: list[int] = []
    for ts in graph.trial_subgraphs.values():
        label = _load_label(annotations, ts.trial_id)
        if label not in ("success", "failure"):
            continue
        chain_probs: list[float] = []
        for ch in ts.chains:
            try:
                res = engine.predict(ch, n_samples=n_samples)
            except KeyError:
                continue
            if not res.edge_contributions:
                continue
            chain_probs.append(res.overall_probability)
        if not chain_probs:
            continue
        probs.append(min(chain_probs))
        labels.append(1 if label == "success" else 0)
    pos = sum(labels)
    if pos == 0 or pos == len(labels) or len(labels) < 4:
        return float("nan"), len(labels), pos
    return _auroc(probs, labels), len(labels), pos


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--initial", default="data/exports/neff100_initial.json")
    ap.add_argument("--annotations", default="data/annotations")
    ap.add_argument("--n-samples", type=int, default=1000)
    ap.add_argument("--out", default="data/dev/edge_attr_experiment_findings.md")
    args = ap.parse_args()

    annotations = Path(args.annotations)
    results = []
    for mode, effect in RUNS:
        os.environ["EROOM_EDGE_ATTR"] = mode
        os.environ["EROOM_EDGE_EFFECT"] = "on" if effect else "off"
        g = GraphStore()
        g.import_snapshot(args.initial)
        reattribute_efficacy(g, annotations)
        stats = edge_stats(g)
        au, n, pos = insample_auroc(g, annotations, args.n_samples)
        results.append((mode, effect, stats, au, n, pos))
        tag = f"{mode}{'+effect' if effect else ''}"
        aff = stats["affects"]
        print(f"\n=== {tag} ===  in-sample AUROC={au:.3f} (n={n}, succ={pos})")
        print(f"  affects: mean={aff['mean']:.3f} std={aff['std']:.3f} "
              f"strength={aff['strength']:.1f} n={aff['n']}")
        for et in BACKBONE:
            s = stats[et.value]
            print(f"  {et.value:20s} mean={s['mean']:.3f} std={s['std']:.3f} "
                  f"strength={s['strength']:.1f} n={s['n']}")

    # Markdown findings
    lines = ["# TASK 2 — per-edge attribution-math experiment\n",
             f"Source: `{args.initial}` re-attributed efficacy-only, "
             f"n_samples={args.n_samples}.\n",
             "IN-SAMPLE AUROC (min over stated chains) — relative ranking is the "
             "signal (shared optimism), NOT an honest out-of-sample number.\n",
             "## Headline — `affects` distribution + AUROC\n",
             "| mode | AUROC | affects mean | affects std | affects strength |",
             "|---|---|---|---|---|"]
    for mode, effect, stats, au, n, pos in results:
        tag = f"{mode}{'+effect' if effect else ''}"
        a = stats["affects"]
        lines.append(f"| {tag} | {au:.3f} | {a['mean']:.3f} | {a['std']:.3f} | "
                     f"{a['strength']:.1f} |")
    lines.append("\n## Per-edge-type mean E[p] (spread in parens)\n")
    lines.append("| edge type | " + " | ".join(
        f"{m}{'+eff' if e else ''}" for m, e, *_ in results) + " |")
    lines.append("|---|" + "|".join("---" for _ in results) + "|")
    for et in BACKBONE:
        cells = []
        for _m, _e, stats, *_ in results:
            s = stats[et.value]
            cells.append(f"{s['mean']:.3f} ({s['std']:.3f})")
        lines.append(f"| {et.value} | " + " | ".join(cells) + " |")
    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
