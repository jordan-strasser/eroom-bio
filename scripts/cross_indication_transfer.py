"""Cross-indication transfer test — the DIRECT north-star measurement.

The thesis: "a mechanism learned in one indication informs another." The random/
temporal holdouts blend within- and cross-indication generalization, so they never
isolate THIS claim. Leave-one-INDICATION-out (LOIO) does: hold out EVERY trial that
touches indication Y from attribution, re-attribute the rest, then predict Y's
trials. Y's own indication-specific edges (biology→Y, endpoint→Y) are now empty
Beta(1,1); the only signal that can rank Y's trials is the SHARED upstream edges
(affects / modulates_via / mechanism_affects) that OTHER indications populated. So
LOIO AUROC > 0.5 = mechanism beliefs transfer across diseases.

Two reads:
  POOLED AUROC over all held-out trials — can the cross-indication signal rank trials
    across the whole held-out set (admits cross-indication base-rate differences).
  MEAN WITHIN-INDICATION AUROC — ranks ONLY within each held-out Y (removes base-rate
    differences). The strictest, purest test of the thesis.
Compared to chance (0.5) and the random-fold holdout (0.534 @ n=250).

    python -m scripts.cross_indication_transfer \
        --initial data/exports/phasec_n250_initial.json \
        --annotated data/exports/phasec_n250_annotated.json \
        --corpus phasec_n250 --min-indication-trials 4
"""
from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
from collections import defaultdict
from pathlib import Path
from statistics import mean

from src.annotation.attributor import _main as attributor_main
from src.graph.store import GraphStore
from src.prediction.path_query import predict_clinical_hypothesis
from scripts.eval_holdout_compose import (
    ANN_DIR,
    _auroc,
    _corpus_ncts,
    _load_canonicalization_cache,
    _overlap_count,
    _resolve_label,
    _training_used_nodes,
    _trial_conditions,
    resolve_chain,
)


def _predict(g: GraphStore, chain: dict, kwargs: dict, n_samples: int):
    """Return (full_chain_prob, upstream_min, upstream_mean) or None.

    full_chain_prob   — predict_clinical_hypothesis as-is (INCLUDES the emptied
                        held-out-indication edges → confounded by ~0.5 drag).
    upstream_min/mean — aggregate over ONLY the NON-EMPTY chain edges (evidence
                        from OTHER indications). This is the FAIR transfer test:
                        the emptied biology→Y / endpoint→Y edges (Beta(1,1),
                        evidence_strength≈0) are dropped so they can't wash out
                        the cross-indication-populated upstream signal under min.
    """
    try:
        r = predict_clinical_hypothesis(
            g, chain["compound_id"], chain["indication_id"],
            n_samples=n_samples, **kwargs,
        )
    except KeyError:
        return None
    if not r.edge_contributions:
        return None
    nonempty = [ec.belief.expected_probability for ec in r.edge_contributions
                if ec.belief.evidence_strength > 1e-6]
    if not nonempty:
        return None
    return (r.overall_probability, min(nonempty), mean(nonempty))


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--initial", required=True)
    ap.add_argument("--annotated", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--min-indication-trials", type=int, default=4,
                    help="score an indication only if it has ≥ this many both-class scorable trials")
    ap.add_argument("--min-overlap", type=int, default=5)
    ap.add_argument("--n-samples", type=int, default=2000)
    ap.add_argument("--out", default="data/dev/cross_indication_transfer_findings.md")
    args = ap.parse_args()

    full = GraphStore()
    full.import_snapshot(args.annotated)
    training_used = _training_used_nodes(full)
    canon = _load_canonicalization_cache()

    # Scorable trials — SAME filters as the holdout AUROC, plus a resolved indication.
    scorable: list[tuple[str, str, dict, dict, str]] = []  # nct,label,chain,kwargs,indication
    for nct in _corpus_ncts(args.corpus):
        ext_p = ANN_DIR / f"{nct}_extraction.json"
        if not ext_p.exists():
            continue
        try:
            extraction = json.loads(ext_p.read_text())
        except json.JSONDecodeError:
            continue
        cls_p = ANN_DIR / f"{nct}_classification.json"
        classification = json.loads(cls_p.read_text()) if cls_p.exists() else None
        label = _resolve_label(extraction, classification)
        if label not in ("success", "failure"):
            continue
        chain = resolve_chain(extraction, _trial_conditions(extraction), full, canon)
        if chain["target_id"] not in training_used or chain["indication_id"] not in training_used:
            continue
        if _overlap_count(chain, training_used) < args.min_overlap:
            continue
        kwargs = {k: chain[k] for k in
                  ("target_id", "mechanism_id", "biology_id", "endpoint_id", "population_id")
                  if chain[k] and chain[k] != "UNKNOWN"}
        scorable.append((nct, label, chain, kwargs, chain["indication_id"]))

    # Group by indication; keep those with ≥min both-class trials.
    by_ind: dict[str, list] = defaultdict(list)
    for row in scorable:
        by_ind[row[4]].append(row)
    eval_inds = {
        ind: rows for ind, rows in by_ind.items()
        if len(rows) >= args.min_indication_trials
        and any(r[1] == "success" for r in rows)
        and any(r[1] == "failure" for r in rows)
    }
    print(f"scorable trials: {len(scorable)} across {len(by_ind)} indications")
    print(f"LOIO-evaluable indications (≥{args.min_indication_trials} both-class): {len(eval_inds)}")

    # All NCTs (ANY chain) per indication → the holdout set when that indication is Y.
    ind_to_all_ncts: dict[str, set[str]] = defaultdict(set)
    for nct, _l, _c, _k, ind in scorable:
        ind_to_all_ncts[ind].add(nct)
    # also catch non-scorable trials of Y via their subgraph chains
    for ts in full.trial_subgraphs.values():
        for ch in ts.chains:
            if ch.indication_id in eval_inds:
                ind_to_all_ncts[ch.indication_id].add(ts.trial_id)

    METHODS = ("full_chain", "upstream_min", "upstream_mean")
    pooled: dict[str, list[tuple[float, int]]] = {m: [] for m in METHODS}
    per_ind_auroc: dict[str, dict[str, float]] = {m: {} for m in METHODS}
    n_scored: dict[str, int] = {}
    for ind in sorted(eval_inds, key=lambda i: -len(eval_inds[i])):
        rows = eval_inds[ind]
        holdout_ncts = sorted(ind_to_all_ncts[ind])
        print(f"  LOIO {ind[:30]:30s}: hold out {len(holdout_ncts)} trials, score {len(rows)}...",
              flush=True)
        with tempfile.NamedTemporaryFile(suffix="_loio.json", delete=False) as fh:
            tmp = fh.name
        try:
            await attributor_main(str(ANN_DIR), args.initial, tmp,
                                  exclude_from_attribution=holdout_ncts)
            gy = GraphStore()
            gy.import_snapshot(tmp)
        finally:
            Path(tmp).unlink(missing_ok=True)
        series: dict[str, list[float]] = {m: [] for m in METHODS}
        labels: list[int] = []
        for nct, label, chain, kwargs, _ind in rows:
            preds = _predict(gy, chain, kwargs, args.n_samples)
            if preds is None:
                continue
            y = 1 if label == "success" else 0
            labels.append(y)
            for m, p in zip(METHODS, preds):
                series[m].append(p)
                pooled[m].append((p, y))
        n_scored[ind] = len(labels)
        if len(set(labels)) == 2 and len(labels) >= 4:
            for m in METHODS:
                per_ind_auroc[m][ind] = _auroc(series[m], labels)

    def pooled_auroc(m: str) -> float:
        ys = [y for _, y in pooled[m]]
        return _auroc([p for p, _ in pooled[m]], ys) if len(set(ys)) == 2 else float("nan")

    def mean_within(m: str) -> float:
        v = per_ind_auroc[m]
        return mean(v.values()) if v else float("nan")

    npooled = len(pooled["full_chain"])
    nys = [y for _, y in pooled["full_chain"]]
    print("\n── Cross-indication transfer (leave-one-indication-out) ──")
    print(f"  held-out indications scored : {len(per_ind_auroc['full_chain'])}")
    print(f"  pooled trials               : {npooled} "
          f"(success={sum(nys)}, failure={npooled-sum(nys)})")
    print(f"  {'method':14s}  POOLED   MEAN-WITHIN   (vs chance 0.5, random-fold 0.534)")
    for m in METHODS:
        tag = {"full_chain": "(confounded)", "upstream_min": "(FAIR test)",
               "upstream_mean": "(FAIR test)"}[m]
        print(f"  {m:14s}  {pooled_auroc(m):.3f}    {mean_within(m):.3f}        {tag}")
    print("\n  per-indication AUROC (upstream_min | upstream_mean):")
    inds_sorted = sorted(per_ind_auroc["upstream_min"],
                         key=lambda i: -per_ind_auroc["upstream_min"][i])
    for ind in inds_sorted:
        print(f"    {ind[:30]:30s} {per_ind_auroc['upstream_min'][ind]:.3f} | "
              f"{per_ind_auroc['upstream_mean'][ind]:.3f}  (n={n_scored[ind]})")

    lines = [
        "# Cross-indication transfer (leave-one-indication-out) — findings\n",
        f"Source: `{args.annotated}` ({args.corpus}), n_samples={args.n_samples}.\n",
        "Hold out EVERY trial touching indication Y from attribution; predict Y's trials "
        "from the SHARED upstream edges other indications populated. AUROC > 0.5 = transfer.\n",
        f"- LOIO-evaluable indications: **{len(per_ind_auroc)}** "
        f"(≥{args.min_indication_trials} both-class trials)",
        f"- pooled held-out trials: **{len(pooled)}** "
        f"(succ {sum(pooled_labels)} / fail {len(pooled)-sum(pooled_labels)})",
        f"- **POOLED LOIO AUROC = {pooled_auroc:.3f}** (chance 0.5; random-fold holdout 0.534)",
        f"- **MEAN within-indication AUROC = {mean_within:.3f}** (strictest — removes base-rate)\n",
        "## per-indication AUROC",
        "| indication | AUROC | n |", "|---|---|---|",
    ]
    for ind, au in sorted(per_ind_auroc.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {ind} | {au:.3f} | {len(eval_inds[ind])} |")
    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
