"""K-fold TRUE-holdout AUROC — the honest out-of-sample counterpart to the
in-sample corpus AUROC in eval_holdout_compose.

WHY re-attribution, not belief-replay: stripping a trial's evidence records and
replaying Beta(alpha,beta) does NOT faithfully reconstruct the stored belief —
node-merge concatenates evidence across merged nodes and the AE SOC roll-up is a
separate propagation path, so a sequential replay diverges (a self-check proves
it). The exact way is to RE-ATTRIBUTE the pre-attribution `initial.json` with
the held-out trials excluded (`--exclude-from-attribution` pre-marks them so
attribution skips them and their evidence never lands), then predict them
against the resulting graph — real generalization through the real pipeline.

SCOPE: an incrementally-appended snapshot (base + --add-corpus) freezes the
base's attribution into `initial.json`'s beliefs, so only the APPENDED trials
are cleanly excludable. This holds out each appended trial from a graph trained
on the base + the other appended trials — a valid out-of-sample estimate over
the fresh slice. A full-corpus holdout needs a from-scratch initial (follow-up).

Scalar only (field would need per-fold re-materialization). Reports holdout
AUROC + the in-sample→holdout gap = the leakage the in-sample number hides.

    python -m scripts.eval_holdout_kfold \
        --initial   data/exports/onco_scale_500_initial.json \
        --annotated data/exports/onco_scale_500_annotated.json \
        --corpus    onco_scale_500 --holdout-corpus onco_scale_248_add --k 5
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import tempfile
from pathlib import Path

from src.annotation.attributor import _main as attributor_main
from src.graph.store import GraphStore
from src.prediction.field_prediction import (
    build_st_desc_map,
    load_edge_fields,
    localized_chain_probability,
)
from src.prediction.path_query import (
    PredictionEngine,
    predict_clinical_hypothesis,
    seed_prediction_rng,
)
from scripts.eval_holdout_compose import (
    ANN_DIR,
    CORPORA_DIR,
    _auroc,
    _binary_accuracy,
    _corpus_ncts,
    _load_canonicalization_cache,
    _overlap_count,
    _resolve_label,
    _training_used_nodes,
    _trial_conditions,
    resolve_chain,
)


def _fold(nct: str, k: int) -> int:
    # deterministic, balanced-ish fold assignment (no Math.random / Date in build)
    return int(hashlib.md5(nct.encode()).hexdigest(), 16) % k


def _predict(g: GraphStore, chain: dict, kwargs: dict, n_samples: int):
    try:
        r = predict_clinical_hypothesis(
            g, chain["compound_id"], chain["indication_id"],
            n_samples=n_samples, **kwargs,
        )
    except KeyError:
        return None
    return r.overall_probability if r.edge_contributions else None


async def graph_holdout_predictions(
    full: GraphStore,
    scorable: list[tuple[str, str, dict, dict]],
    initial: str,
    k: int,
    n_samples: int,
    annotations_dir: str | None = None,
    seed: int | None = 42,
):
    """Per-trial graph predictions, in-sample and TRUE K-fold holdout.

    Returns ``(insample, holdout)`` dicts ``nct -> prob | None``. The holdout
    re-attributes the pre-attribution ``initial`` once per fold, excluding that
    fold's trials, then predicts them — real generalization through the real
    pipeline. Factored out of ``main`` so the baselines harness can score the
    graph on the IDENTICAL folds it scores the ML/base-rate models on (paired
    DeLong needs same trials, same fold assignment)."""
    ann = annotations_dir or str(ANN_DIR)
    # Pin the prediction RNG so the holdout AUROC is exactly reproducible. The
    # scorable order is deterministic (corpus-file order), so a single seeded
    # stream shared across all predict() calls yields identical numbers per run.
    seed_prediction_rng(seed)
    insample = {nct: _predict(full, ch, kw, n_samples) for nct, _, ch, kw in scorable}
    folds: dict[int, list] = {}
    for row in scorable:
        folds.setdefault(_fold(row[0], k), []).append(row)
    holdout: dict[str, float | None] = {}
    for f in sorted(folds):
        fold_rows = folds[f]
        fold_ncts = [r[0] for r in fold_rows]
        print(f"  fold {f + 1}/{k}: re-attribute initial excluding "
              f"{len(fold_ncts)} held-out trials...", flush=True)
        with tempfile.NamedTemporaryFile(suffix=f"_kfold{f}.json", delete=False) as fh:
            tmp = fh.name
        try:
            await attributor_main(ann, initial, tmp, exclude_from_attribution=fold_ncts)
            gf = GraphStore()
            gf.import_snapshot(tmp)
        finally:
            Path(tmp).unlink(missing_ok=True)
        for nct, _label, ch, kw in fold_rows:
            holdout[nct] = _predict(gf, ch, kw, n_samples)
    return insample, holdout


def _field_loo(scorable, full: GraphStore, field_path: str, n_samples: int):
    """Field leave-one-out via anchor-drop (no re-attribution needed — the
    kernel query is additive, so removing a trial's (s,t) anchors removes
    exactly its contribution). Returns rows (nct, label, p_field_insample,
    p_field_loo) for trials whose chain localizes against the field."""
    from src.graph.biolord_embeddings import embed_text

    field_map = load_edge_fields(field_path)
    cache: dict = {}
    rows = []
    for nct, label, chain, kwargs in scorable:
        try:
            result = predict_clinical_hypothesis(
                full, chain["compound_id"], chain["indication_id"],
                n_samples=n_samples, **kwargs,
            )
        except KeyError:
            continue
        if not result.edge_contributions:
            continue
        st_map = build_st_desc_map(full, nct)
        if not st_map:
            continue
        _ps, pl_in, per_edge = localized_chain_probability(
            result.edge_contributions, field_map, st_map,
            embed_fn=embed_text, embed_cache=cache,
        )
        if not any(e["is_localized"] for e in per_edge):
            continue
        # LOO: drop THIS trial's anchors on the chain edges, re-query.
        keys = {(ec.source_id, ec.target_id, ec.edge_type.value)
                for ec in result.edge_contributions}
        loo_map = dict(field_map)
        for k in keys:
            if k in loo_map:
                loo_map[k] = loo_map[k].without_trial(nct)
        _ps2, pl_loo, _pe2 = localized_chain_probability(
            result.edge_contributions, loo_map, st_map,
            embed_fn=embed_text, embed_cache=cache,
        )
        sf = 1.0 - result.safety_penalty
        rows.append((nct, label, pl_in * sf, pl_loo * sf))
    return rows


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--initial", required=True, help="pre-attribution initial.json")
    ap.add_argument("--annotated", required=True,
                    help="trained annotated.json (chain resolution + in-sample)")
    ap.add_argument("--corpus", required=True, help="full corpus (label/filter source)")
    ap.add_argument("--holdout-corpus", default=None,
                    help="restrict the held-out set to these NCTs (the excludable "
                         "appended slice). Defaults to the full corpus.")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--min-overlap", type=int, default=5)
    ap.add_argument("--n-samples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42,
                    help="seed for the prediction RNG so holdout AUROC is "
                         "exactly reproducible (Monte-Carlo Beta sampling).")
    ap.add_argument("--field", default=None,
                    help="(s,t) belief-field snapshot (nct-tagged anchors). Adds "
                         "a FIELD leave-one-out via anchor-drop — full-corpus "
                         "capable, no re-attribution.")
    args = ap.parse_args()

    full = GraphStore()
    full.import_snapshot(args.annotated)
    training_used = _training_used_nodes(full)
    canon = _load_canonicalization_cache()

    holdout_ids = (
        set(_corpus_ncts(args.holdout_corpus)) if args.holdout_corpus else None
    )

    # Scorable = passes the SAME filters as the in-sample AUROC, with a clean
    # label, and (if a holdout-corpus is given) is in the excludable slice.
    scorable: list[tuple[str, str, dict, dict]] = []
    for nct in _corpus_ncts(args.corpus):
        if holdout_ids is not None and nct not in holdout_ids:
            continue
        ext_p = ANN_DIR / f"{nct}_extraction.json"
        if not ext_p.exists():
            continue
        try:
            extraction = json.loads(ext_p.read_text())
        except json.JSONDecodeError:
            continue
        cls_p = ANN_DIR / f"{nct}_classification.json"
        classification = (
            json.loads(cls_p.read_text()) if cls_p.exists() else None
        )
        label = _resolve_label(extraction, classification)
        if label not in ("success", "failure"):
            continue
        chain = resolve_chain(extraction, _trial_conditions(extraction), full, canon)
        if chain["target_id"] not in training_used:
            continue
        if chain["indication_id"] not in training_used:
            continue
        if _overlap_count(chain, training_used) < args.min_overlap:
            continue
        kwargs = {
            k: chain[k] for k in
            ("target_id", "mechanism_id", "biology_id", "endpoint_id", "population_id")
            if chain[k] and chain[k] != "UNKNOWN"
        }
        scorable.append((nct, label, chain, kwargs))

    n_succ = sum(1 for _, lab, _, _ in scorable if lab == "success")
    print(f"scorable held-out trials: {len(scorable)} "
          f"(success={n_succ}, failure={len(scorable) - n_succ})", flush=True)
    if len(scorable) < 4 or n_succ == 0 or n_succ == len(scorable):
        print("not enough labeled held-out trials for AUROC")
        return 1

    # In-sample + TRUE K-fold holdout (re-attribute per fold excluding it).
    insample, holdout = await graph_holdout_predictions(
        full, scorable, args.initial, args.k, args.n_samples, seed=args.seed
    )

    # Report over trials scored BOTH ways.
    pairs = [
        (insample[n], holdout[n], lab)
        for n, lab, _, _ in scorable
        if insample.get(n) is not None and holdout.get(n) is not None
    ]
    if len(pairs) < 4:
        print("too few trials scored both in-sample and holdout")
        return 1
    is_p = [a for a, _, _ in pairs]
    ho_p = [b for _, b, _ in pairs]
    y = [1 if lab == "success" else 0 for _, _, lab in pairs]
    is_auroc = _auroc(is_p, y)
    ho_auroc = _auroc(ho_p, y)
    ho_acc, tp, tn, fp, fn = _binary_accuracy(ho_p, y)

    print(f"\n── K-fold true-holdout (K={args.k}, n={len(pairs)}, "
          f"success={sum(y)}, failure={len(y) - sum(y)}) ──")
    print(f"  AUROC in-sample          = {is_auroc:.3f}   (upper bound — leakage)")
    print(f"  AUROC holdout            = {ho_auroc:.3f}   (TRUE out-of-sample)")
    print(f"  in-sample → holdout gap  = {is_auroc - ho_auroc:+.3f}")
    print(f"  holdout binary acc       = {ho_acc:.3f}   "
          f"(TP={tp}, TN={tn}, FP={fp}, FN={fn})")

    if args.field:
        print("\n  computing FIELD leave-one-out (anchor-drop)...", flush=True)
        frows = _field_loo(scorable, full, args.field, args.n_samples)
        fy = [1 if lab == "success" else 0 for _, lab, _, _ in frows]
        if frows and 0 < sum(fy) < len(fy):
            f_in = _auroc([a for _, _, a, _ in frows], fy)
            f_lo = _auroc([b for _, _, _, b in frows], fy)
            print(f"\n── FIELD true-holdout (anchor-drop, n={len(frows)}, "
                  f"success={sum(fy)}, failure={len(fy) - sum(fy)}) ──")
            print(f"  AUROC field in-sample    = {f_in:.3f}")
            print(f"  AUROC field holdout      = {f_lo:.3f}   (drop each trial's (s,t) anchors)")
            print(f"  field in-sample → holdout gap = {f_in - f_lo:+.3f}")
        else:
            print("  field-LOO: too few localizable scored trials for AUROC")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
