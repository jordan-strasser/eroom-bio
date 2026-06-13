"""B1 Step 1 — HONEST field leave-one-fold-out with per-fold re-materialization.

The existing eval_holdout_kfold `--field` does anchor-drop LOO on the FULL
annotated field, whose marginal fallback still carries the held-out trial's
scalar (leaky on singleton edges — exactly where the field has no anchors left
and reverts to the pooled marginal). This harness closes that leak: per fold it
(1) re-attributes the initial EXCLUDING the fold (clean scalar — same discipline
as eval_holdout_kfold:105), (2) re-materializes the (s,t) field on that clean
graph (anchors AND marginal exclude the fold), (3) predicts the held-out trials
both scalar and field. Apples-to-apples on one clean graph.

Usage:
  .venv/bin/python -m scratch.diagnostics.field_holdout \
      --initial data/exports/multi_500_initial.json \
      --annotated data/exports/multi_500_annotated.json \
      --corpus multi_500 --k 5 [--folds 0]   # --folds limits for a cost probe
"""
from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
import time
from pathlib import Path

from src.annotation.attributor import _main as attributor_main
from src.graph.store import GraphStore
from src.graph.biolord_embeddings import embed_text
from src.prediction.field_prediction import (
    build_st_desc_map,
    load_edge_fields,
    localized_chain_probability,
)
from src.prediction.path_query import predict_clinical_hypothesis
from scripts.materialize_belief_field import materialize_field
from scripts.eval_holdout_kfold import _fold, _predict
from scripts.eval_holdout_compose import (
    ANN_DIR,
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


def _scorable(full: GraphStore, corpus: str, min_overlap: int):
    training_used = _training_used_nodes(full)
    canon = _load_canonicalization_cache()
    out = []
    for nct in _corpus_ncts(corpus):
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
        if chain["target_id"] not in training_used:
            continue
        if chain["indication_id"] not in training_used:
            continue
        if _overlap_count(chain, training_used) < min_overlap:
            continue
        kwargs = {
            k: chain[k] for k in
            ("target_id", "mechanism_id", "biology_id", "endpoint_id", "population_id")
            if chain[k] and chain[k] != "UNKNOWN"
        }
        out.append((nct, label, chain, kwargs))
    return out


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--initial", required=True)
    ap.add_argument("--annotated", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--min-overlap", type=int, default=5)
    ap.add_argument("--n-samples", type=int, default=2000)
    ap.add_argument("--folds", type=int, nargs="*", default=None,
                    help="restrict to these fold indices (cost probe); default all")
    ap.add_argument("--out", default="scratch/diagnostics/field_holdout_results.json")
    args = ap.parse_args()

    full = GraphStore()
    full.import_snapshot(args.annotated)
    scorable = _scorable(full, args.corpus, args.min_overlap)
    n_succ = sum(1 for _, lab, _, _ in scorable if lab == "success")
    print(f"scorable: {len(scorable)} (success={n_succ}, failure={len(scorable)-n_succ})",
          flush=True)

    folds_map: dict[int, list] = {}
    for row in scorable:
        folds_map.setdefault(_fold(row[0], args.k), []).append(row)
    want = set(args.folds) if args.folds is not None else set(folds_map)

    rows = []  # (nct, label, p_scalar_holdout, p_field_holdout)
    cache: dict = {}
    for f in sorted(folds_map):
        if f not in want:
            continue
        fold_rows = folds_map[f]
        fold_ncts = [r[0] for r in fold_rows]
        t0 = time.time()
        print(f"fold {f+1}/{args.k}: re-attribute excluding {len(fold_ncts)} "
              f"trials, re-materialize field...", flush=True)
        with tempfile.NamedTemporaryFile(suffix=f"_fld{f}.json", delete=False) as fh:
            tmp = fh.name
        try:
            await attributor_main(str(ANN_DIR), args.initial, tmp,
                                  exclude_from_attribution=fold_ncts)
            t_attr = time.time()
            mat = materialize_field(tmp)  # writes field INTO tmp, mutates+exports
            t_mat = time.time()
            gf = GraphStore()
            gf.import_snapshot(tmp)
            field_map = load_edge_fields(tmp)
            print(f"  re-attr {t_attr-t0:.0f}s, materialize {t_mat-t_attr:.0f}s "
                  f"({mat['edges_localized']} edges, {mat['anchors_total']} anchors)",
                  flush=True)
        finally:
            Path(tmp).unlink(missing_ok=True)

        for nct, label, ch, kw in fold_rows:
            try:
                result = predict_clinical_hypothesis(
                    gf, ch["compound_id"], ch["indication_id"],
                    n_samples=args.n_samples, **kw,
                )
            except KeyError:
                continue
            if not result.edge_contributions:
                continue
            sf = 1.0 - result.safety_penalty
            p_scalar = result.overall_probability
            st_map = build_st_desc_map(gf, nct)
            if st_map:
                _ps, pl, _pe = localized_chain_probability(
                    result.edge_contributions, field_map, st_map,
                    embed_fn=embed_text, embed_cache=cache,
                )
                p_field = pl * sf
            else:
                p_field = p_scalar
            rows.append((nct, label, p_scalar, p_field))
        print(f"  fold {f+1} done in {time.time()-t0:.0f}s "
              f"(cumulative scored={len(rows)})", flush=True)

    y = [1 if lab == "success" else 0 for _, lab, _, _ in rows]
    if rows and 0 < sum(y) < len(y):
        sc = [r[2] for r in rows]
        fl = [r[3] for r in rows]
        s_au, f_au = _auroc(sc, y), _auroc(fl, y)
        s_acc, *_ = _binary_accuracy(sc, y)
        f_acc, *_ = _binary_accuracy(fl, y)
        print(f"\n── HONEST field holdout (n={len(rows)}, succ={sum(y)}, "
              f"fail={len(y)-sum(y)}) ──")
        print(f"  scalar holdout AUROC = {s_au:.3f}  (acc {s_acc:.3f})")
        print(f"  FIELD  holdout AUROC = {f_au:.3f}  (acc {f_acc:.3f})")
        print(f"  field − scalar       = {f_au - s_au:+.3f}")
        print(f"  vs discrete baseline 0.565")
    else:
        print("too few scored trials for AUROC")

    Path(args.out).write_text(json.dumps(
        {"rows": rows, "folds": sorted(want)}, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
