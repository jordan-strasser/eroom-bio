"""FULL-feature forward-holdout re-test. The earlier learned_aggregator used only
the 6 backbone EFFICACY edges. This adds everything the graph actually carries
that a predictor may legitimately use (NOT the trial's own outcome):

  - 6 backbone efficacy edge beliefs (LOO for train, full for holdout)
  - production efficacy_probability + safety_penalty + overall_probability
    (overall = efficacy x (1-safety); the AE signal I never tested on holdout)
  - modulates_efficacy_of (combo conditioning) mean belief
  - AE structure: n causes_ae, n serious, max target-AE severity
  - population STRUCTURE parsed straight off the PopulationNode (the
    responds_differently belief is hollow, but line/stage/biomarker/gene axes
    are richly populated): per-axis presence flags + feature count
  - is_combination

LEAKAGE GUARD: never use chain.effect_size / p_value / outcome (those ARE the
label). Holdout trials are --exclude-from-attribution so every feature is built
from OTHER trials only (contamination-checked).

Run: python -m scripts.full_feature_holdout \
       --graph data/exports/multi_500_holdout_annotated.json \
       --holdout-ncts data/corpora/holdout_2021_2026.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.graph.models import EdgeType
from src.graph.store import GraphStore
from src.prediction.calibration import auroc, brier_score
from src.prediction.path_query import PredictionEngine, _aggregate_samples, _trust_weight

import scripts.holdout_thesis_analysis as H

ANN = Path("data/annotations")
EFF_EDGES = ["affects", "modulates_via", "mechanism_affects",
             "biology_drives", "reflects_biology", "endpoint_captures"]
POP_AXES = ["line", "stage", "extent", "gene", "biomarker", "severity",
            "functional_class", "prior_tx", "performance", "histology"]
FEATS = (EFF_EDGES
         + ["eff_overall", "safety_penalty", "overall_prob",
            "modeff", "n_ae", "n_serious_ae", "is_combo", "pop_nfeat"]
         + [f"pop_{ax}" for ax in POP_AXES])


def _pop_features(store, pop_id):
    flags = {ax: 0 for ax in POP_AXES}
    n = 0
    if not pop_id or pop_id == "UNKNOWN" or "UNKNOWN" in pop_id:
        return flags, 0
    try:
        nd = store.get_node(pop_id)
    except KeyError:
        return flags, 0
    for f in (nd.get("defining_features") or []):
        ax = f.get("axis") if isinstance(f, dict) else getattr(f, "axis", None)
        ax = getattr(ax, "value", ax)
        if ax in flags:
            flags[ax] = 1
        n += 1
    return flags, n


def _ae_features(store, compound_id):
    n_ae = n_serious = 0
    try:
        for _, _, k, d in store.graph.out_edges(compound_id, keys=True, data=True):
            if k == EdgeType.CAUSES_AE.value:
                n_ae += 1
                tgt = d.get("target_id") if isinstance(d, dict) else None
    except Exception:  # noqa: BLE001
        pass
    # serious flag lives on the AE node; count serious causes_ae targets
    try:
        for _, tgt, k in store.graph.out_edges(compound_id, keys=True):
            if k == EdgeType.CAUSES_AE.value:
                try:
                    if store.get_node(tgt).get("serious"):
                        n_serious += 1
                except KeyError:
                    pass
    except Exception:  # noqa: BLE001
        pass
    return n_ae, n_serious


def trial_features(store, nct, engine, *, exclude_self):
    sg = store.trial_subgraphs.get(nct)
    if not sg or not sg.chains:
        return None
    best = None  # (overall, dict, n_self)
    seen = set()
    for ch in sg.chains:
        ckey = (ch.compound_id, ch.target_id, ch.mechanism_id,
                ch.biology_id, ch.endpoint_id, ch.subgroup_population_id)
        if ckey in seen:
            continue
        seen.add(ckey)
        try:
            res = engine.predict(ch, n_samples=400)
        except Exception:  # noqa: BLE001
            continue
        if not res.edge_contributions:
            continue
        # per-edge LOO means + modeff
        eff = {e: [] for e in EFF_EDGES}
        modeff = []
        n_self = 0
        emeans, eweights = [], []
        for ec in res.edge_contributions:
            et = ec.edge_type.value
            if any(ev.source_id == nct for ev in ec.belief.evidence):
                n_self += 1
            b = H._belief_excluding_set(ec.belief, {nct}, et) if exclude_self else ec.belief
            d = b.alpha + b.beta
            m = b.alpha / d if d > 0 else 0.5
            if et in eff:
                eff[et].append(m)
                emeans.append(m)
                eweights.append(_trust_weight(b))
            elif et == "modulates_efficacy_of":
                modeff.append(m)
        eff_overall = (float(_aggregate_samples([np.array([m]) for m in emeans])[0])
                       if emeans else 0.5)
        flags, nfeat = _pop_features(store, ch.subgroup_population_id)
        n_ae, n_serious = _ae_features(store, ch.compound_id)
        row = {e: (np.mean(eff[e]) if eff[e] else 0.5) for e in EFF_EDGES}
        row.update(
            eff_overall=eff_overall,
            safety_penalty=res.safety_penalty,
            overall_prob=res.overall_probability,
            modeff=float(np.mean(modeff)) if modeff else 0.5,
            n_ae=float(n_ae), n_serious_ae=float(n_serious),
            is_combo=1.0 if ("+" in ch.compound_id or modeff) else 0.0,
            pop_nfeat=float(nfeat),
            **{f"pop_{ax}": float(flags[ax]) for ax in POP_AXES},
        )
        if best is None or res.overall_probability < best[0]:
            best = (res.overall_probability, row, n_self)
    if best is None:
        return None
    return best[1], best[0], best[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--holdout-ncts", required=True)
    a = ap.parse_args()
    store = GraphStore()
    store.import_snapshot(a.graph)
    engine = PredictionEngine(store)
    holdout = {c for ln in Path(a.holdout_ncts).read_text().splitlines()
               if (c := ln.split("#", 1)[0].strip())}

    def label(nct):
        e = H._load_json(ANN / f"{nct}_extraction.json")
        c = H._load_json(ANN / f"{nct}_classification.json")
        return H._resolve_label(e, c) if e else None

    Xtr, ytr = [], []
    Xte, yte, overall_te, eff_te = [], [], [], []
    contam = 0
    for nct in store.trial_subgraphs:
        lab = label(nct)
        if lab not in ("success", "failure"):
            continue
        is_h = nct in holdout
        r = trial_features(store, nct, engine, exclude_self=not is_h)
        if r is None:
            continue
        row, overall, n_self = r
        y = 1 if lab == "success" else 0
        vec = [row[f] for f in FEATS]
        if is_h:
            if n_self:
                contam += 1
            Xte.append(vec); yte.append(y)
            overall_te.append(overall); eff_te.append(row["eff_overall"])
        else:
            Xtr.append(vec); ytr.append(y)

    print(f"graph={a.graph}")
    print(f"train={len(ytr)}  holdout={len(yte)} (succ {sum(yte)}/fail {len(yte)-sum(yte)})  "
          f"contaminated={contam}\n")
    if len(set(yte)) < 2:
        print("one-class holdout"); return

    def show(name, probs):
        print(f"  {name:30s} AUROC {auroc(probs, yte):.3f} | Brier {brier_score(probs, yte):.3f}")

    print("NO-FIT production predictors on holdout:")
    show("efficacy overall (6 edges)", eff_te)
    show("production overall (eff×safety)", overall_te)

    from sklearn.linear_model import LogisticRegression
    Xtr_a, Xte_a = np.array(Xtr), np.array(Xte)
    print("\nLEARNED logistic, fit on train → predict holdout:")
    # (i) 6 efficacy edges only (reproduce prior)
    idx6 = [FEATS.index(e) for e in EFF_EDGES]
    lr6 = LogisticRegression(max_iter=2000).fit(Xtr_a[:, idx6], ytr)
    show("6 efficacy edges (prior)", list(lr6.predict_proba(Xte_a[:, idx6])[:, 1]))
    # (ii) full feature set
    for C in (1.0, 0.1):
        lr = LogisticRegression(max_iter=3000, C=C).fit(Xtr_a, ytr)
        show(f"FULL feature set (C={C})", list(lr.predict_proba(Xte_a)[:, 1]))
    lr = LogisticRegression(max_iter=3000, C=0.1).fit(Xtr_a, ytr)
    coef = sorted(zip(FEATS, lr.coef_[0]), key=lambda x: -abs(x[1]))
    print("\n  top |weight| features (C=0.1):")
    for f, w in coef[:12]:
        print(f"    {f:22s} {w:+.3f}")


if __name__ == "__main__":
    main()
