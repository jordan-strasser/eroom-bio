"""SIGNAL-FLOOR diagnostic: with ORACLE access to each holdout trial's extracted
PRE-TRIAL context, does ANY feature separate success from failure?

This gates the redesign. If nothing separates the trials even with perfect
pre-trial features, the outcomes aren't predictable from available pre-trial
info → rethink the target/holdout. If line-of-therapy / biomarker / stage /
control-type cleanly separate them → THAT is the lever, and we redesign the
graph to carry it.

LEAKAGE DISCIPLINE — features are PRE-TRIAL design/context only. Excluded:
effect_size, p_value, outcome, primary_endpoint_met, classifier failure_modes,
observed AEs. `rationale_strength` is an LLM judgment that MAY encode hindsight,
so it's reported separately and the headline model excludes it.

Oracle features (all knowable before results):
  context: comparator type (placebo/active/single-arm), sample_size, duration
  design:  n_arms, is_combination
  population (data/cache/population_features.json): line (early vs later),
           biomarker/gene selection, stage (early vs metastatic), n prior_tx
  structural: target maturity (# trials sharing the target gene)

Run: python -m scripts.signal_floor_diagnostic \
       --graph data/exports/multi_500_holdout_annotated.json \
       --holdout-ncts data/corpora/holdout_2021_2026.txt
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from src.graph.store import GraphStore
from src.prediction.calibration import auroc

import scripts.holdout_thesis_analysis as H

ANN = Path("data/annotations")
POP_CACHE = Path("data/cache/population_features.json")
_EARLY_LINE = {"first", "adjuvant", "neoadjuvant", "maintenance", "first-line", "1l", "frontline"}
_LATER_LINE = {"second", "third", "later", "relapsed", "refractory", "2l", "3l", "second-line"}
_EARLY_STAGE = {"i", "ii", "iii", "early", "localized", "adjuvant", "resectable", "non-metastatic"}
_METASTATIC = {"iv", "metastatic", "advanced", "unresectable", "metastatic/advanced"}


def _norm(s):
    return (s or "").strip().lower()


def featurize(store, nct, pop_cache):
    ext = H._load_json(ANN / f"{nct}_extraction.json")
    if not ext:
        return None
    ctx = ext.get("context") or {}
    th = ext.get("therapeutic_hypothesis") or {}
    arms = ext.get("arms") or []
    comp = _norm(ctx.get("comparator"))
    f = {}
    f["is_placebo"] = 1.0 if "placebo" in comp else 0.0
    f["is_active_ctrl"] = 1.0 if comp and "placebo" not in comp and comp not in ("none", "", "single", "single-arm", "n/a") else 0.0
    f["is_single_arm"] = 1.0 if comp in ("none", "", "single", "single-arm", "n/a") or not comp else 0.0
    f["sample_size"] = float(ctx.get("sample_size") or 0) or np.nan
    f["duration_wk"] = float(ctx.get("duration_weeks") or 0) or np.nan
    f["n_arms"] = float(len(arms))
    f["is_combo"] = 1.0 if any(len(a.get("compounds") or []) > 1 for a in arms) else 0.0
    # population oracle — parse the FREE-TEXT eligibility (target_population +
    # per-chain population_description). The structured cache doesn't cover these
    # recent NCTs, and ext["subgroups"] holds endpoint landmarks, not strata.
    pop_text = _norm(th.get("target_population"))
    for cr in (ext.get("results_by_chain") or []):
        pd = _norm(cr.get("population_description"))
        if pd and pd not in pop_text:
            pop_text += " | " + pd

    def kw(text, words):
        return 1.0 if any(w in text for w in words) else 0.0

    f["pop_line_early"] = kw(pop_text, ["adjuvant", "neoadjuvant", "residual disease", "resected",
        "after surgery", "perioperative", "maintenance", "consolidation", "first-line", "first line",
        "1l", "treatment-naive", "treatment naive", "previously untreated", "untreated",
        "newly diagnosed", "frontline", "front-line"])
    f["pop_line_later"] = kw(pop_text, ["second-line", "second line", "2l", "third-line", "third line",
        "3l", "previously treated", "pretreated", "pre-treated", "relapsed", "refractory", "recurrent",
        "progressed", "prior therapy", "prior treatment", "after failure", "after progression"])
    f["pop_biomarker_sel"] = kw(pop_text, ["positive", "negative", "mutation", "mutant", "mutated",
        "expressing", "expression", "amplified", "amplification", "her2", "egfr", "alk", "braf", "kras",
        "brca", "pd-l1", "pdl1", "msi", "microsatellite", "biomarker", "harboring", "hormone receptor",
        "hr-", "er-", "er+", "high score", "cps-eg", "high cps"])
    f["pop_stage_early"] = kw(pop_text, ["early", "stage i", "stage ii", "stage iii", "localized",
        "resectable", "non-metastatic", "adjuvant", "neoadjuvant", "residual"])
    f["pop_metastatic"] = kw(pop_text, ["metastatic", "advanced", "stage iv", "stage 4",
        "unresectable", "disseminated", "metastases"])
    f["pop_n_prior_tx"] = f["pop_line_later"]  # proxy: any prior-therapy language
    f["pop_n_features"] = float(len(pop_text) > 0)
    # structural: target maturity
    tgt = _norm(th.get("claimed_target"))
    f["_target"] = tgt
    # possibly-tainted
    rs = _norm(th.get("rationale_strength"))
    f["_rationale_strong"] = 1.0 if rs == "strong" else (0.0 if rs in ("weak", "moderate", "modest") else np.nan)
    return f


def _univ_auroc(vals, y):
    """AUROC of a single feature (NaN-masked). Returns (auroc, n, direction)."""
    v = np.array(vals, float)
    m = ~np.isnan(v)
    if m.sum() < 6 or len(set(np.array(y)[m])) < 2:
        return None
    a = auroc(list(v[m]), list(np.array(y)[m]))
    return a, int(m.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--holdout-ncts", required=True)
    a = ap.parse_args()
    store = GraphStore()
    store.import_snapshot(a.graph)
    holdout = [c for ln in Path(a.holdout_ncts).read_text().splitlines()
               if (c := ln.split("#", 1)[0].strip())]
    pop_cache = json.loads(POP_CACHE.read_text()) if POP_CACHE.exists() else {}

    # target maturity index (structural)
    tgt_trials = defaultdict(set)
    for nct, sg in store.trial_subgraphs.items():
        for ch in sg.chains:
            th = H._load_json(ANN / f"{nct}_extraction.json")

    def label(nct):
        e = H._load_json(ANN / f"{nct}_extraction.json")
        c = H._load_json(ANN / f"{nct}_classification.json")
        return H._resolve_label(e, c) if e else None

    rows, y, ncts = [], [], []
    targets = []
    for nct in holdout:
        lab = label(nct)
        if lab not in ("success", "failure"):
            continue
        f = featurize(store, nct, pop_cache)
        if f is None:
            continue
        rows.append(f); y.append(1 if lab == "success" else 0); ncts.append(nct)
        targets.append(f["_target"])
    # target maturity from the full graph
    tmat_idx = defaultdict(set)
    for nct, sg in store.trial_subgraphs.items():
        e = H._load_json(ANN / f"{nct}_extraction.json")
        t = _norm((e.get("therapeutic_hypothesis") or {}).get("claimed_target")) if e else ""
        if t:
            tmat_idx[t].add(nct)
    for f, nct in zip(rows, ncts):
        f["target_maturity"] = float(len(tmat_idx.get(f["_target"], set()) - {nct}))

    n = len(y)
    print(f"graph={a.graph}\nholdout binary trials: {n} (succ {sum(y)} / fail {n-sum(y)}, base {sum(y)/n:.2f})\n")

    CLEAN = ["is_placebo", "is_active_ctrl", "is_single_arm", "sample_size", "duration_wk",
             "n_arms", "is_combo", "pop_line_early", "pop_line_later", "pop_biomarker_sel",
             "pop_stage_early", "pop_metastatic", "pop_n_prior_tx", "pop_n_features",
             "target_maturity"]

    # ── univariate separation ──
    print("UNIVARIATE separation (|AUROC-0.5| ranked; >0.5 ⇒ feature high→success):")
    print(f"  {'feature':20s} {'AUROC':>7s} {'n':>4s}  {'detail (success-rate by level)'}")
    scored = []
    for name in CLEAN:
        vals = [r.get(name, np.nan) for r in rows]
        res = _univ_auroc(vals, y)
        if res is None:
            continue
        au, nn = res
        scored.append((abs(au - 0.5), au, name, nn, vals))
    for sep, au, name, nn, vals in sorted(scored, reverse=True):
        v = np.array(vals, float)
        detail = ""
        uniq = sorted(set(v[~np.isnan(v)]))
        if len(uniq) <= 3:  # binary/categorical → success rate per level
            parts = []
            for u in uniq:
                msk = v == u
                if msk.sum():
                    parts.append(f"{u:g}:{np.mean(np.array(y)[msk]):.2f}(n{int(msk.sum())})")
            detail = "  ".join(parts)
        print(f"  {name:20s} {au:>7.3f} {nn:>4d}  {detail}")

    # ── multivariate oracle ceiling + generalization ──
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    X = np.array([[r.get(k, np.nan) for k in CLEAN] for r in rows], float)
    Xi = SimpleImputer(strategy="median").fit_transform(X)
    Xs = StandardScaler().fit_transform(Xi)
    yv = np.array(y)

    insample = LogisticRegression(max_iter=3000, C=1.0).fit(Xs, yv).predict_proba(Xs)[:, 1]
    # LOO-CV (n small)
    oof = np.zeros(n)
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        lr = LogisticRegression(max_iter=3000, C=0.5).fit(Xs[tr], yv[tr])
        oof[i] = lr.predict_proba(Xs[i:i+1])[:, 1][0]
    print(f"\nMULTIVARIATE oracle (clean pre-trial features, n={n}):")
    print(f"  in-sample fit AUROC : {auroc(list(insample), y):.3f}  (overfit CEILING — if ~0.5, NO signal exists at all)")
    print(f"  LOO-CV     AUROC    : {auroc(list(oof), y):.3f}  (generalizable signal)")

    # with the possibly-tainted rationale_strength, for contrast
    rs = np.array([r.get("_rationale_strong", np.nan) for r in rows], float)
    rres = _univ_auroc(list(rs), y)
    if rres:
        print(f"\n  [flagged, possible hindsight] rationale_strength univariate AUROC: {rres[0]:.3f} (n={rres[1]})")

    print("\nINTERPRETATION:")
    print("  in-sample≈0.5  → outcomes unpredictable from pre-trial context (rethink target/holdout)")
    print("  in-sample high, LOO≈0.5 → signal exists but n too small to generalize (need more labels)")
    print("  LOO>0.6 on a feature → that feature is the lever; redesign graph to carry it")


if __name__ == "__main__":
    main()
