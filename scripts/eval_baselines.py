"""Prediction benchmark — base rates + ML baselines vs the graph, on IDENTICAL folds.

Answers the external critique "an LLM/XGBoost/LogReg may match or beat the structured
graph." Full design in BENCHMARK.md. This harness reuses eval_holdout_kfold's folds,
labels, and scorable-set construction so every method is scored on the SAME held-out
trials → paired DeLong/McNemar (the only valid test at n≈129).

THREE things, one harness:
  1. Base-rate FLOORS (must-beat): marginal, per-indication, per-target. BENCHMARK.md
     says the first baselines to beat are base rates, not XGBoost.
  2. ML on the graph's OWN extracted fields (level (b), the fair head-to-head):
     LogReg + XGBoost on the resolved chain ids (compound/target/mechanism/biology/
     indication/endpoint/population) — the exact inputs the graph is built from, so a
     flat model on them isolates the value of typed structure + Bayesian accumulation
     + softmin.
  3. CEILING analysis: XGBoost on non-mechanistic DESIGN fields (enrollment, duration,
     single-arm, combo-size, rationale) vs mechanistic vs all — how much of trial
     success is even mechanistic (BENCHMARK.md Q1).

FAIRNESS — mirror the graph exactly. The graph's K-fold holds out only the appended
slice (the base's attribution is frozen in initial.json); base trials are always in
its training set. So ML's per-fold training set = (all base scorable) + (appended
scorable minus this fold), and only the appended slice is scored. `--holdout-corpus`
defines that scored slice (default: score all scorable with plain K-fold).

Generalization — IID + structured. Random IID favors flat models (interpolation). The
discriminating test for the thesis is leave-one-target-out / leave-one-indication-out:
a flat model collapses to the base rate on an unseen target; the graph should compose
via shared mechanisms. Reported for the ML/base-rate models here; the graph's
structured-generalization evidence is the classic-5 (bevacizumab-via-VEGFA) in
eval_holdout_compose.

ALL methods here use fixed, untuned defaults — the nested-CV tuning pass (BENCHMARK.md
sequencing) tunes graph AND ML on the same inner folds. These numbers are the floor.

Usage:
    # cheap (seconds) — base rates + ML, no graph rebuild:
    python -m scripts.eval_baselines --graph data/exports/onco_scale_500_annotated.json \
        --corpus onco_scale_500 --holdout-corpus onco_scale_248_add

    # + the paired graph holdout (re-attributes per fold; minutes):
    python -m scripts.eval_baselines --graph data/exports/onco_scale_500_annotated.json \
        --corpus onco_scale_500 --holdout-corpus onco_scale_248_add \
        --initial data/exports/onco_scale_500_initial.json --with-graph
"""
from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler

from src.annotation.attributor import _chain_backbone_edges
from src.graph.models import EdgeType
from src.graph.store import GraphStore
from src.prediction.calibration import (
    auc_ci,
    brier_score,
    delong_paired,
    expected_calibration_error,
    mcnemar_paired,
    pr_auc,
    recalibrate_kfold,
    reliability_table,
)
from src.prediction.calibration import auroc as _auroc_np
from scripts.eval_holdout_compose import (
    ANN_DIR,
    _binary_accuracy,
    _corpus_ncts,
    _load_canonicalization_cache,
    _overlap_count,
    _resolve_label,
    _training_used_nodes,
    _trial_conditions,
    resolve_chain,
)
from scripts.eval_holdout_kfold import _fold

# Feature-set keys (the dict keys are namespaced so subsets are selectable).
MECH_KEYS = ("f_compound", "f_target", "f_mechanism", "f_biology",
             "f_indication", "f_endpoint", "f_population")
# Leak-safe design fields: all knowable at design time. duration_weeks is held out
# of the "safe" set because a trial stopped early for futility has a short reported
# duration → that's outcome leakage; it goes only in the explicitly-flagged (+dur) set.
DESIGN_SAFE_KEYS = ("d_sample_size", "d_single_arm", "d_n_compounds", "d_rationale")
DESIGN_KEYS = DESIGN_SAFE_KEYS + ("d_duration_weeks",)

_LINE_GROUP = {  # coarse line-of-therapy (the fan-in / filter conditioner)
    "first": "early", "neoadjuvant": "early", "adjuvant": "early",
    "second": "late", "third_plus": "late", "later": "late",
}


def _line_group(graph: GraphStore, pid: str | None) -> str | None:
    if not pid:
        return None
    try:
        name = graph.get_node(pid).get("name") or ""
    except KeyError:
        return None
    for part in name.split("·"):
        part = part.strip()
        if part.startswith("line:"):
            return _LINE_GROUP.get(part.split(":", 1)[1].strip())
    return None


def _composition(graph: GraphStore, nct: str) -> dict:
    """The trial's MAXIMAL COMPOSITION (owner's path def): the full fanned subgraph
    over ALL the trial's chains. Returns the granular backbone edges, a COARSE variant
    that collapses the mechanism-pathway fan-out (compound→target, target→biology,
    biology→{indication,endpoint}, endpoint→indication), and the (biology, line)
    fan-in cells for the conditional model. Empty when the trial isn't in this graph."""
    try:
        ts = graph.get_trial_subgraph_by_id(nct)
    except KeyError:
        return dict(edges=set(), coarse=set(), bios=set(), lines=set(), bio_lines=set())
    edges: set[str] = set()
    coarse: set[str] = set()
    bios: set[str] = set()
    lines: set[str] = set()
    bio_lines: set[tuple[str, str]] = set()
    for ch in ts.chains:
        lg = _line_group(graph, getattr(ch, "subgroup_population_id", None))
        if lg:
            lines.add(lg)
        for src, tgt, et in _chain_backbone_edges(ch):
            edges.add(f"{et.value}|{src}->{tgt}")
        c, t, b = ch.compound_id, ch.target_id, ch.biology_id
        ind, ep = ch.indication_id, ch.endpoint_id
        U = "UNKNOWN"
        if c and t and t != U:
            coarse.add(f"affects|{c}->{t}")
        if t and b and t != U and b != U:  # target→biology, mechanism fan-out collapsed
            coarse.add(f"t2b|{t}->{b}")
        if b and ind and b != U and ind != U:
            coarse.add(f"biology_drives|{b}->{ind}")
        if b and ep and b != U and ep != U:
            coarse.add(f"reflects_biology|{b}->{ep}")
            if ind and ind != U:
                coarse.add(f"endpoint_captures|{ep}->{ind}")
        if b and b != U:
            bios.add(b)
            if lg:
                bio_lines.add((b, lg))
    return dict(edges=edges, coarse=coarse, bios=bios, lines=lines, bio_lines=bio_lines)


@dataclass
class Trial:
    nct: str
    y: int                      # 1 success, 0 failure
    label: str
    is_eval: bool               # in the scored (held-out) slice
    target: str
    indication: str
    feats: dict                 # all features, namespaced (MECH_KEYS + DESIGN_KEYS)
    chain: dict
    kwargs: dict = field(default_factory=dict)
    # maximal-composition fields (from the graph trial-subgraph; see _composition)
    comp_edges: set = field(default_factory=set)
    coarse_edges: set = field(default_factory=set)
    bios: set = field(default_factory=set)
    lines: set = field(default_factory=set)
    bio_lines: set = field(default_factory=set)

    def select(self, keys) -> dict:
        return {k: v for k, v in self.feats.items() if k in keys}


# ── Feature extraction (design-time only — NO results, NO classification) ──


def _design_features(extraction: dict) -> dict:
    ctx = extraction.get("context") or {}
    th = extraction.get("therapeutic_hypothesis") or {}
    arms = extraction.get("arms") or []
    n_compounds = max((len(a.get("compounds") or []) for a in arms), default=0)
    comparator = (ctx.get("comparator") or "").strip().lower()
    single_arm = 1 if comparator in ("", "none", "no comparator", "n/a") else 0
    return {
        "d_sample_size": float(ctx.get("sample_size") or 0.0),
        "d_duration_weeks": float(ctx.get("duration_weeks") or 0.0),
        "d_single_arm": single_arm,
        "d_n_compounds": float(n_compounds),
        "d_rationale": (th.get("rationale_strength") or "unknown").lower(),
    }


def _mech_features(chain: dict) -> dict:
    return {
        "f_compound": chain.get("compound_id") or "novel",
        "f_target": chain.get("target_id") or "UNKNOWN",
        "f_mechanism": chain.get("mechanism_id") or "UNKNOWN",
        "f_biology": chain.get("biology_id") or "UNKNOWN",
        "f_indication": chain.get("indication_id") or "UNKNOWN",
        "f_endpoint": chain.get("endpoint_id") or "UNKNOWN",
        "f_population": chain.get("population_id") or "UNKNOWN",
    }


# ── Scorable-set construction (SAME filters as eval_holdout_kfold) ──────────


def build_trials(graph: GraphStore, corpus: str, holdout_corpus: str | None,
                 min_overlap: int) -> list[Trial]:
    training_used = _training_used_nodes(graph)
    canon = _load_canonicalization_cache()
    eval_ids = set(_corpus_ncts(holdout_corpus)) if holdout_corpus else None
    trials: list[Trial] = []
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
        chain = resolve_chain(extraction, _trial_conditions(extraction), graph, canon)
        if chain["target_id"] not in training_used:
            continue
        if chain["indication_id"] not in training_used:
            continue
        if _overlap_count(chain, training_used) < min_overlap:
            continue
        feats = {**_mech_features(chain), **_design_features(extraction)}
        kwargs = {k: chain[k] for k in
                  ("target_id", "mechanism_id", "biology_id", "endpoint_id", "population_id")
                  if chain[k] and chain[k] != "UNKNOWN"}
        comp = _composition(graph, nct)
        trials.append(Trial(
            nct=nct, y=1 if label == "success" else 0, label=label,
            is_eval=(eval_ids is None or nct in eval_ids),
            target=chain["target_id"], indication=chain["indication_id"],
            feats=feats, chain=chain, kwargs=kwargs,
            comp_edges=comp["edges"], coarse_edges=comp["coarse"],
            bios=comp["bios"], lines=comp["lines"], bio_lines=comp["bio_lines"],
        ))
    return trials


# ── Models: each is (train_rows, test_rows, seed) -> list[prob] ────────────


def _rate(rows: list[Trial]) -> float:
    return float(np.mean([r.y for r in rows])) if rows else 0.5


def m_marginal(train, test, seed):
    return [_rate(train)] * len(test)


def _group_rate(keyattr):
    def model(train, test, seed):
        agg: dict = defaultdict(list)
        for r in train:
            agg[getattr(r, keyattr)].append(r.y)
        marg = _rate(train)
        return [float(np.mean(agg[getattr(r, keyattr)])) if agg.get(getattr(r, keyattr))
                else marg for r in test]
    return model


def _ml_model(keys, kind):
    def model(train, test, seed):
        ytr = [r.y for r in train]
        if len(set(ytr)) < 2:            # degenerate single-class fold
            return [float(np.mean(ytr))] * len(test)
        Xtr = [r.select(keys) for r in train]
        Xte = [r.select(keys) for r in test]
        if kind == "logreg":
            clf = make_pipeline(
                DictVectorizer(sparse=True),
                MaxAbsScaler(),
                LogisticRegression(C=1.0, max_iter=4000),
            )
            clf.fit(Xtr, ytr)
            return clf.predict_proba(Xte)[:, 1].tolist()
        import xgboost as xgb
        dv = DictVectorizer(sparse=True)
        Xtr_m = dv.fit_transform(Xtr)
        Xte_m = dv.transform(Xte)
        clf = xgb.XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            eval_metric="logloss", random_state=seed, n_jobs=4, verbosity=0,
        )
        clf.fit(Xtr_m, np.asarray(ytr))
        return clf.predict_proba(Xte_m)[:, 1].tolist()
    return model


# ── Triangulation: outcome-driven edge-weight inference (NO LLM beliefs) ────
# logit P(success) = Σ_{e ∈ maximal-composition} w_e, fit as L2-logistic — the
# coefficients ARE the edge weights, triangulated jointly across overlapping paths.

def _comp_feats(r) -> dict:        # granular maximal-composition edge incidence
    return {f"E|{e}": 1.0 for e in r.comp_edges}


def _comp_filter_feats(r) -> dict:  # + fan-in/filter: line + biology×line conditioners
    f = {f"E|{e}": 1.0 for e in r.comp_edges}
    f.update({f"L|{lg}": 1.0 for lg in r.lines})
    f.update({f"BL|{b}|{lg}": 1.0 for (b, lg) in r.bio_lines})
    return f


def _coarse_feats(r) -> dict:      # mechanism-pathway fan-out collapsed (target→biology)
    return {f"C|{e}": 1.0 for e in r.coarse_edges}


def _edge_model(feat_fn):
    """L2-logistic over a feature_fn(trial)->{feat:1.0} incidence, with C chosen by
    INNER CV on the TRAINING fold only (never the test fold). Coefficients are the
    inferred edge weights; rare/entangled edges shrink toward 0 (neutral)."""
    def model(train, test, seed):
        ytr = [r.y for r in train]
        if len(set(ytr)) < 2:
            return [float(np.mean(ytr))] * len(test)
        Xtr = [feat_fn(r) for r in train]
        Xte = [feat_fn(r) for r in test]
        Cs = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
        try:
            clf = make_pipeline(
                DictVectorizer(sparse=True),
                LogisticRegressionCV(Cs=Cs, cv=3, scoring="roc_auc", max_iter=4000),
            )
            clf.fit(Xtr, ytr)
        except Exception:  # noqa: BLE001 — degenerate inner fold → fixed strong L2
            clf = make_pipeline(
                DictVectorizer(sparse=True),
                LogisticRegression(C=0.1, max_iter=4000),
            )
            clf.fit(Xtr, ytr)
        return clf.predict_proba(Xte)[:, 1].tolist()
    return model


def _cond_model(k_min: int = 2, prior_strength: float = 2.0):
    """Conditional biology×line pooling (the case-based probe) as a train→test closure,
    so its previously-optimistic same-corpus LOO number gets an HONEST k-fold /
    leave-group-out score on identical folds."""
    def model(train, test, seed):
        base = float(np.mean([r.y for r in train])) if train else 0.5
        a0, b0 = base * prior_strength, (1 - base) * prior_strength
        bl: dict = defaultdict(dict)   # (bio,line) -> {nct: y}  distinct trials
        bo: dict = defaultdict(dict)   # bio       -> {nct: y}
        for r in train:
            for b in r.bios:
                bo[b][r.nct] = r.y
            for (b, lg) in r.bio_lines:
                bl[(b, lg)][r.nct] = r.y

        def cell(idx, key):
            d = idx.get(key, {})
            return ((sum(d.values()) + a0) / (len(d) + a0 + b0)) if len(d) >= k_min else None

        out = []
        for r in test:
            line_of = {b: lg for (b, lg) in r.bio_lines}
            ps = []
            for b in r.bios:
                lg = line_of.get(b)
                p = (cell(bl, (b, lg)) if lg else None)
                if p is None:
                    p = cell(bo, b)
                ps.append(p if p is not None else base)
            out.append(float(np.mean(ps)) if ps else base)
        return out
    return model


def make_models():
    return {
        "base:marginal": m_marginal,
        "base:per-indication": _group_rate("indication"),
        "base:per-target": _group_rate("target"),
        "logreg:mech(b)": _ml_model(MECH_KEYS, "logreg"),
        "xgb:mech(b)": _ml_model(MECH_KEYS, "xgb"),
        "xgb:design(plan)": _ml_model(DESIGN_SAFE_KEYS, "xgb"),
        "xgb:all(safe)": _ml_model(MECH_KEYS + DESIGN_SAFE_KEYS, "xgb"),
        # triangulation + conditional models (NO LLM beliefs):
        "logreg:edges(comp)": _edge_model(_comp_feats),
        "logreg:edges+filt": _edge_model(_comp_filter_feats),
        "logreg:edges(coarse)": _edge_model(_coarse_feats),
        "cond:bio×line": _cond_model(),
    }


# ── Evaluation modes ───────────────────────────────────────────────────────


def run_iid(trials, models, k, seed) -> dict[str, dict[str, float]]:
    """Per-fold: train = base(always) + (eval minus fold); test = eval fold.
    Mirrors the graph's K-fold exactly. Returns {model: {nct: prob}}."""
    preds: dict[str, dict[str, float]] = {m: {} for m in models}
    eval_rows = [r for r in trials if r.is_eval]
    base_rows = [r for r in trials if not r.is_eval]
    by_fold: dict[int, list] = defaultdict(list)
    for r in eval_rows:
        by_fold[_fold(r.nct, k)].append(r)
    for f in sorted(by_fold):
        test = by_fold[f]
        train = base_rows + [r for r in eval_rows if _fold(r.nct, k) != f]
        for name, model in models.items():
            for r, p in zip(test, model(train, test, seed)):
                preds[name][r.nct] = float(p)
    return preds


def run_leave_group_out(trials, models, groupattr, seed) -> dict[str, dict[str, float]]:
    """Hold out each group (target/indication) entirely; train on all other groups,
    score that group's eval trials. The structured-generalization test."""
    preds: dict[str, dict[str, float]] = {m: {} for m in models}
    groups = sorted({getattr(r, groupattr) for r in trials})
    for g in groups:
        test = [r for r in trials if getattr(r, groupattr) == g and r.is_eval]
        if not test:
            continue
        train = [r for r in trials if getattr(r, groupattr) != g]
        if not train:
            continue
        for name, model in models.items():
            for r, p in zip(test, model(train, test, seed)):
                preds[name][r.nct] = float(p)
    return preds


# ── Reporting ──────────────────────────────────────────────────────────────


def _metrics(preds_by_nct: dict[str, float], trials_by_nct: dict[str, Trial]) -> dict:
    ncts = [n for n in preds_by_nct if n in trials_by_nct]
    p = [preds_by_nct[n] for n in ncts]
    y = [trials_by_nct[n].y for n in ncts]
    if len(set(y)) < 2:
        return dict(n=len(y), auroc=float("nan"), lo=float("nan"), hi=float("nan"),
                    prauc=float("nan"), brier=brier_score(p, y),
                    ece=expected_calibration_error(p, y), acc=float("nan"))
    acc = _binary_accuracy(p, y, 0.5)[0]
    a, lo, hi = auc_ci(p, y)
    return dict(n=len(y), auroc=a, lo=lo, hi=hi, prauc=pr_auc(p, y),
                brier=brier_score(p, y), ece=expected_calibration_error(p, y), acc=acc)


def _print_table(title: str, preds: dict[str, dict[str, float]],
                 trials_by_nct: dict[str, Trial], base_rate: float) -> None:
    print(f"\n── {title} ──")
    print(f"  {'model':<22}{'n':>5}{'AUROC':>8}{'95% CI':>15}{'PR-AUC':>8}{'Brier':>8}"
          f"{'ECE':>7}{'Acc':>7}{'BrierSS':>9}")
    brier_marg = base_rate * (1 - base_rate)   # Brier of the marginal predictor
    for name, pmap in preds.items():
        m = _metrics(pmap, trials_by_nct)
        if not m["n"]:
            continue
        bss = 1.0 - (m["brier"] / brier_marg) if brier_marg > 0 else float("nan")
        def f(x, d=3):
            return f"{x:.{d}f}" if x == x else "  —"
        ci = (f"[{m['lo']:.2f}-{m['hi']:.2f}]"
              if m["lo"] == m["lo"] else "—")
        print(f"  {name:<22}{m['n']:>5}{f(m['auroc']):>8}{ci:>15}{f(m['prauc']):>8}"
              f"{f(m['brier']):>8}{f(m['ece']):>7}{f(m['acc']):>7}{f(bss):>9}")
    print(f"  (base rate = {base_rate:.3f}; BrierSS = skill vs the marginal predictor. "
          f"For near-constant base-rate models AUROC≈0.5 ± fold-rate noise — read their "
          f"Brier/Acc, not AUROC. BrierSS<0 = worse-calibrated than always-guess-base-rate)")


def _print_reliability(name: str, pmap: dict[str, float],
                       trials_by_nct: dict[str, Trial]) -> None:
    ncts = [n for n in pmap if n in trials_by_nct]
    p = [pmap[n] for n in ncts]
    y = [trials_by_nct[n].y for n in ncts]
    print(f"\n  reliability — {name} (a calibrated model has pred≈obs per bin):")
    print(f"    {'bin':<13}{'count':>7}{'mean_pred':>11}{'obs_freq':>10}")
    for b in reliability_table(p, y, n_bins=10):
        if b.count == 0:
            continue
        print(f"    [{b.lo:.1f},{b.hi:.1f}){'':<4}{b.count:>7}"
              f"{b.mean_pred:>11.3f}{b.frac_pos:>10.3f}")


def _print_recalibration(g_ho: dict[str, float], trials_by_nct: dict[str, Trial],
                         k: int, seed: int) -> None:
    """Show what RECALIBRATION buys on the graph holdout — the bedrock fix for the
    documented pessimism. Honest: a second-level K-fold (recalibrate_kfold) fits
    the calibrator on held-out-from-test predictions, so 'after' isn't an in-fit
    number. Brier/ECE/accuracy before vs after Platt + isotonic."""
    ncts = [n for n in g_ho if n in trials_by_nct]
    p = np.array([g_ho[n] for n in ncts])
    y = np.array([trials_by_nct[n].y for n in ncts])
    if len(set(y.tolist())) < 2:
        return
    print("\n── Recalibration of the graph holdout (2nd-level CV — the bedrock fix) ──")
    print(f"  {'variant':<18}{'AUROC':>8}{'Brier':>8}{'ECE':>7}{'Acc':>7}")

    def row(label, pp):
        a = _auroc_np(pp, y)
        br = brier_score(pp, y)
        ec = expected_calibration_error(pp, y)
        ac = _binary_accuracy(list(pp), list(y), 0.5)[0]
        print(f"  {label:<18}{a:>8.3f}{br:>8.3f}{ec:>7.3f}{ac:>7.3f}")

    row("raw", p)
    platt = recalibrate_kfold(p, y, k=k, method="platt", seed=seed)
    row("+ Platt", platt)
    iso = recalibrate_kfold(p, y, k=k, method="isotonic", seed=seed)
    row("+ isotonic", iso)
    print("  (AUROC is rank-invariant ⇒ unchanged by monotone recalibration; the win "
          "is Brier/ECE/Acc — fixing the pessimism lifts accuracy off the FN floor)")
    # reliability of the Platt-recalibrated predictions
    print("\n  reliability — graph:holdout + Platt:")
    print(f"    {'bin':<13}{'count':>7}{'mean_pred':>11}{'obs_freq':>10}")
    for b in reliability_table(list(platt), list(y), n_bins=10):
        if b.count == 0:
            continue
        print(f"    [{b.lo:.1f},{b.hi:.1f}){'':<4}{b.count:>7}"
              f"{b.mean_pred:>11.3f}{b.frac_pos:>10.3f}")


def _print_paired(graph_pmap, model_preds, trials_by_nct, thr=0.5) -> None:
    """Paired DeLong (AUROC) + McNemar (acc) of the graph vs each baseline on the
    SAME held-out trials — the only valid comparison at small n."""
    print("\n── Paired graph-vs-baseline (DeLong AUROC, McNemar accuracy) ──")
    common0 = [n for n in graph_pmap if n in trials_by_nct]
    print(f"  graph scored on {len(common0)} held-out trials")
    print(f"  {'baseline':<22}{'ΔAUROC(graph-base)':>20}{'DeLong p':>10}"
          f"{'McNemar p':>11}")
    for name, pmap in model_preds.items():
        ncts = [n for n in common0 if n in pmap]
        if len(ncts) < 4:
            continue
        gp = [graph_pmap[n] for n in ncts]
        bp = [pmap[n] for n in ncts]
        y = [trials_by_nct[n].y for n in ncts]
        if len(set(y)) < 2:
            continue
        d = delong_paired(gp, bp, y)
        gc = [(gp[i] >= thr) == bool(y[i]) for i in range(len(y))]
        bc = [(bp[i] >= thr) == bool(y[i]) for i in range(len(y))]
        mc = mcnemar_paired(gc, bc)
        print(f"  {name:<22}{d.auc_a - d.auc_b:>+20.3f}{d.p_value:>10.3f}"
              f"{mc.p_value:>11.3f}")
    print("  (ΔAUROC>0 ⇒ graph ranks better; p<0.05 ⇒ the difference is significant)")


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", required=True, help="trained annotated.json")
    ap.add_argument("--corpus", required=True, help="full corpus (label/feature source)")
    ap.add_argument("--holdout-corpus", default=None,
                    help="restrict the SCORED slice to these NCTs (matches the graph "
                         "K-fold's excludable appended slice). Base trials stay in train.")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--min-overlap", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--with-graph", action="store_true",
                    help="also run the graph's in-sample + K-fold holdout (re-attributes "
                         "per fold; needs --initial) for the paired comparison.")
    ap.add_argument("--initial", default=None, help="pre-attribution initial.json (--with-graph)")
    ap.add_argument("--n-samples", type=int, default=2000)
    ap.add_argument("--dump-graph-preds", default=None,
                    help="write the graph in-sample+holdout preds to JSON (so "
                         "recalibration can be iterated without re-attributing).")
    ap.add_argument("--graph-preds", default=None,
                    help="load graph preds from a prior --dump-graph-preds JSON "
                         "instead of re-attributing (instant recalibration iteration).")
    args = ap.parse_args()
    if args.with_graph and not args.initial:
        ap.error("--with-graph requires --initial")

    np.random.seed(args.seed)
    graph = GraphStore()
    graph.import_snapshot(args.graph)
    trials = build_trials(graph, args.corpus, args.holdout_corpus, args.min_overlap)
    eval_trials = [t for t in trials if t.is_eval]
    trials_by_nct = {t.nct: t for t in trials}
    eval_by_nct = {t.nct: t for t in eval_trials}

    n_succ = sum(t.y for t in eval_trials)
    base_rate = n_succ / len(eval_trials) if eval_trials else float("nan")
    print(f"graph: {args.graph}")
    print(f"scorable: {len(trials)} total "
          f"({len(trials) - len(eval_trials)} always-train / {len(eval_trials)} scored)")
    print(f"scored slice: {len(eval_trials)} trials "
          f"(success={n_succ}, failure={len(eval_trials) - n_succ}, base rate={base_rate:.3f})")
    if len(eval_trials) < 8 or n_succ in (0, len(eval_trials)):
        print("not enough labeled scored trials")
        return 1

    models = make_models()
    iid = run_iid(trials, models, args.k, args.seed)
    _print_table("IID K-fold (base-always-train; mirrors the graph holdout)",
                 iid, eval_by_nct, base_rate)
    _print_reliability("xgb:mech(b)", iid["xgb:mech(b)"], eval_by_nct)

    loto = run_leave_group_out(trials, models, "target", args.seed)
    _print_table("Leave-one-TARGET-out (structured generalization)",
                 loto, trials_by_nct, base_rate)
    loio = run_leave_group_out(trials, models, "indication", args.seed)
    _print_table("Leave-one-INDICATION-out (structured generalization)",
                 loio, trials_by_nct, base_rate)

    g_in = g_ho = None
    if args.graph_preds:
        blob = json.loads(Path(args.graph_preds).read_text())
        g_in = {n: p for n, p in blob.get("insample", {}).items() if p is not None}
        g_ho = {n: p for n, p in blob.get("holdout", {}).items() if p is not None}
        print(f"\nloaded graph preds from {args.graph_preds} "
              f"({len(g_ho)} holdout, {len(g_in)} in-sample)")
    elif args.with_graph:
        from scripts.eval_holdout_kfold import graph_holdout_predictions
        scorable = [(t.nct, t.label, t.chain, t.kwargs) for t in eval_trials]
        print("\ncomputing graph in-sample + K-fold holdout (re-attribution)...", flush=True)
        g_in, g_ho = await graph_holdout_predictions(
            graph, scorable, args.initial, args.k, args.n_samples)
        g_in = {n: p for n, p in g_in.items() if p is not None}
        g_ho = {n: p for n, p in g_ho.items() if p is not None}
        if args.dump_graph_preds:
            Path(args.dump_graph_preds).write_text(
                json.dumps({"insample": g_in, "holdout": g_ho}, indent=2))
            print(f"dumped graph preds → {args.dump_graph_preds}")

    if g_ho:
        graph_preds = {"graph:in-sample": g_in, "graph:holdout": g_ho}
        _print_table("Graph (scalar) — same scored slice", graph_preds, eval_by_nct, base_rate)
        _print_reliability("graph:holdout", g_ho, eval_by_nct)
        _print_recalibration(g_ho, eval_by_nct, args.k, args.seed)
        _print_paired(g_ho, iid, eval_by_nct)

    print("\nNOTE: all methods use fixed untuned defaults — the nested-CV pass tunes "
          "graph + ML on the same inner folds. These are the floor, not the verdict.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
