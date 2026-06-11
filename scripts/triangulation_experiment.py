"""Outcome-driven edge-weight inference via path triangulation.

A fundamentally different architecture from the deployed per-edge Beta + softmin:
NO LLM belief assignment anywhere. The graph is fully resolved (nodes + edges, no
beliefs); the LLM's only job was node resolution (mapping each trial to its path).
Edge weights are LATENT and inferred PURELY from outcome patterns across overlapping
paths, by logistic regression:

    logit P(success_i) = Σ_{e ∈ path_i} w_e            (+ intercept)

i.e. each trial is one observation, the features are binary "does trial i's path
include edge e", the outcome is success/failure, and the fitted coefficients ARE the
edge weights. L2 regularization (C) shrinks rarely-traversed / entangled edges toward
0 (neutral), and triangulation happens automatically: an edge in 10 wins / 2 losses
gets a positive weight; co-occurring edges that never separate share the weight.

HARMONIZED + EVOLVED. The faithful spec is the `edges` feature set. The architecture
diagnosis (data/dev/architecture_diagnosis.md) found the predictive signal lives at the
BIOLOGY node conditioned on LINE-OF-THERAPY, not the mechanism edges — so we run the
SAME logistic framework over several feature sets to see what the joint fit can extract:
  edges        backbone edge incidence (compound→target→mechanism→biology→…)  [the spec]
  nodes        node incidence (target/mechanism/biology/indication)
  bio          biology-node incidence only
  edges+line   edges + line-of-therapy indicators
  bio+bioline  biology nodes + biology×(early/late line) interactions
  all          edges + nodes + biology×line interactions

Compared to: deployed edge-marginal softmin (~0.534) and the conditional biology×line
case-based probe (0.64–0.71). KEY QUESTION: does LOO AUROC > 0.60 — does triangulation
across overlapping paths discover edge weights that GENERALIZE out-of-sample?

    python -m scripts.triangulation_experiment \
        --graph data/exports/phasec_n250_annotated.json --C 1.0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from scripts.eval_holdout_compose import _auroc, _resolve_label
from src.annotation.attributor import _chain_backbone_edges
from src.graph.store import GraphStore

_LINE_GROUP = {
    "first": "early", "neoadjuvant": "early", "adjuvant": "early",
    "second": "late", "third_plus": "late", "later": "late",
}
DEPLOYED_AUROC = 0.534
COND_AUROC = "0.64-0.71"


def _line_of(g: GraphStore, pid: str | None) -> str | None:
    if not pid:
        return None
    try:
        name = g.get_node(pid).get("name") or ""
    except KeyError:
        return None
    for part in name.split("·"):
        part = part.strip()
        if part.startswith("line:"):
            return part.split(":", 1)[1].strip()
    return None


def build_trials(g: GraphStore, ann: Path) -> list[dict]:
    trials = []
    for ts in g.trial_subgraphs.values():
        ext_p = ann / f"{ts.trial_id}_extraction.json"
        if not ext_p.exists() or not ts.chains:
            continue
        ext = json.loads(ext_p.read_text())
        cls_p = ann / f"{ts.trial_id}_classification.json"
        cls = json.loads(cls_p.read_text()) if cls_p.exists() else None
        label = _resolve_label(ext, cls)
        if label not in ("success", "failure"):
            continue
        edges: set[str] = set()
        nodes: set[str] = set()
        bios: set[str] = set()
        lines: set[str] = set()
        bio_line: set[str] = set()
        for ch in ts.chains:
            lg = _LINE_GROUP.get(_line_of(g, getattr(ch, "subgroup_population_id", None)))
            if lg:
                lines.add(f"L|{lg}")
            for src, tgt, et in _chain_backbone_edges(ch):
                edges.add(f"E|{et.value}|{src}->{tgt}")
            for attr in ("target_id", "mechanism_id", "biology_id", "indication_id"):
                nid = getattr(ch, attr, None)
                if nid and nid != "UNKNOWN":
                    nodes.add(f"N|{nid}")
            bid = getattr(ch, "biology_id", None)
            if bid and bid != "UNKNOWN":
                bios.add(f"B|{bid}")
                if lg:
                    bio_line.add(f"BL|{bid}|{lg}")
        trials.append({
            "nct": ts.trial_id, "y": 1 if label == "success" else 0,
            "edges": edges, "nodes": nodes, "bio": bios,
            "lines": lines, "bio_line": bio_line,
        })
    return trials


FEATURE_SETS = {
    "edges":       lambda t: t["edges"],
    "nodes":       lambda t: t["nodes"],
    "bio":         lambda t: t["bio"],
    "edges+line":  lambda t: t["edges"] | t["lines"],
    "bio+bioline": lambda t: t["bio"] | t["bio_line"],
    "all":         lambda t: t["edges"] | t["nodes"] | t["bio_line"],
}


def matrix(trials: list[dict], extract, min_trav: int = 1
           ) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Binary trial×feature incidence. ``min_trav`` prunes features traversed by
    fewer than this many trials — triangulation REQUIRES overlap, so an edge in 1
    trial is pure memorization (its weight can only fit that one outcome), not an
    inferable latent. Pruning them is the principled version of 'shrink to 0'."""
    counts: dict[str, int] = {}
    for t in trials:
        for f in extract(t):
            counts[f] = counts.get(f, 0) + 1
    vocab = sorted(f for f, c in counts.items() if c >= min_trav)
    idx = {f: i for i, f in enumerate(vocab)}
    X = np.zeros((len(trials), len(vocab)), dtype=float)
    for r, t in enumerate(trials):
        for f in extract(t):
            if f in idx:
                X[r, idx[f]] = 1.0
    y = np.array([t["y"] for t in trials])
    return X, y, vocab


def _fit(X, y, C):
    return LogisticRegression(C=C, max_iter=2000).fit(X, y)  # default L2/lbfgs


def loo_auroc(X: np.ndarray, y: np.ndarray, C: float) -> float:
    n = len(y)
    preds = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool); mask[i] = False
        ytr = y[mask]
        if len(set(ytr.tolist())) < 2:
            preds[i] = ytr.mean()
            continue
        preds[i] = _fit(X[mask], ytr, C).predict_proba(X[i:i + 1])[0, 1]
    return _auroc(preds.tolist(), y.tolist())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--annotations", default="data/annotations")
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--min-traversal", type=int, default=3,
                    help="drop features traversed by < this many trials (triangulation "
                         "needs overlap; 1 = the faithful no-prune spec)")
    ap.add_argument("--out", default="data/dev/triangulation_weights.md")
    args = ap.parse_args()

    g = GraphStore(); g.import_snapshot(args.graph)
    trials = build_trials(g, Path(args.annotations))
    base = float(np.mean([t["y"] for t in trials]))
    print(f"trials: {len(trials)}  base success rate: {base:.3f}  "
          f"(refs: deployed {DEPLOYED_AUROC}, conditional {COND_AUROC})\n")

    def node_name(nid: str) -> str:
        try:
            return g.get_node(nid).get("name") or nid
        except KeyError:
            return nid

    def feat_desc(f: str) -> str:
        kind, rest = f.split("|", 1)
        if kind == "E":
            et, st = rest.split("|", 1); src, tgt = st.split("->", 1)
            return f"{node_name(src)} → {node_name(tgt)} [{et}]"
        if kind in ("N", "B"):
            return f"{node_name(rest)} [{ 'node' if kind=='N' else 'biology'}]"
        if kind == "L":
            return f"line={rest}"
        if kind == "BL":
            bid, lg = rest.rsplit("|", 1)
            return f"{node_name(bid)} × line={lg}"
        return f

    # HONEST out-of-sample AUROC for this approach lives in eval_baselines.py
    # (`logreg:edges(comp)` / `edges(coarse)` on k-fold + leave-one-indication/target-out,
    # paired vs the deployed graph). The earlier same-corpus LOO here was the p≫n +
    # LOO-optimism artifact (in-sample 1.000 ⇒ unstable held-out signs). THIS script is
    # now the edge-weight EXPLAINER: fit on full data (weights are descriptive) and report
    # what triangulation learned + which edges are entangled (unidentifiable).
    mt = args.min_traversal
    X, y, vocab = matrix(trials, FEATURE_SETS["edges"], mt)
    insample = _auroc(_fit(X, y, args.C).predict_proba(X)[:, 1].tolist(), y.tolist())
    print(f"`edges` (maximal composition): {len(vocab)} edges with ≥{mt} traversals; "
          f"in-sample AUROC {insample:.3f}\n  (IN-SAMPLE = leakage; the HONEST number is "
          f"eval_baselines `logreg:edges(comp/coarse)`).")
    clf = _fit(X, y, args.C)
    print(f"\n[edge weights: min-trav {mt}, C={args.C:g}]")
    coef = clf.coef_[0]
    ntrav = X.sum(axis=0)
    srate = np.array([y[X[:, j] > 0].mean() if ntrav[j] > 0 else float("nan")
                      for j in range(len(vocab))])
    order = np.argsort(coef)

    def show(js, header):
        print(f"\n{header}")
        for j in js:
            print(f"  w={coef[j]:+.3f}  n={int(ntrav[j]):3d}  succ={srate[j]:.2f}  "
                  f"{feat_desc(vocab[j])[:60]}")
    show(order[::-1][:10], "TOP +10 edges (push toward SUCCESS):")
    show(order[:10], "TOP −10 edges (push toward FAILURE):")

    # ── entanglement: edges with an IDENTICAL support set always co-occur, so the
    #    fit cannot separate their individual weights (L2 splits them) — report them
    #    as "entangled, need more data to separate" (owner's spec). ──
    support: dict[frozenset, list[int]] = {}
    for j in range(len(vocab)):
        s = frozenset(np.where(X[:, j] > 0)[0].tolist())
        support.setdefault(s, []).append(j)
    clusters = sorted((js for js in support.values() if len(js) >= 2),
                      key=lambda js: -len(js))
    n_ent = sum(len(js) for js in clusters)
    print(f"\nENTANGLED clusters (identical traversal set → unidentifiable weights): "
          f"{len(clusters)} clusters covering {n_ent}/{len(vocab)} edges")
    for js in clusters[:6]:
        print(f"  {len(js)} edges co-occur (n={int(ntrav[js[0]])} trials, "
              f"succ={srate[js[0]]:.2f}); split weight {coef[js[0]]:+.3f} each:")
        for j in js[:4]:
            print(f"     {feat_desc(vocab[j])[:56]}")

    # ── decompose one predicted-failure trial ──
    probs = clf.predict_proba(X)[:, 1]
    fail_idx = [i for i in range(len(trials)) if trials[i]["y"] == 0]
    if fail_idx:
        i = min(fail_idx, key=lambda i: probs[i])
        cols = np.where(X[i] > 0)[0]
        contribs = sorted(((coef[j], vocab[j]) for j in cols), key=lambda c: c[0])
        print(f"\nDECOMPOSE failure trial {trials[i]['nct']} "
              f"(predicted P(success)={probs[i]:.3f}, actual=FAIL):")
        for w, f in contribs[:6]:
            print(f"  {w:+.3f}  {feat_desc(f)[:60]}")

    lines = [f"# Triangulation edge weights (interpretability)\n",
             f"Graph `{args.graph}`, {len(trials)} trials, base {base:.3f}, "
             f"min-traversal {mt}, C={args.C:g}. Fit on FULL data (descriptive weights). "
             f"In-sample AUROC {insample:.3f} (leakage). HONEST AUROC: see eval_baselines "
             f"`logreg:edges(comp/coarse)` (k-fold + leave-group-out).\n",
             f"Entangled clusters (unidentifiable weights): {len(clusters)} covering "
             f"{n_ent}/{len(vocab)} edges.\n",
             "## Top +edges (→ success)\n| weight | n | succ | edge |", "|---|---|---|---|"]
    for j in order[::-1][:12]:
        lines.append(f"| {coef[j]:+.3f} | {int(ntrav[j])} | {srate[j]:.2f} | "
                     f"{feat_desc(vocab[j])} |")
    lines += ["\n## Top −edges (→ failure)\n| weight | n | succ | edge |", "|---|---|---|---|"]
    for j in order[:12]:
        lines.append(f"| {coef[j]:+.3f} | {int(ntrav[j])} | {srate[j]:.2f} | "
                     f"{feat_desc(vocab[j])} |")
    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
