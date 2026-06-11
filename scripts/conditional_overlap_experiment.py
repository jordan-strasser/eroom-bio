"""Edge-assignment first-principles test: does CONDITIONAL (context-dependent)
outcome-pooling beat the current MARGINAL edge belief?

Hypothesis (owner): assigning each edge a single scalar Beta from its own trials'
outcomes MARGINALIZES over the conditioning context. Chain A (n1..n6, right pop)
succeeds, chain B (same n1..n6, wrong pop n7=x) fails → B's failure drags the
SHARED edges toward 0.5, so both predict the cohort average and a held-out trial
can't be ranked. The fix is CONDITIONAL: pool outcomes over trials sharing the
sub-configuration (triangulation overlap), so we learn "n1..n6 works WHEN n7=x".

This tests that WITHOUT re-architecting anything: a pure CASE-BASED predictor (zero
edge attribution) with leave-one-TRIAL-out holdout. For a held-out trial, each of
its chains is scored by the empirical success rate of OTHER trials whose chain
overlaps it on a node set S (distinct-trial counts, Beta-smoothed, hierarchical
backoff when S is sparse); the trial score aggregates its chains. We compare S sets:

  marginal_mech   {mechanism}            ≈ today's altitude (the marginal)
  mech_line       {mechanism, line}      + line-of-therapy conditioner  (the test)
  mech_linegrp    {mechanism, early/late}  coarser line (denser cells)
  mech_pop        {mechanism, population}  full population slug
  biology_line    {biology, line}
  marginal_ind    {indication}           dumb base-rate-by-disease baseline
  backbone        {target,mechanism,biology} backoff   the full mechanistic chain

If a CONDITIONAL set lifts holdout AUROC above marginal_mech AND above the
random-fold 0.534, the conditional signal is real and estimable → justifies
re-architecting attribution. If not, it isn't estimable at this corpus size.

    python -m scripts.conditional_overlap_experiment \
        --graph data/exports/phasec_n250_annotated.json --k-min 2
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from scripts.eval_holdout_compose import _auroc, _resolve_label
from src.graph.store import GraphStore

_LINE_GROUP = {  # coarsen line-of-therapy into early/late for denser cells
    "first": "early", "neoadjuvant": "early", "adjuvant": "early",
    "second": "late", "third_plus": "late", "later": "late",
}


def _pop_axes(name: str) -> dict[str, str]:
    axes: dict[str, str] = {}
    for part in (name or "").split("·"):
        part = part.strip()
        if ":" in part:
            k, v = part.split(":", 1)
            axes[k.strip()] = v.strip()
    return axes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--annotations", default="data/annotations")
    ap.add_argument("--k-min", type=int, default=2,
                    help="min DISTINCT other-trials in a cell to trust it (else backoff)")
    ap.add_argument("--prior-strength", type=float, default=2.0)
    ap.add_argument("--out", default="data/dev/conditional_overlap_findings.md")
    args = ap.parse_args()

    g = GraphStore()
    g.import_snapshot(args.graph)
    ann = Path(args.annotations)

    # ── build chain-level cases with their conditioning attributes ──
    trials: list[dict] = []
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
        y = 1 if label == "success" else 0
        chains = []
        for ch in ts.chains:
            pid = getattr(ch, "subgroup_population_id", None)
            line = None
            if pid:
                try:
                    line = _pop_axes(g.get_node(pid).get("name") or "").get("line")
                except KeyError:
                    pass
            axes = {}
            if pid:
                try:
                    axes = _pop_axes(g.get_node(pid).get("name") or "")
                except KeyError:
                    pass
            chains.append({
                "target": getattr(ch, "target_id", None),
                "mech": getattr(ch, "mechanism_id", None),
                "bio": getattr(ch, "biology_id", None),
                "ind": getattr(ch, "indication_id", None),
                "pop": pid,
                "line": line,
                "linegrp": _LINE_GROUP.get(line) if line else None,
                "severity": axes.get("severity"),
                "stage": axes.get("stage"),
            })
        trials.append({"nct": ts.trial_id, "y": y, "chains": chains})

    base_rate = mean(t["y"] for t in trials)
    a0 = base_rate * args.prior_strength
    b0 = (1 - base_rate) * args.prior_strength
    print(f"trials: {len(trials)}  base success rate: {base_rate:.3f}")

    # ── STEP 0: does line predict outcome (marginally + within-mechanism)? ──
    line_outcomes: dict[str, list[int]] = defaultdict(list)
    seen = set()
    for t in trials:
        for c in t["chains"]:
            if c["line"] and (t["nct"], c["line"]) not in seen:
                seen.add((t["nct"], c["line"]))
                line_outcomes[c["line"]].append(t["y"])
    print("\nSTEP 0 — marginal success rate by line-of-therapy (distinct trials):")
    for ln, ys in sorted(line_outcomes.items(), key=lambda kv: -len(kv[1])):
        print(f"  {ln:14s} n={len(ys):3d}  success={mean(ys):.3f}")
    # within-mechanism early-vs-late
    mech_line: dict[tuple, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for t in trials:
        per = {}
        for c in t["chains"]:
            if c["mech"] and c["linegrp"]:
                per[(c["mech"], c["linegrp"])] = t["y"]
        for (m, lg), y in per.items():
            mech_line[m][lg].append(y)
    early_late_pairs = [(m, d) for m, d in mech_line.items() if "early" in d and "late" in d]
    if early_late_pairs:
        de = mean(y for _m, d in early_late_pairs for y in d["early"])
        dl = mean(y for _m, d in early_late_pairs for y in d["late"])
        print(f"  within-mechanism (mechanisms w/ both early+late, n={len(early_late_pairs)}): "
              f"early success {de:.3f} vs late {dl:.3f}  (Δ={de - dl:+.3f})")

    # ── STEP 1: case-based LOO predictor, several overlap schemes ──
    # A scheme = ordered backoff ladder of key-functions (finest first).
    def K(*attrs):
        def fn(c):
            vals = [c.get(x) for x in attrs]
            return tuple(vals) if all(v is not None for v in vals) else None
        return fn

    schemes: dict[str, list] = {
        "marginal_mech": [K("mech")],
        "mech_line":     [K("mech", "line"), K("mech")],
        "mech_linegrp":  [K("mech", "linegrp"), K("mech")],
        "mech_pop":      [K("mech", "pop"), K("mech")],
        # biology unit + its conditioners — marginal_bio is the CONTROL that
        # tells us if biology_line's lift is the LINE conditioner or just biology.
        "marginal_bio":  [K("bio")],
        "bio_line":      [K("bio", "line"), K("bio")],
        "bio_linegrp":   [K("bio", "linegrp"), K("bio")],
        "bio_pop":       [K("bio", "pop"), K("bio")],
        "bio_severity":  [K("bio", "severity"), K("bio")],
        "bio_stage":     [K("bio", "stage"), K("bio")],
        "bio_linegrp_sev": [K("bio", "linegrp", "severity"), K("bio", "linegrp"), K("bio")],
        "marginal_ind":  [K("ind")],
        "backbone":      [K("target", "mech", "bio"), K("mech", "bio"), K("mech")],
    }

    # index per (scheme, ladder-level): key -> {trial_id: y}  (distinct trials)
    indices: dict[str, list[dict]] = {}
    for name, ladder in schemes.items():
        idx_levels = []
        for keyfn in ladder:
            idx: dict = defaultdict(dict)
            for t in trials:
                for c in t["chains"]:
                    k = keyfn(c)
                    if k is not None:
                        idx[k][t["nct"]] = t["y"]
            idx_levels.append((keyfn, idx))
        indices[name] = idx_levels

    def predict(name: str, trial: dict, agg) -> float | None:
        chain_ps = []
        for c in trial["chains"]:
            p = None
            for keyfn, idx in indices[name]:
                k = keyfn(c)
                if k is None:
                    continue
                others = {tid: y for tid, y in idx.get(k, {}).items() if tid != trial["nct"]}
                if len(others) >= args.k_min:
                    s = sum(others.values())
                    p = (s + a0) / (len(others) + a0 + b0)
                    break
            if p is None:
                p = base_rate
            chain_ps.append(p)
        return agg(chain_ps) if chain_ps else None

    print(f"\nSTEP 1 — case-based LOO holdout AUROC (k_min={args.k_min}); "
          f"vs chance 0.5, random-fold 0.534:")
    print(f"  {'scheme':14s}  {'AUROC(mean)':>11s}  {'AUROC(min)':>10s}")
    results = {}
    for name in schemes:
        for agg_name, agg in (("mean", mean), ("min", min)):
            probs, ys = [], []
            for t in trials:
                p = predict(name, t, agg)
                if p is None:
                    continue
                probs.append(p); ys.append(t["y"])
            au = _auroc(probs, ys) if len(set(ys)) == 2 else float("nan")
            results[(name, agg_name)] = au
        print(f"  {name:14s}  {results[(name,'mean')]:>11.3f}  {results[(name,'min')]:>10.3f}")

    base = results[("marginal_mech", "mean")]
    lines = ["# Conditional-overlap experiment — findings\n",
             f"Graph: `{args.graph}`, {len(trials)} trials, base rate {base_rate:.3f}, "
             f"k_min={args.k_min}. Case-based LOO (no edge attribution).\n",
             "Does conditioning the outcome-pool on context (line/population) beat the "
             "marginal `{mechanism}`? Compared to random-fold holdout 0.534.\n",
             "| scheme | AUROC(mean) | AUROC(min) | Δ vs marginal_mech |",
             "|---|---|---|---|"]
    for name in schemes:
        m = results[(name, "mean")]
        lines.append(f"| {name} | {m:.3f} | {results[(name,'min')]:.3f} | {m - base:+.3f} |")
    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
