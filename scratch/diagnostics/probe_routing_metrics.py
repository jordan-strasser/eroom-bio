"""A3/A4 aggregate metrics: contamination (#2) + efficacy-edge spread (#3).

Compares a BASELINE annotated snapshot (EROOM_ROUTING off) against a ROUTED one
(EROOM_ROUTING on), both re-attributed from the SAME initial.json so the only
difference is the flag.

  #2  % of failure-trials (trial_outcome=failure) that touch the EFFICACY SPINE
      (affects / modulates_via / mechanism_affects / biology_drives). Baseline
      ~100% (every failure downvotes the spine); routing should drop it toward
      the EFFICACY+MEASUREMENT share (~32%) since SAFETY/OPERATIONAL censor.
  #3  Efficacy-edge posterior spread (mean / sd / median of E[p] over evidenced
      efficacy edges), vs the global base rate. Removing contamination should
      RAISE the mean off the below-base-rate centering and widen the spread.

Read-only. Usage:
  .venv/bin/python -m scratch.diagnostics.probe_routing_metrics \
      data/exports/multi_500_baseline_reattr.json \
      data/exports/multi_500_routed_reattr.json
"""
from __future__ import annotations

import sys
from collections import Counter

import numpy as np

import scratch.diagnostics._common as C

SPINE = {"affects", "modulates_via", "mechanism_affects", "biology_drives"}


def _failure_ncts() -> set[str]:
    """NCTs whose classifier trial_outcome == failure (the P4 baseline pop)."""
    out = set()
    for nct in _ALL_TRIAL_NCTS:
        _ext, cls = C.load_annotation(nct)
        if cls and cls.get("trial_outcome") == "failure":
            out.add(nct)
    return out


def _branch_breakdown(failure_ncts: set[str]) -> Counter:
    """Theoretical routing branch per failure trial, from its primary mode."""
    from src.annotation.taxonomy import FailureMode, routing_branch_for
    c: Counter = Counter()
    for nct in failure_ncts:
        _ext, cls = C.load_annotation(nct)
        mode_str = C.primary_failure_mode(cls)
        try:
            mode = FailureMode(mode_str) if mode_str else None
        except ValueError:
            mode = None
        c[routing_branch_for(mode).value] += 1
    return c


def _spine_touch_ncts(g) -> set[str]:
    """NCTs that landed any trial-sourced record on an efficacy-spine edge."""
    touched: set[str] = set()
    for _u, _v, key, b in C.iter_edges(g):
        if key not in SPINE or b is None:
            continue
        for nct in C.trial_evidence_ncts(b):
            touched.add(nct)
    return touched


def _class_spread(g) -> dict[str, tuple[int, float, float, float]]:
    """(n, mean, sd, median) of E[p] over evidenced edges, by class."""
    by_cls: dict[str, list[float]] = {}
    for _u, _v, key, b in C.iter_edges(g):
        cls = C.EDGE_CLASS.get(key, "?")
        if b is None or b.evidence_strength <= 0:
            continue
        by_cls.setdefault(cls, []).append(b.expected_probability)
    out = {}
    for cls, vals in by_cls.items():
        a = np.array(vals)
        out[cls] = (len(a), float(a.mean()), float(a.std()), float(np.median(a)))
    return out


def _base_rate(g) -> tuple[float, int]:
    from scripts.eval_holdout_compose import _resolve_label
    labels = []
    for nct in g.trial_subgraphs:
        ext, cls = C.load_annotation(nct)
        if ext is None:
            continue
        lab = _resolve_label(ext, cls)
        if lab in ("success", "failure"):
            labels.append(1 if lab == "success" else 0)
    return (float(np.mean(labels)) if labels else float("nan"), len(labels))


def _report(tag: str, g, failure_ncts: set[str]) -> None:
    print(f"\n========== {tag} ==========")
    in_graph_fail = {n for n in failure_ncts if n in g.trial_subgraphs}
    touch = _spine_touch_ncts(g)
    ftouch = in_graph_fail & touch
    pct = 100 * len(ftouch) / max(len(in_graph_fail), 1)
    print(f"#2 failure-trials touching efficacy spine: "
          f"{len(ftouch)}/{len(in_graph_fail)} = {pct:.1f}%")
    br, nbr = _base_rate(g)
    print(f"    [ref] mechanistic binary base success rate: {br:.3f} (n={nbr})")
    print("#3 posterior E[p] spread by edge class (evidenced edges):")
    spread = _class_spread(g)
    for cls in (C.EFFICACY, C.MEASUREMENT, C.SAFETY, C.MODULATION):
        if cls in spread:
            n, mean, sd, med = spread[cls]
            print(f"    {cls:12} n={n:5} mean={mean:.3f} sd={sd:.3f} med={med:.3f}")


if __name__ == "__main__":
    baseline_path = sys.argv[1]
    routed_path = sys.argv[2]

    gb = C.load_graph(baseline_path)
    gr = C.load_graph(routed_path)
    # Trial NCTs to consider: union of both graphs' subgraphs.
    _ALL_TRIAL_NCTS = set(gb.trial_subgraphs) | set(gr.trial_subgraphs)
    failure_ncts = _failure_ncts()
    print(f"failure-trials (trial_outcome=failure) in annotations: {len(failure_ncts)}")
    print("routing-branch breakdown over failure-trials (theoretical):")
    for branch, n in sorted(_branch_breakdown(failure_ncts).items(),
                            key=lambda kv: -kv[1]):
        print(f"    {branch:12} {n:4} ({100*n/max(len(failure_ncts),1):.0f}%)")
    spine_touchers = {"efficacy", "measurement", "unknown"}
    bd = _branch_breakdown(failure_ncts)
    expect = sum(n for b, n in bd.items() if b in spine_touchers)
    print(f"  → expect ~{100*expect/max(len(failure_ncts),1):.0f}% to touch the "
          f"spine under routing (efficacy+measurement+unknown)")

    _report("BASELINE (EROOM_ROUTING off)", gb, failure_ncts)
    _report("ROUTED (EROOM_ROUTING on)", gr, failure_ncts)
