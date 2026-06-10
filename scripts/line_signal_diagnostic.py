"""Does the STRUCTURED population axis (line/stage/biomarker) carry cross-trial
signal for trial success? — the measurement that gates the #1 responds_differently
fix.

The free-text line test didn't transfer last session, but it used crude parsing.
This reads the graph's STRUCTURED PopulationNode.defining_features and asks, on
the train trials' own binary labels:

  1. (verify the bug) how many responds_differently edges exist / carry evidence,
     and how many PopulationNodes carry a `line` axis.
  2. success base-rate by (axis, level), parent-enrollment population, with n.
  3. an honest LOO signal check: predict each trial from the leave-one-out base
     rate of its population axes; AUROC vs the binary label.

No graph mutation. Run:
  python -m scripts.line_signal_diagnostic --graph data/exports/multi_500_annotated.json
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from src.graph.store import GraphStore
from src.graph.models import EdgeType
from src.prediction.calibration import auroc

import scripts.holdout_thesis_analysis as H

ANN = Path("data/annotations")


def _node_get(node, key, default=None):
    """Snapshot nodes import as dicts; be defensive about dict vs model."""
    if isinstance(node, dict):
        return node.get(key, default)
    return getattr(node, key, default)


def pop_axes(graph: GraphStore, pop_id: str) -> list[tuple[str, str]]:
    """[(axis, level)] for a population node, gene/biomarker keyed by gene."""
    if not pop_id or pop_id == "UNKNOWN":
        return []
    try:
        node = graph.get_node(pop_id)
    except KeyError:
        return []
    feats = _node_get(node, "defining_features") or []
    out: list[tuple[str, str]] = []
    for f in feats:
        axis = (f.get("axis") if isinstance(f, dict) else getattr(f, "axis", "")) or ""
        key = (f.get("key") if isinstance(f, dict) else getattr(f, "key", "")) or ""
        level = (f.get("level") if isinstance(f, dict) else getattr(f, "level", "")) or ""
        axis, key, level = axis.lower(), key.lower(), level.lower()
        if axis in ("gene", "biomarker"):
            out.append((f"biomarker:{key}", level))
        elif axis:
            out.append((axis, level))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    a = ap.parse_args()

    store = GraphStore()
    store.import_snapshot(a.graph)

    # ── 1. verify the bug ──────────────────────────────────────────────
    rd_total = rd_evidenced = 0
    for u, v, key, data in store._graph.edges(keys=True, data=True):  # noqa: SLF001
        if key != EdgeType.RESPONDS_DIFFERENTLY.value:
            continue
        rd_total += 1
        belief = data.get("belief")
        es = getattr(belief, "evidence_strength", None)
        if es is None and isinstance(belief, dict):
            es = belief.get("evidence_strength")
        if es and es > 0.0:
            rd_evidenced += 1

    pop_total = pop_with_line = 0
    for nid, node in store._graph.nodes(data=True):  # noqa: SLF001
        if _node_get(node, "node_type") != "population" and not str(nid).startswith(("line", "stage", "biomarker")):
            # population nodes don't have a uniform id prefix; detect by features
            if not _node_get(node, "defining_features"):
                continue
        feats = _node_get(node, "defining_features")
        if not feats:
            continue
        pop_total += 1
        axes = {(_node_get(f, "axis") or "").lower() for f in feats} if feats else set()
        if "line" in axes:
            pop_with_line += 1

    print(f"graph = {a.graph}")
    print(f"responds_differently edges: {rd_total} total, {rd_evidenced} with evidence")
    print(f"PopulationNodes (with defining_features): {pop_total}, carrying a `line` axis: {pop_with_line}")

    # ── trials + labels ────────────────────────────────────────────────
    ncts_labels: list[tuple[str, int]] = []
    for nct in store.trial_subgraphs:
        ext = H._load_json(ANN / f"{nct}_extraction.json")
        cls = H._load_json(ANN / f"{nct}_classification.json")
        lab = H._resolve_label(ext, cls) if ext else None
        if lab in ("success", "failure"):
            ncts_labels.append((nct, 1 if lab == "success" else 0))
    print(f"\nbinary-labeled trials: {len(ncts_labels)} "
          f"({sum(y for _, y in ncts_labels)} success / "
          f"{sum(1 - y for _, y in ncts_labels)} failure)")

    # ── per-trial population axes (parent enrollment cohort) ───────────
    # parent_population_id is the enrollment cohort; its axes are what the
    # trial's overall outcome reflects. Also collect chain subgroup pops.
    trial_axes: dict[str, list[tuple[str, str]]] = {}
    n_parent_pop = n_any_axis = 0
    for nct, _y in ncts_labels:
        ts = store.trial_subgraphs.get(nct)
        if not ts:
            continue
        parent_id = getattr(ts, "parent_population_id", None)
        axes = pop_axes(store, parent_id) if parent_id else []
        if axes:
            n_parent_pop += 1
        if not axes:
            # fall back to union of chain subgroup pops (within-trial strata)
            seen = set()
            for ch in ts.chains:
                pid = getattr(ch, "subgroup_population_id", None)
                if pid and pid not in seen:
                    seen.add(pid)
                    axes.extend(pop_axes(store, pid))
        axes = list(dict.fromkeys(axes))  # dedup, keep order
        trial_axes[nct] = axes
        if axes:
            n_any_axis += 1
    print(f"trials with a parent-pop axis: {n_parent_pop}; "
          f"with ANY axis (parent or subgroup): {n_any_axis}")

    # ── 2. success base-rate by (axis, level) ──────────────────────────
    cell: dict[tuple[str, str], list[int]] = defaultdict(list)
    for nct, y in ncts_labels:
        for (axis, level) in trial_axes.get(nct, []):
            cell[(axis, level)].append(y)
    axis_levels: dict[str, list[tuple[str, list[int]]]] = defaultdict(list)
    for (axis, level), ys in cell.items():
        axis_levels[axis].append((level, ys))

    overall_rate = sum(y for _, y in ncts_labels) / len(ncts_labels)
    print(f"\noverall success rate = {overall_rate:.3f}\n")
    print(f"  {'axis':18s} {'level':16s} {'n':>4s} {'succ_rate':>9s} {'Δ_overall':>9s}")
    for axis in sorted(axis_levels):
        rows = sorted(axis_levels[axis], key=lambda r: -len(r[1]))
        for level, ys in rows:
            if len(ys) < 3:
                continue
            r = sum(ys) / len(ys)
            print(f"  {axis:18s} {level:16s} {len(ys):>4d} {r:>9.3f} {r - overall_rate:>+9.3f}")

    # ── 3. honest LOO signal check: population-axis-only predictor ──────
    # For each trial, score = mean leave-one-out base rate of its axes (cells
    # with >=3 OTHER trials). Trials with no usable axis fall back to the global
    # LOO rate. AUROC vs label — does the structured axis beat chance?
    probs: list[float] = []
    ys: list[int] = []
    n_axis_used = 0
    for nct, y in ncts_labels:
        axis_scores: list[float] = []
        for (axis, level) in trial_axes.get(nct, []):
            ys_cell = cell[(axis, level)]
            # leave-one-out: this trial contributes exactly one y to the cell
            others = list(ys_cell)
            others.remove(y)  # drop one occurrence of this trial's label
            if len(others) >= 3:
                axis_scores.append(sum(others) / len(others))
        if axis_scores:
            score = sum(axis_scores) / len(axis_scores)
            n_axis_used += 1
        else:
            # global LOO base rate
            tot = sum(yy for _, yy in ncts_labels) - y
            score = tot / (len(ncts_labels) - 1)
        probs.append(score)
        ys.append(y)
    if len(set(ys)) == 2:
        au = auroc(probs, ys)
        print(f"\nLOO population-axis-only predictor (ALL axes averaged): AUROC = {au:.3f} "
              f"(n={len(ys)}, axis-informed for {n_axis_used}/{len(ys)} trials; "
              f"rest = global base rate)")

    # ── 3b. per-axis LOO AUROC (does any SINGLE axis predict, undiluted?) ──
    # For a given axis family, restrict to trials carrying it; score each by the
    # leave-one-out success rate of its (axis, level) cell; AUROC vs label.
    print(f"\n  per-axis LOO AUROC (only trials carrying that axis; min 15 trials, "
          f"each level >=3):")
    print(f"  {'axis':18s} {'n_trials':>8s} {'n_levels':>8s} {'AUROC':>7s}")
    axis_families = sorted({ax for (ax, _lv) in cell})
    # group biomarkers together as one family too
    families = list(axis_families) + ["biomarker:*"]
    for fam in families:
        rows: list[tuple[float, int]] = []
        levels_used: set[str] = set()
        for nct, y in ncts_labels:
            axes = [(ax, lv) for (ax, lv) in trial_axes.get(nct, [])
                    if (ax == fam) or (fam == "biomarker:*" and ax.startswith("biomarker:"))]
            scores = []
            for (ax, lv) in axes:
                others = list(cell[(ax, lv)])
                others.remove(y)
                if len(others) >= 3:
                    scores.append(sum(others) / len(others))
                    levels_used.add(f"{ax}={lv}")
            if scores:
                rows.append((sum(scores) / len(scores), y))
        if len(rows) >= 15 and len({y for _, y in rows}) == 2:
            au = auroc([p for p, _ in rows], [y for _, y in rows])
            print(f"  {fam:18s} {len(rows):>8d} {len(levels_used):>8d} {au:>7.3f}")


if __name__ == "__main__":
    main()
