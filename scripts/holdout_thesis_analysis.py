"""North-star thesis harness — three decoupled axes + the cross-indication ablation.

Tests the claim: *the system predicts a held-out trial's failure, and the
deciding (weakest-link) edge's belief is built from OTHER trials — ideally from
OTHER indications entirely.* We decouple it so each piece is honestly auditable:

  AXIS 1 — CROSS-INDICATION LEARNING (structural, no holdout needed)
    Does one edge's belief get co-updated by trials from ≥2 indications? Counts
    BiologyNode/MechanismNode bridges (find_biology_bridges/find_mechanism_bridges).
    Necessary substrate; proves pooling exists, not that it predicts.

  AXIS 2 — PREDICTIVE EFFICACY (holdout, out-of-sample)
    Does the chain-walk P(success) rank real CT.gov SUCCESS above FAILURE?
    AUROC (threshold-free) + Brier + accuracy@threshold vs the base rate.
    Label = _resolve_label (real primary_endpoint_met; non-mechanistic failures
    dropped to ambiguous). Run scripts.eval_baselines for the LogReg/XGBoost tie-line.

  AXIS 3 — THE INTERSECTION (the differentiated claim) + ABLATION
    Among holdout trials whose DECIDING edge carries cross-indication evidence:
    hit-rate, and the ablation — re-aggregate efficacy with the cross-indication
    evidence REMOVED. A trial where the full graph calls failure correctly and
    the ablated graph does NOT is a prediction *only cross-indication transfer
    enables* (a flat model structurally cannot do this). That flip is the proof,
    not the overlap.

HONESTY CHECKS baked in:
  * Out-of-sample: expects the holdout trials to be IN the graph's trial_subgraphs
    but EXCLUDED from attribution (build with --exclude-from-attribution), so
    their chains resolve yet they contributed zero evidence. The harness VERIFIES
    this per trial (deciding edge must have NO self-evidence) and loudly flags any
    contamination.
  * Coverage reported (trials whose chain doesn't resolve are excluded, counted).
  * Denominator first, spotlight second — no cherry-picking.

Usage:
    python -m scripts.holdout_thesis_analysis --graph data/exports/<demo>_annotated.json \
        --holdout-ncts data/corpora/holdout_2021_2026.txt --threshold 0.5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.graph.models import EdgeBeliefState
from src.graph.store import GraphStore
from src.prediction.calibration import auroc, brier_score, expected_calibration_error
from src.prediction.path_query import (
    _aggregate_samples,
    _sample_edge,
    _trust_weight,
    predict_clinical_hypothesis,
)
from src.prediction.provenance import (
    _NCT_RE,
    _edge_provenance,
    _replay_records,
    build_nct_indication_index,
    find_biology_bridges,
    find_mechanism_bridges,
    is_non_disease,
    therapeutic_area,
)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.eval_holdout_compose import _resolve_label  # noqa: E402

ANN_DIR = Path("data/annotations")


# ── ablation primitives (set-generalization of provenance._belief_excluding) ──

def _belief_excluding_set(
    belief: EdgeBeliefState, exclude: set[str], edge_type_value: str
) -> EdgeBeliefState:
    """Edge belief with every record whose source_id is in ``exclude`` removed,
    via the same delta-adjust as provenance._belief_excluding (anchor on the
    stored belief; subtract only the replayed contribution of the dropped
    records; floor at Beta(1,1))."""
    kept = [ev for ev in belief.evidence if ev.source_id not in exclude]
    if len(kept) == len(belief.evidence):
        return EdgeBeliefState(alpha=belief.alpha, beta=belief.beta, evidence=belief.evidence)
    full = _replay_records(belief.evidence, edge_type_value)
    excl = _replay_records(kept, edge_type_value)
    alpha = belief.alpha + (excl.alpha - full.alpha)
    beta = belief.beta + (excl.beta - full.beta)
    if alpha < 1.0 or beta < 1.0 or (alpha + beta - 2.0) <= 0.0:
        return EdgeBeliefState(alpha=1.0, beta=1.0, evidence=kept)
    return EdgeBeliefState(alpha=alpha, beta=beta, evidence=kept)


def _efficacy_excluding(contributions, exclude: set[str], n_samples: int):
    """Re-aggregate chain efficacy with ``exclude`` source NCTs removed from every
    edge (edges left below their prior are dropped, as the engine drops Beta(1,1)).
    Mirrors provenance._self_excluded_efficacy. Returns (efficacy|None, n_edges)."""
    rng = np.random.default_rng()
    samples, weights = [], []
    for c in contributions:
        b = _belief_excluding_set(c.belief, exclude, c.edge_type.value)
        if b.evidence_strength <= 0.0:
            continue
        samples.append(_sample_edge(rng, b, n_samples))
        weights.append(_trust_weight(b))
    if not samples:
        return None, 0
    return float(np.mean(_aggregate_samples(samples, weights))), len(samples)


# ── per-trial analysis ────────────────────────────────────────────────────

def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _decisive_result(store: GraphStore, nct: str, n_samples: int):
    """Replicate trace_holdout's choice: predict each distinct (compound,
    indication) the trial poses, return the PredictionResult with the LOWEST
    overall probability (the most decisive chain)."""
    sg = store.trial_subgraphs.get(nct)
    if sg is None:
        return None
    seen, results = set(), []
    for ch in sg.chains:
        key = (ch.compound_id, ch.indication_id)
        if key in seen:
            continue
        seen.add(key)
        try:
            res = predict_clinical_hypothesis(
                store, ch.compound_id, ch.indication_id, n_samples=n_samples
            )
        except KeyError:
            continue
        if res.edge_contributions:
            results.append(res)
    if not results:
        return None
    return min(results, key=lambda r: r.overall_probability)


def analyze_trial(store, nct, nct_index, n_samples, threshold) -> dict:
    out: dict = {"nct": nct, "covered": False, "label": None}
    ext = _load_json(ANN_DIR / f"{nct}_extraction.json")
    cls = _load_json(ANN_DIR / f"{nct}_classification.json")
    out["label"] = _resolve_label(ext, cls) if ext else None

    res = _decisive_result(store, nct, n_samples)
    if res is None:
        return out  # uncovered: chain doesn't resolve against the graph
    out["covered"] = True
    self_inds = nct_index.get(nct, set())
    decider = res.weakest_link
    dkey = (decider.source_id, decider.target_id, decider.edge_type) if decider else None
    provs = [
        _edge_provenance(c, store=store, nct_index=nct_index, self_nct=nct,
                         self_indications=self_inds,
                         is_deciding=(c.source_id, c.target_id, c.edge_type) == dkey)
        for c in res.edge_contributions
    ]
    dprov = next((p for p in provs if p.is_deciding), None)

    cross: set[str] = set()
    for c in res.edge_contributions:
        for ev in c.belief.evidence:
            s = ev.source_id
            if _NCT_RE.match(s) and s != nct and (nct_index.get(s, set()) - self_inds):
                cross.add(s)
    abl_eff, _ = _efficacy_excluding(res.edge_contributions, cross, n_samples)
    abl_p = abl_eff * (1.0 - res.safety_penalty) if abl_eff is not None else None

    full_p = res.overall_probability
    out.update(
        compound=res.trial_hypothesis.split(" -> ")[0],
        hypothesis=res.trial_hypothesis,
        full_p=full_p, efficacy=res.efficacy_probability, safety=res.safety_penalty,
        deciding=dprov, abl_p=abl_p, n_cross=len(cross),
        self_contaminated=bool(dprov and dprov.self_ncts),
        deciding_cross=bool(dprov and dprov.cross_indication_ncts),
    )
    if out["label"] in ("success", "failure"):
        pred_fail = full_p < threshold
        out["pred_fail"] = pred_fail
        out["correct"] = pred_fail == (out["label"] == "failure")
    # ablation flip: full calls failure, removing cross-indication loses that call
    out["flip"] = abl_p is not None and full_p < threshold <= abl_p
    return out


# ── reports ────────────────────────────────────────────────────────────────

def axis1(store) -> None:
    print("\n" + "=" * 78)
    print("AXIS 1 — CROSS-INDICATION LEARNING (structural; the substrate)")
    print("=" * 78)
    bio = find_biology_bridges(store)
    mech = find_mechanism_bridges(store)
    bio_cross = [b for b in bio if b.spans_oncology_and_other]
    mech_cross = [m for m in mech if m.spans_oncology_and_other]
    print(f"BiologyNode bridges (one biology drives ≥2 indications): {len(bio)} "
          f"({len(bio_cross)} span oncology AND a non-onco disease)")
    for b in sorted(bio_cross, key=lambda x: -x.belief_spread)[:6]:
        s, w = b.strongest, b.weakest
        if s and w:
            print(f"   {b.biology_name[:42]:42s} {s.indication_id}={s.expected_probability:.2f} "
                  f"↔ {w.indication_id}={w.expected_probability:.2f} (spread {b.belief_spread:.2f})")
    print(f"MechanismNode bridges (one belief co-updated by trials from ≥2 indications): "
          f"{len(mech)} ({len(mech_cross)} cross-area)")
    for m in sorted(mech_cross, key=lambda x: -len(x.indications))[:6]:
        print(f"   {m.mechanism_name[:42]:42s} {', '.join(m.indications[:4])}")


def axis2(records, threshold) -> None:
    print("\n" + "=" * 78)
    print("AXIS 2 — PREDICTIVE EFFICACY (holdout, out-of-sample)")
    print("=" * 78)
    total = len(records)
    covered = [r for r in records if r["covered"]]
    contam = [r for r in covered if r.get("self_contaminated")]
    scored = [r for r in covered if r["label"] in ("success", "failure") and "full_p" in r]
    print(f"holdout trials: {total} | covered (chain resolves): {len(covered)} "
          f"| binary-labeled & scored: {len(scored)} "
          f"(ambiguous/unlabeled dropped: {len(covered) - len(scored)})")
    if contam:
        print(f"  ⚠ CONTAMINATION: {len(contam)} trials have SELF-evidence on the deciding edge "
              f"→ NOT out-of-sample. Rebuild with --exclude-from-attribution for these: "
              f"{', '.join(r['nct'] for r in contam[:8])}")
    if not scored:
        print("  no binary-scored trials yet")
        return
    probs = [r["full_p"] for r in scored]
    y = [0 if r["label"] == "failure" else 1 for r in scored]   # 1=success
    base = sum(y) / len(y)
    acc = sum((p >= threshold) == (yy == 1) for p, yy in zip(probs, y)) / len(y)
    print(f"  AUROC {auroc(probs, y):.3f} | Brier {brier_score(probs, y):.3f} | "
          f"ECE {expected_calibration_error(probs, y):.3f} | acc@{threshold:.2f} {acc:.3f} | "
          f"base-rate {base:.3f} (success={sum(y)}, failure={len(y)-sum(y)})")
    print("  (AUROC is threshold-free — ranks success>failure; run scripts.eval_baselines "
          "for the LogReg/XGBoost tie-line on the same trials)")


def axis3(records, threshold) -> None:
    print("\n" + "=" * 78)
    print("AXIS 3 — INTERSECTION: efficacy GIVEN cross-indication transfer + ABLATION")
    print("=" * 78)
    scored = [r for r in records if r["covered"] and r["label"] in ("success", "failure")
              and "full_p" in r]
    xind = [r for r in scored if r.get("deciding_cross")]
    correct_x = [r for r in xind if r.get("correct")]
    flips = [r for r in xind if r.get("flip")]
    print(f"binary-scored covered: {len(scored)}")
    print(f"  deciding edge carries cross-indication evidence: {len(xind)}")
    print(f"  …of those, binary-correct: {len(correct_x)}")
    print(f"  …ablation FLIPS (remove cross-indication → loses the failure call): {len(flips)}")
    # softer, threshold-free signal: how much removing cross-indication raises P
    # for the correct-failure cross-indication trials (positive ⇒ transfer is
    # actively suppressing the prediction even without a clean 0.5 crossing).
    deltas = [r["abl_p"] - r["full_p"] for r in correct_x
              if r["label"] == "failure" and r.get("abl_p") is not None]
    if deltas:
        near = sum(1 for d in deltas if d >= 0.05)
        print(f"  …ablation Δ on correct failures (abl_p − full_p): "
              f"mean {sum(deltas)/len(deltas):+.3f}, {near}/{len(deltas)} rise ≥0.05 "
              f"(transfer actively lowering the call)")
    spotlight = [r for r in flips if r.get("correct") and r["label"] == "failure"]
    print(f"\nSPOTLIGHT — correct failure ∧ cross-indication-deciding ∧ ablation-flips "
          f"({len(spotlight)}): the predictions ONLY cross-indication transfer enables")
    for r in spotlight:
        d = r["deciding"]
        xinds = {k: v for k, v in (d.cross_indication_ncts or {}).items() if not is_non_disease(k)}
        areas = sorted({therapeutic_area(k) for k in xinds})
        print(f"\n  {r['nct']}  {r['hypothesis']}")
        print(f"    predicted FAIL  full_p={r['full_p']:.2f}  →  ablated (no cross-ind) "
              f"p={r['abl_p']:.2f}   [actual: FAILURE ✓]")
        print(f"    deciding edge: {d.source_name[:30]}→{d.target_name[:30]} "
              f"({d.edge_type})  E[p]={d.expected_probability:.2f}  from {d.n_records} records")
        print(f"    cross-indication support ({len(xinds)} diseases, areas={areas}):")
        for ind, ncts in list(xinds.items())[:4]:
            print(f"       {ind}: {', '.join(ncts[:4])}")


def per_trial_table(records, threshold) -> None:
    print("\n" + "=" * 78)
    print("PER-TRIAL DETAIL")
    print("=" * 78)
    print(f"{'nct':12} {'label':8} {'full_p':>7} {'pred':>6} {'ok':>3} "
          f"{'abl_p':>7} {'xind':>5} {'flip':>5}  deciding edge")
    for r in sorted(records, key=lambda x: (not x["covered"], x.get("full_p", 1.0))):
        if not r["covered"]:
            print(f"{r['nct']:12} {'—':8} {'(uncovered: chain does not resolve)'}")
            continue
        d = r.get("deciding")
        de = f"{d.source_name[:18]}→{d.target_name[:18]}({d.edge_type})" if d else "-"
        pred = ("FAIL" if r.get("pred_fail") else "ok") if "pred_fail" in r else "-"
        ok = ("✓" if r.get("correct") else "✗") if "correct" in r else "-"
        abl = f"{r['abl_p']:.2f}" if r.get("abl_p") is not None else "-"
        print(f"{r['nct']:12} {str(r['label'])[:8]:8} {r.get('full_p', 0):>7.2f} "
              f"{pred:>6} {ok:>3} {abl:>7} {'Y' if r.get('deciding_cross') else '-':>5} "
              f"{'Y' if r.get('flip') else '-':>5}  {de}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True, help="train(+excluded-holdout) snapshot")
    ap.add_argument("--holdout-ncts", required=True,
                    help="file of holdout NCT ids (one per line, # comments ok)")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="P(success) cutoff for the binary failure call + flip test")
    ap.add_argument("--n-samples", type=int, default=4000)
    a = ap.parse_args()

    store = GraphStore()
    store.import_snapshot(a.graph)
    nct_index = build_nct_indication_index(store)
    ncts = [ln.strip() for ln in Path(a.holdout_ncts).read_text().splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")]
    print(f"graph: {a.graph}  |  holdout trials: {len(ncts)}  |  threshold {a.threshold}")

    axis1(store)
    records = [analyze_trial(store, nct, nct_index, a.n_samples, a.threshold) for nct in ncts]
    axis2(records, a.threshold)
    axis3(records, a.threshold)
    per_trial_table(records, a.threshold)


if __name__ == "__main__":
    main()
