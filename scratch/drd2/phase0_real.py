"""Phase 0 real-substrate proof: does the direction backoff FIRE on the built
DRD2 graph (not just the synthetic unit test)?

The substrate is thin — one shared both-direction edge (DRD2 modulates_via
"Dopamine receptors"). This confirms that, routed through predict_clinical_hypothesis
(the canonical entry) on the REAL snapshot, agonist vs antagonist stated chains get
different predictions under backoff, and that backoff differs from flat. If it does
NOT move here, the eval is uninterpretable and we STOP.
"""
from __future__ import annotations

from collections import defaultdict

from src.graph.store import GraphStore
from src.prediction.path_query import (
    DIRECTION_BACKOFF, DIRECTION_FLAT, DIRECTION_SAME_ONLY,
    predict_clinical_hypothesis, seed_prediction_rng,
)

SNAP = "data/exports/drd2_subset_annotated.json"
DRD2 = "ENSG00000149295"
DRUGS = {
    "pramipexol": "agonist", "ropinirole": "agonist", "ropinirol": "agonist",
    "rotigotine": "agonist", "apomorphine": "agonist", "cabergoline": "agonist",
    "haloperidol": "antagonist", "risperidone": "antagonist",
    "olanzapine": "antagonist", "metoclopramide": "antagonist",
    "prochlorperazine": "antagonist",
}


def _drug_of(cid: str) -> str | None:
    for d in DRUGS:
        if d in cid.lower():
            return d
    return None


def main() -> int:
    g = GraphStore(); g.import_snapshot(SNAP)
    nodes = g._graph.nodes

    # stated (compound, indication, direction) pairs whose chain walks THROUGH DRD2
    pairs = {}  # (cid, ind) -> direction
    for ts in g.trial_subgraphs.values():
        for ch in ts.chains:
            if ch.target_id != DRD2:
                continue
            drug = _drug_of(ch.compound_id)
            if not drug:
                continue
            pairs[(ch.compound_id, ch.indication_id)] = (DRUGS[drug], drug,
                                                          ch.direction)
    print(f"DRD2-walking stated (compound,indication) pairs: {len(pairs)}\n")

    rows = []
    for (cid, ind), (exp_dir, drug, chain_dir) in sorted(pairs.items()):
        r = {}
        for mode in (DIRECTION_BACKOFF, DIRECTION_FLAT, DIRECTION_SAME_ONLY):
            seed_prediction_rng(42)
            try:
                res = predict_clinical_hypothesis(g, cid, ind, direction_mode=mode)
                r[mode] = res.overall_probability
            except Exception as e:  # noqa: BLE001
                r[mode] = None
        rows.append((drug, exp_dir, chain_dir, cid, ind, r))

    # print a sample
    print(f"{'drug':14s} {'dir':10s} {'backoff':>8s} {'flat':>8s} {'same':>8s}  indication")
    moved = 0
    for drug, exp_dir, chain_dir, cid, ind, r in rows:
        bo, fl, so = r[DIRECTION_BACKOFF], r[DIRECTION_FLAT], r[DIRECTION_SAME_ONLY]
        if bo is None:
            continue
        indn = nodes[ind].get("name", ind) if ind in nodes else ind
        delta = abs(bo - fl) if (bo is not None and fl is not None) else 0
        if delta > 1e-4:
            moved += 1
        print(f"{drug:14s} {chain_dir:10s} {bo:8.4f} "
              f"{(fl if fl is not None else -1):8.4f} "
              f"{(so if so is not None else -1):8.4f}  {indn[:34]}")

    # aggregate: does backoff separate agonist vs antagonist means?
    by_dir = defaultdict(list)
    for drug, exp_dir, chain_dir, cid, ind, r in rows:
        if r[DIRECTION_BACKOFF] is not None and chain_dir in ("agonist", "antagonist"):
            by_dir[chain_dir].append(r[DIRECTION_BACKOFF])
    print("\n=== PHASE 0 REAL-SUBSTRATE VERDICT ===")
    for d in ("agonist", "antagonist"):
        v = by_dir[d]
        if v:
            print(f"  {d:11s} backoff overall: n={len(v)} mean={sum(v)/len(v):.4f}")
    print(f"  pairs where backoff != flat (direction moved prediction): {moved}/{len(rows)}")
    fired = moved > 0
    print(f"\n  DIRECTION BACKOFF FIRES ON REAL GRAPH: {fired}")
    if not fired:
        print("  *** GATE FAILS: backoff does not move any real prediction — STOP ***")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
