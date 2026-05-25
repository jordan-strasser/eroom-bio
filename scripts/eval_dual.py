"""Dual P(success) eval on the classic 5 holdouts: scalar marginal belief vs
(s,t)-localized "multi-dimensional edge math", on the causal chain.

For each holdout it resolves the chain, then aggregates the same edges two ways
(identical softmin): (a) each edge's scalar Beta mean, (b) each localizable edge's
field mean queried at the trial's own (s,t). Reports both + direction-correctness
vs the literature outcome.

In-sample (ceiling): the holdout IS in --graph (built + attributed), so this is
the upper bound. True holdout: point --graph at a snapshot without the holdout.

Usage:
    python -m scripts.eval_dual --graph data/exports/mi_v2_ceiling_annotated.json \
        --field <private>/mi_v2_ceiling_annotated_belief_field.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.graph.store import GraphStore
from src.prediction.field_prediction import (
    load_edge_fields,
    localized_chain_probability,
    trial_chain_descriptions,
)
from src.prediction.path_query import predict_clinical_hypothesis

# NCT -> (label, literature outcome). The classic 5.
HOLDOUTS = {
    "NCT01844505": ("nivolumab CheckMate-067", "success"),
    "NCT01127633": ("solanezumab EXPEDITION", "failure"),
    "NCT00112918": ("bevacizumab AVANT", "failure"),
    "NCT00134264": ("torcetrapib ILLUMINATE", "failure"),
    "NCT00970359": ("selumetinib thyroid", "success"),
}
ANN = Path("data/annotations")


def _correct(p: float, lit: str) -> str:
    pred = "success" if p >= 0.5 else "failure"
    return "✓" if pred == lit else "✗"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--field", required=True, help="private belief_field snapshot")
    args = ap.parse_args()

    from src.graph.biolord_embeddings import embed_text

    g = GraphStore()
    g.import_snapshot(args.graph)
    field_map = load_edge_fields(args.field)
    print(f"graph: {g._graph.number_of_nodes()} nodes, {len(g.trial_subgraphs)} trials; "  # noqa: SLF001
          f"field: {len(field_map)} localized edges\n")

    cache: dict = {}
    rows = []
    for nct, (name, lit) in HOLDOUTS.items():
        ts = g.trial_subgraphs.get(nct)
        if not ts or not ts.chains:
            print(f"  {nct} {name}: NOT in graph (skip — needs in-sample build for ceiling)")
            continue
        ch = ts.chains[0]
        try:
            result = predict_clinical_hypothesis(
                g, ch.compound_id, ch.indication_id, n_samples=2000,
            )
        except Exception as e:  # noqa: BLE001
            print(f"  {nct} {name}: predict failed: {e}")
            continue
        if not result.edge_contributions:
            print(f"  {nct} {name}: no evidenced edges")
            continue
        descs = trial_chain_descriptions(ANN / f"{nct}_extraction.json")
        try:
            ind_name = g.get_node(ch.indication_id).get("name", "")
        except KeyError:
            ind_name = ch.indication_id
        p_scalar, p_local, per_edge = localized_chain_probability(
            result.edge_contributions, field_map, descs,
            embed_fn=embed_text, indication_name=ind_name, embed_cache=cache,
        )
        n_loc = sum(1 for e in per_edge if e["is_localized"])
        rows.append((nct, name, lit, p_scalar, p_local, n_loc, per_edge))

    # table
    print(f"{'NCT':12} {'holdout':26} {'lit':8} {'P_scalar':>9} {'P_(s,t)':>9} {'loc.edges':>10}")
    print("-" * 92)
    sc_ok = st_ok = 0
    for nct, name, lit, ps, pl, nloc, _ in rows:
        sc_ok += _correct(ps, lit) == "✓"
        st_ok += _correct(pl, lit) == "✓"
        print(f"{nct:12} {name:26} {lit:8} {ps:8.3f}{_correct(ps,lit)} {pl:8.3f}{_correct(pl,lit)} {nloc:>10}")
    n = len(rows)
    print("-" * 92)
    print(f"direction-correct:  scalar {sc_ok}/{n}   (s,t) {st_ok}/{n}")

    # per-edge localization detail (where (s,t) moved the belief)
    print("\nlocalized-edge detail (scalar -> (s,t)):")
    for nct, name, _, _, _, _, per_edge in rows:
        moved = [e for e in per_edge if e["is_localized"] and abs(e["scalar"] - e["localized"]) >= 0.02]
        if moved:
            print(f"  {nct} {name}:")
            for e in moved:
                print(f"    {e['edge']}:  {e['scalar']} -> {e['localized']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
