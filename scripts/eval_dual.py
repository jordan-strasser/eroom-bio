"""Dual P(success) eval on the classic 5 holdouts: scalar marginal belief vs
(s,t)-localized "multi-dimensional edge math", on the causal chain.

Each localizable edge is queried at the holdout's OWN arm descriptions (via
build_st_desc_map) — the multi-arm-correct (s,t). Both aggregates use the same
softmin, so the only difference is localization.

In-sample (ceiling): the holdout IS in --graph. True holdout: --graph excludes it.
--sweep tunes the kernel bandwidth (query-time, no re-materialize) and reports
(s,t) directionality at each width — smaller bandwidth = sharper, less
cross-dimension bleed.

Usage:
    python -m scripts.eval_dual --graph G --field F
    python -m scripts.eval_dual --graph G --field F --bandwidth 0.15
    python -m scripts.eval_dual --graph G --field F --sweep 0.05,0.1,0.15,0.2,0.25,0.35,0.5
"""

from __future__ import annotations

import argparse

from src.graph.store import GraphStore
from src.prediction.field_prediction import (
    build_st_desc_map,
    load_edge_fields,
    localized_chain_probability,
)
from src.prediction.path_query import predict_clinical_hypothesis

# NCT -> (label, literature outcome, drug-of-interest slug). The drug slug picks
# the chain to predict on for multi-arm trials (e.g. bevacizumab's arm in AVANT,
# not the chemo-only arm).
HOLDOUTS = {
    "NCT01844505": ("nivolumab CheckMate-067", "success", "nivolumab"),
    "NCT01127633": ("solanezumab EXPEDITION", "failure", "solanezumab"),
    "NCT00112918": ("bevacizumab AVANT", "failure", "bevacizumab"),
    "NCT00134264": ("torcetrapib ILLUMINATE", "failure", "torcetrapib"),
    "NCT00970359": ("selumetinib thyroid", "success", "selumetinib"),
}


def _ok(p, lit):
    return ("success" if p >= 0.5 else "failure") == lit


def _evaluate(g, field_map, embed_fn, cache, bandwidth=None):
    rows = []
    for nct, (name, lit, drug) in HOLDOUTS.items():
        ts = g.trial_subgraphs.get(nct)
        if not ts or not ts.chains:
            rows.append((nct, name, lit, None, None, None, "not-in-graph"))
            continue
        # predict on the chain whose compound is the drug of interest (the arm
        # that matters for a multi-arm trial), else fall back to the first chain.
        matching = [c for c in ts.chains if drug in (c.compound_id or "").lower()]
        ch = (matching or ts.chains)[0]
        try:
            result = predict_clinical_hypothesis(g, ch.compound_id, ch.indication_id, n_samples=2000)
        except Exception as e:  # noqa: BLE001
            rows.append((nct, name, lit, None, None, None, f"predict-fail:{e}"))
            continue
        if not result.edge_contributions:
            rows.append((nct, name, lit, None, None, None, "no-edges"))
            continue
        st_map = build_st_desc_map(g, nct)
        ps, pl, per_edge = localized_chain_probability(
            result.edge_contributions, field_map, st_map,
            embed_fn=embed_fn, bandwidth=bandwidth, embed_cache=cache,
        )
        rows.append((nct, name, lit, ps, pl, per_edge, None))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--field", required=True)
    ap.add_argument("--bandwidth", type=float, default=None)
    ap.add_argument("--sweep", default=None, help="comma-separated bandwidths")
    args = ap.parse_args()

    from src.graph.biolord_embeddings import embed_text

    g = GraphStore()
    g.import_snapshot(args.graph)
    field_map = load_edge_fields(args.field)
    print(f"graph: {g._graph.number_of_nodes()} nodes, {len(g.trial_subgraphs)} trials; "  # noqa: SLF001
          f"field: {len(field_map)} localized edges\n")
    cache: dict = {}

    if args.sweep:
        widths = [float(x) for x in args.sweep.split(",")]
        print(f"{'bandwidth':>10} {'scalar dir':>11} {'(s,t) dir':>10}   per-trial (s,t) P")
        print("-" * 80)
        base = _evaluate(g, field_map, embed_text, cache, bandwidth=None)
        sc = sum(1 for r in base if r[3] is not None and _ok(r[3], r[2]))
        n = sum(1 for r in base if r[3] is not None)
        for b in widths:
            rows = _evaluate(g, field_map, embed_text, cache, bandwidth=b)
            st = sum(1 for r in rows if r[4] is not None and _ok(r[4], r[2]))
            ps = " ".join(f"{r[0][-5:]}:{r[4]:.2f}{'✓' if _ok(r[4],r[2]) else '✗'}"
                          for r in rows if r[4] is not None)
            print(f"{b:>10.3f} {f'{sc}/{n}':>11} {f'{st}/{n}':>10}   {ps}")
        return 0

    rows = _evaluate(g, field_map, embed_text, cache, bandwidth=args.bandwidth)
    print(f"{'NCT':12} {'holdout':26} {'lit':8} {'P_scalar':>9} {'P_(s,t)':>9}")
    print("-" * 80)
    sc = st = n = 0
    for nct, name, lit, ps, pl, per_edge, skip in rows:
        if ps is None:
            print(f"{nct:12} {name:26} {lit:8}   ({skip})")
            continue
        n += 1; sc += _ok(ps, lit); st += _ok(pl, lit)
        print(f"{nct:12} {name:26} {lit:8} {ps:8.3f}{'✓' if _ok(ps,lit) else '✗'} "
              f"{pl:8.3f}{'✓' if _ok(pl,lit) else '✗'}")
    print("-" * 80)
    print(f"direction-correct:  scalar {sc}/{n}   (s,t) {st}/{n}"
          f"   (bandwidth={args.bandwidth or 'default 0.25'})")
    print("\nlocalized-edge detail (scalar -> (s,t)):")
    for nct, name, lit, ps, pl, per_edge, skip in rows:
        if not per_edge:
            continue
        moved = [e for e in per_edge if e["is_localized"]]
        if moved:
            print(f"  {nct} {name}:")
            for e in moved:
                print(f"    {e['edge']}:  {e['scalar']} -> {e['localized']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
