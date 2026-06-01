"""Throwaway: compare bottom-up vs top-down annotated graphs (#1/#2/#4)."""

from __future__ import annotations

from src.graph.store import GraphStore
from src.prediction.path_query import predict_clinical_hypothesis

_ROLES = ("compound_id", "target_id", "mechanism_id", "biology_id",
          "indication_id", "endpoint_id", "subgroup_population_id")


def _concepts(g):
    out = set()
    for ts in g.trial_subgraphs.values():
        for ch in ts.chains:
            for a in _ROLES:
                v = getattr(ch, a, None)
                if v and v != "UNKNOWN":
                    out.add(v)
    return out


def _ewb(g):
    return sum(1 for *_e, d in g._graph.edges(keys=True, data=True)
               if (d.get("belief") or {}).get("evidence"))


def _edge_ep(g):
    out = {}
    for s, t, k, d in g._graph.edges(keys=True, data=True):
        b = d.get("belief")
        if not b:
            continue
        a, be = b.get("alpha", 1.0), b.get("beta", 1.0)
        out[(s, t, k)] = a / (a + be)
    return out


def _chain_keys(g):
    out = {}
    for ts in g.trial_subgraphs.values():
        for ch in ts.chains:
            c, i = ch.compound_id, ch.indication_id
            if c and i and "UNKNOWN" not in (c, i):
                out[(c, i)] = ch
    return out


def main():
    bu = GraphStore(); bu.import_snapshot("data/exports/mi_bu52_annotated.json")
    td = GraphStore(); td.import_snapshot("data/exports/mi_v2_annotated.json")

    print("=== #1 structure (bottom-up vs top-down) ===")
    print(f"  bottom-up: {bu._graph.number_of_nodes()} nodes  {bu._graph.number_of_edges()} edges  "
          f"{_ewb(bu)} w/belief  {len(bu.trial_subgraphs)} trials")
    print(f"  top-down:  {td._graph.number_of_nodes()} nodes  {td._graph.number_of_edges()} edges  "
          f"{_ewb(td)} w/belief  {len(td.trial_subgraphs)} trials")
    cbu, ctd = _concepts(bu), _concepts(td)
    print(f"  chain concepts: bu={len(cbu)} td={len(ctd)}  "
          f"missing(td not in bu)={len(ctd - cbu)}  extra(bu not in td)={len(cbu - ctd)}")
    if ctd - cbu:
        print(f"    missing sample: {sorted(ctd - cbu)[:8]}")
    if cbu - ctd:
        print(f"    extra sample:   {sorted(cbu - ctd)[:8]}")
    only_td = set(td.trial_subgraphs) - set(bu.trial_subgraphs)
    only_bu = set(bu.trial_subgraphs) - set(td.trial_subgraphs)
    print(f"  trials only top-down: {sorted(only_td)}   only bottom-up: {sorted(only_bu)}")

    print("\n=== #4 per-edge Beta on shared edges ===")
    ebu, etd = _edge_ep(bu), _edge_ep(td)
    shared = set(ebu) & set(etd)
    if shared:
        diffs = [abs(ebu[k] - etd[k]) for k in shared]
        print(f"  shared edges={len(shared)}  mean|dE[p]|={sum(diffs)/len(diffs):.4f}  "
              f"max={max(diffs):.4f}  (#>0.05: {sum(1 for d in diffs if d > 0.05)})")

    print("\n=== #2 prediction parity (shared compound->indication chains) ===")
    kbu, ktd = _chain_keys(bu), _chain_keys(td)
    shared_ch = set(kbu) & set(ktd)
    pdiffs, dir_diff, worst = [], 0, []
    for key in shared_ch:
        try:
            pbu = predict_clinical_hypothesis(bu, key[0], key[1], n_samples=2000).overall_probability
            ptd = predict_clinical_hypothesis(td, key[0], key[1], n_samples=2000).overall_probability
        except Exception:
            continue
        pdiffs.append(abs(pbu - ptd))
        worst.append((abs(pbu - ptd), key[0], key[1], pbu, ptd))
        if (pbu >= 0.5) != (ptd >= 0.5):
            dir_diff += 1
    if pdiffs:
        print(f"  shared chains={len(pdiffs)}  mean|dP|={sum(pdiffs)/len(pdiffs):.4f}  "
              f"max={max(pdiffs):.4f}  direction-differs={dir_diff}")
        for d, c, i, pbu, ptd in sorted(worst, reverse=True)[:5]:
            print(f"    dP={d:.3f}  bu={pbu:.3f} td={ptd:.3f}  {c} -> {i}")


if __name__ == "__main__":
    main()
