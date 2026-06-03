"""Throwaway: isolate the description-biology effect from the prompt/safety confound.

Three graphs:
  mi_bu52   bottom-up, DESCRIPTION-identity biology, NEW annotations
  mi_td_new top-down,   LEGACY ontology biology,      NEW annotations  (same annotations as mi_bu52)
  mi_v2     top-down,   LEGACY ontology biology,      OLD annotations

Comparisons:
  A) mi_bu52 vs mi_td_new  -> description-biology effect with prompt/safety HELD CONSTANT
  B) mi_td_new vs mi_v2    -> how much the new prompt/annotations alone moved top-down (the confound)

Per chain we split overall = efficacy x (1 - safety_penalty) so a safety-driven
move (re-annotation) is distinguishable from an efficacy-driven move (biology).
"""

from __future__ import annotations

from src.graph.store import GraphStore
from src.prediction.path_query import predict_clinical_hypothesis

FOCUS = [
    ("pembrolizumab", "melanoma"),
    ("nivolumab", "melanoma"),
    ("ipilimumab", "melanoma"),
    ("atezolizumab", "melanoma"),
    ("tiragolumab", "melanoma"),
    ("mek162", "melanoma"),
    ("irinotecan_hydrochloride", "colorectal_cancer"),
]


def _load(name):
    g = GraphStore()
    g.import_snapshot(f"data/exports/{name}_annotated.json")
    return g


def _chain_keys(g):
    out = {}
    for ts in g.trial_subgraphs.values():
        for ch in ts.chains:
            c, i = ch.compound_id, ch.indication_id
            if c and i and "UNKNOWN" not in (c, i):
                out[(c, i)] = ch
    return out


def _pred(g, c, i):
    try:
        r = predict_clinical_hypothesis(g, c, i, n_samples=2000)
        return (r.overall_probability, r.efficacy_probability, r.safety_penalty)
    except Exception:
        return None


def compare(g1, l1, g2, l2):
    print(f"\n=== {l1}  vs  {l2}  (overall | efficacy | safety_penalty) ===")
    k1, k2 = _chain_keys(g1), _chain_keys(g2)
    shared = set(k1) & set(k2)
    diffs, flips, worst = [], 0, []
    for c, i in shared:
        p1, p2 = _pred(g1, c, i), _pred(g2, c, i)
        if not p1 or not p2:
            continue
        d = abs(p1[0] - p2[0])
        diffs.append(d)
        worst.append((d, c, i, p1, p2))
        if (p1[0] >= 0.5) != (p2[0] >= 0.5):
            flips += 1
    if diffs:
        print(f"  shared={len(diffs)}  mean|dP|={sum(diffs)/len(diffs):.4f}  "
              f"max={max(diffs):.4f}  direction-flips={flips}")
        print(f"  worst chains:")
        for d, c, i, p1, p2 in sorted(worst, reverse=True)[:7]:
            print(f"    dP={d:.3f}  {l1}=[{p1[0]:.3f} eff={p1[1]:.3f} saf={p1[2]:.3f}]  "
                  f"{l2}=[{p2[0]:.3f} eff={p2[1]:.3f} saf={p2[2]:.3f}]  {c}->{i}")


def focus_table(graphs):
    print("\n=== focus chains: overall (efficacy, safety) across builds ===")
    hdr = "  {:32s}".format("compound -> indication")
    for name, _ in graphs:
        hdr += f"  {name:>22s}"
    print(hdr)
    for c, i in FOCUS:
        row = "  {:32s}".format(f"{c[:20]}->{i[:10]}")
        for _, g in graphs:
            p = _pred(g, c, i)
            row += f"  {('%.2f(e%.2f,s%.2f)'%(p[0],p[1],p[2])) if p else 'n/a':>22s}"
        print(row)


def main():
    bu = _load("mi_bu52")     # bottom-up, description biology, new annotations
    tdn = _load("mi_td_new")  # top-down, ontology biology, new annotations
    v2 = _load("mi_v2")       # top-down, ontology biology, old annotations

    focus_table([("bu_desc", bu), ("td_new", tdn), ("v2_old", v2)])
    compare(bu, "bu_desc(new)", tdn, "td_ontology(new)")   # A: biology effect, confound held
    compare(tdn, "td_ontology(new)", v2, "v2_ontology(old)")  # B: confound magnitude


if __name__ == "__main__":
    main()
