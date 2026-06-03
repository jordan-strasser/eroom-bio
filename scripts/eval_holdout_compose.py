"""Compose-and-scan holdout eval — predict test trials WITHOUT building them in.

The honest true-holdout: a test trial is NOT added to the graph. Its chain is
*composed* from known anchors and *scanned* for compatibility against the trained
graph — predicted ONLY if it lands, else reported "unknown" rather than
fabricating a number on a chain the corpus never established.

A holdout is, by definition, a NOVEL compound — so we DON'T require its compound
in the graph (that would defeat the point). We anchor on its TARGET: a new
anti-PD-1 should be predictable from what other PD-1 trials taught the graph.
`predict_clinical_hypothesis(compound_id=None, target_id=...)` natively supports
this ("novel compound, familiar target") — it skips the affects edge and predicts
the target-onward chain from the trained beliefs. Compound-specific binding and
off-target safety are deliberately NOT modeled: the question is "would an
anti-{target} in {indication} succeed, given what we've learned?"

Compatibility gate: scorable when the TARGET and INDICATION resolve to graph
nodes AND a path exists. Targets absent from the corpus (e.g. APP, VEGFA in a
small oncology corpus) are honest "unknown"s — the graph has no knowledge to
generalize from. Contrast `eval_dual.py`, which builds the holdout in (even
excluded from attribution) and so reads the holdout's OWN populate-time priors;
here the training graph stays pristine and the prediction uses only what the
OTHER trials established — which is what generalization actually means.

Usage:
    python -m scripts.eval_holdout_compose --graph data/exports/mi_oc52_annotated.json
"""

from __future__ import annotations

import argparse

from src.graph.store import GraphStore
from src.prediction.path_query import predict_clinical_hypothesis

# Classic 5 case-study holdouts: label -> (compound, indication_slug,
# target_gene_symbol, literature_direction). Indication is the slug the trial's
# condition canonicalizes to; target is the gene the novel compound engages.
HOLDOUTS: dict[str, tuple[str, str, str, str]] = {
    "nivolumab CheckMate-067": ("nivolumab", "melanoma", "PDCD1", "success"),
    "solanezumab EXPEDITION": ("solanezumab", "alzheimer_disease", "APP", "failure"),
    "bevacizumab AVANT": ("bevacizumab", "colorectal_cancer", "VEGFA", "failure"),
    "torcetrapib ILLUMINATE": ("torcetrapib", "cardiovascular_diseases", "CETP", "failure"),
    "selumetinib thyroid": ("selumetinib", "differentiated_thyroid_cancer", "MAP2K1", "success"),
}


def _target_index(g: GraphStore) -> dict[str, str]:
    return {
        (g._graph.nodes[n].get("gene_symbol") or "").upper(): n  # noqa: SLF001
        for n in g._graph  # noqa: SLF001
        if g._graph.nodes[n].get("node_type") == "TargetNode"  # noqa: SLF001
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--n-samples", type=int, default=2000)
    args = ap.parse_args()

    g = GraphStore()
    g.import_snapshot(args.graph)
    tgt_idx = _target_index(g)

    def present(nid: str) -> bool:
        return nid in g._graph  # noqa: SLF001

    print(f"compose-and-scan (target-anchored) holdout eval — graph: {args.graph}")
    print(f"{'holdout':26} | landing      | prediction                                  | direction")
    print("-" * 112)
    scored = correct = 0
    for label, (drug, ind, gene, lit) in HOLDOUTS.items():
        tid = tgt_idx.get(gene.upper())
        i_ok = present(ind)
        landing = f"tgt={'Y' if tid else 'n'} ind={'Y' if i_ok else 'n'}"
        pred, direction = "— (target absent — no knowledge)", "unknown"
        if tid and i_ok:
            try:
                r = predict_clinical_hypothesis(
                    g, None, ind, target_id=tid, n_samples=args.n_samples,
                )
                if r.edge_contributions:
                    overall = r.overall_probability
                    call = "success" if overall >= 0.5 else "failure"
                    ok = call == lit
                    scored += 1
                    correct += ok
                    path = "→".join(ec.target_id[:13] for ec in r.edge_contributions)
                    pred = f"P={overall:.3f} ({len(r.edge_contributions)}e: {path[:34]})"
                    direction = ("CORRECT" if ok else "WRONG") + f" [{call} vs {lit}]"
                else:
                    pred = "no path from target"
            except Exception as e:  # noqa: BLE001
                pred = f"err: {str(e)[:42]}"
        print(f"{label:26} | {landing:12} | {pred:43} | {direction}")
    print("-" * 112)
    rate = f"{correct}/{scored}" if scored else "0/0"
    print(f"scorable (target lands): {scored}/{len(HOLDOUTS)}    direction-correct: {rate}")
    print("\nNOTE: target-absent holdouts are HONEST unknowns — a small corpus simply has"
          "\nno knowledge of that biology to generalize from. The graph stays unmutated;"
          "\npredictions use only what the OTHER trials established on the shared chain.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
