"""Build data/cache/biology_ontology_map.json — the desc -> GO-BP map B1 keys on.

For every biology description in the corpus (post-merge node descriptions AND the
pre-merge per-chain biology_descriptions, so a fresh populate can map any chain),
record the nearest GO-biological-process term (BioLORD cosine) over the corpus GO-BP
vocabulary (union of QuickGO gene annotations). The gate is applied by the CONSUMER
(src/graph/biology_ontology.py), so we store the raw nearest term + cosine here.
Deterministic + offline once written.
"""
import glob
import json
from collections import OrderedDict

import numpy as np

SNAP = "data/exports/multi_500_annotated.json"
OUT = "data/cache/biology_ontology_map.json"


def normkey(s):
    return " ".join((s or "").lower().split())


def main():
    vocab = {}
    for f in glob.glob("data/cache/quickgo/*.json"):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        for x in d if isinstance(d, list) else []:
            if x.get("aspect") == "biological_process" and x.get("stable_id") and x.get("display_name"):
                vocab[x["stable_id"]] = x["display_name"]
    go_ids = list(vocab)
    go_labels = [vocab[g] for g in go_ids]
    print(f"GO-BP vocab: {len(go_ids)} terms")

    d = json.load(open(SNAP))
    nodes = d["graph"]["nodes"]
    ts = d["trial_subgraphs"]
    descs = set()
    for n in nodes:
        if n.get("node_type") == "BiologyNode":
            descs.add((n.get("description") or n.get("name") or "").strip())
    for t in ts.values():
        for ch in t.get("chains", []):
            bd = (ch.get("biology_description") or "").strip()
            if bd:
                descs.add(bd)
    descs = sorted(d for d in descs if d)
    print(f"unique biology descriptions to map: {len(descs)}")

    from src.graph.biolord_embeddings import embed_texts
    GE = np.asarray(embed_texts(go_labels), dtype=np.float32)
    GE /= np.linalg.norm(GE, axis=1, keepdims=True) + 1e-9
    DE = np.asarray(embed_texts(descs), dtype=np.float32)
    DE /= np.linalg.norm(DE, axis=1, keepdims=True) + 1e-9
    S = DE @ GE.T
    nn = S.argmax(axis=1)
    cos = S.max(axis=1)

    out = OrderedDict()
    for i, desc in enumerate(descs):
        out[normkey(desc)] = {
            "go_id": go_ids[nn[i]],
            "go_label": go_labels[nn[i]],
            "cos": round(float(cos[i]), 4),
        }
    json.dump(out, open(OUT, "w"), indent=0)
    print(f"wrote {OUT}: {len(out)} entries")
    # quick gate summary
    for g in (0.55, 0.60, 0.65):
        m = sum(1 for v in out.values() if v["cos"] >= g)
        print(f"  gate {g}: {m}/{len(out)} ({100*m/len(out):.0f}%) descriptions map")


if __name__ == "__main__":
    main()
