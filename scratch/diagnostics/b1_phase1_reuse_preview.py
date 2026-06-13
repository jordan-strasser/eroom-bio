"""Phase 1 decision number — post-collapse biology TRIAL-reuse histogram.

If biology were re-keyed onto a GO-BP controlled vocabulary (nearest term by
BioLORD, gated; unmappable -> content-hash fallback), what does the biology-node
integer-reuse histogram become? This is the headline scoreboard metric (move
%singleton down and %>=8 up from 5.2% toward mechanism's 18.7%) and the honest
Phase-3 preview. Reuse here = # DISTINCT host trials (chains), summed over the
singleton nodes a GO term pools (the real merge), NOT node counts.
"""
import glob
import json
import statistics as st
from collections import defaultdict

import numpy as np

SNAP = "data/exports/multi_500_annotated.json"


def hist(vals, label):
    vals = sorted(vals)
    n = len(vals)
    s1 = sum(1 for v in vals if v <= 1)
    g8 = sum(1 for v in vals if v >= 8)
    g4 = sum(1 for v in vals if v >= 4)
    print(f"  {label:<34} n={n:>4} median={st.median(vals):>3} mean={np.mean(vals):.2f} "
          f"%singleton={100*s1/n:>4.1f} %>=4={100*g4/n:>4.1f} %>=8={100*g8/n:>4.1f} max={max(vals)}")


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

    d = json.load(open(SNAP))
    nodes = d["graph"]["nodes"]
    ts = d["trial_subgraphs"]
    bio = {n["id"]: n for n in nodes if n.get("node_type") == "BiologyNode"}
    host = defaultdict(set)
    for nct, t in ts.items():
        for ch in t.get("chains", []):
            if ch.get("biology_id"):
                host[ch["biology_id"]].add(nct)
    bids = list(bio)
    bdescs = [bio[b].get("description") or bio[b].get("name") or "" for b in bids]

    from src.graph.biolord_embeddings import embed_texts
    GE = np.asarray(embed_texts(go_labels), dtype=np.float32)
    BE = np.asarray(embed_texts(bdescs), dtype=np.float32)
    GE /= np.linalg.norm(GE, axis=1, keepdims=True) + 1e-9
    BE /= np.linalg.norm(BE, axis=1, keepdims=True) + 1e-9
    S = BE @ GE.T
    nn = S.argmax(axis=1)
    nncos = S.max(axis=1)

    # current (content-hash) reuse
    cur = [len(host.get(b, set())) for b in bids]
    print("=== BIOLOGY NODE TRIAL-REUSE: current vs GO-re-keyed ===")
    hist(cur, "current (bio:<sha1> content-hash)")

    for GATE in (0.55, 0.60, 0.65, 0.70):
        # new node id: GO term if cos>=gate else its own content-hash (fallback => stays separate)
        group_trials = defaultdict(set)
        for i, b in enumerate(bids):
            if nncos[i] >= GATE:
                key = ("GO", go_ids[nn[i]])
            else:
                key = ("HASH", b)
            group_trials[key] |= host.get(b, set())
        new = [len(v) for v in group_trials.values()]
        mapped = sum(1 for i in range(len(bids)) if nncos[i] >= GATE)
        hist(new, f"GO-re-keyed @ gate {GATE} (map {mapped}/{len(bids)})")

    # also: what % of biology-incident EFFICACY structure (mechanism_affects/biology_drives/
    # reflects_biology) would recur, approximated by node reuse >=2 share
    print("\n  (interpretation: %>=8 is the scoreboard target — mechanism is 18.7%)")


if __name__ == "__main__":
    main()
