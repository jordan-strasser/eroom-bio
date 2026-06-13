"""Phase 1B (robust) — map biology descriptions to a controlled GO-BP vocabulary
by BioLORD nearest-neighbor, network-free.

Vocabulary = the union of GO biological_process terms annotated to the corpus's
own genes (data/cache/quickgo/*.json) — 2367 terms, exactly the GO-BP space the
mechanism layer already lives in. For each of the 212 biology descriptions we take
the nearest GO-BP term by BioLORD cosine. This is direction-robust (BioLORD encodes
the concept, downweighting inhibition/activation) and deterministic (no OLS lexical
ranking, no SSL flakiness, no drug/CHEBI false hits).

  coverage   = % biology nodes whose nearest GO-BP term clears a cosine gate
  collapse   = % singleton nodes whose nearest GO-BP term is SHARED with >=1 other
               biology node (the real merge an ontology id would buy)
Calibrate the gate by eyeballing mapped pairs at each cosine band.
"""
import glob
import json
from collections import Counter, defaultdict

import numpy as np

SNAP = "data/exports/multi_500_annotated.json"


def main():
    # GO-BP vocab
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

    # biology nodes + reuse
    d = json.load(open(SNAP))
    nodes = d["graph"]["nodes"]
    ts = d["trial_subgraphs"]
    bio = {n["id"]: n for n in nodes if n.get("node_type") == "BiologyNode"}
    host = defaultdict(set)
    for nct, t in ts.items():
        for ch in t.get("chains", []):
            if ch.get("biology_id"):
                host[ch["biology_id"]].add(nct)
    reuse = {b: len(host.get(b, set())) for b in bio}
    bids = list(bio)
    bdescs = [bio[b].get("description") or bio[b].get("name") or "" for b in bids]
    singles = [b for b in bids if reuse[b] <= 1]

    # embed (cached on disk; model loads once)
    from src.graph.biolord_embeddings import embed_texts
    print("Embedding GO labels + biology descriptions via BioLORD ...")
    GE = np.asarray(embed_texts(go_labels), dtype=np.float32)
    BE = np.asarray(embed_texts(bdescs), dtype=np.float32)
    GE /= np.linalg.norm(GE, axis=1, keepdims=True) + 1e-9
    BE /= np.linalg.norm(BE, axis=1, keepdims=True) + 1e-9
    S = BE @ GE.T                          # 212 x 2367
    nn = S.argmax(axis=1)
    nncos = S.max(axis=1)

    near = {bids[i]: (go_ids[nn[i]], go_labels[nn[i]], float(nncos[i])) for i in range(len(bids))}

    # coverage at gates
    print("\n=== COVERAGE: nearest GO-BP term clears cosine gate ===")
    for g in (0.50, 0.55, 0.60, 0.65, 0.70):
        m = [b for b in bids if near[b][2] >= g]
        sm = [b for b in singles if near[b][2] >= g]
        print(f"  gate {g:.2f}: mapped {len(m)}/{len(bids)} ({100*len(m)/len(bids):.0f}%) | "
              f"singletons {len(sm)}/{len(singles)} ({100*len(sm)/len(singles):.0f}%)")

    # eyeball calibration: sample mapped pairs across cosine bands
    print("\n=== CALIBRATION: nearest-GO mappings by cosine band (is it correct?) ===")
    order = sorted(bids, key=lambda b: -near[b][2])
    for lo, hi in [(0.75, 1.01), (0.65, 0.75), (0.58, 0.65), (0.50, 0.58)]:
        band = [b for b in order if lo <= near[b][2] < hi]
        print(f"  -- band [{lo:.2f},{hi:.2f}): {len(band)} nodes --")
        for b in band[:6]:
            gid, lab, c = near[b]
            print(f"     {c:.3f}  '{bdescs[bids.index(b)]}'  ->  {gid} '{lab}'")

    # collapse at a calibrated gate
    for GATE in (0.55, 0.60, 0.65):
        members = defaultdict(list)
        for b in bids:
            if near[b][2] >= GATE:
                members[near[b][0]].append(b)
        coll = sum(1 for b in singles if near[b][2] >= GATE and len(members[near[b][0]]) >= 2)
        smap = sum(1 for b in singles if near[b][2] >= GATE)
        sizes = sorted((len(m) for m in members.values()), reverse=True)
        print(f"\n=== COLLAPSE @ gate {GATE} (shared nearest GO-BP term) ===")
        print(f"  singleton collapse: {coll}/{len(singles)} ({100*coll/len(singles):.0f}% of all singletons) "
              f"| {100*coll/max(1,smap):.0f}% of {smap} mapped singletons")
        print(f"  re-keyed GO groups: {len(members)}; top sizes {sizes[:10]}; >=8: {sum(1 for s in sizes if s>=8)}")
        # show the multi-node GO groups (what would merge)
        big = sorted(members.items(), key=lambda kv: -len(kv[1]))
        shown = 0
        for gid, m in big:
            if len(m) < 2:
                continue
            print(f"    {gid} '{vocab[gid]}' <- " + "; ".join(f"'{bdescs[bids.index(x)]}'" for x in m[:5])
                  + (" ..." if len(m) > 5 else ""))
            shown += 1
            if shown >= 10:
                break


if __name__ == "__main__":
    main()
