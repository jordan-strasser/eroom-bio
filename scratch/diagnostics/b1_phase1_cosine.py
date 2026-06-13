"""Phase 1A — are the singleton biology nodes collapsible paraphrases?

For the 71% singleton biology nodes, compute BioLORD cosine to the nearest OTHER
biology node. What fraction have a neighbor at >= {0.70, 0.75, 0.80, 0.85}? Tells
us whether they sit just under the 0.85 Tier-3 bar (paraphrases the merge barely
missed) or are genuinely far apart (distinct biology at n~=500).

Uses the on-disk BioLORD cache (data/cache/biolord_embeddings.json) keyed by
normalized-lowercased description — the SAME vectors node_merge Tier-3 compared.
No model load needed if coverage is complete (reported).
"""
import json
import math
import statistics as st
from collections import defaultdict

SNAP = "data/exports/multi_500_annotated.json"
CACHE = "data/cache/biolord_embeddings.json"


def normkey(s: str) -> str:
    return " ".join((s or "").lower().split())


def main():
    d = json.load(open(SNAP))
    nodes = d["graph"]["nodes"]
    ts = d["trial_subgraphs"]
    bio = {n["id"]: n for n in nodes if n.get("node_type") == "BiologyNode"}

    # integer reuse
    host = defaultdict(set)
    for nct, t in ts.items():
        for ch in t.get("chains", []):
            if ch.get("biology_id"):
                host[ch["biology_id"]].add(nct)
    reuse = {b: len(host.get(b, set())) for b in bio}

    cache = json.load(open(CACHE))
    # vectors for each biology node (by its description)
    import numpy as np
    vecs = {}
    missing = []
    for b, n in bio.items():
        key = normkey(n.get("description") or n.get("name") or "")
        v = cache.get(key)
        if v is None:
            missing.append((b, key))
        else:
            vecs[b] = np.asarray(v, dtype=np.float32)
    print(f"BioLORD-cache coverage: {len(vecs)}/{len(bio)} biology nodes "
          f"({100*len(vecs)/len(bio):.0f}%); missing={len(missing)}")
    if missing[:5]:
        for b, k in missing[:5]:
            print("   MISSING:", b, repr(k[:60]))

    ids = list(vecs)
    M = np.stack([vecs[i] for i in ids])
    M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    S = M @ M.T
    np.fill_diagonal(S, -1.0)
    nn = S.max(axis=1)               # nearest-OTHER-node cosine
    nn_idx = S.argmax(axis=1)

    singles = [i for i, b in enumerate(ids) if reuse[ids[i]] <= 1]
    multis = [i for i, b in enumerate(ids) if reuse[ids[i]] >= 2]
    print(f"\nsingletons in cache: {len(singles)} ; multi-reuse: {len(multis)}")

    def frac_at(idxs, thr):
        if not idxs:
            return 0.0
        return 100 * sum(1 for i in idxs if nn[i] >= thr) / len(idxs)

    print("\n=== nearest-neighbor cosine: fraction with a neighbor >= threshold ===")
    print(f"{'group':<22}{'n':>5}  {'>=0.70':>8}{'>=0.75':>8}{'>=0.80':>8}{'>=0.85':>8}{'>=0.90':>8}  median_nn")
    for label, idxs in [("ALL biology", list(range(len(ids)))),
                        ("singletons(<=1)", singles),
                        ("multi-reuse(>=2)", multis)]:
        med = st.median([nn[i] for i in idxs]) if idxs else 0
        print(f"{label:<22}{len(idxs):>5}  "
              f"{frac_at(idxs,0.70):>7.0f}%{frac_at(idxs,0.75):>7.0f}%{frac_at(idxs,0.80):>7.0f}%"
              f"{frac_at(idxs,0.85):>7.0f}%{frac_at(idxs,0.90):>7.0f}%  {med:.3f}")

    # qualitative: 25 singletons + their nearest neighbor + cosine, sorted by cosine desc
    print("\n=== SINGLETON nearest-neighbor pairs (sorted by cosine) — paraphrase or distinct? ===")
    pairs = []
    for i in singles:
        j = nn_idx[i]
        pairs.append((nn[i], bio[ids[i]].get("description"), bio[ids[j]].get("description"),
                      reuse[ids[i]], reuse[ids[j]]))
    pairs.sort(reverse=True)
    print("  -- top 18 (closest singletons) --")
    for c, a, b, ra, rb in pairs[:18]:
        print(f"   {c:.3f}  '{a}'  <->  '{b}' (nbr reuse {rb})")
    print("  -- bottom 8 (most isolated singletons) --")
    for c, a, b, ra, rb in pairs[-8:]:
        print(f"   {c:.3f}  '{a}'  <->  '{b}'")

    # how many singletons would join SOME node at each bar (collapse rate by cosine)
    print("\n=== cosine-collapse rate: singletons that would merge at a looser bar ===")
    for thr in (0.70, 0.75, 0.80, 0.85):
        c = sum(1 for i in singles if nn[i] >= thr)
        print(f"   bar {thr}: {c}/{len(singles)} singletons join a neighbor = {100*c/len(singles):.0f}%")


if __name__ == "__main__":
    main()
