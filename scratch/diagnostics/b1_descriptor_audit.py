"""Phase 0 — BiologyNode descriptor provenance audit (read-only).

Traces every descriptor string a BiologyNode feeds into the four consumers and
checks whether the Tier-3 BioLORD merge is comparing the correct, consistent,
*rich* biological claim — or an impoverished/inconsistent stub. Pure measurement.

Consumers (file:line from the task spec):
  (a) id hash      bio:<sha1(norm desc)>      populate.py:650-651
  (b) SapBERT      name_id                    node_merge.py:118-120 (Tier-2)
  (c) BioLORD T3   _node_text=description     node_merge.py:134, 293-299 (Tier-3, the
                                              one that's supposed to collapse paraphrases)
  (d) field        per-chain biology_description  field_prediction.build_st_desc_map:82

All on data/exports/multi_500_annotated.json (n=472 in-graph trials).
"""
import hashlib
import json
import re
import statistics as st
from collections import Counter, defaultdict

SNAP = "data/exports/multi_500_annotated.json"


def norm(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def bio_hash(desc: str) -> str:
    return "bio:" + hashlib.sha1(norm(desc).encode("utf-8")).hexdigest()[:12]


def main():
    d = json.load(open(SNAP))
    nodes = d["graph"]["nodes"]
    ts = d["trial_subgraphs"]

    bio_nodes = {n["id"]: n for n in nodes if n.get("node_type") == "BiologyNode"}
    print(f"# biology nodes: {len(bio_nodes)}")

    # --- per-chain biology_description, grouped by biology_id; integer reuse = #distinct host trials
    chain_descs = defaultdict(list)          # bio_id -> [per-chain desc, ...]
    host_trials = defaultdict(set)           # bio_id -> {nct,...}
    for nct, t in ts.items():
        for ch in t.get("chains", []):
            bid = ch.get("biology_id")
            if bid is None:
                continue
            host_trials[bid].add(nct)
            chain_descs[bid].append((ch.get("biology_description") or "").strip())

    # --- integer reuse control (should reproduce MERGE_POOLING_MAP: median 1, 71% singleton, 5.2% >=8)
    reuse = {bid: len(host_trials.get(bid, set())) for bid in bio_nodes}
    vals = sorted(reuse.values())
    n = len(vals)
    singleton = sum(1 for v in vals if v <= 1)
    unobs = sum(1 for v in vals if v == 0)
    ge8 = sum(1 for v in vals if v >= 8)
    print("\n=== REUSE CONTROL (integer = #distinct host trials w/ a chain referencing the node) ===")
    print(f"  median={st.median(vals)} mean={st.mean(vals):.2f} max={max(vals)}")
    print(f"  %unobserved(==0)={100*unobs/n:.1f}  %singleton(<=1)={100*singleton/n:.1f}  %>=8={100*ge8/n:.1f}")
    print("  (MERGE_POOLING_MAP P1: median 1, mean 2.65, 0% unobs, 71% singleton, 5.2% >=8)")

    # --- descriptor richness: node description AND per-chain biology_description
    def lenstats(strings, label):
        words = [len(s.split()) for s in strings if s]
        chars = [len(s) for s in strings if s]
        print(f"\n=== RICHNESS — {label} (n={len(words)}) ===")
        if not words:
            print("  (none)")
            return
        def q(x, p):
            x = sorted(x); import math
            i = min(len(x)-1, int(p*len(x)))
            return x[i]
        print(f"  words: median={st.median(words)} mean={st.mean(words):.1f} p90={q(words,0.9)} max={max(words)} min={min(words)}")
        print(f"  chars: median={st.median(chars)} mean={st.mean(chars):.1f} p90={q(chars,0.9)} max={max(chars)}")
        wc = Counter(words)
        print(f"  word-count histogram (words: #nodes): " +
              ", ".join(f"{w}:{c}" for w, c in sorted(wc.items())[:12]))

    node_descs = [bn.get("description") or "" for bn in bio_nodes.values()]
    lenstats(node_descs, "post-merge NODE.description (= Tier-3 _node_text input)")
    all_chain_descs = [c for cs in chain_descs.values() for c in cs]
    lenstats(all_chain_descs, "per-chain biology_description (= field input)")

    # --- consistency: is each consumer's string the SAME string?
    # (1) does id == sha1(norm(node.description))?  -> id-hash input vs Tier-3 input identical?
    id_matches_desc = 0
    id_matches_name = 0
    name_eq_desc = 0
    for bid, bn in bio_nodes.items():
        desc = bn.get("description") or ""
        name = bn.get("name") or ""
        ont = bn.get("ontology_id") or bid
        if bio_hash(desc) == ont:
            id_matches_desc += 1
        if bio_hash(name) == ont:
            id_matches_name += 1
        if norm(desc) == norm(name):
            name_eq_desc += 1
    print("\n=== CONSISTENCY across the 4 consumers ===")
    print(f"  id == sha1(norm(node.description)):  {id_matches_desc}/{len(bio_nodes)}  "
          "(id-hash input == Tier-3 _node_text input)")
    print(f"  id == sha1(norm(node.name)):         {id_matches_name}/{len(bio_nodes)}")
    print(f"  norm(description) == norm(name):     {name_eq_desc}/{len(bio_nodes)}  "
          "(name_id ~ description ~ same phrase?)")

    # (2) is the post-merge node.description the SAME as the per-chain biology_description(s)?
    #     i.e. does the field see a richer string than the merge?
    node_eq_chain = 0
    node_richer = 0
    chain_richer = 0
    examples_chain_richer = []
    for bid, bn in bio_nodes.items():
        nd = norm(bn.get("description") or "")
        cds = {norm(c) for c in chain_descs.get(bid, []) if c}
        if not cds:
            continue
        if cds == {nd}:
            node_eq_chain += 1
        else:
            # differ: compare median lengths
            nlen = len(nd.split())
            clens = [len(c.split()) for c in cds]
            if max(clens) > nlen:
                chain_richer += 1
                if len(examples_chain_richer) < 6:
                    examples_chain_richer.append((bn.get("description"), sorted(cds, key=len, reverse=True)[:3]))
            else:
                node_richer += 1
    print(f"\n  per-chain biology_description vs node.description:")
    print(f"    identical (all chains == node):  {node_eq_chain}")
    print(f"    a chain is LONGER than node:     {chain_richer}")
    print(f"    node longer than all chains:     {node_richer}")
    if examples_chain_richer:
        print("    (examples where field sees a longer per-chain string than the node/merge:)")
        for nd, cds in examples_chain_richer:
            print(f"       node='{nd}'  chains={cds}")

    # --- format / session drift markers
    def fmt_markers(strings):
        m = Counter()
        for s in strings:
            if not s:
                continue
            if "#" in s: m["has_#"] += 1
            if "**" in s: m["has_**markdown"] += 1
            if s.strip().endswith("."): m["ends_with_period"] += 1
            if re.search(r"[.!?]\s+[A-Z]", s): m["multi_sentence"] += 1
            if len(s.split()) >= 8: m["ge8_words(sentence-like)"] += 1
            if s and s[0].isupper(): m["starts_upper"] += 1
            if s == s.lower(): m["all_lowercase"] += 1
        return m
    print("\n=== FORMAT / SESSION-DRIFT MARKERS — node.description ===")
    for k, v in fmt_markers(node_descs).most_common():
        print(f"  {k}: {v}/{len(bio_nodes)} ({100*v/len(bio_nodes):.0f}%)")
    print("=== FORMAT / SESSION-DRIFT MARKERS — per-chain biology_description ===")
    nchain = len([c for c in all_chain_descs if c])
    for k, v in fmt_markers(all_chain_descs).most_common():
        print(f"  {k}: {v}/{nchain} ({100*v/max(1,nchain):.0f}%)")

    # metadata keys present on biology nodes (look for created-at / schema / version)
    metakeys = Counter()
    for bn in bio_nodes.values():
        for k in (bn.get("metadata") or {}):
            metakeys[k] += 1
    print("\n=== node metadata keys (look for created-at/schema/version) ===")
    for k, v in metakeys.most_common():
        print(f"  {k}: {v}")

    # --- SAMPLE 20: side-by-side id-hash string vs Tier-3 string vs raw per-chain desc
    print("\n=== SAMPLE 20 BIOLOGY NODES — descriptor side-by-side ===")
    # diverse sample: sort by reuse, take a spread
    ordered = sorted(bio_nodes.items(), key=lambda kv: reuse[kv[0]])
    idx = [int(i * (len(ordered) - 1) / 19) for i in range(20)]
    for i in sorted(set(idx)):
        bid, bn = ordered[i]
        idhash_input = norm(bn.get("description") or "")           # (a) what got hashed
        tier3_input = (bn.get("description") or bn.get("name") or "").strip()  # (c) _node_text
        nameid = bn.get("name_id") or ""                          # (b)
        chains = chain_descs.get(bid, [])
        uniq_chains = sorted({c for c in chains if c})
        print(f"\n  [{bid}] reuse={reuse[bid]}")
        print(f"    (a) id-hash input (norm desc): '{idhash_input}'")
        print(f"    (c) Tier-3 _node_text        : '{tier3_input}'")
        print(f"    (b) SapBERT name_id          : '{nameid}'")
        print(f"    (d) field per-chain desc[{len(uniq_chains)}u]: " +
              ("; ".join(f'\'{c}\'' for c in uniq_chains[:3]) + (" ..." if len(uniq_chains) > 3 else "")))


if __name__ == "__main__":
    main()
