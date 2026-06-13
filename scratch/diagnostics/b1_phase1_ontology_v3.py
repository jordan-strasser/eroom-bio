"""Phase 1B v3 — robust ontology coverage + collapse (ancestor-free).

v1 mis-gated on direction; v2 under-stemmed (apoptosis != apoptotic) and its
ancestor roll-up collapsed everything onto the BFO process root. v3:

  - stem content words (5-char prefix after suffix strip) so apoptosis~apoptotic.
  - gate: accept an ontology hit if the description's biological head-noun(s)
    appear (stemmed) in the term label — direction/regulation/generic words ignored.
  - coverage reported for GO (primary) and for GO U EFO U MONDO U HPO.
  - collapse WITHOUT ancestors: two biology nodes collapse if their accepted-term
    labels share a stemmed biological noun (= the canonical concept a controlled-
    vocab id would key on). No DAG fetch, no BFO-root artifact.
  - characterize the UNMAPPED descriptions (the clinical-outcome story).

Uses the two on-disk OLS caches (multi-ontology + GO-only).
"""
import json
import os
import re
from collections import Counter, defaultdict

SNAP = "data/exports/multi_500_annotated.json"
MULTI = "scratch/diagnostics/ols_cache.json"
GO = "scratch/diagnostics/ols_go_cache.json"

STOP = set("""a an the of in to and or via for with on at by is be as into onto from up
regulation regulated regulate negative positive response process processes signaling
signalling pathway pathways activity function functional mediated mediation host cell
cellular reduction reduced reduce inhibition inhibit inhibited inhibitory suppression
suppress suppressed enhancement enhance enhanced activation activate activated increase
increased decrease decreased modulation modulate modulating modulator production improvement
improve improved control level levels symbiont effector during after entry disruption
disrupted promotion promote restoration restore attenuation attenuate homeostasis general
involved type related associated other release secretion accumulation formation
development differentiation growth survival death dependent independent system organismal
multicellular biological biosynthetic metabolic catabolic anti via mediated""".split())


def stem(w):
    for suf in ("ation", "ication", "ically", "ical", "tion", "sion", "ing", "ed",
                "ogenesis", "esis", "osis", "otic", "ity", "ies", "ation", "ation"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            w = w[: -len(suf)]
            break
    return w[:5]


def bio_stems(s):
    return {stem(w) for w in re.findall(r"[a-z0-9]+", (s or "").lower())
            if w not in STOP and len(w) > 2}


def load(p):
    return json.load(open(p)) if os.path.exists(p) else {}


def main():
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
    descs = {b: (n.get("description") or n.get("name") or "") for b, n in bio.items()}
    singles = [b for b in bio if reuse[b] <= 1]

    multi = load(MULTI)
    go = load(GO)

    def best_hit(q, only_go):
        """Best ontology hit whose label shares a stemmed biological noun w/ q."""
        qs = bio_stems(q)
        cands = []
        for src in ([go] if only_go else [go, multi]):
            for x in (src.get(q) or []) if isinstance(src.get(q), list) else []:
                lab, oid, ont = x.get("label"), x.get("obo_id"), x.get("ontology_name")
                if not lab or not oid:
                    continue
                if only_go and not str(oid).startswith("GO:"):
                    continue
                shared = qs & bio_stems(lab)
                if shared:
                    cands.append((len(shared), oid, lab, (ont or "?"), shared))
        if not cands:
            return None
        cands.sort(reverse=True)
        _, oid, lab, ont, shared = cands[0]
        return (oid, lab, ont, shared)

    go_map = {b: best_hit(q, True) for b, q in descs.items()}
    any_map = {b: best_hit(q, False) for b, q in descs.items()}

    def cov(m, label):
        nm = sum(1 for a in m.values() if a)
        sm = sum(1 for b in singles if m[b])
        print(f"  {label:<26} mapped {nm}/{len(bio)} ({100*nm/len(bio):.0f}%)  | "
              f"singletons {sm}/{len(singles)} ({100*sm/len(singles):.0f}%)")
        onts = Counter(a[2].upper() for a in m.values() if a)
        print(f"      ontology source: " + ", ".join(f"{k}:{v}" for k, v in onts.most_common()))
        return m

    print("=== COVERAGE (stem-aware biological-noun overlap) ===")
    cov(go_map, "GO biological-process")
    cov(any_map, "GO U EFO U MONDO U HPO")

    # collapse via shared stemmed biological concept (the controlled-vocab key)
    def collapse(m, name):
        concept_nodes = defaultdict(set)
        node_concepts = {}
        for b, a in m.items():
            if not a:
                continue
            node_concepts[b] = a[3]
            for c in a[3]:
                concept_nodes[c].add(b)
        nmapped_single = sum(1 for b in singles if b in node_concepts)
        coll = 0
        examples = []
        for b in singles:
            if b not in node_concepts:
                continue
            partners = set()
            for c in node_concepts[b]:
                partners |= concept_nodes[c]
            partners.discard(b)
            if partners:
                coll += 1
                if len(examples) < 14:
                    p = min(partners)
                    examples.append((descs[b], descs[p], node_concepts[b] & node_concepts[p]))
        print(f"\n=== COLLAPSE via shared {name} concept ===")
        print(f"  singleton collapse: {coll}/{len(singles)} ({100*coll/len(singles):.0f}% of ALL singletons)")
        print(f"  of {nmapped_single} MAPPED singletons -> {100*coll/max(1,nmapped_single):.0f}%")
        # group-size histogram (preview of reuse if re-keyed by concept)
        # representative concept = the most populated concept a node touches
        rep = {}
        for b, cs in node_concepts.items():
            rep[b] = max(cs, key=lambda c: len(concept_nodes[c]))
        groups = Counter(rep.values())
        sizes = sorted(groups.values(), reverse=True)
        print(f"  re-keyed groups: {len(groups)}; top sizes {sizes[:12]}; >=8: {sum(1 for s in sizes if s>=8)}")
        print("  sample concept merges (singleton ~ partner [shared stems]):")
        for a, b, sh in examples[:12]:
            print(f"    '{a}'  ~  '{b}'  [{','.join(sorted(sh))}]")
        return node_concepts

    collapse(go_map, "GO")
    collapse(any_map, "any-ontology")

    # characterize the UNMAPPED (GO) descriptions
    unmapped = [descs[b] for b in bio if not go_map[b]]
    print(f"\n=== UNMAPPED by GO ({len(unmapped)}/{len(bio)}) — what kind of biology? ===")
    buckets = {
        "cardio/vascular": r"cardiac|myocard|vascular|vasodil|blood pressure|hypertens|atheroscler|coagul|thrombo|ventric|arter|heart",
        "metabolic/endocrine": r"glycemi|glucose|insulin|lipid|cholesterol|ldl|hdl|metaboli|diabet|hormone|natriur|renal|uric",
        "neuro/cognitive": r"cogniti|neuro|cns|dopamin|anxiol|depress|seizure|pain|neuronal|alzheim",
        "immune/inflam": r"immun|inflamm|cytokine|t-cell|b-cell|autoimmun|tnf|interleuk|lymphocyte",
        "onco/cell-process": r"apopto|prolifer|angiogen|mitot|tumou?r|cell cycle|cytotox|growth inhib|metasta|dna",
        "tissue/organ-fx": r"mucosal|epitheli|bone|muscle|skin|hepati|pulmonar|gastric|detrusor|contractil",
    }
    bc = Counter()
    for u in unmapped:
        hit = False
        for name, pat in buckets.items():
            if re.search(pat, u.lower()):
                bc[name] += 1
                hit = True
                break
        if not hit:
            bc["other/uncategorized"] += 1
    for k, v in bc.most_common():
        print(f"  {k:<22} {v}")
    print("\n  sample unmapped descriptions:")
    for u in unmapped[:18]:
        print(f"    - {u}")


if __name__ == "__main__":
    main()
