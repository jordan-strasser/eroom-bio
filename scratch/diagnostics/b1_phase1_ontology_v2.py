"""Phase 1B v2 — ontology coverage + collapse, done right.

v1 used a BioLORD cosine(desc, bare-GO-label) gate, which is miscalibrated: BioLORD
downweights direction, so cos("angiogenesis inhibition","angiogenesis")=0.59 < gate
and a CORRECT mapping was rejected. v1 also measured collapse at the maximally-
specific OLS top-hit, where paraphrases never coincide. Both flaws understate B1.

v2 fixes both:
  - GO-only OLS search (force the primary vocabulary the task names).
  - direction-robust gate: accept the top GO hit if it shares a biological CONTENT
    word with the description after stripping direction/regulation/generic words
    (so "angiogenesis inhibition" -> "negative regulation of angiogenesis" via
    "angiogenesis", but "blood pressure reduction" -> "post-exercise hypotension"
    is rejected — no shared biological noun).
  - roll-up collapse: fetch GO ancestors (OLS) and measure how many DISTINCT
    biology nodes share a non-generic GO ancestor (the real merge a B1 build buys),
    not just an identical leaf term.

Coverage = % biology nodes with an accepted GO mapping.
Collapse  = % singleton nodes that share an accepted-leaf OR coarse-ancestor term
            with >=1 other biology node.
OLS responses cache to disk.
"""
import json
import os
import re
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import threading

SNAP = "data/exports/multi_500_annotated.json"
GO_CACHE = "scratch/diagnostics/ols_go_cache.json"
ANC_CACHE = "scratch/diagnostics/ols_anc_cache.json"

# direction / regulation / generic words to ignore when testing biological-noun overlap
STOP = set("""a an the of in to and or via for with on at by is be as into onto from
regulation regulated regulate negative positive response process processes signaling
signalling pathway pathways activity function functional mediated mediation host cell
cellular reduction reduced reduce inhibition inhibit inhibited inhibitory suppression
suppress suppressed enhancement enhance enhanced activation activate activated increase
increased decrease decreased modulation modulate modulating production improvement
improve improved control level levels via symbiont effector during after re entry
disruption disrupted promotion promote restoration restore attenuation attenuate
homeostasis general involved type related associated other""".split())
# ultra-generic GO ancestors that don't count as a meaningful shared concept
GENERIC_GO = {
    "GO:0008150",  # biological_process
    "GO:0009987",  # cellular process
    "GO:0008152",  # metabolic process
    "GO:0065007",  # biological regulation
    "GO:0050789",  # regulation of biological process
    "GO:0050794",  # regulation of cellular process
    "GO:0050896",  # response to stimulus
    "GO:0032501",  # multicellular organismal process
    "GO:0032502",  # developmental process
    "GO:0023052",  # signaling
    "GO:0007154",  # cell communication
    "GO:0051716",  # cellular response to stimulus
    "GO:0048518", "GO:0048519",  # pos/neg regulation of biological process
    "GO:0048522", "GO:0048523",  # pos/neg regulation of cellular process
    "GO:0010468", "GO:0019222",  # regulation of gene expression / metabolic process
    "GO:0050793",  # regulation of developmental process
    "GO:0002376",  # immune system process  (kept? it's broad — count as generic)
}


def content_words(s):
    return {w for w in re.findall(r"[a-z0-9]+", (s or "").lower())
            if w not in STOP and len(w) > 2}


def cache_load(p):
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
    uniq = sorted(set(descs.values()))

    # 1) GO-only OLS search
    go_cache = cache_load(GO_CACHE)
    lock = threading.Lock()

    def fetch_go(q):
        if q in go_cache:
            return
        params = urllib.parse.urlencode({"q": q, "ontology": "go", "type": "class", "rows": 10})
        try:
            r = urllib.request.urlopen("https://www.ebi.ac.uk/ols4/api/search?" + params, timeout=15)
            docs = json.loads(r.read()).get("response", {}).get("docs", [])
        except Exception as e:
            docs = {"__error__": f"{type(e).__name__}:{e}"}
        with lock:
            go_cache[q] = docs

    todo = [q for q in uniq if q not in go_cache]
    print(f"GO search: {len(todo)} to fetch (warm {len(uniq)-len(todo)})")
    with ThreadPoolExecutor(max_workers=12) as ex:
        list(ex.map(fetch_go, todo))
    json.dump(go_cache, open(GO_CACHE, "w"))

    # 2) accept best GO hit by biological-noun overlap (direction-robust)
    assign = {}  # bio_id -> (go_id, label, overlap_words)
    for b, q in descs.items():
        qcw = content_words(q)
        docs = go_cache.get(q) or []
        if not isinstance(docs, list):
            continue
        best = None
        for x in docs:
            lab, oid = x.get("label"), x.get("obo_id")
            if not lab or not oid or not str(oid).startswith("GO:"):
                continue
            shared = qcw & content_words(lab)
            if shared and (best is None or len(shared) > len(best[2])):
                best = (oid, lab, shared)
        if best:
            assign[b] = best

    nmap = len(assign)
    smap = sum(1 for b in assign if reuse[b] <= 1)
    print(f"\n=== COVERAGE (direction-robust GO-noun overlap) ===")
    print(f"  mapped {nmap}/{len(bio)} ({100*nmap/len(bio):.0f}%)  | singletons {smap}/150 "
          f"({100*smap/150:.0f}%)")

    # 3) leaf-term collapse (exact GO id shared)
    leaf_members = defaultdict(list)
    for b, a in assign.items():
        leaf_members[a[0]].append(b)
    singles = [b for b in bio if reuse[b] <= 1]
    leaf_collapse = sum(1 for b in singles if b in assign and len(leaf_members[assign[b][0]]) >= 2)
    print(f"\n=== COLLAPSE @ exact GO leaf term ===")
    print(f"  singleton collapse: {leaf_collapse}/{len(singles)} ({100*leaf_collapse/len(singles):.0f}%)")
    print(f"  terms with >=2 nodes: {sum(1 for m in leaf_members.values() if len(m)>=2)}")

    # 4) roll-up collapse: fetch ancestors, share a non-generic ancestor
    anc_cache = cache_load(ANC_CACHE)
    term_ids = sorted({a[0] for a in assign.values()})

    def fetch_anc(gid):
        if gid in anc_cache:
            return
        iri = urllib.parse.quote(urllib.parse.quote(
            f"http://purl.obolibrary.org/obo/{gid.replace(':', '_')}", safe=""), safe="")
        url = f"https://www.ebi.ac.uk/ols4/api/ontologies/go/terms/{iri}/hierarchicalAncestors?size=200"
        try:
            r = urllib.request.urlopen(url, timeout=15)
            terms = json.loads(r.read()).get("_embedded", {}).get("terms", [])
            anc = [t.get("obo_id") for t in terms if t.get("obo_id")]
        except Exception as e:
            anc = {"__error__": f"{type(e).__name__}:{e}"}
        with lock:
            anc_cache[gid] = anc

    todo = [g for g in term_ids if g not in anc_cache]
    print(f"\nGO ancestor fetch: {len(todo)} terms (warm {len(term_ids)-len(todo)})")
    with ThreadPoolExecutor(max_workers=12) as ex:
        list(ex.map(fetch_anc, todo))
    json.dump(anc_cache, open(ANC_CACHE, "w"))

    # each node's concept-set = its leaf + non-generic ancestors
    def concept_set(gid):
        anc = anc_cache.get(gid)
        anc = anc if isinstance(anc, list) else []
        s = {gid} | set(anc)
        return {t for t in s if t not in GENERIC_GO}

    node_concepts = {b: concept_set(a[0]) for b, a in assign.items()}
    # collapse: a singleton shares >=1 non-generic concept with another mapped node
    concept_to_nodes = defaultdict(set)
    for b, cs in node_concepts.items():
        for c in cs:
            concept_to_nodes[c].add(b)
    rollup_collapse = 0
    rollup_examples = []
    for b in singles:
        if b not in node_concepts:
            continue
        partners = set()
        for c in node_concepts[b]:
            partners |= concept_to_nodes[c]
        partners.discard(b)
        if partners:
            rollup_collapse += 1
            if len(rollup_examples) < 12:
                p = next(iter(partners))
                shared = node_concepts[b] & node_concepts[p]
                rollup_examples.append((descs[b], descs[p], shared))
    print(f"\n=== COLLAPSE @ coarse GO ancestor (the real B1 merge) ===")
    print(f"  singleton collapse: {rollup_collapse}/{len(singles)} ({100*rollup_collapse/len(singles):.0f}%)")
    print(f"  (of {smap} mapped singletons -> {100*rollup_collapse/max(1,smap):.0f}% of MAPPED singletons)")

    # 5) reuse histogram if we re-id biology by its coarse concept (preview Phase-3 lift)
    #    group mapped nodes by a representative coarse concept (most-shared ancestor)
    rep = {}
    for b, cs in node_concepts.items():
        if not cs:
            continue
        rep[b] = max(cs, key=lambda c: len(concept_to_nodes[c]))
    groups = defaultdict(list)
    for b, r in rep.items():
        groups[r].append(b)
    sizes = sorted((len(m) for m in groups.values()), reverse=True)
    print(f"\n=== PREVIEW: biology nodes re-grouped by coarse GO concept ===")
    print(f"  {nmap} mapped nodes -> {len(groups)} coarse groups; "
          f"group sizes: {sizes[:12]}{' ...' if len(sizes)>12 else ''}")
    print(f"  groups pooling >=8 distinct nodes: {sum(1 for s in sizes if s>=8)}")

    print("\n=== sample coarse-ancestor merges (singleton <-> partner, shared concept) ===")
    for a, b, shared in rollup_examples:
        sj = list(shared)[:3]
        print(f"   '{a}'  ~  '{b}'   [{', '.join(sj)}]")


if __name__ == "__main__":
    main()
