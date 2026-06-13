"""Phase 1B — ontology collapse rate for the singleton biology layer.

Map every biology node description to its best controlled-vocabulary term
(GO biological-process primary; EFO/MONDO/HPO for disease/phenotype/physiology)
via the EBI OLS4 search API, gated by BioLORD semantic similarity between the
description and the matched term label. Then:

  coverage      = % of biology nodes that map confidently to a term
  collapse rate = % of singleton nodes whose assigned term is SHARED with >=1
                  other biology node (i.e. the ontology would merge them)

This is the curated-vocabulary analogue of the raw-cosine measure in
b1_phase1_cosine.py — it asks whether an EXTERNAL, reliability-tracking grouping
(not embedding clustering) collapses the singletons. OLS responses cache to disk.
"""
import json
import os
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict

SNAP = "data/exports/multi_500_annotated.json"
OLS_CACHE = "scratch/diagnostics/ols_cache.json"
ONTOLOGIES = "go,efo,mondo,hpo"


def normkey(s):
    return " ".join((s or "").lower().split())


def load_cache():
    if os.path.exists(OLS_CACHE):
        return json.load(open(OLS_CACHE))
    return {}


def ols_search(query, cache):
    if query in cache:
        return cache[query]
    params = urllib.parse.urlencode({
        "q": query, "ontology": ONTOLOGIES, "type": "class",
        "rows": 8, "fieldList": "obo_id,label,ontology_name,type",
    })
    url = "https://www.ebi.ac.uk/ols4/api/search?" + params
    try:
        r = urllib.request.urlopen(url, timeout=12)
        docs = json.loads(r.read()).get("response", {}).get("docs", [])
    except Exception as e:
        docs = {"__error__": f"{type(e).__name__}:{e}"}
    cache[query] = docs
    time.sleep(0.15)
    return docs


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

    # 1. OLS search for each unique description (parallel; warm cache reused)
    cache = load_cache()
    uniq = sorted(set(descs.values()))
    todo = [q for q in uniq if q not in cache]
    print(f"OLS: {len(uniq)} unique descriptions, {len(todo)} to fetch "
          f"(warm cache {len(uniq)-len(todo)}) across [{ONTOLOGIES}] ...")
    from concurrent.futures import ThreadPoolExecutor
    import threading
    lock = threading.Lock()

    def fetch(q):
        params = urllib.parse.urlencode({
            "q": q, "ontology": ONTOLOGIES, "type": "class", "rows": 8})
        url = "https://www.ebi.ac.uk/ols4/api/search?" + params
        try:
            r = urllib.request.urlopen(url, timeout=15)
            docs = json.loads(r.read()).get("response", {}).get("docs", [])
        except Exception as e:
            docs = {"__error__": f"{type(e).__name__}:{e}"}
        with lock:
            cache[q] = docs
        return q

    with ThreadPoolExecutor(max_workers=12) as ex:
        for i, _ in enumerate(ex.map(fetch, todo)):
            if (i + 1) % 50 == 0:
                with lock:
                    json.dump(cache, open(OLS_CACHE, "w"))
                print(f"  ...{i+1}/{len(todo)}")
    errors = sum(1 for v in cache.values() if isinstance(v, dict) and "__error__" in v)
    json.dump(cache, open(OLS_CACHE, "w"))
    if errors:
        print(f"  OLS errors on {errors}/{len(uniq)} queries (network). Sample:",
              next((v["__error__"] for v in cache.values() if isinstance(v, dict) and "__error__" in v), ""))

    # 2. BioLORD-gate: embed each desc + all candidate labels, pick max-cosine candidate
    from src.graph.biolord_embeddings import embed_texts, cosine_similarity
    cand_labels = set()
    for q in uniq:
        docs = cache.get(q) or []
        if isinstance(docs, list):
            for x in docs:
                if x.get("label"):
                    cand_labels.add(x["label"])
    all_text = sorted(set(uniq) | cand_labels)
    print(f"Embedding {len(all_text)} strings (desc + candidate labels) via BioLORD ...")
    vecs = dict(zip(all_text, embed_texts(all_text)))

    assign = {}   # bio_id -> (obo_id, label, ontology, cosine) or None
    for b, q in descs.items():
        docs = cache.get(q) or []
        best = None
        if isinstance(docs, list):
            for x in docs:
                lab, oid, ont = x.get("label"), x.get("obo_id"), x.get("ontology_name")
                if not lab or not oid:
                    continue
                c = cosine_similarity(vecs[q], vecs[lab])
                if best is None or c > best[3]:
                    best = (oid, lab, ont, c)
        assign[b] = best

    # 3. Coverage at semantic gates + ontology breakdown
    print("\n=== COVERAGE: biology nodes mapping to a term at semantic-cosine gate ===")
    for gate in (0.0, 0.55, 0.65, 0.75, 0.85):
        mapped = [b for b, a in assign.items() if a and a[3] >= gate]
        sing = [b for b in mapped if reuse[b] <= 1]
        print(f"  gate cos>={gate:.2f}: mapped {len(mapped)}/{len(bio)} "
              f"({100*len(mapped)/len(bio):.0f}%)  | singletons mapped {len(sing)}/150")

    # ontology source breakdown at a moderate gate
    GATE = 0.65
    onts = Counter()
    for b, a in assign.items():
        if a and a[3] >= GATE:
            onts[(a[2] or "?").upper()] += 1
    print(f"\n  ontology source of confident maps (gate {GATE}): " +
          ", ".join(f"{k}:{v}" for k, v in onts.most_common()))

    # 4. COLLAPSE RATE: distinct biology nodes sharing an assigned term
    for GATE in (0.55, 0.65, 0.75):
        term_members = defaultdict(list)
        for b, a in assign.items():
            if a and a[3] >= GATE:
                term_members[a[0]].append(b)
        singles = [b for b in bio if reuse[b] <= 1]
        sing_collapsed = sum(
            1 for b in singles
            if assign[b] and assign[b][3] >= GATE and len(term_members[assign[b][0]]) >= 2
        )
        sing_mapped = sum(1 for b in singles if assign[b] and assign[b][3] >= GATE)
        # how many ontology-terms have >=2 distinct biology nodes (real merges)
        merge_terms = {t: m for t, m in term_members.items() if len(m) >= 2}
        nodes_merged = sum(len(m) for m in merge_terms.values())
        n_terms = len(term_members)
        print(f"\n=== COLLAPSE @ gate {GATE} ===")
        print(f"  confident maps: {sum(len(m) for m in term_members.values())} nodes -> {n_terms} distinct terms")
        print(f"  terms with >=2 biology nodes (real merges): {len(merge_terms)}  pooling {nodes_merged} nodes")
        print(f"  SINGLETON collapse: {sing_collapsed}/{len(singles)} singletons "
              f"({100*sing_collapsed/len(singles):.0f}%) share a term with another node "
              f"[of {sing_mapped} mapped singletons]")

    # 5. sample: the biggest ontology merge-groups (gate 0.65) — paraphrase or collapse?
    GATE = 0.65
    term_members = defaultdict(list)
    term_label = {}
    for b, a in assign.items():
        if a and a[3] >= GATE:
            term_members[a[0]].append(b)
            term_label[a[0]] = (a[1], a[2])
    print(f"\n=== TOP ONTOLOGY MERGE-GROUPS (gate {GATE}) — what gets pooled ===")
    for t, m in sorted(term_members.items(), key=lambda kv: -len(kv[1]))[:14]:
        if len(m) < 2:
            continue
        lab, ont = term_label[t]
        members = [descs[b] for b in m]
        print(f"  {t} [{ont}] '{lab}'  <-  {len(m)} nodes: " +
              "; ".join(f"'{x}'" for x in members[:5]) + (" ..." if len(members) > 5 else ""))


if __name__ == "__main__":
    main()
