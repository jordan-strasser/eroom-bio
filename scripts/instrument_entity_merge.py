"""TASK 5 — generalized ENTITY-merge verification (the SapBERT-tier gate).

`instrument_mechanism_merge.py` checks DESCRIPTION nodes (chain-descriptions on a
node, BioLORD). This is its ENTITY-node counterpart: it audits the 4a/AE SapBERT
tiers (Indication / Population / Endpoint / AdverseEvent — plus Compound / Target
for reference) that node_merge applies as an entity-linker on the node NAME. Those
tiers were added config-only and UNVERIFIED; this answers the owner's question —
"is SapBERT working for indication/population/endpoint/AE?" — with measured
under/over-merge rates, BEFORE we trust them.

Two failure modes, both via SapBERT name cosine (`cambridgeltl/SapBERT-from-
PubMedBERT-fulltext`, the same model + threshold the merge uses):

  UNDER-MERGE — two SEPARATE nodes of one type whose canonical NAMES are SapBERT
    synonyms (cosine ≥ --under-threshold, default = the merge threshold 0.80) ⇒
    they SHOULD have collapsed but didn't ("NSCLC" vs "non-small cell lung cancer").
    Sibling diseases are a known false-positive (breast vs ovarian ≈ 0.65 sits
    BELOW 0.80, so a 0.80 flag is usually a real synonym) — the cosine is printed
    so borderline (0.80–0.85) vs decisive (>0.92) is visible.

  OVER-MERGE — a node whose ``metadata.merged_from`` (scope-stripped) spans ≥2
    DISTINCT base entities with LOW SapBERT cosine ⇒ the merge fused unlike things
    (a MedDRA term swallowing an unrelated PT; breast+ovarian into one node).
    Pure ``id#NCT`` scope-variants of ONE base id are the expected Tier-1 id-merge
    and are NOT flagged. Opaque ids (ENSG/R-HSA/CHEBI/bio:) that differ are reported
    separately as "distinct-id merges" (can't embed an id — human inspect).

Read-only. Run on a 4a+AE build:
  python -m scripts.instrument_entity_merge --graph data/exports/phaseb_n50b_annotated.json
"""
from __future__ import annotations

import argparse
import itertools
import re
import statistics as stx
from collections import defaultdict

from src.graph import sapbert_embeddings as SB
from src.graph.node_merge import MergeConfig
from src.graph.store import GraphStore

# Entity node types the SapBERT entity-linker tier governs (4a/AE) + the id-merged
# reference types. Mechanism/Biology are DESCRIPTION nodes → instrument_mechanism_merge.
ENTITY_TYPES = [
    "IndicationNode",
    "PopulationNode",
    "EndpointNode",
    "AdverseEventNode",
    "CompoundNode",
    "InterventionNode",
    "TargetNode",
]

# A merged_from base token that is an opaque external id (not embeddable as a name).
_OPAQUE_ID = re.compile(r"^(ENSG\d|ENST\d|R-HSA-|R-GGP-|GO:|CHEBI:|CHEMBL\d|bio:|mech:)", re.I)


def _strip_scope(token: str) -> str:
    """Drop the ``#NCT...`` trial-scope suffix node_merge stamps on pre-merge ids."""
    return token.split("#", 1)[0]


def _slug_to_text(token: str) -> str:
    """A slug id ('non_small_cell_lung_cancer', 'line_first__severity_moderate')
    → readable text for embedding. Axis-compound population slugs ('a__b') → 'a b'."""
    return token.replace("__", " ").replace("_", " ").strip()


def _surface_forms(name: str, merged_from: list[str]) -> tuple[set[str], set[str]]:
    """Distinct (name_like, opaque_id) surface forms for an over-merge check.

    name_like: embeddable text — the canonical name + any non-opaque, non-scope
      base token rendered to text. opaque_id: differing external ids (ENSG/R-HSA…)
      that merged — reported, not embedded.
    """
    name_like: set[str] = set()
    if name and name.strip():
        name_like.add(name.strip())
    opaque: set[str] = set()
    for raw in merged_from or []:
        base = _strip_scope(str(raw))
        if not base:
            continue
        if _OPAQUE_ID.match(base):
            opaque.add(base)
        else:
            name_like.add(_slug_to_text(base))
    return name_like, opaque


def audit_type(store: GraphStore, node_type: str, under_thr: float, over_thr: float):
    nodes = store.get_nodes_by_type(node_type)
    # (id, canonical name, merged_from list)
    rows = []
    for n in nodes:
        nid = n["id"]
        name = (n.get("name") or "").strip()
        md = n.get("metadata") or {}
        mf = md.get("merged_from") or n.get("aliases") or []
        if not isinstance(mf, list):
            mf = []
        rows.append((nid, name, mf))

    # ── embed every distinct surface string once (canonical names + over-merge
    #    name_like forms), batched + cached ──
    texts: set[str] = set()
    for _nid, name, mf in rows:
        nl, _op = _surface_forms(name, mf)
        texts.update(nl)
    texts_l = sorted(t for t in texts if t)
    emb = dict(zip(texts_l, SB.embed_compound_names(texts_l))) if texts_l else {}

    def cos(x: str, y: str):
        if x not in emb or y not in emb or x == y:
            return None
        return SB.cosine_similarity(emb[x], emb[y])

    # ── UNDER-MERGE: separate nodes with synonym-level name cosine ──
    named = [(nid, name) for nid, name, _ in rows if name]
    under = []
    for (ia, na), (ib, nb) in itertools.combinations(named, 2):
        if na == nb:
            continue  # identical surface but separate ids: a pure id miss, list it
        c = cos(na, nb)
        if c is not None and c >= under_thr:
            under.append((c, na, nb))
    # identical-name-but-separate-node (definite id-merge miss)
    name_to_ids = defaultdict(set)
    for nid, name, _ in rows:
        if name:
            name_to_ids[name].add(nid)
    dup_names = {nm: ids for nm, ids in name_to_ids.items() if len(ids) > 1}
    under.sort(reverse=True)

    # ── OVER-MERGE: a node fusing ≥2 distinct low-cosine surface forms ──
    over = []
    distinct_id_merges = []
    for nid, name, mf in rows:
        nl, opaque = _surface_forms(name, mf)
        nl_l = sorted(nl)
        if len(nl_l) >= 2:
            coss = [c for c in (cos(x, y) for x, y in itertools.combinations(nl_l, 2))
                    if c is not None]
            if coss and min(coss) < over_thr:
                over.append((min(coss), name, nl_l))
        if len(opaque) >= 2:
            distinct_id_merges.append((name, sorted(opaque)))
    over.sort()

    return {
        "n": len(nodes),
        "n_named": len(named),
        "under": under,
        "dup_names": dup_names,
        "over": over,
        "distinct_id_merges": distinct_id_merges,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--under-threshold", type=float,
                    default=MergeConfig().sapbert_threshold,  # 0.80 — the merge threshold
                    help="cosine ≥ this between two SEPARATE same-type nodes ⇒ under-merge")
    ap.add_argument("--over-threshold", type=float, default=0.50,
                    help="min intra-node cosine < this across merged surface forms ⇒ over-merge")
    ap.add_argument("--types", nargs="*", default=ENTITY_TYPES)
    a = ap.parse_args()

    store = GraphStore()
    store.import_snapshot(a.graph)
    print(f"=== ENTITY-merge verification: {a.graph} ===")
    print(f"SapBERT name cosine · under≥{a.under_threshold} · over<{a.over_threshold}\n")

    summary = []
    for nt in a.types:
        r = audit_type(store, nt, a.under_threshold, a.over_threshold)
        if r["n"] == 0:
            continue
        n_under = len(r["under"]) + sum(len(ids) - 1 for ids in r["dup_names"].values())
        n_over = len(r["over"])
        summary.append((nt, r["n"], n_under, n_over, len(r["distinct_id_merges"])))
        print(f"── {nt}  ({r['n']} nodes, {r['n_named']} named) ──")
        print(f"   UNDER-merge synonym pairs ≥{a.under_threshold}: {len(r['under'])}"
              f"   |   identical-name separate nodes: {len(r['dup_names'])}")
        for c, na, nb in r["under"][:8]:
            print(f"     cos {c:.3f}  {na[:34]!r}  ~  {nb[:34]!r}")
        for nm, ids in list(r["dup_names"].items())[:4]:
            print(f"     SAME NAME on {len(ids)} nodes: {nm[:40]!r}")
        print(f"   OVER-merge (fused low-cosine surface forms <{a.over_threshold}): {n_over}"
              f"   |   distinct-id merges: {len(r['distinct_id_merges'])}")
        for c, nm, forms in r["over"][:6]:
            print(f"     min-cos {c:.3f}  node {nm[:30]!r} ← {[f[:24] for f in forms][:4]}")
        for nm, ids in r["distinct_id_merges"][:4]:
            print(f"     {nm[:30]!r} ← {len(ids)} distinct ids {ids[:3]}")
        print()

    print("=== SUMMARY (per type) ===")
    print(f"{'type':18s} {'nodes':>6s} {'under':>6s} {'over':>6s} {'id-merge':>9s}")
    for nt, n, nu, no, nid in summary:
        print(f"{nt:18s} {n:6d} {nu:6d} {no:6d} {nid:9d}")


if __name__ == "__main__":
    main()
