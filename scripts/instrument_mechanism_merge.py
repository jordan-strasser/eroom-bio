"""Instrument the MechanismNode merge — quantify the #2 (mapping) + #4 (merge) noise.

Per MechanismNode: the Reactome label (node name/description) + the set of DISTINCT
chain ``mechanism_description``s that landed on it (chains where mechanism_id == node).
Then three rates via BioLORD cosine:
  OVER-MERGE   : a node carrying ≥2 semantically-DISTINCT chain-descriptions (low
                 intra-node cosine) ⇒ SapBERT-name geometry merged unlike mechanisms.
  LABEL DIVERGE: cosine(node Reactome label, chain-descriptions) low ⇒ the canonical
                 label disagrees with the trial's stated mechanism (pathway-ranker miss).
  UNDER-MERGE  : the same chain-description on >1 node ⇒ one mechanism split.

Read-only. Run:
  python -m scripts.instrument_mechanism_merge --graph data/exports/neff100_annotated.json
  # works for BiologyNode too: --node-type BiologyNode --id-attr biology_id --desc-attr biology_description
"""
from __future__ import annotations

import argparse
import itertools
import statistics as stx
from collections import defaultdict

from src.graph import biolord_embeddings as BE
from src.graph.store import GraphStore


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--node-type", default="MechanismNode")
    ap.add_argument("--id-attr", default="mechanism_id")
    ap.add_argument("--desc-attr", default="mechanism_description")
    ap.add_argument("--cos-threshold", type=float, default=0.5)
    a = ap.parse_args()

    store = GraphStore()
    store.import_snapshot(a.graph)

    def node_label(nid: str) -> tuple[str, str]:
        try:
            n = store.get_node(nid)
        except KeyError:
            return ("", "")
        return ((n.get("name") or ""), (n.get("description") or ""))

    # node_id -> list of chain descriptions that landed on it
    node_descs: dict[str, list[str]] = defaultdict(list)
    for _nct, ts in store.trial_subgraphs.items():
        for ch in ts.chains:
            nid = getattr(ch, a.id_attr, None)
            if not nid or nid == "UNKNOWN":
                continue
            d = (getattr(ch, a.desc_attr, "") or "").strip()
            if d:
                node_descs[nid].append(d)

    # embed every distinct description + node label, batched
    texts: set[str] = set()
    for nid, descs in node_descs.items():
        texts.update(descs)
        nm, nd = node_label(nid)
        texts.update(t for t in (nm, nd) if t)
    texts_l = sorted(t for t in texts if t)
    emb = dict(zip(texts_l, BE.embed_texts(texts_l)))

    def cos(x: str, y: str):
        if x not in emb or y not in emb:
            return None
        return BE.cosine_similarity(emb[x], emb[y])

    thr = a.cos_threshold
    multi = {nid: sorted(set(d)) for nid, d in node_descs.items() if len(set(d)) >= 2}
    print(f"=== {a.node_type} merge instrumentation: {a.graph} ===")
    print(f"{len(node_descs)} nodes with ≥1 chain; {len(multi)} with ≥2 distinct chain-descriptions\n")

    # ── OVER-MERGE ────────────────────────────────────────────────────
    over = []
    for nid, descs in multi.items():
        coss = [c for c in (cos(x, y) for x, y in itertools.combinations(descs, 2))
                if c is not None]
        if coss:
            over.append((nid, descs, min(coss), stx.mean(coss)))
    over.sort(key=lambda r: r[2])
    n_over = sum(1 for *_, mn, _ in over if mn < thr)
    print(f"OVER-MERGE: {n_over}/{len(multi)} multi-desc nodes have min intra-node "
          f"cosine < {thr} (distinct mechanisms collapsed onto one node)")
    for nid, descs, mn, mean in over[:10]:
        nm, _ = node_label(nid)
        print(f"  {nid[:22]:22s} '{nm[:26]:26s}' ({len(descs)}d min-cos {mn:.2f}): {descs[:4]}")

    # ── LABEL DIVERGENCE ──────────────────────────────────────────────
    div = []
    for nid, descs in node_descs.items():
        nm, nd = node_label(nid)
        label = nd or nm
        if not label:
            continue
        coss = [c for c in (cos(label, d) for d in set(descs)) if c is not None]
        if coss:
            div.append((nid, nm, stx.mean(coss)))
    div.sort(key=lambda r: r[2])
    n_div = sum(1 for *_, m in div if m < thr)
    print(f"\nLABEL DIVERGENCE: {n_div}/{len(div)} nodes have mean cosine(Reactome label, "
          f"chain-descs) < {thr}; overall mean {stx.mean([m for *_, m in div]):.3f}")
    for nid, nm, m in div[:10]:
        print(f"  '{nm[:26]:26s}' cos {m:.2f}  vs chains: {sorted(set(node_descs[nid]))[:3]}")

    # ── UNDER-MERGE ───────────────────────────────────────────────────
    desc_to_nodes: dict[str, set[str]] = defaultdict(set)
    for nid, descs in node_descs.items():
        for d in set(descs):
            desc_to_nodes[d].add(nid)
    split = {d: nids for d, nids in desc_to_nodes.items() if len(nids) > 1}
    print(f"\nUNDER-MERGE: {len(split)} distinct chain-descriptions appear on >1 node "
          f"(same mechanism split across nodes)")
    for d, nids in sorted(split.items(), key=lambda x: -len(x[1]))[:10]:
        labels = [node_label(n)[0][:18] for n in list(nids)[:3]]
        print(f"  '{d[:38]:38s}' on {len(nids)} nodes: {labels}")


if __name__ == "__main__":
    main()
