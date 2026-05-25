"""Phase A verification audit: per-stage failure checks (Part 1) + end-to-end
chain traces (Part 2). Writes a markdown report. No LLM (pure inspection).

See future_ideas/phase_a_verification_audit.md. Usage:
    python -m scripts.audit_chain \
        --graph data/exports/multi_indication_52_annotated.json \
        --out audit/phase_a_verification.md
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from collections import Counter
from pathlib import Path

from src.graph.models import EdgeType
from src.graph.store import GraphStore

UNKNOWN = "unknown"
ANN = Path("data/annotations")
BOX_PATH = Path(
    "/Users/jordanstrasser/Code/eroom-bio/eroom-enterprise/artifacts/manifold1_boxes.json"
)
FIELD_PATH = Path(
    "/Users/jordanstrasser/Code/eroom-bio/eroom-enterprise/artifacts/"
    "multi_indication_52_annotated_belief_field.json"
)

# Chain edges: (edge_type, source chain-attr, target chain-attr, src label, tgt label)
CHAIN_EDGES = [
    (EdgeType.AFFECTS, "compound_id", "target_id", "Compound", "Target"),
    (EdgeType.MODULATES_VIA, "target_id", "mechanism_id", "Target", "Mechanism"),
    (EdgeType.MECHANISM_AFFECTS, "mechanism_id", "biology_id", "Mechanism", "Biology"),
    (EdgeType.BIOLOGY_DRIVES, "biology_id", "indication_id", "Biology", "Indication"),
    (EdgeType.REFLECTS_BIOLOGY, "biology_id", "endpoint_id", "Biology", "Endpoint"),
    (EdgeType.ENDPOINT_CAPTURES, "endpoint_id", "indication_id", "Endpoint", "Indication"),
    (EdgeType.RESPONDS_DIFFERENTLY, "subgroup_population_id", "indication_id", "Population", "Indication"),
]


def embedded_text(node: dict) -> tuple[str, bool]:
    """The exact text A.1 embeds: description if present, else name (fallback)."""
    desc = (node.get("description") or "").strip()
    if desc:
        return desc, False
    return (node.get("name") or "").strip(), True


def trial_descs() -> dict[str, dict[str, str]]:
    """NCT -> representative (mechanism, biology, population) descriptions = the
    (s,t) text each trial contributes."""
    out: dict[str, dict[str, str]] = {}
    for p in glob.glob(str(ANN / "*_extraction.json")):
        try:
            d = json.loads(Path(p).read_text())
        except json.JSONDecodeError:
            continue
        nct = d.get("nct_id")
        if not nct:
            continue
        m = b = pop = ""
        for cr in d.get("results_by_chain", []) or []:
            m = m or (cr.get("mechanism_description") or "").strip()
            b = b or (cr.get("biology_description") or "").strip()
            pop = pop or (cr.get("population_description") or "").strip()
        out[nct] = {"mechanism": m, "biology": b, "population": pop}
    return out


# ── Part 1: per-stage checks ─────────────────────────────────────────────────


def part1(g: GraphStore, boxes: dict, field_edges: dict) -> list[str]:
    L = ["## Part 1 — per-stage failure-point checks", ""]

    # A.0 description coverage by type
    L.append("**A.0/A.0b — embedded text (description vs name-fallback):**")
    L.append("")
    L.append("| node type | total | has description | name-fallback |")
    L.append("|---|--:|--:|--:|")
    by_type: dict[str, list[int]] = {}
    for _, data in g._graph.nodes(data=True):  # noqa: SLF001
        nt = data.get("node_type", "?")
        has = bool((data.get("description") or "").strip())
        t = by_type.setdefault(nt, [0, 0])
        t[0] += 1
        t[1] += int(has)
    for nt in ["MechanismNode", "BiologyNode", "PopulationNode", "InterventionNode",
               "TargetNode", "EndpointNode", "IndicationNode"]:
        if nt in by_type:
            tot, has = by_type[nt]
            L.append(f"| {nt} | {tot} | {has} | {tot - has} |")
    L.append("")

    # A.1 caches
    bio_cache = Path("data/cache/biolord_embeddings.json")
    n_bio = len(json.loads(bio_cache.read_text())) if bio_cache.exists() else 0
    L.append(f"**A.1 — BioLORD cache:** {n_bio} cached embeddings "
             f"({'present' if n_bio else 'MISSING'}).")
    L.append("")

    # A.2 boxes + Reactome consistency
    bio_ids = [n["id"] for n in g.get_nodes_by_type("BiologyNode")]
    boxed = [b for b in bio_ids if b in boxes]
    L.append(f"**A.2 — boxes:** {len(boxed)}/{len(bio_ids)} biology nodes have a fitted box.")
    try:
        from src.graph.box_embeddings import relation
        from src.graph.reactome_hierarchy import fetch_reactome_ancestors
        viol = checked = 0
        for nid in boxed:
            if not nid.startswith("R-"):
                continue
            for anc in fetch_reactome_ancestors(nid):
                if anc in boxes:
                    checked += 1
                    rel = relation(boxes[anc], boxes[nid])  # ancestor vs descendant
                    if rel == "merge":  # should be parent_of, never merge
                        viol += 1
        L.append(f"  Reactome ancestor pairs checked: {checked}; "
                 f"wrongly box-relation=='merge': {viol} (want 0).")
    except Exception as e:
        L.append(f"  (Reactome consistency check skipped: {e})")
    L.append("")

    # A.4 merger off + sibling distinctness
    from src.graph.biology_merge import embedding_merge_enabled
    pd1, ctla4 = "R-HSA-389948", "R-HSA-389513"
    both = all(x in g._graph for x in (pd1, ctla4))  # noqa: SLF001
    L.append(f"**A.4 — merger:** embedding merge enabled = {embedding_merge_enabled()} "
             f"(want False). PD-1 ({pd1}) and CTLA4 ({ctla4}) both present as "
             f"distinct nodes = {both} (want True — siblings not pooled).")
    L.append("")

    # A.6 modulates_via bucket distribution
    bk = Counter()
    for u, v, key, data in g._graph.edges(data=True, keys=True):  # noqa: SLF001
        if key == EdgeType.MODULATES_VIA.value:
            for ev in (data.get("belief", {}).get("evidence", []) or []):
                bk[ev.get("support", "?")] += 1
    tot = sum(bk.values())
    amb = bk.get("ambiguous", 0)
    L.append(f"**A.6 — modulates_via buckets (graph edges):** {tot} records, "
             f"{amb} ambiguous = {round(100 * amb / max(tot, 1))}%. {dict(bk)}")
    L.append("")
    return L


# ── Part 2: chain trace ──────────────────────────────────────────────────────


def trace(g: GraphStore, compound: str, indication: str, boxes: dict,
          tdesc: dict, field_edges: dict) -> list[str]:
    # pick one representative chain (prefer a subgroup chain with a real outcome)
    chosen = None
    for ts in g.trial_subgraphs.values():
        for ch in ts.chains:
            if ch.compound_id == compound and ch.indication_id == indication:
                chosen = (ts, ch)
                if ch.outcome.value != "unknown":
                    break
        if chosen and chosen[1].outcome.value != "unknown":
            break
    if not chosen:
        return [f"### {compound} → {indication}", "_(no chain found)_", ""]
    ts, ch = chosen
    L = [f"### Chain: {compound} → {indication}  (trial {ts.trial_id}, arm {ch.arm_id})", ""]

    # Nodes
    L.append("**Nodes — what text is embedded:**")
    L.append("")
    L.append("| role | node id | embedded text | src | box |")
    L.append("|---|---|---|---|---|")
    for _, src_attr, _, src_label, _ in CHAIN_EDGES:
        pass
    roles = [("Compound", ch.compound_id), ("Target", ch.target_id),
             ("Mechanism", ch.mechanism_id), ("Biology", ch.biology_id),
             ("Endpoint", ch.endpoint_id), ("Indication", ch.indication_id),
             ("Population", ch.subgroup_population_id)]
    for label, nid in roles:
        if nid == UNKNOWN or nid not in g._graph:  # noqa: SLF001
            L.append(f"| {label} | `{nid}` | _(unresolved)_ | — | — |")
            continue
        node = g.get_node(nid)
        txt, fb = embedded_text(node)
        src = "NAME-fallback" if fb else "description"
        boxinfo = "yes" if nid in boxes else "—"
        L.append(f"| {label} | `{nid}` | {txt[:70]} | {src} | {boxinfo} |")
    L.append("")

    # Edges
    L.append("**Edges — scalar belief, contributing trials, and (s,t) text:**")
    L.append("")
    for et, sattr, tattr, slab, tlab in CHAIN_EDGES:
        s = getattr(ch, sattr)
        t = getattr(ch, tattr)
        if s == UNKNOWN or t == UNKNOWN:
            continue
        try:
            belief = g.get_edge_belief(s, t, et)
        except KeyError:
            continue
        ep = belief.alpha / (belief.alpha + belief.beta)
        conc = belief.alpha + belief.beta - 2.0
        ncts = Counter((ev.source_id, ev.support) for ev in belief.evidence)
        n_anchors = len(field_edges.get((s, t, et.value), []))
        L.append(f"- **{et.value}** ({slab}→{tlab}): "
                 f"Beta(α={belief.alpha:.1f}, β={belief.beta:.1f}) E[p]={ep:.2f} "
                 f"conc={conc:.1f} · {len(belief.evidence)} evidence from "
                 f"{len(set(n for n, _ in ncts))} trials · field anchors={n_anchors}")
        # show up to 4 contributing trials + their (s,t) text where applicable
        shown = 0
        for (nct, bucket), cnt in ncts.most_common():
            st = ""
            if et == EdgeType.MECHANISM_AFFECTS and nct in tdesc:
                st = f"  (s,t)=({tdesc[nct]['mechanism'][:34]!r} , {tdesc[nct]['biology'][:34]!r})"
            elif et == EdgeType.MODULATES_VIA and nct in tdesc:
                st = f"  t-desc={tdesc[nct]['mechanism'][:50]!r}"
            elif et == EdgeType.RESPONDS_DIFFERENTLY and nct in tdesc:
                st = f"  s-desc={tdesc[nct]['population'][:50]!r}"
            L.append(f"    - {nct} [{bucket}]{st}")
            shown += 1
            if shown >= 4:
                break
    L.append("")
    return L


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="data/exports/multi_indication_52_annotated.json")
    ap.add_argument("--out", default="audit/phase_a_verification.md")
    ap.add_argument("--chains", default="trametinib:melanoma,ezetimibe:hypercholesterolemia,rosuvastatin:hypercholesterolemia")
    args = ap.parse_args()

    g = GraphStore()
    g.import_snapshot(args.graph)

    boxes = {}
    if BOX_PATH.exists():
        from src.graph.box_embeddings import load_boxes
        boxes = load_boxes(BOX_PATH)

    # field anchors per (src, tgt, edge_type) from the private belief-field snapshot
    field_edges: dict = {}
    if FIELD_PATH.exists():
        fd = json.loads(FIELD_PATH.read_text())
        links = fd["graph"].get("links") or fd["graph"].get("edges") or []
        for e in links:
            bf = (e.get("belief") or {}).get("belief_field")
            if bf and bf.get("anchors"):
                field_edges[(e.get("source"), e.get("target"), e.get("key"))] = bf["anchors"]

    tdesc = trial_descs()

    out = ["# Phase A verification audit", "",
           f"Graph: `{args.graph}` — {g._graph.number_of_nodes()} nodes, "  # noqa: SLF001
           f"{g._graph.number_of_edges()} edges, {len(g.trial_subgraphs)} trial subgraphs.",
           f"Boxes: {len(boxes)} · belief-field edges (private): {len(field_edges)}.", ""]
    out += part1(g, boxes, field_edges)
    out += ["## Part 2 — end-to-end chain traces", ""]
    for pair in args.chains.split(","):
        comp, ind = pair.split(":")
        out += trace(g, comp.strip(), ind.strip(), boxes, tdesc, field_edges)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(out))
    print(f"wrote {args.out}  ({len(out)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
