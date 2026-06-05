"""Materialize the manifold-2 belief field on a snapshot (post-hoc), PRIVATE.

For each backbone edge (all seven types in ``EDGE_SPECS``), this replays the
edge's *existing* evidence records — the same ``(n_eff, p_obs)`` the scalar
update used — but localized at ``(s, t) = BioLORD(per-chain A.0b descriptions)``.

**Per-chain localization (Finding-2 fix).** An evidence record carries only its
NCT (``source_id``), not an arm. So each record is placed at the descriptions of
the chain(s) in that trial that actually *traverse this edge* — matched by
``chain.<src_attr>/<tgt_attr> == edge.(source, target)`` against
``g.trial_subgraphs[nct].chains``, with per-arm descriptions from
``chain_descriptions_by_arm``. A trial with K distinct chain (s,t) pairs on the
edge splits the record's ``n_eff`` evenly across them, so the field's marginal
still tracks the public scalar while distinct chains stay separable. (Previously
a trial collapsed to one first-non-empty (s,t) — which could even blend two arms'
descriptions.) Trials with no per-chain description fall back to the trial-level
first-non-empty pair. Writes a private snapshot (carrying ``belief_field``) under
``EROOM_PRIVATE_ROOT`` (point it at ``eroom-enterprise/artifacts``).

This is a post-hoc materialization (no live-pipeline change). Auto-populating
the field inside the attributor is the deferred productionization; this proves
and stores the capability on the real graph now.

Usage:
    EROOM_PRIVATE_ROOT=/path/to/eroom-enterprise/artifacts \
      python -m scripts.materialize_belief_field \
      --graph data/exports/multi_indication_52_annotated.json
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from src.boundary import private_root
from src.graph.box_embeddings import chain_descriptions_by_arm
from src.graph.models import EdgeBeliefState, EdgeType
from src.graph.store import GraphStore
from src.inference.belief_field import (
    BeliefField,
    apply_virtual_evidence_local,
    expected_p,
)
from src.inference.beliefs import (
    SupportBucket,
    effective_n_for_evidence,
    p_obs_for_bucket,
)


def _trial_descriptions(ann_dir: Path) -> dict[str, dict[str, str]]:
    """NCT id → representative per-trial A.0b descriptions (first non-empty)."""
    out: dict[str, dict[str, str]] = {}
    for path in sorted(glob.glob(str(ann_dir / "*_extraction.json"))):
        try:
            d = json.loads(Path(path).read_text())
        except json.JSONDecodeError:
            continue
        nct = d.get("nct_id")
        if not nct:
            continue
        mech = bio = pop = ""
        for cr in d.get("results_by_chain", []) or []:
            mech = mech or (cr.get("mechanism_description") or "").strip()
            bio = bio or (cr.get("biology_description") or "").strip()
            pop = pop or (cr.get("population_description") or "").strip()
        out[nct] = {"mechanism": mech, "biology": bio, "population": pop}
    return out


# (s,t) on EVERY backbone edge. Per edge: (chain src attr, chain tgt attr,
# src desc-source, tgt desc-source). A desc-source of "mechanism"/"biology"/
# "population" is a per-arm A.0b key (varies per trial-arm → localizes); "node"
# uses the endpoint's own v2 description (fixed per node → that edge's field
# collapses toward the scalar). Now that every node is described (v2), every
# edge is a queryable surface — incl. the decisive modulates_via.
EDGE_SPECS = {
    EdgeType.AFFECTS: ("compound_id", "target_id", "node", "node"),
    EdgeType.MODULATES_VIA: ("target_id", "mechanism_id", "node", "mechanism"),
    EdgeType.MECHANISM_AFFECTS: ("mechanism_id", "biology_id", "mechanism", "biology"),
    EdgeType.BIOLOGY_DRIVES: ("biology_id", "indication_id", "biology", "node"),
    EdgeType.REFLECTS_BIOLOGY: ("biology_id", "endpoint_id", "biology", "node"),
    EdgeType.ENDPOINT_CAPTURES: ("endpoint_id", "indication_id", "node", "node"),
    EdgeType.RESPONDS_DIFFERENTLY: ("subgroup_population_id", "indication_id", "population", "node"),
}
_PER_ARM = {"mechanism", "biology", "population"}


def _node_desc(g, node_id: str) -> str:
    try:
        nd = g.get_node(node_id)
    except KeyError:
        return ""
    return (nd.get("description") or nd.get("name") or "").strip()


def _chain_st_pairs(
    g, by_arm, et, src_id, tgt_id,
) -> dict[str, list[tuple[str, str]]]:
    """``nct -> [distinct (s_desc, t_desc) pairs]`` for the chains of each trial
    that traverse this edge. Per-arm endpoints (mechanism/biology/population) use
    the arm's A.0b description (so two trials' evidence on the same edge stays
    separable); fixed endpoints use the node's own description."""
    s_attr, t_attr, s_src, t_src = EDGE_SPECS[et]
    s_node = _node_desc(g, src_id) if s_src not in _PER_ARM else None
    t_node = _node_desc(g, tgt_id) if t_src not in _PER_ARM else None
    out: dict[str, list[tuple[str, str]]] = {}
    for nct, ts in g.trial_subgraphs.items():
        seen: set[tuple[str, str]] = set()
        for ch in ts.chains:
            if getattr(ch, s_attr) != src_id or getattr(ch, t_attr) != tgt_id:
                continue
            d = by_arm.get((nct, ch.arm_id), {})
            s_desc = d.get(s_src, "") if s_src in _PER_ARM else s_node
            t_desc = d.get(t_src, "") if t_src in _PER_ARM else t_node
            if not s_desc or not t_desc:
                continue
            pair = (s_desc, t_desc)
            if pair not in seen:
                seen.add(pair)
                out.setdefault(nct, []).append(pair)
    return out


def materialize_field(
    graph_path: str, *, annotations_dir: str = "data/annotations",
) -> dict:
    """Replay every backbone edge's *existing* evidence localized at
    ``(s, t) = BioLORD(per-arm descriptions)`` and write the private belief-field
    snapshot under ``EROOM_PRIVATE_ROOT``. No new evidence — a strict refinement
    of the scalar marginal (see module docstring). Returns a summary dict
    ``{edges_localized, anchors_total, out, sample?}``.

    Reused by ``build_graph --assemble`` so the field is regenerated with every
    build (the field is post-hoc by construction — boundary strips it from the
    public snapshot — so without this it drifts stale relative to the graph)."""
    from src.graph.biolord_embeddings import embed_text, embed_texts

    g = GraphStore()
    g.import_snapshot(graph_path)
    by_arm = chain_descriptions_by_arm(annotations_dir)  # per-(nct,arm) A.0b descs
    emb_cache: dict[str, list[float]] = {}

    # Pre-embed every (s,t) description the localization below will request, in
    # ONE batch (one model.encode + one 45 MB cache round-trip) instead of one
    # embed_text per text — each of which re-read the whole cache, the ~3 h
    # field-materialization blocker. After this warm-up, embed() is pure local-
    # cache hits. The walk mirrors the apply loop's pair discovery exactly.
    wanted: set[str] = set()
    for et in EDGE_SPECS:
        for e in g.get_edges_by_type(et):
            belief = EdgeBeliefState.model_validate(e["belief"])
            if not belief.evidence:
                continue
            pairs_by_nct = _chain_st_pairs(g, by_arm, et, e["source_id"], e["target_id"])
            sd, td_ = _node_desc(g, e["source_id"]), _node_desc(g, e["target_id"])
            node_pair = (sd, td_) if sd and td_ else None
            for ev in belief.evidence:
                pairs = pairs_by_nct.get(ev.source_id) or ([node_pair] if node_pair else None)
                if not pairs:
                    continue
                for s_desc, t_desc in pairs:
                    wanted.add(s_desc)
                    wanted.add(t_desc)
    if wanted:
        ordered = sorted(wanted)
        emb_cache.update(zip(ordered, embed_texts(ordered)))

    def embed(text: str) -> list[float]:
        if text not in emb_cache:
            emb_cache[text] = embed_text(text)  # belt-and-suspenders (warm-up missed it)
        return emb_cache[text]

    edges_localized = 0
    anchors_total = 0
    sample = None
    for et in EDGE_SPECS:
        for e in g.get_edges_by_type(et):
            belief = EdgeBeliefState.model_validate(e["belief"])
            if not belief.evidence:
                continue
            pairs_by_nct = _chain_st_pairs(g, by_arm, et, e["source_id"], e["target_id"])
            # node-level (s,t) fallback for records whose nct has no matching chain
            sd, td_ = _node_desc(g, e["source_id"]), _node_desc(g, e["target_id"])
            node_pair = (sd, td_) if sd and td_ else None
            field = BeliefField()
            for ev in belief.evidence:
                pairs = pairs_by_nct.get(ev.source_id) or ([node_pair] if node_pair else None)
                if not pairs:
                    continue
                n_eff = effective_n_for_evidence(
                    ev.source_type, ev.quality_score, n_obs=ev.n_obs,
                    edge_type=et.value,
                )
                p_obs = p_obs_for_bucket(SupportBucket(ev.support))
                share = n_eff / len(pairs)  # split keeps the field marginal == scalar
                for s_desc, t_desc in pairs:
                    apply_virtual_evidence_local(
                        field, s=embed(s_desc), t=embed(t_desc), n_eff=share,
                        p_obs=p_obs, nct=ev.source_id,  # provenance for field-LOO
                    )
            if field.anchors:
                # carry the scalar marginal so far-from-anchor queries fall back
                # to the pooled edge belief, not 0.5 (field = refinement of scalar)
                field.marginal_alpha = belief.alpha
                field.marginal_beta = belief.beta
                belief.belief_field = field.to_dict()
                # write back to the in-memory edge belief
                e_data = g._graph.get_edge_data(  # noqa: SLF001
                    e["source_id"], e["target_id"], key=et.value,
                )
                e_data["belief"] = belief.model_dump(mode="json")
                edges_localized += 1
                anchors_total += len(field.anchors)
                if sample is None and len(field.anchors) >= 2:
                    sample = (e["source_id"], e["target_id"], et, field)

    root = private_root(create=True)
    out = root / (Path(graph_path).stem + "_belief_field.json")
    g.export_private_snapshot(str(out))
    result: dict = {
        "edges_localized": edges_localized,
        "anchors_total": anchors_total,
        "out": str(out),
    }
    if sample is not None:
        sid, tid, et, field = sample
        # Separability spot-check: query at each anchor's own (s,t).
        ps = [expected_p(field, a.s, a.t) for a in field.anchors]
        result["sample"] = {
            "src": sid, "tgt": tid, "etype": et.value,
            "anchors": len(field.anchors), "p_min": min(ps), "p_max": max(ps),
        }
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="Materialize manifold-2 belief field (A.3).")
    ap.add_argument("--graph", default="data/exports/multi_indication_52_annotated.json")
    ap.add_argument("--annotations", default="data/annotations")
    args = ap.parse_args()

    r = materialize_field(args.graph, annotations_dir=args.annotations)
    print(f"localized {r['edges_localized']} edges, {r['anchors_total']} anchors total")
    print(f"private snapshot (with belief_field) -> {r['out']}")
    s = r.get("sample")
    if s is not None:
        print(f"sample edge {s['src']} --{s['etype']}--> {s['tgt']}: {s['anchors']} anchors, "
              f"localized P range [{s['p_min']:.2f}, {s['p_max']:.2f}] "
              f"(spread {s['p_max'] - s['p_min']:.2f} = evidence kept separable)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
