"""Materialize the manifold-2 belief field on a snapshot (post-hoc), PRIVATE.

For each ``mechanism_affects`` and ``responds_differently`` edge, this replays
the edge's *existing* evidence records — the same ``(n_eff, p_obs)`` the scalar
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


# edge_type -> (chain src attr, chain tgt attr, src desc-key, tgt desc-key, tgt special).
# RESPONDS_DIFFERENTLY's target is the indication node name (no A.0b description).
EDGE_SPECS = {
    EdgeType.MECHANISM_AFFECTS: (
        "mechanism_id", "biology_id", "mechanism", "biology", None,
    ),
    EdgeType.RESPONDS_DIFFERENTLY: (
        "subgroup_population_id", "indication_id", "population", None, "indication_name",
    ),
}


def _chain_st_pairs(
    g, by_arm, et, src_id, tgt_id, tgt_name,
) -> dict[str, list[tuple[str, str]]]:
    """``nct -> [distinct (s_desc, t_desc) pairs]`` for the chains of each trial
    that traverse this edge (chain endpoints == edge endpoints), using per-arm
    A.0b descriptions. This is what lets two trials' evidence on the *same* edge
    land at different (s,t) — and a combo trial's distinct arms stay distinct."""
    s_attr, t_attr, s_key, t_key, t_special = EDGE_SPECS[et]
    out: dict[str, list[tuple[str, str]]] = {}
    for nct, ts in g.trial_subgraphs.items():
        seen: set[tuple[str, str]] = set()
        for ch in ts.chains:
            if getattr(ch, s_attr) != src_id or getattr(ch, t_attr) != tgt_id:
                continue
            d = by_arm.get((nct, ch.arm_id), {})
            s_desc = d.get(s_key, "")
            t_desc = tgt_name if t_special == "indication_name" else d.get(t_key, "")
            if not s_desc or not t_desc:
                continue
            pair = (s_desc, t_desc)
            if pair not in seen:
                seen.add(pair)
                out.setdefault(nct, []).append(pair)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Materialize manifold-2 belief field (A.3).")
    ap.add_argument("--graph", default="data/exports/multi_indication_52_annotated.json")
    ap.add_argument("--annotations", default="data/annotations")
    args = ap.parse_args()

    from src.graph.biolord_embeddings import embed_text

    g = GraphStore()
    g.import_snapshot(args.graph)
    by_arm = chain_descriptions_by_arm(args.annotations)  # per-(nct,arm) A.0b descs
    descs = _trial_descriptions(Path(args.annotations))    # trial-level fallback
    emb_cache: dict[str, list[float]] = {}

    def embed(text: str) -> list[float]:
        if text not in emb_cache:
            emb_cache[text] = embed_text(text)
        return emb_cache[text]

    edges_localized = 0
    anchors_total = 0
    sample = None
    for et, (_s_attr, _t_attr, src_key, tgt_key, tgt_special) in EDGE_SPECS.items():
        for e in g.get_edges_by_type(et):
            belief = EdgeBeliefState.model_validate(e["belief"])
            if not belief.evidence:
                continue
            tgt_name = ""
            if tgt_special == "indication_name":
                try:
                    tgt_name = g.get_node(e["target_id"]).get("name", "") or e["target_id"]
                except KeyError:
                    tgt_name = e["target_id"]
            pairs_by_nct = _chain_st_pairs(
                g, by_arm, et, e["source_id"], e["target_id"], tgt_name,
            )
            field = BeliefField()
            for ev in belief.evidence:
                pairs = pairs_by_nct.get(ev.source_id)
                if not pairs:  # no chain matched this edge → trial-level fallback
                    td = descs.get(ev.source_id, {})
                    s_desc = td.get(src_key, "")
                    t_desc = tgt_name if tgt_special == "indication_name" else td.get(tgt_key, "")
                    if not s_desc or not t_desc:
                        continue
                    pairs = [(s_desc, t_desc)]
                n_eff = effective_n_for_evidence(
                    ev.source_type, ev.quality_score, n_obs=ev.n_obs,
                    edge_type=et.value,
                )
                p_obs = p_obs_for_bucket(SupportBucket(ev.support))
                share = n_eff / len(pairs)  # split keeps the field marginal == scalar
                for s_desc, t_desc in pairs:
                    apply_virtual_evidence_local(
                        field, s=embed(s_desc), t=embed(t_desc), n_eff=share, p_obs=p_obs,
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
    out = root / (Path(args.graph).stem + "_belief_field.json")
    g.export_private_snapshot(str(out))
    print(f"localized {edges_localized} edges, {anchors_total} anchors total")
    print(f"private snapshot (with belief_field) -> {out}")

    if sample is not None:
        sid, tid, et, field = sample
        # Separability spot-check: query at each anchor's own (s,t).
        ps = [expected_p(field, a.s, a.t) for a in field.anchors]
        print(f"sample edge {sid} --{et.value}--> {tid}: {len(field.anchors)} anchors, "
              f"localized P range [{min(ps):.2f}, {max(ps):.2f}] "
              f"(spread {max(ps) - min(ps):.2f} = evidence kept separable)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
