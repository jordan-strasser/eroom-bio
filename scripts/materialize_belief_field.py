"""Materialize the manifold-2 belief field on a snapshot (post-hoc), PRIVATE.

For each ``mechanism_affects`` and ``responds_differently`` edge, this replays
the edge's *existing* evidence records — the same ``(n_eff, p_obs)`` the scalar
update used — but localized at ``(s, t) = BioLORD(per-trial A.0b descriptions)``.
So the field's marginal tracks the public scalar, while distinct trials' evidence
stays separable. Writes a private snapshot (carrying ``belief_field``) under
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


def main() -> int:
    ap = argparse.ArgumentParser(description="Materialize manifold-2 belief field (A.3).")
    ap.add_argument("--graph", default="data/exports/multi_indication_52_annotated.json")
    ap.add_argument("--annotations", default="data/annotations")
    args = ap.parse_args()

    from src.graph.biolord_embeddings import embed_text

    g = GraphStore()
    g.import_snapshot(args.graph)
    descs = _trial_descriptions(Path(args.annotations))
    emb_cache: dict[str, list[float]] = {}

    def embed(text: str) -> list[float]:
        if text not in emb_cache:
            emb_cache[text] = embed_text(text)
        return emb_cache[text]

    # (edge_type, source-desc-key, target-desc-key). responds_differently's
    # target is the indication node name (indications have no A.0b description).
    edge_specs = {
        EdgeType.MECHANISM_AFFECTS: ("mechanism", "biology", None),
        EdgeType.RESPONDS_DIFFERENTLY: ("population", None, "indication_name"),
    }

    edges_localized = 0
    anchors_total = 0
    sample = None
    for et, (src_key, tgt_key, tgt_special) in edge_specs.items():
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
            field = BeliefField()
            for ev in belief.evidence:
                td = descs.get(ev.source_id, {})
                s_desc = td.get(src_key, "")
                t_desc = td.get(tgt_key, "") if tgt_key else tgt_name
                if not s_desc or not t_desc:
                    continue
                n_eff = effective_n_for_evidence(
                    ev.source_type, ev.quality_score, n_obs=ev.n_obs,
                    edge_type=et.value,
                )
                p_obs = p_obs_for_bucket(SupportBucket(ev.support))
                apply_virtual_evidence_local(
                    field, s=embed(s_desc), t=embed(t_desc), n_eff=n_eff, p_obs=p_obs,
                )
            if field.anchors:
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
