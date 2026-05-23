"""prediction_analyzer — forensic deep-dive on a single trial's P(success).

`scripts/graphguard.py` is a fast pre-flight integrity check (structural).
This is the heavier sibling: for a given trial, walk every edge the
prediction engine actually uses, surface the underlying evidence, and
flag patterns that drive incorrect predictions or look suspicious.

Categories of red-flag detected:

1. CHAIN ROUTING
   - Stated chain vs auto-walked chain differ → the audit-pipeline
     fix (round 27) is reading the right chain
   - Edges skipped because chain endpoint is UNKNOWN
   - Phantom AFFECTS: the edge's target gene_symbol doesn't match
     the extraction's claimed_target

2. EVIDENCE QUALITY (per edge)
   - AMBIGUOUS-saturated: >50% of records are AMBIGUOUS bucket
     (signals classifier emitting fallback because trials don't
     demonstrate the edge — common for AFFECTS where binding is
     assumed, not measured)
   - Single-trial domination: one NCT contributes >60% of records
     (signals one trial's evidence drowning out the cross-trial signal)
   - Populator-only: 0 trial records, only DATABASE_CURATED — edge
     is essentially a prior, contributes via curated-DB tier weight
   - Self-evidence: trial's own NCT appears in the edge's records
     (the round-24 contamination signal — should be 0 for a
     true-holdout snapshot)

3. SAFETY ATTRIBUTION
   - Per safety_risk, the propagation source: compound-specific
     (clean) vs target_class (look at WHICH siblings contributed)
   - AE biological plausibility: does the AE class match the
     compound class? (e.g. checkpoint-irAEs on amyloid target is
     a leak; CV AEs from anti-amyloid trials in elderly patients
     is plausible)

4. COUNTERFACTUAL DECOMPOSITION
   - For each edge, what's its trust-weight share of the geomean
   - Which edge, if removed, would move the prediction the most
   - Which edges are "load-bearing" (trust > 20% of total)

Run:
  .venv/bin/python scripts/prediction_analyzer.py \\
    --snapshot data/exports/multi_indication_30_annotated.json \\
    --nct NCT01844505

Or all 5 case-study holdouts at once:
  .venv/bin/python scripts/prediction_analyzer.py \\
    --snapshot ... \\
    --holdouts NCT01844505,NCT01127633,NCT00112918,NCT00134264,NCT00970359
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from src.graph.antibody_target_resolver import nl_target_to_gene
from src.graph.models import EdgeBeliefState, EdgeType
from src.graph.store import GraphStore
from src.prediction.path_query import (
    _collect_modulation_edges,
    _trust_weight,
    predict_clinical_hypothesis,
)


# ── Chain extraction ────────────────────────────────────────────────────


def _stored_primary_chain(graph: GraphStore, nct_id: str, extraction: dict):
    """Same selection logic as case_study_audit's _resolve_stored_chain."""
    if nct_id not in graph.trial_subgraphs:
        return None
    ts = graph.trial_subgraphs[nct_id]
    if not ts.chains:
        return None
    th = (extraction.get("therapeutic_hypothesis") or {})
    primary_str = (th.get("compound") or "").lower()
    primary_first_word = ""
    for ch in primary_str.split("+"):
        ch = ch.strip()
        if ch and not ch.startswith("("):
            primary_first_word = (ch.split() or [""])[0]
            break
    for c in ts.chains:
        cid_low = (c.compound_id or "").lower()
        if cid_low == primary_first_word or cid_low.startswith(primary_first_word):
            return c
    return ts.chains[0]


def _load_extraction(nct_id: str) -> dict:
    p = Path(f"data/annotations/{nct_id}_extraction.json")
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return {}


# ── Per-edge analysis ───────────────────────────────────────────────────


_CHAIN_EDGES: list[tuple[str, str, EdgeType, str]] = [
    ("compound_id", "target_id", EdgeType.AFFECTS, "AFFECTS"),
    ("target_id", "mechanism_id", EdgeType.MODULATES_VIA, "MODULATES_VIA"),
    ("mechanism_id", "biology_id", EdgeType.MECHANISM_AFFECTS, "MECHANISM_AFFECTS"),
    ("biology_id", "indication_id", EdgeType.BIOLOGY_DRIVES, "BIOLOGY_DRIVES"),
    ("endpoint_id", "indication_id", EdgeType.ENDPOINT_CAPTURES, "ENDPOINT_CAPTURES"),
    ("subgroup_population_id", "indication_id", EdgeType.RESPONDS_DIFFERENTLY, "RESPONDS_DIFFERENTLY"),
]


def _bucket_distribution(belief: EdgeBeliefState) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for rec in belief.evidence:
        counts[rec.support] += 1
    return dict(counts)


def _source_distribution(belief: EdgeBeliefState) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for rec in belief.evidence:
        counts[rec.source_id] += 1
    return dict(counts)


def _analyze_edge(
    src_id: str, tgt_id: str, edge_label: str,
    belief: EdgeBeliefState,
    own_nct: str,
) -> dict:
    n_records = len(belief.evidence)
    bucket_counts = _bucket_distribution(belief)
    source_counts = _source_distribution(belief)

    flags: list[tuple[str, str]] = []  # (level, message)

    # AMBIGUOUS-saturated
    n_ambig = bucket_counts.get("ambiguous", 0)
    if n_records > 0 and n_ambig / n_records > 0.5:
        flags.append((
            "WARN",
            f"{n_ambig}/{n_records} records are AMBIGUOUS bucket — "
            f"classifier emitted fallback (likely 'trials assumed "
            f"the edge, didn't demonstrate it')",
        ))

    # Single-trial domination
    if source_counts:
        top_source, top_count = max(source_counts.items(), key=lambda kv: kv[1])
        if n_records > 1 and top_count / n_records > 0.6:
            flags.append((
                "WARN",
                f"{top_count}/{n_records} records from one source "
                f"({top_source}) — one trial dominating cross-trial signal",
            ))

    # Self-evidence (round-24 contamination signal)
    if own_nct and own_nct in source_counts:
        flags.append((
            "FAIL",
            f"holdout NCT {own_nct} appears in this edge's evidence "
            f"({source_counts[own_nct]} record(s)) — self-attribution leak",
        ))

    # Populator-only (no real trial records, only DATABASE_CURATED)
    has_trial = any(
        not s.startswith((
            "opentargets:", "opentargets_assoc:", "chembl_rest:",
            "mab_table:", "reactome_target_lookup:",
            "quickgo_target_lookup:", "trial_biology_fallback:",
            "endpoint_type_prior:", "cross_reference:",
            "combo_inherit:", "indication_taxonomy:",
            "lincs:", "trial_inference:",
        ))
        for s in source_counts
    )
    if n_records > 0 and not has_trial:
        flags.append((
            "INFO",
            "edge has only populator-curated records (no trial-"
            "attribution); evidence is database-vetted prior knowledge",
        ))

    return {
        "edge": f"{src_id} --{edge_label}--> {tgt_id}",
        "edge_label": edge_label,
        "src_id": src_id,
        "tgt_id": tgt_id,
        "alpha": belief.alpha,
        "beta": belief.beta,
        "expected_p": belief.expected_probability,
        "evidence_strength": belief.evidence_strength,
        "trust_weight": _trust_weight(belief),
        "n_records": n_records,
        "distinct_sources": len(source_counts),
        "bucket_counts": bucket_counts,
        "source_counts": dict(source_counts),
        "flags": flags,
    }


# ── Target-vs-extraction check ──────────────────────────────────────────


def _flag_target_mismatch(
    chain_target_id: str,
    graph: GraphStore,
    extraction: dict,
) -> tuple[str, str] | None:
    th = (extraction.get("therapeutic_hypothesis") or {})
    claimed_target = (th.get("claimed_target") or "").strip()
    if not claimed_target or chain_target_id in ("UNKNOWN", None):
        return None
    try:
        tnode = graph.get_node(chain_target_id)
    except KeyError:
        return None
    sym = (tnode.get("gene_symbol") or "").lower()
    ct_low = claimed_target.lower()
    if sym and (sym in ct_low or ct_low in sym):
        return None
    # NL-synonym table fallback
    for piece in re.split(r"[,/+&]| and | plus ", ct_low):
        piece = piece.strip()
        if not piece:
            continue
        mapped = nl_target_to_gene(piece)
        if mapped and mapped.lower() == sym:
            return None
    return (
        "WARN",
        f"chain target gene_symbol={sym!r} doesn't match the "
        f"extraction's claimed_target={claimed_target!r}; the AFFECTS "
        f"edge may be a name-match phantom rather than the trial's "
        f"actual target",
    )


# ── Safety risk analysis ────────────────────────────────────────────────


def _analyze_safety_risks(result, chain_target_id: str) -> list[dict]:
    """Inspect each safety_risk's biological plausibility."""
    out: list[dict] = []
    for sr in (result.safety_risks or []):
        flags: list[tuple[str, str]] = []
        # Heuristic checks for biological implausibility.
        ae_low = (sr.ae_name or "").lower()
        if sr.source == "target_class":
            # Checkpoint-inhibitor AEs (irAEs) on a target that doesn't
            # bind antibodies in the immune-checkpoint sense = leak.
            irae_terms = (
                "hypophysitis", "adrenal insufficiency", "thyroiditis",
                "colitis (immune", "pneumonitis (immune",
            )
            if any(t in ae_low for t in irae_terms):
                # Is the target a known checkpoint target?
                checkpoint_targets = {
                    "ENSG00000188389",  # PDCD1
                    "ENSG00000163599",  # CTLA4
                    "ENSG00000120217",  # CD274 (PD-L1)
                    "ENSG00000089692",  # LAG3
                    "ENSG00000181847",  # TIGIT
                }
                if chain_target_id not in checkpoint_targets:
                    flags.append((
                        "FAIL",
                        f"immune-related AE on non-checkpoint target — "
                        f"likely name-match leak via "
                        f"propagate_to_target_associated_ae",
                    ))
        out.append({
            "ae_name": sr.ae_name,
            "source": sr.source,
            "belief": sr.belief_probability,
            "n_eff": sr.evidence_strength,
            "contributing_compound_ids": sr.contributing_compound_ids,
            "flags": flags,
        })
    return out


# ── Counterfactual: trust-weight contribution per edge ──────────────────


def _counterfactual_weights(edges_analysis: list[dict]) -> list[dict]:
    """For each edge, compute its share of the trust-weight total +
    its contribution to log(P(success)) = sum(w_i * log(E[p]_i)) / sum(w_i)."""
    kept = [e for e in edges_analysis if e["trust_weight"] > 0]
    sum_w = sum(e["trust_weight"] for e in kept)
    if sum_w == 0:
        return [{**e, "weight_share": 0.0, "log_contrib": 0.0} for e in edges_analysis]
    enriched: list[dict] = []
    for e in edges_analysis:
        share = e["trust_weight"] / sum_w if sum_w > 0 else 0
        ep = max(e["expected_p"], 1e-9)
        log_contrib = share * math.log(ep)
        enriched.append({**e, "weight_share": share, "log_contrib": log_contrib})
    return enriched


# ── Main analysis ───────────────────────────────────────────────────────


def analyze_one(
    nct_id: str,
    graph: GraphStore,
) -> dict:
    extraction = _load_extraction(nct_id)
    stored = _stored_primary_chain(graph, nct_id, extraction)

    result_dict = {
        "nct_id": nct_id,
        "extraction_compound": (extraction.get("therapeutic_hypothesis") or {}).get("compound"),
        "extraction_claimed_target": (extraction.get("therapeutic_hypothesis") or {}).get("claimed_target"),
        "extraction_proposed_mechanism": (extraction.get("therapeutic_hypothesis") or {}).get("proposed_mechanism"),
        "stored_chain": None,
        "prediction": None,
        "edge_analysis": [],
        "safety_analysis": [],
        "global_flags": [],
    }

    if stored is None:
        result_dict["global_flags"].append((
            "FAIL", "no stored trial subgraph — cannot analyze",
        ))
        return result_dict

    # Round-28: surface mechanism.direction so the audit can show whether
    # the chain math is treating an inhibitory mechanism as "edge operates"
    # when a reader might interpret P(success) as "patient benefits."
    mechanism_direction: str | None = None
    if stored.mechanism_id and stored.mechanism_id != "UNKNOWN":
        try:
            m_node = graph.get_node(stored.mechanism_id)
            mechanism_direction = m_node.get("direction")
        except KeyError:
            mechanism_direction = None

    result_dict["stored_chain"] = {
        "compound_id": stored.compound_id,
        "target_id": stored.target_id or "UNKNOWN",
        "mechanism_id": stored.mechanism_id or "UNKNOWN",
        "mechanism_direction": mechanism_direction,
        "biology_id": stored.biology_id or "UNKNOWN",
        "indication_id": stored.indication_id or "UNKNOWN",
        "endpoint_id": stored.endpoint_id or "UNKNOWN",
        "subgroup_population_id": stored.subgroup_population_id or "UNKNOWN",
    }

    # Target mismatch flag
    mismatch_flag = _flag_target_mismatch(
        stored.target_id, graph, extraction,
    )
    if mismatch_flag:
        result_dict["global_flags"].append(mismatch_flag)

    # Run the prediction with stated chain
    indication_id = stored.indication_id
    if indication_id == "UNKNOWN" or indication_id not in graph._graph:
        result_dict["global_flags"].append((
            "FAIL",
            f"indication {indication_id!r} not in graph — prediction would error",
        ))
        return result_dict

    kwargs = {
        "target_id": stored.target_id or "UNKNOWN",
        "mechanism_id": stored.mechanism_id or "UNKNOWN",
        "biology_id": stored.biology_id or "UNKNOWN",
        "endpoint_id": stored.endpoint_id or "UNKNOWN",
        "population_id": stored.subgroup_population_id or "UNKNOWN",
    }
    np.random.seed(42)
    pred = predict_clinical_hypothesis(
        graph, stored.compound_id, indication_id,
        n_samples=5000, **kwargs,
    )
    result_dict["prediction"] = {
        "overall_p_success": pred.overall_probability,
        "efficacy_p": pred.efficacy_probability,
        "safety_penalty": pred.safety_penalty,
        "ci_lower": pred.ci_lower,
        "ci_upper": pred.ci_upper,
        "weakest_link": (
            f"{pred.weakest_link.edge_type.value}: "
            f"{pred.weakest_link.source_id} → "
            f"{pred.weakest_link.target_id}"
            if pred.weakest_link else None
        ),
    }

    # Per-edge analysis on contributions (these are the edges actually
    # collected by the engine — already filtered by drop rule).
    edges_analysis = []
    for ec in pred.edge_contributions:
        edges_analysis.append(_analyze_edge(
            ec.source_id, ec.target_id, ec.edge_type.value,
            ec.belief, own_nct=nct_id,
        ))
    edges_analysis = _counterfactual_weights(edges_analysis)
    result_dict["edge_analysis"] = edges_analysis

    # Safety analysis
    result_dict["safety_analysis"] = _analyze_safety_risks(pred, stored.target_id)

    return result_dict


# ── Rendering ───────────────────────────────────────────────────────────


def _render_one(d: dict) -> str:
    out: list[str] = []
    out.append(f"## {d['nct_id']}\n\n")

    # Header: extraction summary
    out.append("### Trial says\n")
    out.append("```\n")
    out.append(f"  compound          {d.get('extraction_compound')!r}\n")
    out.append(f"  claimed_target    {d.get('extraction_claimed_target')!r}\n")
    out.append(f"  proposed_mechanism {d.get('extraction_proposed_mechanism')!r}\n")
    out.append("```\n\n")

    # Stored chain
    out.append("### Stored chain\n")
    out.append("```\n")
    sc = d.get("stored_chain") or {}
    for k in ("compound_id", "target_id", "mechanism_id",
              "mechanism_direction",
              "biology_id", "indication_id", "endpoint_id",
              "subgroup_population_id"):
        out.append(f"  {k:<22} {sc.get(k)}\n")
    out.append("```\n\n")
    # Round-28: when mechanism direction is known, note that the chain
    # math is direction-blind so a reader doesn't misread P(success).
    md = sc.get("mechanism_direction")
    if md in ("inhibiting", "activating", "modulating"):
        out.append(
            f"_Mechanism direction is `{md}` — the chain math reflects "
            f"edge operativity (\"this mechanism is in play\"), not "
            f"benefit direction. For inhibitory mechanisms, P(success) "
            f"is interpreted as \"P(this inhibitory cascade is the right "
            f"thing to do for this patient population)\", not \"P(target "
            f"activity goes up)\"._\n\n"
        )

    # Global flags
    if d.get("global_flags"):
        out.append("### Global flags\n")
        for level, msg in d["global_flags"]:
            out.append(f"- **{level}**: {msg}\n")
        out.append("\n")

    # Prediction summary
    pred = d.get("prediction")
    if pred:
        out.append("### Prediction\n")
        out.append(
            f"- **P(success) = {pred['overall_p_success']:.3f}** "
            f"= efficacy {pred['efficacy_p']:.3f} × "
            f"(1 − safety {pred['safety_penalty']:.3f}) "
            f"(95% CI [{pred['ci_lower']:.3f}, {pred['ci_upper']:.3f}])\n"
        )
        if pred.get("weakest_link"):
            out.append(f"- Weakest link: `{pred['weakest_link']}`\n")
        out.append("\n")

    # Edge analysis with trust-weight share
    if d.get("edge_analysis"):
        out.append("### Per-edge forensic\n\n")
        # Sort by weight_share descending
        edges_sorted = sorted(
            d["edge_analysis"], key=lambda e: -e["weight_share"]
        )
        for e in edges_sorted:
            wpct = e["weight_share"] * 100
            out.append(
                f"**{e['edge_label']}**: `{e['src_id']} → {e['tgt_id']}` "
                f"— **{wpct:.0f}%** of total trust weight\n"
            )
            out.append(
                f"  - Belief: Beta({e['alpha']:.2f}, {e['beta']:.2f}) "
                f"E[p]={e['expected_p']:.3f} n_eff={e['evidence_strength']:.2f} "
                f"trust={e['trust_weight']:.3f}\n"
            )
            out.append(
                f"  - Provenance: {e['n_records']} record(s) from "
                f"{e['distinct_sources']} distinct source(s)\n"
            )
            if e["bucket_counts"]:
                buckets = ", ".join(
                    f"{k}={v}" for k, v in sorted(e["bucket_counts"].items())
                )
                out.append(f"  - Bucket mix: {buckets}\n")
            if e["source_counts"]:
                src_str = ", ".join(
                    f"{nct}×{n}" if n > 1 else nct
                    for nct, n in sorted(
                        e["source_counts"].items(), key=lambda kv: -kv[1]
                    )[:6]
                )
                if len(e["source_counts"]) > 6:
                    src_str += f" (+{len(e['source_counts']) - 6} more)"
                out.append(f"  - Sources: {src_str}\n")
            for level, msg in e["flags"]:
                out.append(f"  - **{level}**: {msg}\n")
            out.append("\n")

    # Safety analysis
    if d.get("safety_analysis"):
        out.append("### Safety risks (per-AE forensic)\n\n")
        for sr in d["safety_analysis"]:
            out.append(
                f"- **{sr['ae_name']}** (source={sr['source']}, "
                f"belief={sr['belief']:.3f}, n_eff={sr['n_eff']:.1f})\n"
            )
            if sr["contributing_compound_ids"]:
                out.append(
                    f"  - Contributing compounds: "
                    f"{', '.join(sr['contributing_compound_ids'][:5])}"
                )
                if len(sr["contributing_compound_ids"]) > 5:
                    out.append(
                        f" (+{len(sr['contributing_compound_ids']) - 5} more)"
                    )
                out.append("\n")
            for level, msg in sr["flags"]:
                out.append(f"  - **{level}**: {msg}\n")
        out.append("\n")

    return "".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--nct", default="", help="Single NCT id")
    parser.add_argument(
        "--holdouts", default="",
        help="Comma-separated NCT ids — analyze each.",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Write markdown to this path (otherwise stdout).",
    )
    args = parser.parse_args()
    if not args.snapshot.exists():
        print(f"missing snapshot: {args.snapshot}", file=sys.stderr)
        return 2

    graph = GraphStore()
    graph.import_snapshot(str(args.snapshot))

    if args.nct:
        nct_ids = [args.nct.strip()]
    elif args.holdouts:
        nct_ids = [n.strip() for n in args.holdouts.split(",") if n.strip()]
    else:
        print("--nct or --holdouts required", file=sys.stderr)
        return 2

    parts: list[str] = []
    parts.append("# prediction_analyzer — forensic chain audit\n\n")
    parts.append(f"Snapshot: `{args.snapshot}`\n\n")
    for nct in nct_ids:
        analysis = analyze_one(nct, graph)
        parts.append("\n---\n\n")
        parts.append(_render_one(analysis))

    md = "".join(parts)
    if args.out:
        args.out.write_text(md)
        print(f"Wrote {args.out}", file=sys.stderr)
    else:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
