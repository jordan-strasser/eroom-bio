"""GraphGuard — chain-integrity audit for graph snapshots.

The sole mission of this project is mapping trial hypotheses into the
graph. When that mapping breaks (dose-laden compound slugs, UNKNOWN
target resolution, phantom name-match AFFECTS edges, etc.), any
downstream P(success) result is noise. GraphGuard scans a snapshot
for those failure modes and emits a per-trial integrity report
BEFORE any prediction or audit is published.

It does NOT touch the graph; it's read-only. Run it as a gate:

  .venv/bin/python scripts/graphguard.py \\
    --snapshot data/exports/multi_indication_30_annotated.json

Exit codes:
  0  — all trials PASS or only have WARN-level issues
  1  — at least one trial has a FAIL-level issue (P0); don't publish
       audit numbers until these are resolved.

Each trial gets one of three statuses:
  PASS  — all chain layers grounded; no canonicalization regression
  WARN  — at least one chain edge skipped/missing or population
          defaulted to unselected; results legible but caveated
  FAIL  — canonicalization regression OR chain has < 2 traversable
          edges (prediction reduces to a single endpoint-prior).

Filterable to a single NCT via --nct, or to a list of expected holdouts
via --holdouts. Useful in a CI loop or before re-running case studies.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

from src.graph.antibody_target_resolver import (
    NL_TARGET_TO_GENE,
    nl_target_to_gene,
)
from src.graph.models import EdgeBeliefState, EdgeType
from src.graph.store import GraphStore

# Compound slug patterns that indicate the dose-strip fix didn't fire
# (or the snapshot pre-dates round 27). Matches the round-26 regression
# (`bevacizumab_7_5_mg_kg`) and similar.
_DOSE_LADEN_SLUG = re.compile(
    r"(_\d+_\d+_mg(_kg|_m_2|_m\^?2)?$"
    r"|_\d+_mg(_kg|_m_2|_m\^?2)?$"
    r"|_\d+_(iu|miu|mu|units?)$"
    r"|_q\d+h$"
    r"|_[bt]?id$|_qd$|_po$|_iv$|_sc$|_im$)",
    re.IGNORECASE,
)


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def _load_extraction(nct_id: str) -> dict | None:
    p = Path(f"data/annotations/{nct_id}_extraction.json")
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def _check_trial(
    nct_id: str,
    graph: GraphStore,
) -> dict:
    """Run all integrity checks for one trial. Return a result dict."""
    issues: list[tuple[str, str]] = []  # (level, message)

    try:
        ts = graph.get_trial_subgraph_by_id(nct_id)
    except KeyError:
        return {
            "nct_id": nct_id,
            "status": "FAIL",
            "issues": [("FAIL", "no trial_subgraph in snapshot")],
            "chain_summary": None,
        }

    # Load extraction for cross-check (what the trial says it tests)
    extraction = _load_extraction(nct_id) or {}
    th = (extraction.get("therapeutic_hypothesis") or {})
    claimed_target = (th.get("claimed_target") or "").strip()
    claimed_compound = (th.get("compound") or "").strip()

    if not ts.chains:
        issues.append(("FAIL", "trial_subgraph has no chains"))

    # Find the primary chain (matching the extraction's primary compound)
    primary_first_word = ""
    for ch_str in claimed_compound.split("+"):
        ch_str = ch_str.strip()
        if ch_str and not ch_str.startswith("("):
            primary_first_word = (ch_str.split() or [""])[0].lower()
            break
    primary_chain = None
    for c in ts.chains:
        cid_low = (c.compound_id or "").lower()
        if cid_low == primary_first_word or cid_low.startswith(primary_first_word):
            primary_chain = c
            break
    if primary_chain is None and ts.chains:
        primary_chain = ts.chains[0]
        issues.append((
            "WARN",
            f"primary chain didn't match extraction.therapeutic_hypothesis "
            f"compound={claimed_compound!r}; fell back to first chain",
        ))

    if primary_chain is None:
        return {
            "nct_id": nct_id,
            "status": "FAIL",
            "issues": issues,
            "chain_summary": None,
        }

    # ── P0: dose-laden compound slug ───────────────────────────────────
    if _DOSE_LADEN_SLUG.search(primary_chain.compound_id or ""):
        issues.append((
            "FAIL",
            f"compound_id {primary_chain.compound_id!r} contains a dose "
            f"suffix — canonicalization didn't strip it; round-27 fix "
            f"not applied to this snapshot",
        ))

    # ── P0: compound is drug-like but chembl_id is None ───────────────
    try:
        comp_node = graph.get_node(primary_chain.compound_id)
        if comp_node.get("chembl_id") is None and primary_chain.compound_id != "UNKNOWN":
            issues.append((
                "FAIL",
                f"compound {primary_chain.compound_id!r} has chembl_id=None; "
                f"OT/ChEMBL canonicalization did not resolve it",
            ))
    except KeyError:
        issues.append((
            "FAIL",
            f"compound {primary_chain.compound_id!r} is not a node in the graph",
        ))

    # ── P0: target=UNKNOWN when extraction stated a target ────────────
    if primary_chain.target_id == "UNKNOWN" and claimed_target:
        issues.append((
            "FAIL",
            f"chain.target_id=UNKNOWN but extraction.claimed_target="
            f"{claimed_target!r} — target resolution failed",
        ))

    # ── P1: AFFECTS edge target's gene_symbol matches extraction's claimed_target?
    if primary_chain.target_id and primary_chain.target_id != "UNKNOWN":
        try:
            tnode = graph.get_node(primary_chain.target_id)
            sym = (tnode.get("gene_symbol") or "").lower()
            ct_low = claimed_target.lower() if claimed_target else ""
            if claimed_target and sym and sym not in ct_low and ct_low not in sym:
                # Use the round-24 antibody/NL-target synonym table.
                # If any token of the claimed_target maps to the
                # chain's gene_symbol, the chain is consistent.
                consistent = False
                for piece in re.split(r"[,/+&]| and | plus ", ct_low):
                    piece = piece.strip()
                    if not piece:
                        continue
                    mapped = nl_target_to_gene(piece)
                    if mapped and mapped.lower() == sym:
                        consistent = True
                        break
                if not consistent:
                    issues.append((
                        "WARN",
                        f"target gene_symbol={sym!r} doesn't match "
                        f"extraction.claimed_target={claimed_target!r}",
                    ))
        except KeyError:
            pass

    # ── P1: biology is a synthesized slug (trial_biology_fallback) ────
    if primary_chain.biology_id and primary_chain.biology_id != "UNKNOWN":
        try:
            bnode = graph.get_node(primary_chain.biology_id)
            if (bnode.get("metadata") or {}).get("source") == "trial_biology_fallback":
                issues.append((
                    "WARN",
                    f"biology {primary_chain.biology_id!r} is a synthesized "
                    f"fallback slug; the chain doesn't ground at a real "
                    f"GO/Reactome pathway",
                ))
        except KeyError:
            pass

    # ── P1: population defaulted to unselected ────────────────────────
    if primary_chain.subgroup_population_id and primary_chain.subgroup_population_id.endswith("__unselected"):
        issues.append((
            "WARN",
            f"population resolved to default `__unselected` — no subgroup features",
        ))

    # ── P0: too few traversable chain edges ───────────────────────────
    traversable = sum(
        1 for k in (
            "compound_id", "target_id", "mechanism_id", "biology_id",
            "indication_id", "endpoint_id", "subgroup_population_id",
        ) if getattr(primary_chain, k, None) not in (None, "", "UNKNOWN")
    )
    if traversable < 4:
        issues.append((
            "FAIL",
            f"chain has only {traversable} resolved fields; a "
            f"prediction would walk too few edges to be meaningful",
        ))

    # Determine overall status
    if any(level == "FAIL" for level, _ in issues):
        status = "FAIL"
    elif any(level == "WARN" for level, _ in issues):
        status = "WARN"
    else:
        status = "PASS"

    return {
        "nct_id": nct_id,
        "status": status,
        "issues": issues,
        "chain_summary": {
            "compound_id": primary_chain.compound_id,
            "target_id": primary_chain.target_id,
            "mechanism_id": primary_chain.mechanism_id,
            "biology_id": primary_chain.biology_id,
            "indication_id": primary_chain.indication_id,
            "endpoint_id": primary_chain.endpoint_id,
            "population_id": primary_chain.subgroup_population_id,
        },
    }


def _emit_markdown(results: list[dict], snapshot_path: Path) -> str:
    out: list[str] = []
    out.append(f"# GraphGuard — chain-integrity report\n")
    out.append(f"Snapshot: `{snapshot_path}`\n\n")
    status_counts = Counter(r["status"] for r in results)
    out.append(
        f"**Summary**: "
        f"{status_counts.get('PASS', 0)} PASS, "
        f"{status_counts.get('WARN', 0)} WARN, "
        f"{status_counts.get('FAIL', 0)} FAIL "
        f"(of {len(results)} trials)\n\n"
    )
    if status_counts.get("FAIL", 0):
        out.append(
            "**Audit numbers should NOT be published until FAIL "
            "issues are resolved** — the chain mapping is broken; "
            "predictions will be noise.\n\n"
        )

    # Failures and warnings first; PASS trials grouped at the end.
    rank = {"FAIL": 0, "WARN": 1, "PASS": 2}
    for r in sorted(results, key=lambda r: (rank[r["status"]], r["nct_id"])):
        out.append(f"## {r['nct_id']} — {r['status']}\n\n")
        if r["chain_summary"]:
            cs = r["chain_summary"]
            out.append("Chain:\n")
            out.append("```\n")
            for k in (
                "compound_id", "target_id", "mechanism_id",
                "biology_id", "indication_id", "endpoint_id",
                "population_id",
            ):
                out.append(f"  {k:<13} {cs[k]}\n")
            out.append("```\n\n")
        if r["issues"]:
            for level, msg in r["issues"]:
                out.append(f"- **{level}**: {msg}\n")
        else:
            out.append("_(no issues)_\n")
        out.append("\n")
    return "".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument(
        "--nct", default="",
        help="Run on a single NCT id (otherwise scans all trial subgraphs).",
    )
    parser.add_argument(
        "--holdouts", default="",
        help="Comma-separated NCT ids — run only on these.",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Write markdown report to this path (otherwise prints to stdout).",
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
        nct_ids = sorted(graph.trial_subgraphs.keys())

    results = [_check_trial(nct, graph) for nct in nct_ids]
    md = _emit_markdown(results, args.snapshot)
    if args.out:
        args.out.write_text(md)
        print(f"Wrote {args.out}", file=sys.stderr)
    else:
        print(md)

    # Brief stderr summary for CI usage
    counts = Counter(r["status"] for r in results)
    print(
        f"GraphGuard: {counts.get('PASS', 0)} PASS / "
        f"{counts.get('WARN', 0)} WARN / "
        f"{counts.get('FAIL', 0)} FAIL",
        file=sys.stderr,
    )
    return 0 if counts.get("FAIL", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
