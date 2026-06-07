"""Provenance trace — what evidence (from which indications) decides a prediction?

For one trial, predict it faithfully (``predict_clinical_hypothesis``) and print
the deciding (weakest-link) edge plus that edge's evidence records grouped by the
SOURCE trial's indication — so cross-indication transfer ("this trial's fate was
informed by trials in other diseases") is concrete and auditable.

    # one trial
    python -m scripts.trace_provenance --graph <annotated.json> --nct NCT01142336
    # discover trials whose DECIDING edge has cross-indication support (the "find the 5"):
    python -m scripts.trace_provenance --graph <annotated.json> --scan --limit 25
    python -m scripts.trace_provenance --graph <g> --scan --corpus onco_scale_500 --json out.json

See ``src/prediction/provenance.py`` (`trace_holdout`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.graph.store import GraphStore
from src.prediction.provenance import (
    HoldoutTrace,
    deciding_cross_indications,
    therapeutic_area,
    trace_holdout,
)

CORPORA_DIR = Path("data/corpora")


def _corpus_ncts(name: str) -> list[str]:
    path = CORPORA_DIR / f"{name}.txt"
    return [
        ln.strip()
        for ln in path.read_text().splitlines()
        if ln.strip() and not ln.startswith("#")
    ]


def _print_edge(e, indent: str = "    ") -> None:
    mark = " ◄ DECIDING (weakest link)" if e.is_deciding else ""
    print(
        f"{indent}{e.edge_type}: {e.source_name} → {e.target_name}{mark}\n"
        f"{indent}  E[p]={e.expected_probability:.2f}  n_eff={e.evidence_strength:.1f}  "
        f"bottleneck={e.bottleneck_score:.2f}  records={e.n_records} "
        f"({e.n_database_records} db)"
    )
    if e.self_ncts:
        print(f"{indent}  self: {', '.join(e.self_ncts)}")
    if e.same_indication_ncts:
        print(f"{indent}  same-indication: {', '.join(e.same_indication_ncts[:6])}")
    for ind, ncts in sorted(e.cross_indication_ncts.items()):
        print(f"{indent}  ✦ CROSS-INDICATION [{ind}]: {', '.join(ncts[:6])}")
    if e.self_excluded_evidence_strength > 0:
        print(
            f"{indent}  without THIS trial: E[p]={e.self_excluded_expected_probability:.2f} "
            f"(n_eff={e.self_excluded_evidence_strength:.1f})"
        )
    elif e.self_ncts:
        print(f"{indent}  without THIS trial: unobserved (this trial was the only evidence)")


def _print_trace(t: HoldoutTrace) -> None:
    print("=" * 78)
    print(f"TRACE — {t.nct}  ({', '.join(t.indications)})  compound={t.compound_id}")
    print(
        f"  prediction: overall={t.overall_probability:.3f}  "
        f"efficacy={t.efficacy_probability:.3f}  safety_penalty={t.safety_penalty:.3f}"
    )
    print(f"  hypothesis: {t.hypothesis}")
    print(
        f"  deciding edge has cross-indication support: "
        f"{'YES ✦' if t.deciding_edge_has_cross_indication else 'no'}"
    )
    if t.self_excluded_efficacy is not None:
        print(
            f"  efficacy from OTHER trials alone (self-excluded): "
            f"{t.self_excluded_efficacy:.3f} over {t.self_excluded_n_edges} surviving edge(s)"
        )
    else:
        print("  efficacy from OTHER trials alone: n/a (no edge survives without this trial)")
    print("\n  DECIDING EDGE:")
    if t.deciding_edge:
        _print_edge(t.deciding_edge, indent="    ")
    else:
        print("    (none)")
    print("\n  ALL CHAIN EDGES:")
    for e in sorted(t.edges, key=lambda e: e.bottleneck_score, reverse=True):
        _print_edge(e, indent="    ")


def _scan(store: GraphStore, ncts: list[str], n_samples: int) -> list[HoldoutTrace]:
    traces: list[HoldoutTrace] = []
    for nct in ncts:
        try:
            traces.append(trace_holdout(store, nct, n_samples=n_samples))
        except KeyError:
            continue
    return traces


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--graph", required=True, help="trained annotated.json snapshot")
    ap.add_argument("--nct", default=None, help="trace a single trial")
    ap.add_argument(
        "--scan",
        action="store_true",
        help="scan trials and rank those whose deciding edge has cross-indication support",
    )
    ap.add_argument(
        "--corpus",
        default=None,
        help="restrict --scan to this corpus's NCTs (data/corpora/<name>.txt); "
        "default = all trials in the graph",
    )
    ap.add_argument("--limit", type=int, default=25, help="max trials shown in --scan")
    ap.add_argument(
        "--cross-area",
        action="store_true",
        help="--scan: only trials whose deciding edge spans ≥2 therapeutic areas "
        "(the differentiated onco↔other transfer, not the chemo backbone)",
    )
    ap.add_argument(
        "--max-breadth",
        type=int,
        default=None,
        help="--scan: drop trials whose deciding edge draws on more than N "
        "cross-indications (filters the generic DNA-damage/mitotic backbone)",
    )
    ap.add_argument("--n-samples", type=int, default=4000)
    ap.add_argument("--json", default=None, help="dump full traces to this path")
    args = ap.parse_args()

    store = GraphStore()
    store.import_snapshot(args.graph)

    if args.nct:
        _print_trace(trace_holdout(store, args.nct, n_samples=args.n_samples))
        return 0

    if not args.scan:
        ap.error("pass --nct <NCT> or --scan")

    ncts = _corpus_ncts(args.corpus) if args.corpus else sorted(store.trial_subgraphs)
    traces = _scan(store, ncts, args.n_samples)

    # Annotate each cross-indication trace with (real cross diseases, areas,
    # spans-areas) so we can rank SPECIFIC CROSS-AREA demos above the generic
    # DNA-damage/mitotic-arrest backbone (supported by ~50 cancers).
    rows = []
    for t in traces:
        if not t.deciding_edge_has_cross_indication:
            continue
        real, areas, spans = deciding_cross_indications(t)
        if not real:
            continue  # cross support was only tox/non-disease slugs
        if args.cross_area and not spans:
            continue
        if args.max_breadth is not None and len(real) > args.max_breadth:
            continue
        rows.append((t, real, areas, spans))

    # rank: spans areas first, then non-onco holdout, then most SPECIFIC, then evidence
    def _key(row):
        t, real, areas, spans = row
        h_areas = {therapeutic_area(i) for i in t.indications}
        return (spans, any(a != "oncology" for a in h_areas), -len(real), t.deciding_edge.evidence_strength)
    rows.sort(key=_key, reverse=True)

    print("=" * 78)
    print(f"SCANNED {len(traces)} predictable trials")
    print(f"  deciding edge has REAL cross-indication evidence: {len(rows)}", end="")
    if args.cross_area:
        print("  (cross-area only)", end="")
    if args.max_breadth is not None:
        print(f"  (breadth ≤ {args.max_breadth})", end="")
    print("  ← candidate demos\n" + "=" * 78)
    for t, real, areas, spans in rows[: args.limit]:
        de = t.deciding_edge
        tag = "  ✦ CROSS-AREA" if spans else ""
        print(
            f"\n{t.nct}  P={t.overall_probability:.2f}  [{', '.join(t.indications)}]"
            f"  areas={'/'.join(sorted(areas))}{tag}"
            f"\n  deciding: {de.edge_type} {de.source_name} → {de.target_name} "
            f"(E[p]={de.expected_probability:.2f}, n_eff={de.evidence_strength:.1f})"
            f"\n  cross-indication support from: {', '.join(real)}"
        )

    if args.json:
        with open(args.json, "w") as fh:
            json.dump([t.model_dump() for t in traces], fh, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
