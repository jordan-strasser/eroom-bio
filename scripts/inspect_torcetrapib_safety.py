"""Q4: why does torcetrapib have safety_penalty=0.000 despite 3 CETP siblings?

Three hypotheses to verify in order:
  (a) CETP node has NO target_associated_ae edges at all
  (b) Edges exist but below the round-20 threshold (min_belief=0.55,
      min_evidence=1.0)
  (c) Chain target_id doesn't match where AE edges sit

Run: .venv/bin/python scripts/inspect_torcetrapib_safety.py
"""

from __future__ import annotations

import json
from pathlib import Path

from src.graph.models import EdgeBeliefState, EdgeType
from src.graph.store import GraphStore

SNAPSHOT = Path("data/exports/multi_indication_52_annotated.json")
TORCETRAPIB_NCT = "NCT00134264"
CETP_TARGET = "ENSG00000087237"
CETP_SIBLINGS = ["torcetrapib", "anacetrapib", "evacetrapib", "dalcetrapib"]
MIN_BELIEF_PENALTY = 0.55
MIN_EVIDENCE_PENALTY = 1.0


def _belief_summary(edge: dict) -> str:
    try:
        b = EdgeBeliefState.model_validate(edge["belief"])
    except Exception as exc:
        return f"<belief parse error: {exc}>"
    sources = sorted({rec.source_id for rec in b.evidence}) if b.evidence else []
    src_str = ", ".join(sources[:5])
    if len(sources) > 5:
        src_str += f" (+{len(sources) - 5} more)"
    return (
        f"E[p]={b.expected_probability:.3f}  "
        f"alpha={b.alpha:.2f}  beta={b.beta:.2f}  "
        f"n_eff={b.evidence_strength:.2f}  "
        f"records={len(b.evidence)}  sources=[{src_str}]"
    )


def _passes_penalty_threshold(edge: dict) -> bool:
    try:
        b = EdgeBeliefState.model_validate(edge["belief"])
    except Exception:
        return False
    return (
        b.expected_probability >= MIN_BELIEF_PENALTY
        and b.evidence_strength >= MIN_EVIDENCE_PENALTY
    )


def main() -> int:
    if not SNAPSHOT.exists():
        print(f"missing snapshot: {SNAPSHOT}")
        return 1
    print(f"# Round 24 Q4 — Torcetrapib Safety Propagation Diagnostic\n")
    print(f"Snapshot: `{SNAPSHOT.name}`\n")
    print(f"Round-20 penalty thresholds: "
          f"min_belief={MIN_BELIEF_PENALTY}, min_evidence={MIN_EVIDENCE_PENALTY}\n")

    store = GraphStore()
    store.import_snapshot(str(SNAPSHOT))

    # Step 1. Confirm the chain's actual target_id (hypothesis c check)
    try:
        sg = store.get_trial_subgraph_by_id(TORCETRAPIB_NCT)
    except KeyError:
        print(f"FATAL: {TORCETRAPIB_NCT} not in snapshot")
        return 1
    print(f"## Torcetrapib chain target_ids in {TORCETRAPIB_NCT}\n")
    chain_targets: set[str] = set()
    for chain in sg.chains:
        if chain.compound_id == "torcetrapib":
            chain_targets.add(chain.target_id)
    print(f"Distinct target_ids on torcetrapib chains: {sorted(chain_targets)}")
    if CETP_TARGET in chain_targets:
        print(f"✓ Chain uses {CETP_TARGET} (CETP) — hypothesis (c) NOT the cause\n")
    else:
        print(f"✗ Chain does NOT use CETP_TARGET={CETP_TARGET} — hypothesis (c) IS the cause\n")

    # Step 2. CETP target_associated_ae edges (hypothesis a/b check)
    print(f"## CETP ({CETP_TARGET}) → target_associated_ae edges\n")
    try:
        cetp_edges = store.get_neighboring_edges(
            CETP_TARGET, edge_types=[EdgeType.TARGET_ASSOCIATED_AE],
        )
    except KeyError:
        print(f"FATAL: {CETP_TARGET} not in graph at all\n")
        cetp_edges = []
    cetp_outgoing = [e for e in cetp_edges if e["source_id"] == CETP_TARGET]
    print(f"target_associated_ae count: **{len(cetp_outgoing)}**\n")
    if not cetp_outgoing:
        print("→ Hypothesis (a) confirmed: CETP has NO target_associated_ae edges. "
              "Propagation never produced any.\n")
    else:
        print("| ae_id | belief | passes penalty threshold? |")
        print("|-------|--------|---------------------------|")
        for e in sorted(cetp_outgoing, key=lambda x: x["target_id"]):
            passes = "✓" if _passes_penalty_threshold(e) else "✗"
            print(f"| {e['target_id']} | {_belief_summary(e)} | {passes} |")
        passing = sum(1 for e in cetp_outgoing if _passes_penalty_threshold(e))
        print(f"\n{passing} of {len(cetp_outgoing)} edges pass the round-20 penalty threshold.\n")
        if passing == 0:
            print("→ Hypothesis (b) confirmed: edges exist but all below threshold.\n")

    # Step 3. Per-sibling causes_ae inventory (root-cause for missing propagation)
    print(f"## causes_ae edges per CETP sibling\n")
    for sib in CETP_SIBLINGS:
        try:
            store.get_node(sib)
        except KeyError:
            print(f"### {sib} — NOT IN GRAPH\n")
            continue
        try:
            edges = store.get_neighboring_edges(
                sib, edge_types=[EdgeType.CAUSES_AE],
            )
        except KeyError:
            print(f"### {sib} — no neighbors lookup possible\n")
            continue
        outgoing = [e for e in edges if e["source_id"] == sib]
        print(f"### {sib} — {len(outgoing)} causes_ae edges\n")
        if not outgoing:
            print(f"(no AE attributions in corpus for {sib})\n")
            continue
        print("| ae_id | belief |")
        print("|-------|--------|")
        for e in sorted(outgoing, key=lambda x: x["target_id"])[:15]:
            print(f"| {e['target_id']} | {_belief_summary(e)} |")
        if len(outgoing) > 15:
            print(f"\n_({len(outgoing) - 15} additional AEs omitted)_\n")
        print()

    # Step 4. Cross-tabulate — which AEs are attributed to ≥2 siblings?
    print(f"## AEs attributed to multiple CETP siblings (propagation candidates)\n")
    ae_to_siblings: dict[str, list[tuple[str, EdgeBeliefState]]] = {}
    for sib in CETP_SIBLINGS:
        try:
            edges = store.get_neighboring_edges(
                sib, edge_types=[EdgeType.CAUSES_AE],
            )
        except KeyError:
            continue
        for e in edges:
            if e["source_id"] != sib:
                continue
            try:
                belief = EdgeBeliefState.model_validate(e["belief"])
            except Exception:
                continue
            ae_to_siblings.setdefault(e["target_id"], []).append((sib, belief))
    multi_sib = {ae: lst for ae, lst in ae_to_siblings.items() if len(lst) >= 2}
    print(f"AEs with ≥2 CETP-sibling attributions: **{len(multi_sib)}**\n")
    if multi_sib:
        print("| ae_id | siblings | beliefs |")
        print("|-------|----------|---------|")
        for ae, lst in sorted(multi_sib.items()):
            sibs = ", ".join(s for s, _ in lst)
            beliefs = " | ".join(f"{s}: E[p]={b.expected_probability:.3f}, n_eff={b.evidence_strength:.2f}" for s, b in lst)
            print(f"| {ae} | {sibs} | {beliefs} |")
        eligible = sum(
            1 for lst in multi_sib.values()
            if sum(1 for _, b in lst if b.evidence_strength >= 1.0) >= 2
        )
        print(f"\n{eligible} AEs have ≥2 siblings with evidence_strength ≥ 1.0 "
              f"(meets propagation prerequisite).\n")
    else:
        print("→ ROOT CAUSE: no AE was extracted from ≥2 CETP-sibling trials. "
              "Even if propagation ran, there'd be nothing to propagate.\n")

    # Verdict
    print("---\n\n## Verdict\n")
    if not cetp_outgoing:
        if not multi_sib:
            print("**Root cause: corpus coverage**. The 3 sibling trials (REVEAL "
                  "anacetrapib, ACCELERATE evacetrapib, dal-OUTCOMES dalcetrapib) "
                  "either didn't have AE attributions extracted, or each AE was "
                  "attributed to only one sibling. Propagation requires ≥2 siblings "
                  "with shared AEs (`_MIN_COMPOUNDS_FOR_TARGET_AE=2`).\n")
            print("**Fix path:** improve AE attribution coverage for CV trials, OR "
                  "lower `_MIN_COMPOUNDS_FOR_TARGET_AE` to 1 (target-class propagation "
                  "from a single sibling — likely too aggressive without further "
                  "tuning of the contribution weight).\n")
        else:
            print("**Root cause: propagation never ran on this snapshot**, even though "
                  "the prerequisites are met. `propagate_to_target_associated_ae` "
                  "needs to be invoked after each causes_ae update — verify the "
                  "attributor chain.\n")
    elif passing == 0:
        print("**Root cause: edges exist but all below the round-20 penalty threshold "
              "(min_belief=0.55, min_evidence=1.0).** Either lower the threshold for "
              "safety propagation, or improve causes_ae evidence quality for the sibling "
              "trials.\n")
    else:
        print("Propagation produced edges that PASS the threshold. The safety_penalty=0.000 "
              "verdict in the audit must be looking at a different chain or compound — "
              "re-check the audit pipeline.\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
