"""Q1: train/test leakage check for the n=52 + 5-holdout result.

For each case study (5 holdouts), tabulate training trials that overlap by
compound, target, mechanism, or indication. Output markdown so we can
sanity-check whether the 4/5 direction-correct result is learned
generalization or compound-disease memorization.

Run: .venv/bin/python scripts/check_holdout_leakage.py

Outputs markdown to stdout; redirect to audit/round_24_q1_leakage.md if you
want to keep it.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

SNAPSHOT = Path("data/exports/multi_indication_52_annotated.json")
HOLDOUTS = {
    "NCT01844505": ("nivolumab", "melanoma", "success"),
    "NCT01127633": ("solanezumab", "alzheimer", "failure"),
    "NCT00112918": ("bevacizumab", "colorectal_cancer__line_adjuvant", "failure"),
    "NCT00134264": ("torcetrapib", "cardiovascular", "failure (safety)"),
    "NCT00970359": ("selumetinib", "thyroid_cancer", "success"),
}


def _chain_summary(chain: dict) -> dict:
    return {
        "compound_id": chain.get("compound_id"),
        "target_id": chain.get("target_id"),
        "mechanism_id": chain.get("mechanism_id"),
        "biology_id": chain.get("biology_id"),
        "indication_id": chain.get("indication_id"),
    }


def _subgraph_signature(subgraph: dict) -> dict[str, set[str]]:
    """Collect the union of compound/target/mechanism/indication ids touched."""
    sig: dict[str, set[str]] = {
        "compound_ids": set(),
        "target_ids": set(),
        "mechanism_ids": set(),
        "indication_ids": set(),
    }
    for chain in subgraph.get("chains", []):
        for field, key in (
            ("compound_id", "compound_ids"),
            ("target_id", "target_ids"),
            ("mechanism_id", "mechanism_ids"),
            ("indication_id", "indication_ids"),
        ):
            val = chain.get(field)
            if val and val != "UNKNOWN":
                sig[key].add(val)
    return sig


def _closeness_score(holdout_sig: dict[str, set[str]], train_sig: dict[str, set[str]]) -> tuple[int, dict]:
    """Score how close a training trial is to a holdout.

    Tiers: compound match (8) > target match (4) > mechanism match (2) >
    indication match (1). Compound+indication together = the canonical
    'memorization risk' case.
    """
    overlap = {
        "compound": holdout_sig["compound_ids"] & train_sig["compound_ids"],
        "target": holdout_sig["target_ids"] & train_sig["target_ids"],
        "mechanism": holdout_sig["mechanism_ids"] & train_sig["mechanism_ids"],
        "indication": holdout_sig["indication_ids"] & train_sig["indication_ids"],
    }
    score = (
        8 * (1 if overlap["compound"] else 0)
        + 4 * (1 if overlap["target"] else 0)
        + 2 * (1 if overlap["mechanism"] else 0)
        + 1 * (1 if overlap["indication"] else 0)
    )
    return score, overlap


def _fmt_set(ids: set[str], cap: int = 3) -> str:
    if not ids:
        return "-"
    items = sorted(ids)
    if len(items) <= cap:
        return ", ".join(items)
    return ", ".join(items[:cap]) + f" (+{len(items) - cap} more)"


def main() -> int:
    if not SNAPSHOT.exists():
        print(f"missing snapshot: {SNAPSHOT}")
        return 1
    data = json.loads(SNAPSHOT.read_text())
    subgraphs = data["trial_subgraphs"]
    nct_ids = sorted(subgraphs.keys())
    holdout_ids = set(HOLDOUTS.keys())
    training_ids = [n for n in nct_ids if n not in holdout_ids]

    print("# Round 24 Q1 — Train/Test Leakage Check\n")
    print(f"Snapshot: `{SNAPSHOT.name}` — {len(nct_ids)} subgraphs "
          f"({len(training_ids)} training + {len(holdout_ids)} holdout)\n")
    print("Closeness score: compound=8, target=4, mechanism=2, indication=1. "
          "Higher = greater leakage risk.\n")

    holdout_sigs = {nct: _subgraph_signature(subgraphs[nct]) for nct in holdout_ids}
    train_sigs = {nct: _subgraph_signature(subgraphs[nct]) for nct in training_ids}

    summary_rows = []

    for holdout_nct, (compound, indication, outcome) in HOLDOUTS.items():
        if holdout_nct not in subgraphs:
            print(f"\n## {holdout_nct} {compound} — NOT IN SNAPSHOT\n")
            continue
        h_sig = holdout_sigs[holdout_nct]
        print(f"\n## {holdout_nct} — {compound} → {indication} ({outcome})\n")
        print(f"Holdout signature: "
              f"compounds={_fmt_set(h_sig['compound_ids'])}, "
              f"targets={_fmt_set(h_sig['target_ids'])}, "
              f"mechanisms={_fmt_set(h_sig['mechanism_ids'])}, "
              f"indications={_fmt_set(h_sig['indication_ids'])}\n")

        rows = []
        for train_nct in training_ids:
            score, overlap = _closeness_score(h_sig, train_sigs[train_nct])
            if score == 0:
                continue
            rows.append((score, train_nct, overlap))
        rows.sort(key=lambda x: -x[0])

        compound_matches = sum(1 for r in rows if r[2]["compound"])
        indication_matches = sum(1 for r in rows if r[2]["indication"])
        both_matches = sum(1 for r in rows if r[2]["compound"] and r[2]["indication"])
        target_matches = sum(1 for r in rows if r[2]["target"])

        print(f"Training overlap totals: "
              f"compound={compound_matches}, target={target_matches}, "
              f"indication={indication_matches}, "
              f"**compound+indication={both_matches}**\n")

        verdict = _verdict(compound_matches, target_matches, indication_matches, both_matches)
        print(f"**Verdict:** {verdict}\n")

        summary_rows.append({
            "holdout": holdout_nct,
            "compound": compound,
            "compound_match": compound_matches,
            "target_match": target_matches,
            "indication_match": indication_matches,
            "compound_plus_indication": both_matches,
            "verdict": verdict,
        })

        print("| score | NCT | compound | target | mechanism | indication |")
        print("|------:|-----|----------|--------|-----------|------------|")
        for score, train_nct, overlap in rows[:20]:
            print(f"| {score} | {train_nct} | "
                  f"{_fmt_set(overlap['compound']) if overlap['compound'] else '-'} | "
                  f"{_fmt_set(overlap['target']) if overlap['target'] else '-'} | "
                  f"{_fmt_set(overlap['mechanism']) if overlap['mechanism'] else '-'} | "
                  f"{_fmt_set(overlap['indication']) if overlap['indication'] else '-'} |")
        if len(rows) > 20:
            print(f"\n_({len(rows) - 20} additional overlapping training trials omitted)_\n")

    print("\n---\n\n## Summary\n")
    print("| holdout | compound | cpd-match | tgt-match | ind-match | cpd+ind | verdict |")
    print("|---------|----------|----------:|----------:|----------:|--------:|---------|")
    for r in summary_rows:
        print(f"| {r['holdout']} | {r['compound']} | "
              f"{r['compound_match']} | {r['target_match']} | "
              f"{r['indication_match']} | **{r['compound_plus_indication']}** | "
              f"{r['verdict']} |")

    return 0


def _verdict(cpd: int, tgt: int, ind: int, both: int) -> str:
    if both >= 3:
        return "MEMORIZATION RISK — multiple training trials with same compound+indication"
    if both >= 1:
        return "WEAK LEAKAGE — at least one direct compound+indication training trial"
    if cpd >= 1:
        return "COMPOUND-ONLY OVERLAP — same drug in different indication, indirect"
    if tgt >= 3:
        return "TARGET-CLASS LEARNING — multiple training trials share the target"
    if ind >= 3:
        return "INDICATION-CLASS LEARNING — multiple training trials in same indication"
    return "GENERALIZATION — minimal training overlap"


if __name__ == "__main__":
    raise SystemExit(main())
