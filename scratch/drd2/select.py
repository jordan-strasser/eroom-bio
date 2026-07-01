"""Phase 3 selection — pick ~100 balanced trials and write the corpus.

Reads scratch/drd2/ctgov_raw.json. Selects a ~100-trial set balanced across
direction (≈50 agonist / ≈50 antagonist) and spread over 3 core indications per
direction so leave-one-indication-out (LOIO) has ≥3 rotatable folds each side.
Within every (direction, indication) cell, trials are ranked (results > phase 3/2 >
completed/terminated) and filled round-robin across drugs so no blockbuster
(risperidone) dominates and every drug appears.

Writes data/corpora/drd2_subset.txt and scratch/drd2/selection.json.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
RAW = HERE / "ctgov_raw.json"
CORPUS = Path("data/corpora/drd2_subset.txt")

DIRECTION = {
    "pramipexole": "agonist", "ropinirole": "agonist", "rotigotine": "agonist",
    "apomorphine": "agonist", "cabergoline": "agonist",
    "haloperidol": "antagonist", "risperidone": "antagonist",
    "olanzapine": "antagonist", "metoclopramide": "antagonist",
    "prochlorperazine": "antagonist",
}

# (direction, indication) -> target n. Sums to ~100, 50/50 by direction.
PLAN = {
    ("agonist", "parkinsons"): 25,
    ("agonist", "restless_legs"): 17,
    ("agonist", "hyperprolactinemia"): 8,
    ("antagonist", "schizophrenia"): 25,
    ("antagonist", "nausea_vomiting"): 17,
    ("antagonist", "bipolar_mania"): 8,
}
PER_DRUG_CELL_CAP = 9  # max trials one drug can contribute to a single cell


def phase_rank(phase: str) -> int:
    d = max([c for c in phase if c.isdigit()], default="0")
    return {"3": 0, "2": 1, "4": 2, "1": 3, "0": 4}.get(d, 5)


def rank_key(r: dict) -> tuple:
    has_signal = any(s in r["status"].upper()
                     for s in ("COMPLETED", "TERMINATED", "WITHDRAWN", "SUSPENDED"))
    return (
        0 if r["has_results"] else 1,
        phase_rank(r["phase"]),
        0 if has_signal else 1,
        r["nct_id"],
    )


def primary_drug(r: dict, direction: str) -> str:
    for d in r["drugs"]:
        if DIRECTION.get(d) == direction:
            return d
    return r["drugs"][0]


def main() -> int:
    records = json.loads(RAW.read_text())
    by_nct = {r["nct_id"]: r for r in records}

    # candidate pool per cell
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        if r["direction"] == "MIXED" or not r["processable"]:
            continue
        key = (r["direction"], r["indication"])
        if key in PLAN:
            cells[key].append(r)

    selected: dict[str, dict] = {}
    for key, quota in PLAN.items():
        direction, indication = key
        pool = cells[key]
        # group by primary drug, each sorted by rank
        by_drug: dict[str, list[dict]] = defaultdict(list)
        for r in sorted(pool, key=rank_key):
            by_drug[primary_drug(r, direction)].append(r)
        drug_caps = Counter()
        # round-robin across drugs until quota met or pool exhausted
        picked = 0
        progress = True
        while picked < quota and progress:
            progress = False
            for drug in sorted(by_drug):
                if picked >= quota:
                    break
                if drug_caps[drug] >= PER_DRUG_CELL_CAP:
                    continue
                # next unselected trial for this drug
                while by_drug[drug]:
                    r = by_drug[drug].pop(0)
                    if r["nct_id"] in selected:
                        continue
                    r2 = dict(r)
                    r2["sel_direction"] = direction
                    r2["sel_indication"] = indication
                    r2["sel_drug"] = drug
                    selected[r["nct_id"]] = r2
                    drug_caps[drug] += 1
                    picked += 1
                    progress = True
                    break

    sel = list(selected.values())
    (HERE / "selection.json").write_text(json.dumps(sel, indent=2))

    # ── balance report ──
    print(f"selected {len(sel)} trials\n")
    dir_counts = Counter(r["sel_direction"] for r in sel)
    ind_counts = Counter(r["sel_indication"] for r in sel)
    print("by direction:", dict(dir_counts))
    print("by indication:")
    for ind, n in ind_counts.most_common():
        print(f"    {ind:22s} {n}")
    print("\ndrug x direction x indication:")
    matrix = defaultdict(Counter)
    for r in sel:
        matrix[(r["sel_drug"], r["sel_direction"])][r["sel_indication"]] += 1
    for (drug, direction), inds in sorted(matrix.items(),
                                          key=lambda kv: (kv[0][1], kv[0][0])):
        cells_s = "  ".join(f"{k}={v}" for k, v in inds.most_common())
        print(f"    {drug:16s} {direction:10s} {cells_s}")

    # cells that fell short of quota
    print("\ncell fill vs quota:")
    got = Counter((r["sel_direction"], r["sel_indication"]) for r in sel)
    for key, quota in PLAN.items():
        flag = "" if got[key] >= quota else "  << short"
        print(f"    {key[0]:10s} {key[1]:20s} {got[key]}/{quota}{flag}")

    # ── write corpus ──
    lines = [
        "# DRD2 direction-pooling subset — one DRD2 (ENSG00000149295) node, both",
        "# directions, multi-indication. Built by scratch/drd2/{pull_ctgov,select}.py.",
        "# Each NCT: <drug> | <direction(ChEMBL action_type)> | <indication(CT.gov cond)>",
        "# | phase | status. Direction is stamped natively at build from the ChEMBL",
        "# mechanism cache (primed via scratch/drd2/mech_cache_primed.json).",
        "#",
        f"# {len(sel)} trials | "
        f"agonist={dir_counts['agonist']} antagonist={dir_counts['antagonist']} | "
        f"indications={len(ind_counts)}",
        "",
    ]
    # group corpus by direction then indication for readability
    for direction in ("agonist", "antagonist"):
        lines.append(f"# ── {direction} ──")
        for r in sorted([s for s in sel if s["sel_direction"] == direction],
                        key=lambda r: (r["sel_indication"], r["sel_drug"], r["nct_id"])):
            cond = (r["conditions"][0] if r["conditions"] else "?")[:40]
            res = "results" if r["has_results"] else "noresults"
            lines.append(
                f"{r['nct_id']}  # {r['sel_drug']} | {direction} | "
                f"{r['sel_indication']} | P{r['phase'] or '?'} | {r['status']} | "
                f"{res} | {cond}"
            )
        lines.append("")
    CORPUS.parent.mkdir(parents=True, exist_ok=True)
    CORPUS.write_text("\n".join(lines))
    print(f"\nwrote {CORPUS} ({len(sel)} NCT ids)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
