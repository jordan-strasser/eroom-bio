"""Phase 3 — CT.gov intervention pull for the 10 DRD2 drugs (cheap, metadata).

Searches CT.gov v2 by intervention (reuses the extended
ClinicalTrialsClient.search(intervention=, study_type='INTERVENTIONAL')), buckets
each trial by indication, tags it with its drug's direction, and caches the raw
per-trial records to scratch so selection can be tuned without re-fetching.

No annotation. Writes scratch/drd2/ctgov_raw.json + prints a drug x indication
matrix to inform the ~100-trial selection.
"""
from __future__ import annotations

import asyncio
import json
from collections import Counter, defaultdict
from pathlib import Path

from src.ingestion.clinicaltrials import ClinicalTrialsClient, is_processable_drug_trial

HERE = Path(__file__).parent
RAW_OUT = HERE / "ctgov_raw.json"

# drug -> direction (from finalize_directions.py, all verified)
DRUGS = {
    "pramipexole": "agonist", "ropinirole": "agonist", "rotigotine": "agonist",
    "apomorphine": "agonist", "cabergoline": "agonist",
    "haloperidol": "antagonist", "risperidone": "antagonist",
    "olanzapine": "antagonist", "metoclopramide": "antagonist",
    "prochlorperazine": "antagonist",
}

PER_DRUG_FETCH = 350

# indication buckets — priority order matters (first match wins)
_RULES: list[tuple[str, list[str]]] = [
    ("schizophrenia", ["schizophren", "schizoaffective"]),
    ("parkinsons", ["parkinson"]),
    ("restless_legs", ["restless leg", "restless legs syndrome", "willis-ekbom"]),
    ("bipolar_mania", ["bipolar", "mania", "manic"]),
    ("nausea_vomiting", ["nausea", "vomiting", "emesis", "emetic", "ponv", "cinv",
                         "antiemetic", "hyperemesis"]),
    ("gastroparesis_gi", ["gastroparesis", "dyspepsia", "gastric empty",
                          "gastric emptying", "reflux", "gerd"]),
    ("hyperprolactinemia", ["prolactin"]),
    ("migraine_headache", ["migraine", "headache", "cephalgia"]),
    ("tourette_tics", ["tourette", "tic disorder", "tics"]),
    ("dementia_agitation", ["dementia", "alzheimer", "agitation", "delirium"]),
    ("autism", ["autis", "asperger", "pervasive developmental"]),
    ("depression_anxiety", ["depress", "anxiety", "ocd", "obsessive"]),
    ("huntington_chorea", ["huntington", "chorea"]),
    ("tardive_dyskinesia", ["tardive", "dyskinesia"]),
    ("psychosis_other", ["psychosis", "psychotic", "hallucinat"]),
]

CORE = {"schizophrenia", "parkinsons", "nausea_vomiting", "bipolar_mania",
        "restless_legs", "hyperprolactinemia"}


def indication_bucket(conditions: list[str]) -> str:
    text = " ; ".join(conditions).lower()
    for bucket, keys in _RULES:
        if any(k in text for k in keys):
            return bucket
    return "other"


def _phase_num(phase: str) -> str:
    # for ranking: "2", "3", "2/3" -> take the highest digit present
    digits = [c for c in phase if c.isdigit()]
    return max(digits) if digits else "0"


async def main() -> None:
    ct = ClinicalTrialsClient()
    # nct -> record (dedup across drugs; remember all matched drugs/directions)
    records: dict[str, dict] = {}
    for drug, direction in DRUGS.items():
        trials = await ct.search(intervention=drug, study_type="INTERVENTIONAL",
                                 max_results=PER_DRUG_FETCH)
        print(f"  {drug:16s} ({direction:10s}): fetched {len(trials)}")
        for t in trials:
            bucket = indication_bucket(t.conditions)
            rec = records.get(t.nct_id)
            if rec is None:
                rec = {
                    "nct_id": t.nct_id, "title": t.title, "phase": t.phase,
                    "status": t.status, "conditions": t.conditions,
                    "indication": bucket, "has_results": t.has_results,
                    "n_primary": len(t.primary_outcomes),
                    "processable": is_processable_drug_trial(t),
                    "drugs": [], "directions": [],
                }
                records[t.nct_id] = rec
            if drug not in rec["drugs"]:
                rec["drugs"].append(drug)
            if direction not in rec["directions"]:
                rec["directions"].append(direction)
    # resolve a single direction per trial (drop trials mixing both directions)
    for rec in records.values():
        ds = set(rec["directions"])
        rec["direction"] = ds.pop() if len(ds) == 1 else "MIXED"

    RAW_OUT.write_text(json.dumps(list(records.values()), indent=2))
    print(f"\nwrote {RAW_OUT}: {len(records)} unique trials\n")

    # ── matrix: drug x indication (processable only) ──
    mixed = [r for r in records.values() if r["direction"] == "MIXED"]
    print(f"trials matching BOTH directions (dropped from clean set): {len(mixed)}")
    print(f"  {[m['nct_id'] for m in mixed][:15]}\n")

    by_dir_ind = defaultdict(Counter)
    for r in records.values():
        if r["direction"] == "MIXED" or not r["processable"]:
            continue
        by_dir_ind[r["direction"]][r["indication"]] += 1
    for direction in ("agonist", "antagonist"):
        print(f"=== {direction} (processable) ===")
        for ind, n in by_dir_ind[direction].most_common():
            core = " *CORE*" if ind in CORE else ""
            print(f"    {ind:22s} {n}{core}")
        print()

    # per drug x indication (core only)
    print("drug x core-indication (processable, count):")
    perdrug = defaultdict(Counter)
    for r in records.values():
        if r["direction"] == "MIXED" or not r["processable"]:
            continue
        if r["indication"] in CORE:
            # attribute to each matched drug for the matrix view
            for d in r["drugs"]:
                if d in DRUGS:
                    perdrug[d][r["indication"]] += 1
    for drug in DRUGS:
        row = perdrug[drug]
        cells = "  ".join(f"{k}={v}" for k, v in row.most_common())
        print(f"    {drug:16s} {cells}")


if __name__ == "__main__":
    asyncio.run(main())
