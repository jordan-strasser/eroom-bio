"""Construct the multi_500 corpus: 100 each across cancer / Alzheimer's /
autoimmune / diabetes / cardiology, start ≤2021. Cancer reuses the cached onco
set (free extractions); the four new areas are fetched (with-results + terminated)
and need fresh extraction. Deterministic (sorted, no RNG). Writes
data/corpora/multi_500.txt + a composition report. Throwaway builder."""
import asyncio
import json
from pathlib import Path

from src.ingestion.clinicaltrials import (
    ClinicalTrialsClient,
    is_processable_drug_trial,
)

MAX_YEAR = 2021
TARGET = 100
AREAS = {
    "alzheimers": ["alzheimer disease"],
    "autoimmune": ["rheumatoid arthritis", "systemic lupus erythematosus",
                   "multiple sclerosis", "inflammatory bowel disease"],
    "diabetes": ["type 2 diabetes mellitus"],
    "cardiology": ["heart failure", "coronary artery disease"],
}
DATE_CACHE = Path("data/cache/onco_start_dates.json")


def _yr(d):
    return int(d[:4]) if d and len(d) >= 4 and d[:4].isdigit() else None


async def _cond_candidates(client, cond):
    """≤2021 NCTs for one condition: with-results ∪ terminated/withdrawn."""
    out = {}
    for kwargs in ({"has_results": True, "max_results": 300},
                   {"status": "TERMINATED,WITHDRAWN", "max_results": 150}):
        try:
            for r in await client.search(condition=cond, **kwargs):
                y = _yr(r.start_date)
                if y and y <= MAX_YEAR and is_processable_drug_trial(r):
                    out.setdefault(r.nct_id, y)
        except Exception as e:  # noqa: BLE001
            print(f"  ERR {cond} {kwargs}: {type(e).__name__}")
    return sorted(out)


def _round_robin(per_cond_lists, target, taken):
    """Balance across sub-conditions: round-robin until target reached."""
    picked, i = [], 0
    lists = [list(l) for l in per_cond_lists]
    while len(picked) < target and any(lists):
        lst = lists[i % len(lists)]
        if lst:
            nct = lst.pop(0)
            if nct not in taken and nct not in picked:
                picked.append(nct)
        i += 1
        if i > 100000:
            break
    return picked


import datetime as _dt
_PROMPT_CUT = _dt.datetime(2026, 6, 3).timestamp()  # classification-prompt change


def _is_current_prompt(nct):
    """True iff this NCT's cached extraction AND classification were written with
    the CURRENT prompts (mtime ≥ the 2026-06-03 prompt change) — so reusing them
    is byte-identical to re-extracting now (consistent with the fresh non-onco
    areas). The 10 stale (06-02) onco extractions are excluded."""
    e = f"data/annotations/{nct}_extraction.json"
    c = f"data/annotations/{nct}_classification.json"
    import os
    return (os.path.exists(e) and os.path.exists(c)
            and os.path.getmtime(e) >= _PROMPT_CUT
            and os.path.getmtime(c) >= _PROMPT_CUT)


async def _cancer_slice(client, taken):
    """100 cancer NCTs from the CURRENT-PROMPT cached onco set, start ≤2021 —
    free (cached) AND consistent (same prompts as the fresh non-onco areas)."""
    onco = [l.strip() for l in open("data/corpora/onco_scale_500.txt")
            if l.strip() and not l.startswith("#")]
    onco = [n for n in onco if _is_current_prompt(n)]  # current-prompt cache only
    cache = json.loads(DATE_CACHE.read_text()) if DATE_CACHE.exists() else {}
    sem = asyncio.Semaphore(6)

    async def one(nct):
        if nct in cache:
            return
        async with sem:
            try:
                cache[nct] = (await client.get_study(nct)).start_date
            except Exception:  # noqa: BLE001
                cache[nct] = None
    await asyncio.gather(*(one(n) for n in onco))
    DATE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    DATE_CACHE.write_text(json.dumps(cache))
    le = sorted(n for n in onco if (_yr(cache.get(n)) or 9999) <= MAX_YEAR and n not in taken)
    return le[:TARGET]


async def main():
    client = ClinicalTrialsClient()
    corpus, report = {}, {}

    cancer = await _cancer_slice(client, set())
    for n in cancer:
        corpus[n] = "cancer"
    report["cancer"] = len(cancer)

    for area, conds in AREAS.items():
        per_cond = [await _cond_candidates(client, c) for c in conds]
        picked = _round_robin(per_cond, TARGET, set(corpus))
        for n in picked:
            corpus[n] = area
        report[area] = len(picked)

    out = Path("data/corpora/multi_500.txt")
    out.write_text("\n".join(sorted(corpus)) + "\n")
    print(f"\nwrote {out} — {len(corpus)} NCTs")
    for area, n in report.items():
        print(f"  {area:12s} {n}")
    fresh = sum(n for a, n in report.items() if a != "cancer")
    print(f"cached (cancer, free): {report.get('cancer', 0)}   fresh extraction (paid): {fresh}")


asyncio.run(main())
