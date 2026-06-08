"""Fetch CT.gov start dates for a corpus + report the year distribution, so we
can pick the train/holdout date cutoff. Caches to data/cache so reruns are free.
Throwaway helper for the 2021-split n=500 build."""
import asyncio
import json
from collections import Counter
from pathlib import Path

from src.ingestion.clinicaltrials import ClinicalTrialsClient

CORPUS = "onco_scale_500"
CACHE = Path(f"data/cache/{CORPUS}_start_dates.json")


def _year(d: str | None) -> int | None:
    if not d or len(d) < 4 or not d[:4].isdigit():
        return None
    return int(d[:4])


async def main():
    ncts = [l.strip() for l in open(f"data/corpora/{CORPUS}.txt")
            if l.strip() and not l.startswith("#")]
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    client = ClinicalTrialsClient()
    sem = asyncio.Semaphore(5)

    async def one(nct):
        if nct in cache:
            return
        async with sem:
            try:
                rec = await client.get_study(nct)
                cache[nct] = rec.start_date
            except Exception as e:  # noqa: BLE001
                cache[nct] = None
                print(f"  FAIL {nct}: {type(e).__name__}")

    await asyncio.gather(*(one(n) for n in ncts))
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(cache, indent=0))

    years = Counter(_year(cache.get(n)) for n in ncts)
    missing = years.pop(None, 0)
    print(f"\n{CORPUS}: {len(ncts)} NCTs, {missing} missing a start date")
    print("year distribution:")
    cum = 0
    for y in sorted(k for k in years if k is not None):
        cum += years[y]
        print(f"  {y}: {years[y]:3d}   (cumulative ≤{y}: {cum})")
    for cut in (2019, 2020, 2021):
        le = sum(v for k, v in years.items() if k is not None and k <= cut)
        ge = sum(v for k, v in years.items() if k is not None and k > cut)
        print(f"  cutoff ≤{cut}: train={le}  holdout(>{cut})={ge}  (+{missing} undated)")


asyncio.run(main())
