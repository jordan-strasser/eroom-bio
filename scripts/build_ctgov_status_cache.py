"""Build the CT.gov status cache for the stop-reason routing override.

Fetches ``overallStatus`` + ``whyStopped`` from ClinicalTrials.gov v2 for every
NCT in a corpus and writes/merges ``data/cache/ctgov_status.json``
(``{nct: {"overall_status", "why_stopped"}}``). The attributor consults this
cache (under EROOM_ROUTING) to reroute early-terminated non-efficacy trials to the
censoring branch instead of letting them downvote the biological backbone.

Usage:
    python -m scripts.build_ctgov_status_cache --corpus onco_scale_500
    python -m scripts.build_ctgov_status_cache --corpus onco_scale_500 --only-stopped
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path

CACHE = Path("data/cache/ctgov_status.json")
CORPORA = Path("data/corpora")
API = "https://clinicaltrials.gov/api/v2/studies/{}?fields=protocolSection.statusModule"


def corpus_ncts(name: str) -> list[str]:
    p = CORPORA / (name if name.endswith(".txt") else f"{name}.txt")
    out = []
    for line in p.read_text().splitlines():
        c = line.split("#", 1)[0].strip()
        if c.startswith("NCT"):
            out.append(c)
    return out


def fetch(nct: str) -> tuple[str, str] | None:
    for attempt in range(3):
        try:
            with urllib.request.urlopen(API.format(nct), timeout=25) as r:
                sm = (json.loads(r.read()).get("protocolSection", {})
                      .get("statusModule", {}))
                return sm.get("overallStatus", ""), (sm.get("whyStopped") or "").strip()
        except Exception:  # noqa: BLE001
            if attempt == 2:
                return None
            time.sleep(2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--only-stopped", action="store_true",
                    help="only KEEP terminated/withdrawn/suspended trials in the cache "
                         "(the only ones the override can fire on) — smaller cache")
    ap.add_argument("--refresh", action="store_true",
                    help="re-fetch even NCTs already cached")
    args = ap.parse_args()

    cache = {}
    if CACHE.exists():
        cache = json.loads(CACHE.read_text())

    ncts = corpus_ncts(args.corpus)
    todo = [n for n in ncts if args.refresh or n not in cache]
    print(f"corpus {args.corpus}: {len(ncts)} NCTs, {len(todo)} to fetch")

    stopped = 0
    for i, nct in enumerate(todo):
        res = fetch(nct)
        if res is None:
            print(f"  {nct}: fetch failed, skipping")
            continue
        status, why = res
        is_stopped = any(s in status.upper()
                         for s in ("TERMINATED", "WITHDRAWN", "SUSPENDED"))
        if args.only_stopped and not is_stopped:
            cache.pop(nct, None)
        else:
            cache[nct] = {"overall_status": status, "why_stopped": why}
        stopped += int(is_stopped and bool(why))
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(todo)} fetched ({stopped} stopped-with-reason)", flush=True)
        time.sleep(0.15)

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(cache, indent=2))
    print(f"wrote {CACHE}: {len(cache)} entries "
          f"({stopped} early-stopped with a documented reason this run)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
