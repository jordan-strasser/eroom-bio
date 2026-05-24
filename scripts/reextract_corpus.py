"""Re-extract a corpus with the current extraction prompt (A.0b).

Forces re-extraction (the extractor skips trials whose ``*_extraction.json``
cache exists) by backing up + deleting those caches first, then re-running
extract. Classifications are KEPT — the A.0b per-chain descriptions are additive
metadata the classifier doesn't read, so edge logic is unchanged. Reports
per-chain description coverage and real token cost.

Usage:
    python -m scripts.reextract_corpus --corpus multi_indication_52_train
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
from pathlib import Path

import anthropic

from scripts.build_graph import extract_all, fetch_trials_by_ids
from src.annotation.extractor import Extractor

ANN = Path("data/annotations")
_usage = {"in": 0, "out": 0, "calls": 0}


def _wrap_usage(client: anthropic.AsyncAnthropic) -> None:
    orig = client.messages.create

    async def wrapped(*a, **k):
        resp = await orig(*a, **k)
        u = getattr(resp, "usage", None)
        if u is not None:
            _usage["in"] += getattr(u, "input_tokens", 0)
            _usage["out"] += getattr(u, "output_tokens", 0)
            _usage["calls"] += 1
        return resp

    try:
        client.messages.create = wrapped
    except Exception:
        pass


async def main() -> int:
    ap = argparse.ArgumentParser(description="Re-extract a corpus (A.0b).")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--concurrency", type=int, default=5)
    args = ap.parse_args()

    corpus_path = Path(f"data/corpora/{args.corpus}.txt")
    ncts = [
        ln.strip()
        for ln in corpus_path.read_text().splitlines()
        if ln.strip().startswith("NCT")
    ]
    print(f"corpus {args.corpus}: {len(ncts)} trials")

    backup = Path("/tmp/reextract_backup")
    backup.mkdir(parents=True, exist_ok=True)
    deleted = 0
    for nct in ncts:
        f = ANN / f"{nct}_extraction.json"
        if f.exists():
            shutil.copy(f, backup / f.name)
            f.unlink()
            deleted += 1
    print(f"backed up + deleted {deleted} extraction caches -> {backup} (classifications kept)")

    trials = await fetch_trials_by_ids(ncts)
    print(f"fetched {len(trials)} trial records")
    client = anthropic.AsyncAnthropic(timeout=90.0)
    _wrap_usage(client)
    extractor = Extractor(client)
    extracted = await extract_all(trials, extractor, concurrency=args.concurrency)
    print(f"re-extracted {len(extracted)}/{len(trials)} trials")

    cov = tot = 0
    for nct in ncts:
        f = ANN / f"{nct}_extraction.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        for cr in d.get("results_by_chain", []):
            tot += 1
            if (cr.get("mechanism_description") or "").strip():
                cov += 1
    print(f"per-chain mechanism_description coverage: {cov}/{tot} chains")

    cost = _usage["in"] / 1e6 * 3 + _usage["out"] / 1e6 * 15
    print(f"cost: calls={_usage['calls']} in={_usage['in']} out={_usage['out']} ~${cost:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
