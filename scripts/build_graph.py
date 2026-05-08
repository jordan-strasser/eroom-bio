"""End-to-end graph rebuild: fetch → populate → annotate → attribute.

Single fetch, four phases:
  1. Pull trials from ClinicalTrials.gov (one network call, reused below).
  2. PopulationPipeline.populate_oncology—initial graph + skeleton
     trial subgraphs.
  3. Extractor + Classifier—write per-trial annotations to
     data/annotations/.
  4. attributor._main—apply efficacy + AE updates and save the
     final snapshot.

Default: melanoma, n=10, with-results only. The four-step pipeline is
``ingest → annotate → update → query``; this driver covers the first
three so a single command rebuilds the graph end-to-end.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from rich.console import Console

# Load .env at the repo root so CLUE_API_KEY (and any other secrets) reach
# the subprocess; without this LINCS silently skips even when the key is set.
load_dotenv()

from src.annotation.attributor import _main as attributor_main
from src.annotation.classifier import Classifier
from src.annotation.extractor import Extractor
from src.graph.populate import (
    PopulationPipeline,
    seed_responds_differently_from_extractions,
)
from src.graph.store import GraphStore
from src.ingestion.clinicaltrials import ClinicalTrialsClient, TrialRecord

logger = logging.getLogger(__name__)
console = Console()

EXPORTS_DIR = Path("data/exports")
ANNOTATIONS_DIR = Path("data/annotations")
CORPORA_DIR = Path("data/corpora")


def load_corpus(path: Path) -> list[str]:
    """Read a frozen NCT-id corpus file. Lines starting with ``#`` and
    blank lines are ignored. Order is preserved (insertion order matters
    for any downstream determinism)."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line in seen:
            continue
        seen.add(line)
        out.append(line)
    return out


def save_corpus(path: Path, trials: list[TrialRecord]) -> None:
    """Write a corpus file: one NCT id per line, deterministic, with a
    short header recording how it was built."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Frozen trial corpus for the Eroom Bio graph pipeline.",
        "# Each line is one NCT id. Build_graph reads this file when the path",
        "# is passed via --corpus and skips the CT.gov search query.",
        "",
    ]
    lines.extend(t.nct_id for t in trials)
    path.write_text("\n".join(lines) + "\n")


def wipe_outputs(area: str, keep_annotations: bool = False) -> None:
    for path in EXPORTS_DIR.glob(f"{area}_*.json"):
        path.unlink()
        console.print(f"  removed {path}")
    if not keep_annotations and ANNOTATIONS_DIR.exists():
        for path in ANNOTATIONS_DIR.glob("*.json"):
            path.unlink()
        console.print(f"  cleared {ANNOTATIONS_DIR}/")
    elif keep_annotations:
        console.print(f"  kept {ANNOTATIONS_DIR}/ (--keep-annotations)")
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)


async def fetch_trials(
    condition: str,
    max_trials: int,
    include_terminated: bool,
    corpus_path: Path | None = None,
    corpus_concurrency: int = 8,
) -> list[TrialRecord]:
    """Fetch trials from CT.gov, with optional frozen-corpus support.

    If ``corpus_path`` is given and the file exists, fetch each listed
    NCT id individually via ``ClinicalTrialsClient.get_study``—this is
    the reproducible path. CT.gov ranking drift between runs can shuffle
    which trials a query returns, so a fresh search query at the same
    ``--max-trials`` may produce a different cohort. Pinning the corpus
    avoids that.

    If the corpus path is given but the file does NOT exist yet, run the
    standard search and persist the resulting NCT list to the file. The
    next run with the same path will then be reproducible.
    """
    ct = ClinicalTrialsClient()

    # Reproducible path: load IDs from frozen corpus, fetch each by id.
    if corpus_path is not None and corpus_path.exists():
        nct_ids = load_corpus(corpus_path)
        console.print(
            f"  loading frozen corpus from {corpus_path} ({len(nct_ids)} ids)"
        )
        sem = asyncio.Semaphore(corpus_concurrency)

        async def _one(nct_id: str) -> TrialRecord | None:
            async with sem:
                try:
                    return await ct.get_study(nct_id)
                except Exception:
                    logger.warning(
                        "Frozen-corpus fetch failed for %s", nct_id, exc_info=True,
                    )
                    return None

        results = await asyncio.gather(*(_one(n) for n in nct_ids))
        trials = [t for t in results if t is not None]
        console.print(
            f"  loaded {len(trials)}/{len(nct_ids)} from frozen corpus"
        )
        if max_trials and len(trials) > max_trials:
            trials = trials[:max_trials]
        return trials

    # Search path: query CT.gov as usual, then persist the result so the
    # next run with the same --corpus is reproducible.
    trials = await ct.fetch_oncology_with_results(
        max_results=max_trials, condition=condition,
    )
    console.print(f"  with-results: {len(trials)}")
    if include_terminated:
        terminated = await ct.fetch_oncology_terminated_with_reason(
            max_results=max_trials, condition=condition,
        )
        seen = {t.nct_id for t in trials}
        added = [t for t in terminated if t.nct_id not in seen]
        trials.extend(added)
        console.print(f"  +terminated: {len(added)}")
    if corpus_path is not None:
        save_corpus(corpus_path, trials)
        console.print(f"  saved corpus to {corpus_path}")
    return trials


async def extract_all(
    trials: list[TrialRecord],
    extractor: Extractor,
    concurrency: int = 5,
) -> list[TrialRecord]:
    """Run Extractor on each trial concurrently. Returns trials whose
    extraction succeeded—classification is split into a separate phase
    so the seeder can populate subgroup populations + entity context
    between them.
    """
    sem = asyncio.Semaphore(concurrency)

    async def _one(trial: TrialRecord) -> bool:
        async with sem:
            try:
                console.print(f"  [cyan]{trial.nct_id}[/cyan] extracting…")
                await extractor.extract(trial)
                console.print(f"  [cyan]{trial.nct_id}[/cyan] extracted")
                return True
            except Exception:
                logger.error("extract failed for %s", trial.nct_id, exc_info=True)
                console.print(f"  [red]{trial.nct_id}[/red] extract failed")
                return False

    results = await asyncio.gather(*(_one(t) for t in trials))
    return [t for t, ok in zip(trials, results) if ok]


async def classify_all(
    trials: list[TrialRecord],
    extractor: Extractor,
    classifier: Classifier,
    graph: GraphStore,
    concurrency: int = 5,
) -> int:
    """Run Classifier on each trial with the trial's graph entities as
    prompt context. Looks up TrialSubgraph from the (already-seeded +
    forked) graph so the LLM sees subgroup populations created in step 3b.
    """
    sem = asyncio.Semaphore(concurrency)

    async def _one(trial: TrialRecord) -> bool:
        async with sem:
            try:
                extraction = await extractor.extract(trial)  # cache hit
                try:
                    ts = graph.get_trial_subgraph_by_id(trial.nct_id)
                except KeyError:
                    ts = None
                console.print(f"  [cyan]{trial.nct_id}[/cyan] classifying…")
                await classifier.classify(
                    extraction, graph=graph, trial_subgraph=ts,
                )
                console.print(f"  [green]{trial.nct_id}[/green] classified")
                return True
            except Exception:
                logger.error("classify failed for %s", trial.nct_id, exc_info=True)
                console.print(f"  [red]{trial.nct_id}[/red] classify failed")
                return False

    results = await asyncio.gather(*(_one(t) for t in trials))
    return sum(results)


async def main(
    condition: str,
    max_trials: int,
    include_terminated: bool,
    concurrency: int,
    area: str,
    keep_annotations: bool = False,
    corpus: str | None = None,
) -> None:
    console.rule(
        f"[bold]Rebuilding graph: condition={condition!r}, n={max_trials}"
        + (f", corpus={corpus}" if corpus else "")
        + "[/bold]"
    )

    console.print("[bold]Step 0:[/bold] wiping prior outputs")
    wipe_outputs(area, keep_annotations=keep_annotations)

    initial_path = EXPORTS_DIR / f"{area}_initial.json"
    annotated_path = EXPORTS_DIR / f"{area}_annotated.json"
    corpus_path = (CORPORA_DIR / f"{corpus}.txt") if corpus else None

    console.rule("[bold]Step 1: fetch trials[/bold]")
    trials = await fetch_trials(
        condition, max_trials, include_terminated, corpus_path=corpus_path,
    )
    if not trials:
        console.print("[red]No trials fetched. Aborting.[/red]")
        return
    console.print(f"  total: {len(trials)} trials")
    for t in trials:
        console.print(f"    {t.nct_id}: {t.title[:80]}")

    console.rule("[bold]Step 2: populate (initial graph)[/bold]")
    graph = GraphStore()
    client = anthropic.AsyncAnthropic(timeout=60.0)
    pipeline = PopulationPipeline(graph, anthropic_client=client)
    await pipeline.populate_oncology(
        max_trials=max_trials,
        include_terminated_no_results=include_terminated,
        condition=condition,
        trials=trials,
    )
    graph.export_snapshot(str(initial_path))
    console.print(f"  wrote {initial_path}")

    console.rule("[bold]Step 3a: extract[/bold]")
    extractor = Extractor(client)
    classifier = Classifier(client)
    extracted = await extract_all(trials, extractor, concurrency=concurrency)
    console.print(f"  extracted {len(extracted)}/{len(trials)} trials")

    # Step 3b: seed subgroup populations + responds_differently edges and
    # fork chains. Has to run BEFORE classification so the LLM sees the
    # subgroup PopulationNodes in its entity-context block.
    console.rule("[bold]Step 3b: seed populations + fork chains[/bold]")
    seed_graph = GraphStore()
    seed_graph.import_snapshot(str(initial_path))
    rd_added, chains_added = seed_responds_differently_from_extractions(
        seed_graph, ANNOTATIONS_DIR,
    )
    seed_graph.export_snapshot(str(initial_path))
    console.print(
        f"  seeded {rd_added} responds_differently edges, "
        f"forked {chains_added} subgroup chains"
    )

    # Step 3c: classify each trial against the seeded graph so the
    # classifier prompt can ground edge updates in canonical node IDs.
    console.rule("[bold]Step 3c: classify (with entity context)[/bold]")
    n_classified = await classify_all(
        extracted, extractor, classifier, seed_graph, concurrency=concurrency,
    )
    console.print(f"  classified {n_classified}/{len(extracted)} trials")

    console.rule("[bold]Step 4: attribute[/bold]")
    await attributor_main(
        str(ANNOTATIONS_DIR), str(initial_path), str(annotated_path),
    )

    final = GraphStore()
    final.import_snapshot(str(annotated_path))
    stats = final.stats()
    console.rule("[bold green]Done[/bold green]")
    console.print(f"Final snapshot: {annotated_path}")
    console.print(f"  nodes={stats['node_count']} edges={stats['edge_count']}")
    console.print(f"  node types: {stats['node_types']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rebuild the Eroom Bio graph end-to-end.",
    )
    parser.add_argument(
        "--condition", default="melanoma",
        help="ClinicalTrials.gov condition filter (default: melanoma)",
    )
    parser.add_argument(
        "--max-trials", type=int, default=10,
        help="Max trials to fetch (default: 10)",
    )
    parser.add_argument(
        "--include-terminated", action="store_true",
        help="Also pull terminated trials with why_stopped",
    )
    parser.add_argument(
        "--concurrency", type=int, default=5,
        help="Concurrent extract+classify calls (default: 5)",
    )
    parser.add_argument(
        "--area", default="oncology",
        help="Snapshot filename prefix (default: oncology)",
    )
    parser.add_argument(
        "--keep-annotations", action="store_true",
        help="Skip wiping data/annotations/. Reuses cached extract+classify "
             "results from a prior run—useful when iterating on the "
             "attribute step.",
    )
    parser.add_argument(
        "--corpus", default=None,
        help="Frozen corpus name. Reads/writes data/corpora/<name>.txt—"
             "if the file exists, fetches the listed NCT ids by id "
             "instead of issuing a CT.gov search query (reproducible). "
             "If absent, runs the standard search and saves the result "
             "for future runs.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    asyncio.run(main(
        condition=args.condition,
        max_trials=args.max_trials,
        include_terminated=args.include_terminated,
        keep_annotations=args.keep_annotations,
        concurrency=args.concurrency,
        area=args.area,
        corpus=args.corpus,
    ))
