"""End-to-end graph rebuild: fetch → populate → annotate → attribute.

Single fetch, four phases:
  1. Pull trials from ClinicalTrials.gov (one network call, reused below).
  2. PopulationPipeline.populate_trials—initial graph + skeleton
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
    attach_node_descriptions_from_extractions,
    seed_responds_differently_from_extractions,
)
from src.graph.store import GraphStore
from src.ingestion.clinicaltrials import ClinicalTrialsClient, TrialRecord

logger = logging.getLogger(__name__)
console = Console()

EXPORTS_DIR = Path("data/exports")
ANNOTATIONS_DIR = Path("data/annotations")
CORPORA_DIR = Path("data/corpora")


def _chain_coverage(graph: GraphStore) -> dict[str, int | float]:
    """Count how many of each trial's chains got at least one edge update.

    A chain is "touched" when any of its 6 backbone edges (binds_to,
    modulates_via, mechanism_affects, biology_drives, reflects_biology,
    endpoint_captures) has an EvidenceRecord with the trial's id.
    A trial with chains-built-but-zero-touched is a silent failure —
    populated correctly but the classifier never emitted an edge that
    routed there. Worth tracking run-over-run because it's the single
    cleanest signal that classifier prompts and node canonicalization
    are in sync.
    """
    from collections import defaultdict

    edge_evidence_trials: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for src, tgt, key, data in graph._graph.edges(keys=True, data=True):  # noqa: SLF001
        belief = data.get("belief") or {}
        if hasattr(belief, "evidence"):  # EdgeBeliefState instance
            evidence = belief.evidence
        else:  # plain dict from snapshot
            evidence = belief.get("evidence", []) if isinstance(belief, dict) else []
        for ev in evidence:
            sid = ev.get("source_id") if isinstance(ev, dict) else getattr(ev, "source_id", None)
            if sid:
                edge_evidence_trials[(src, tgt, key)].add(sid)

    chains_total = chains_touched = 0
    trials_full = trials_partial = trials_zero = trials_with_chains = 0
    for trial_id, sub in graph.trial_subgraphs.items():
        chains = sub.chains
        if not chains:
            continue
        trials_with_chains += 1
        touched_in_trial = 0
        for c in chains:
            chains_total += 1
            backbone = [
                (c.compound_id, c.target_id, "binds_to"),
                (c.target_id, c.mechanism_id, "modulates_via"),
                (c.mechanism_id, c.biology_id, "mechanism_affects"),
                (c.biology_id, c.indication_id, "biology_drives"),
                (c.biology_id, c.endpoint_id, "reflects_biology"),
                (c.endpoint_id, c.indication_id, "endpoint_captures"),
            ]
            if any(trial_id in edge_evidence_trials.get(k, set()) for k in backbone):
                chains_touched += 1
                touched_in_trial += 1
        if touched_in_trial == 0:
            trials_zero += 1
        elif touched_in_trial == len(chains):
            trials_full += 1
        else:
            trials_partial += 1

    return {
        "chains_total": chains_total,
        "chains_touched": chains_touched,
        "chain_pct": 100.0 * chains_touched / chains_total if chains_total else 0.0,
        "trials_with_chains": trials_with_chains,
        "trials_full": trials_full,
        "trials_partial": trials_partial,
        "trials_zero": trials_zero,
    }


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
    corpus_concurrency: int = 4,
    include_ncts: list[str] | None = None,
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
        if include_ncts:
            corpus_set = set(nct_ids)
            missing = [n for n in include_ncts if n not in corpus_set]
            if missing:
                raise SystemExit(
                    f"--include lists {len(missing)} NCT id(s) not in corpus "
                    f"{corpus_path.name}: {missing}"
                )
            if max_trials and len(include_ncts) > max_trials:
                raise SystemExit(
                    f"--include lists {len(include_ncts)} NCT ids but "
                    f"--max-trials is {max_trials}; raise --max-trials or "
                    f"shorten --include."
                )
            # Place included ids first (preserving include order), then the
            # rest of the corpus in its original order. The downstream
            # truncation to max_trials is then guaranteed to retain the
            # included set.
            include_order = {n: i for i, n in enumerate(include_ncts)}
            nct_ids = sorted(
                nct_ids,
                key=lambda n: (
                    0 if n in include_order else 1,
                    include_order.get(n, 0),
                ),
            )
            console.print(
                f"  --include pinned {len(include_ncts)} NCT ids to the head of the slice"
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


async def fetch_trials_by_ids(
    nct_ids: list[str],
    corpus_concurrency: int = 4,
) -> list[TrialRecord]:
    """Fetch a specific set of trials by NCT id (round-19 incremental add).

    Mirrors the per-id fetch path inside ``fetch_trials`` (frozen-corpus
    branch) but without the corpus file dance—the caller already knows
    exactly which ids to add to the existing snapshot.
    """
    if not nct_ids:
        return []
    ct = ClinicalTrialsClient()
    sem = asyncio.Semaphore(corpus_concurrency)

    async def _one(nct_id: str) -> TrialRecord | None:
        async with sem:
            try:
                return await ct.get_study(nct_id)
            except Exception:
                logger.warning(
                    "Incremental fetch failed for %s", nct_id, exc_info=True,
                )
                return None

    results = await asyncio.gather(*(_one(n) for n in nct_ids))
    trials = [t for t in results if t is not None]
    console.print(f"  fetched {len(trials)}/{len(nct_ids)} by NCT id")
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
    reannotate: list[str] | None = None,
    corpus: str | None = None,
    include_ncts: list[str] | None = None,
    min_classify_success_rate: float = 0.80,
    allow_partial_classify: bool = False,
    base_snapshot: str | None = None,
    add_trials: list[str] | None = None,
    add_corpus: str | None = None,
    min_subgraph_success_rate: float = 0.75,
    allow_partial_subgraphs: bool = False,
    exclude_from_attribution: list[str] | None = None,
    assemble: bool = True,
    enrich_pubmed: bool = False,
    bottom_up: bool = True,
) -> None:
    incremental = bool(base_snapshot)
    reannotate = reannotate or []
    if reannotate:                       # surgical re-annotation: pin + preserve others
        keep_annotations = True
        include_ncts = sorted(set((include_ncts or []) + reannotate))
        console.print(
            f"[bold]Surgical re-annotation:[/bold] re-running extract+classify for "
            f"{len(reannotate)} trial(s); all other annotations preserved"
        )
    if incremental:
        console.rule(
            f"[bold]Incremental build: base={base_snapshot}[/bold]"
        )
    else:
        console.rule(
            f"[bold]Rebuilding graph: condition={condition!r}, n={max_trials}"
            + (f", corpus={corpus}" if corpus else "")
            + "[/bold]"
        )

    initial_path = EXPORTS_DIR / f"{area}_initial.json"
    annotated_path = EXPORTS_DIR / f"{area}_annotated.json"
    corpus_path = (CORPORA_DIR / f"{corpus}.txt") if corpus else None
    base_snapshot_path = Path(base_snapshot) if base_snapshot else None

    # Round-19 incremental-mode validation. Bad combos must raise BEFORE
    # any wipe / fetch / network call, otherwise a typo could nuke
    # data/annotations/ on its way to SystemExit.
    if incremental:
        assert base_snapshot_path is not None
        if not base_snapshot_path.exists():
            raise SystemExit(
                f"--base-snapshot file not found: {base_snapshot}"
            )
        if not (add_trials or add_corpus):
            raise SystemExit(
                "--base-snapshot requires --add-trials or --add-corpus "
                "(nothing to add otherwise)."
            )
        if add_trials and add_corpus:
            raise SystemExit(
                "--add-trials and --add-corpus are mutually exclusive."
            )
        if corpus is not None:
            raise SystemExit(
                "--corpus cannot be combined with --base-snapshot; "
                "use --add-corpus to add a new corpus to the base."
            )
        if include_ncts:
            raise SystemExit(
                "--include is for fresh builds with --corpus; in "
                "incremental mode, list the NCTs via --add-trials directly."
            )
    else:
        if add_trials or add_corpus:
            raise SystemExit(
                "--add-trials / --add-corpus require --base-snapshot."
            )

    # Bottom-up incremental append (the 200K seam): build_bottomup starts from the
    # loaded base graph, builds Phase 1 ONLY for the new trials, unions them in, and
    # re-runs the Phase-2 merge restricted to new-involving pairs. No raise here —
    # this is the production append path. (Top-down incremental still works as the
    # escape hatch.)

    # Validate CLI inputs BEFORE wiping anything — otherwise a bad flag
    # combo nukes data/annotations/ on its way to the SystemExit.
    if include_ncts and corpus_path is None:
        raise SystemExit("--include requires --corpus (frozen-corpus path only).")
    if include_ncts and corpus_path is not None and corpus_path.exists():
        corpus_ids = set(load_corpus(corpus_path))
        missing = [n for n in include_ncts if n not in corpus_ids]
        if missing:
            raise SystemExit(
                f"--include lists {len(missing)} NCT id(s) not in corpus "
                f"{corpus_path.name}: {missing}"
            )
        if max_trials and len(include_ncts) > max_trials:
            raise SystemExit(
                f"--include lists {len(include_ncts)} NCT ids but "
                f"--max-trials is {max_trials}; raise --max-trials or "
                f"shorten --include."
            )

    if incremental:
        # Round-19: do NOT wipe in incremental mode. The base snapshot
        # and the cached annotations both stay so already-handled
        # trials keep their state; new trials add on top.
        console.print(
            "[bold]Step 0:[/bold] incremental mode — preserving prior outputs"
        )
        EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
        ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    else:
        console.print("[bold]Step 0:[/bold] wiping prior outputs")
        wipe_outputs(area, keep_annotations=keep_annotations)
        for _nct in reannotate:          # surgical: drop only these trials' caches
            for _suffix in ("extraction", "classification"):
                _p = ANNOTATIONS_DIR / f"{_nct}_{_suffix}.json"
                if _p.exists():
                    _p.unlink()
                    console.print(f"  re-annotate: cleared {_p.name} (will re-run)")

    console.rule("[bold]Step 1: fetch trials[/bold]")
    if incremental:
        if add_trials:
            target_ncts = list(add_trials)
        else:
            assert add_corpus is not None
            add_corpus_path = CORPORA_DIR / f"{add_corpus}.txt"
            if not add_corpus_path.exists():
                raise SystemExit(
                    f"--add-corpus file not found: {add_corpus_path}"
                )
            target_ncts = load_corpus(add_corpus_path)
        console.print(f"  fetching {len(target_ncts)} NCT id(s) by id")
        trials = await fetch_trials_by_ids(target_ncts)
    else:
        trials = await fetch_trials(
            condition, max_trials, include_terminated, corpus_path=corpus_path,
            include_ncts=include_ncts,
        )
    if not trials:
        console.print("[red]No trials fetched. Aborting.[/red]")
        return
    console.print(f"  total: {len(trials)} trials")
    for t in trials:
        console.print(f"    {t.nct_id}: {t.title[:80]}")

    console.rule("[bold]Step 2: populate (initial graph)[/bold]")
    graph = GraphStore()
    if incremental:
        assert base_snapshot_path is not None
        graph.import_snapshot(str(base_snapshot_path))
        console.print(
            f"  loaded base snapshot from {base_snapshot_path}: "
            f"{len(graph.trial_subgraphs)} existing trial subgraphs, "
            f"{len(graph.applied_attribution_trial_ids)} already attributed"
        )
        # Skip trials whose subgraph already exists in the base — they've
        # been canonicalized + populated before, and the populator's LLM
        # calls (indication canonicalization, population features) are
        # the costly part. add_node / add_edge are idempotent so anything
        # that slipped through this filter would still be correct, just
        # wasteful.
        new_trials = [
            t for t in trials if t.nct_id not in graph.trial_subgraphs
        ]
        skipped = len(trials) - len(new_trials)
        if skipped:
            console.print(
                f"  skipping {skipped} trials whose subgraph is already "
                f"in the base snapshot"
            )
        trials = new_trials
        if not trials:
            console.print(
                "[yellow]All requested trials already in base snapshot; "
                "nothing to add.[/yellow]"
            )
            return
    client = anthropic.AsyncAnthropic(timeout=60.0)

    # Step 3a: extract — RUNS BEFORE POPULATE (semantic-layers redesign).
    # Extraction has no graph dependency (extract_all needs only the trial +
    # extractor); only classify needs the seeded graph, and it still runs after
    # populate. Doing extract first lets populate define the Mechanism / Biology
    # nodes FROM each trial's extracted descriptions (the description is the
    # node's semantic identity + BioLORD merge substrate) instead of a per-trial
    # Reactome lookup that resolves inconsistently in the isolated bottom-up
    # build and falls back to a low-evidence slug. The extractor + classifier
    # built here stay in scope for the classify step (3c) below.
    console.rule("[bold]Step 3a: extract (before populate)[/bold]")
    extractor = Extractor(client, enrich_pubmed=enrich_pubmed)
    classifier = Classifier(client)
    extracted = await extract_all(trials, extractor, concurrency=concurrency)
    console.print(f"  extracted {len(extracted)}/{len(trials)} trials")

    if bottom_up:
        # Chains-first build: resolve each trial in ISOLATION, then reassemble via
        # the re-runnable node_merge projection (vs top-down's overlap-first shared
        # store). Faithful to top-down on n=10 — chains 61==61 (0 missing, 0
        # splits), belief coverage 205/258 vs 203/257. See populate_bottomup.
        # annotations_dir lets per-trial populate read the just-written
        # extractions so Biology nodes get their identity from the description.
        from src.graph.populate_bottomup import build_bottomup
        graph = await build_bottomup(
            trials, client, condition=condition,
            annotations_dir=str(ANNOTATIONS_DIR),
            # Incremental append: start from the loaded base graph and merge ONLY
            # the new trials' chains into it (Phase 1 just for `trials`, which the
            # round-19 filter already trimmed to the not-yet-present ones). Fresh
            # build: base_graph=None → empty start. Either way the same Phase-2
            # merge runs (restricted to new-involving pairs on append).
            base_graph=graph if incremental else None,
            # Dump the Phase-1 union (pre-assemble) so the visualizer's
            # before-block is FAITHFUL: each chain still references its own
            # per-trial biology/target instance (the merge destroys that
            # mapping, so it can't be reconstructed post-hoc). Additive only.
            premerge_dump_path=str(EXPORTS_DIR / f"{area}_premerge.json"),
        )
        console.print(
            f"  [bold]bottom-up (chains-first) build[/bold]: "
            f"{graph.stats()['node_count']} nodes, "
            f"{len(graph.trial_subgraphs)} trial subgraphs"
            + ("  (incremental append)" if incremental else "")
        )
    else:
        # Top-down (overlap-first) stays on LEGACY ontology biology: it writes
        # straight into a shared store with no Phase-2 geometric merge, so
        # description-identity biology would fragment by paraphrasing here with
        # nothing to consolidate it. Description-identity is bottom-up-only
        # (it needs the merge). So we do NOT thread annotations_dir here.
        pipeline = PopulationPipeline(graph, anthropic_client=client)
        await pipeline.populate_trials(
            max_trials=max_trials,
            include_terminated_no_results=include_terminated,
            condition=condition,
            trials=trials,
        )

    # Phase-4 EFO indication tree: connect ALL→leukemia, uveal→melanoma, … on
    # the assembled graph so the prediction's indication backoff can pool a
    # leaf-anchored chain onto its parent disease. Additive + best-effort
    # (a failed EFO lookup just skips that disease).
    try:
        from src.graph.populate import link_indication_subtypes_via_efo
        from src.ingestion.opentargets import OpenTargetsClient
        n_sub = await link_indication_subtypes_via_efo(graph, OpenTargetsClient())
        console.print(f"  EFO indication hierarchy: +{n_sub} SUBTYPE_OF edges")
    except Exception:  # noqa: BLE001
        logger.debug("EFO indication linking skipped", exc_info=True)

    # Round-20.5 / round-21 followup: silent-drop guard.
    # build_trial_subgraphs skips a trial when its indication / endpoint
    # / arm structure can't be resolved. Without this check, 14 of 50
    # trials in the n=50 multi_indication build were lost silently — the
    # snapshot LOOKED complete but the chains weren't there.
    #
    # Refinement: NOT every drop reason represents silent loss. Drops
    # whose reason matches one of LEGITIMATE_DROP_REASONS are correct
    # rejections (non-therapeutic trials, fundamentally empty CT.gov
    # records). Counting them against the success rate falsely makes a
    # clean build look broken, and ratchets the threshold toward "no
    # search-result corpus will ever pass." Exclude them from the
    # denominator: success rate is computed over trials that COULD have
    # produced a chain, not the raw input count.
    LEGITIMATE_DROP_REASONS = {
        # All interventions are non-drug (diagnostic, behavioral,
        # device, procedure). Round 22's slug-create can't help — no
        # compound to anchor a chain.
        "no_arms_filtered_by_diagnostic",
        # CT.gov returned a trial with no conditions at all (round 22
        # boundary). Can't construct an indication chain side.
        "no_conditions",
        # CT.gov returned no primary outcomes (round 22 boundary).
        # Can't construct an endpoint chain side.
        "no_primary_outcomes",
    }

    from src.graph.populate import _DROPPED_TRIAL_SUBGRAPHS_LOG
    legitimate_drops: set[str] = set()
    if _DROPPED_TRIAL_SUBGRAPHS_LOG.exists():
        import json as _json
        for _line in _DROPPED_TRIAL_SUBGRAPHS_LOG.read_text().splitlines():
            if not _line.strip():
                continue
            try:
                _entry = _json.loads(_line)
            except _json.JSONDecodeError:
                continue
            if _entry.get("reason") in LEGITIMATE_DROP_REASONS:
                legitimate_drops.add(_entry.get("nct_id"))

    built_subgraphs = sum(
        1 for t in trials if t.nct_id in graph.trial_subgraphs
    )
    eligible_count = len(trials) - len(legitimate_drops)
    subgraph_success_rate = (
        built_subgraphs / eligible_count if eligible_count else 1.0
    )
    if (
        subgraph_success_rate < min_subgraph_success_rate
        and not allow_partial_subgraphs
    ):
        raise SystemExit(
            f"\ntrial subgraph build success rate "
            f"{subgraph_success_rate:.1%} below minimum "
            f"{min_subgraph_success_rate:.1%} "
            f"({built_subgraphs} of {eligible_count} eligible trials "
            f"produced a trial subgraph; "
            f"{len(legitimate_drops)} non-therapeutic trials "
            f"excluded from the denominator). Aborting before "
            f"extraction to avoid a silently-truncated snapshot. "
            f"See {_DROPPED_TRIAL_SUBGRAPHS_LOG} for per-trial drop "
            f"reasons. Bug-indicating reasons (no_indication, "
            f"no_endpoint, no_arms_empty_arm_groups, no_arms) point "
            f"to populator gaps to fix. Re-run with "
            f"--allow-partial-subgraphs to force the build to "
            f"continue with the partial set."
        )
    console.print(
        f"  built {built_subgraphs}/{eligible_count} trial subgraphs "
        f"({subgraph_success_rate:.1%}); "
        f"{len(legitimate_drops)} non-therapeutic trials excluded "
        f"from the denominator"
    )

    graph.export_snapshot(str(initial_path))
    console.print(f"  wrote {initial_path}")

    # Step 3b: seed subgroup populations + responds_differently edges and
    # fork chains. Has to run BEFORE classification so the LLM sees the
    # subgroup PopulationNodes in its entity-context block.
    console.rule("[bold]Step 3b: seed populations + fork chains[/bold]")
    seed_graph = GraphStore()
    seed_graph.import_snapshot(str(initial_path))
    rd_added, chains_added = await seed_responds_differently_from_extractions(
        seed_graph, ANNOTATIONS_DIR,
    )
    # A.0: preserve the trials' rich free-text descriptions onto the Mechanism
    # / Biology / Population nodes as the BioLORD embedding substrate (A.1).
    # Reads the same cached extractions — no new LLM call.
    desc_set = attach_node_descriptions_from_extractions(
        seed_graph, ANNOTATIONS_DIR,
    )
    seed_graph.export_snapshot(str(initial_path))
    console.print(
        f"  seeded {rd_added} responds_differently edges, "
        f"forked {chains_added} subgroup chains, "
        f"set {desc_set} node descriptions"
    )

    # Step 3b.5 (v2 / Q4): give EVERY node type a description + canonical
    # name_id, so the manifold-1 substrate (BioLORD embeddings / boxes / (s,t))
    # is meaningful on Target/Compound/Indication/Endpoint too — not just the
    # Mechanism/Biology/Population nodes that extractions describe. Haiku +
    # cache, so rebuilds are cheap. The geometric merge stays a separate,
    # re-runnable projection (scripts/build_groundup.py), not baked in here.
    console.rule("[bold]Step 3b.5: descriptions + name_id (all node types)[/bold]")
    from src.graph.descriptions import assign_name_ids, generate_node_descriptions
    gen_desc = await generate_node_descriptions(
        seed_graph, client, concurrency=concurrency,
    )
    n_name_ids = assign_name_ids(seed_graph)
    seed_graph.export_snapshot(str(initial_path))
    console.print(
        f"  generated {gen_desc} node descriptions (Haiku), "
        f"assigned {n_name_ids} name_ids"
    )

    # Step 3c: classify each trial against the seeded graph so the
    # classifier prompt can ground edge updates in canonical node IDs.
    console.rule("[bold]Step 3c: classify (with entity context)[/bold]")
    n_classified = await classify_all(
        extracted, extractor, classifier, seed_graph, concurrency=concurrency,
    )
    console.print(f"  classified {n_classified}/{len(extracted)} trials")

    # Round-16 safety: abort if a large fraction of trials failed to
    # classify. Without this check, an API outage / credit exhaustion
    # mid-run produces a silently-truncated snapshot that looks complete
    # but has ~half the trials' evidence missing. The downstream
    # attribution + prediction don't know to flag the gap.
    classify_success_rate = (
        n_classified / len(extracted) if extracted else 1.0
    )
    if (
        classify_success_rate < min_classify_success_rate
        and not allow_partial_classify
    ):
        raise SystemExit(
            f"\nclassify success rate {classify_success_rate:.1%} below "
            f"minimum {min_classify_success_rate:.1%} "
            f"({n_classified} of {len(extracted)} trials classified). "
            f"Aborting before attribution to avoid a degraded snapshot. "
            f"Common causes: API credit exhaustion (check the rebuild "
            f"log for 'credit balance is too low'), rate limit, or "
            f"network outage. Re-run after fixing, or pass "
            f"--allow-partial-classify to force the build to continue "
            f"with the partial classifications."
        )

    # Step 3d: prune invalid mechanisms — Reactome/GO entries that aren't a
    # cellular ACTION/process (therapeutic collections, disease modules,
    # pathogen-lifecycle pathways, receptor-family groupings, the 'other'
    # placeholder). Post-classify / pre-attribute, so a trial's outcome is never
    # credited through a non-mechanism. Deterministic name-gate, no LLM (see
    # src/graph/mechanism_validity.py). Off-target hits + sub-mechanisms — real
    # biology — are deliberately untouched.
    console.rule("[bold]Step 3d: prune invalid mechanisms[/bold]")
    from src.graph.mechanism_validity import prune_invalid_mechanisms
    prune_graph = GraphStore()
    prune_graph.import_snapshot(str(initial_path))
    pstats = prune_invalid_mechanisms(prune_graph)
    prune_graph.export_snapshot(str(initial_path))
    console.print(
        f"  dropped {pstats['nodes_dropped']} non-mechanism nodes "
        f"{pstats['by_tier']}, {pstats['chains_dropped']} chain variants; "
        f"{len(pstats['trials_emptied'])} trials left chain-less"
    )

    console.rule("[bold]Step 4: attribute[/bold]")
    await attributor_main(
        str(ANNOTATIONS_DIR), str(initial_path), str(annotated_path),
        exclude_from_attribution=exclude_from_attribution,
    )

    # Step 5 (--assemble): the v2 post-build geometry. The per-(s,t) belief field
    # and the box / is-a geometry are PRIVATE artifacts the boundary strips from
    # the public snapshot, so they're materialized post-hoc — and therefore drift
    # stale unless regenerated on every build (this is exactly how the field fell
    # behind at 2/7 while the code already did 7). Wiring the two former scripts
    # (assemble_v2 + materialize_belief_field) in here keeps the field in lockstep
    # with the graph. Off by default: adds BioLORD embedding compute and needs
    # EROOM_PRIVATE_ROOT. Node-merge stays a separate, re-runnable projection
    # (assemble_v2 --merge), deliberately NOT baked in here.
    if assemble:
        console.rule("[bold]Step 5: assemble geometry + materialize (s,t) field[/bold]")
        # Best-effort: steps 1-4 (incl. paid extractions) are already done and
        # written to annotated_path, so a step-5 failure (offline BioLORD, unset
        # private root, …) must NOT crash the build. The field is regenerable
        # (scripts/materialize_belief_field.py) and post-hoc by construction.
        try:
            from scripts.assemble_v2 import assemble_geometry
            from scripts.materialize_belief_field import materialize_field

            geo = assemble_geometry(str(annotated_path), annotations_dir=str(ANNOTATIONS_DIR))
            console.print(
                f"  geometry: {geo['boxes']} boxes, "
                f"is-a SUBTYPE_OF {geo['subtype_before']}→{geo['subtype_after']} "
                f"(+{geo['subtype_added']})"
            )
            fld = materialize_field(str(annotated_path), annotations_dir=str(ANNOTATIONS_DIR))
            console.print(
                f"  (s,t) field: {fld['edges_localized']} edges localized, "
                f"{fld['anchors_total']} anchors"
            )
            console.print(f"  private artifacts -> {geo['private_root']}")
        except Exception as exc:  # noqa: BLE001 — field is regenerable; don't lose the build
            console.print(
                f"  [red]Step 5 (geometry + (s,t) field) FAILED:[/red] {exc}\n"
                f"  [yellow]The graph snapshot is intact. Regenerate the field with:[/yellow]\n"
                f"  python -m scripts.materialize_belief_field {annotated_path}"
            )
            logger.warning("assemble/materialize step failed", exc_info=True)

    final = GraphStore()
    final.import_snapshot(str(annotated_path))
    stats = final.stats()
    coverage = _chain_coverage(final)
    console.rule("[bold green]Done[/bold green]")
    console.print(f"Final snapshot: {annotated_path}")
    console.print(f"  nodes={stats['node_count']} edges={stats['edge_count']}")
    console.print(f"  node types: {stats['node_types']}")
    console.print(
        f"  chain coverage: {coverage['chains_touched']}/{coverage['chains_total']} "
        f"chains ({coverage['chain_pct']:.0f}%) touched by ≥1 update from their own trial"
    )
    console.print(
        f"                  {coverage['trials_full']}/{coverage['trials_with_chains']} "
        f"trials full, {coverage['trials_partial']} partial, "
        f"[red]{coverage['trials_zero']} zero[/red]"
    )

    # Phantom-edge guard: every edge type the PREDICTION consumes must be
    # PRODUCED by the populator. A type consumed but instantiated by no producer
    # is a phantom (the reflects_biology gap — defined in the schema, walked by
    # the attributor, in the field EDGE_SPECS, but created by no populate method,
    # so it silently never carried a belief). Warn loudly if any backbone edge
    # type is absent from the built graph so the gap can't reopen unnoticed.
    from src.prediction.path_query import CONSUMED_BACKBONE_EDGE_TYPES
    present_edge_types = {k for *_, k in final._graph.edges(keys=True)}  # noqa: SLF001
    missing_backbone = sorted(
        et.value for et in CONSUMED_BACKBONE_EDGE_TYPES
        if et.value not in present_edge_types
    )
    if missing_backbone:
        console.print(
            f"  [red]⚠ PHANTOM EDGE(S): the prediction consumes {missing_backbone} "
            f"but the build produced ZERO of them[/red] — they will never carry a "
            f"belief or materialize into the (s,t) field. A producer is missing "
            f"(see path_query.CONSUMED_BACKBONE_EDGE_TYPES)."
        )
    else:
        console.print(
            f"  [green]backbone complete[/green]: all "
            f"{len(CONSUMED_BACKBONE_EDGE_TYPES)} consumed edge types are present"
        )


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
        "--reannotate", default="",
        help="Surgical re-annotation (append-only; NO global wipe): comma-"
             "separated NCT ids to RE-EXTRACT + RE-CLASSIFY under the current "
             "prompts. Deletes ONLY those trials' cached "
             "<nct>_{extraction,classification}.json and rebuilds them, leaving "
             "every other trial's annotations untouched. Implies "
             "--keep-annotations and pins the ids via --include (so requires "
             "--corpus). Replaces the destructive 'drop --keep-annotations → "
             "wipe 120 files to rebuild 10' pattern — the chains-first append-"
             "only way to iterate on prompts.",
    )
    parser.add_argument(
        "--corpus", default=None,
        help="Frozen corpus name. Reads/writes data/corpora/<name>.txt—"
             "if the file exists, fetches the listed NCT ids by id "
             "instead of issuing a CT.gov search query (reproducible). "
             "If absent, runs the standard search and saves the result "
             "for future runs.",
    )
    parser.add_argument(
        "--include", default="",
        help="Comma-separated NCT ids that must appear in the slice "
             "(e.g. the standard inspection set). The included ids are "
             "placed at the head of the corpus order so they survive the "
             "--max-trials cap. Requires --corpus and all ids must be "
             "present in the corpus file.",
    )
    parser.add_argument(
        "--min-classify-success-rate", type=float, default=0.80,
        help="Minimum fraction of trials that must classify successfully "
             "(default 0.80). Build aborts before attribution if below "
             "this, preventing a silently-truncated snapshot from API "
             "credit exhaustion / outages.",
    )
    parser.add_argument(
        "--allow-partial-classify", action="store_true",
        help="Override the min-classify-success-rate check and continue "
             "building from whatever subset classified. Use only when you "
             "explicitly want a partial snapshot for debugging.",
    )
    parser.add_argument(
        "--base-snapshot", default=None,
        help="Round-19 incremental mode. Path to an existing annotated "
             "snapshot to extend instead of rebuilding from scratch. "
             "Skips the Step 0 wipe and the populator only adds trials "
             "not already present. Pairs with --add-trials or "
             "--add-corpus; --corpus is rejected in this mode. "
             "The output annotated snapshot at "
             "data/exports/<area>_annotated.json overwrites the base "
             "when --area matches.",
    )
    parser.add_argument(
        "--add-trials", default="",
        help="Round-19 incremental mode. Comma-separated NCT ids to fetch "
             "and add on top of --base-snapshot. Mutually exclusive with "
             "--add-corpus.",
    )
    parser.add_argument(
        "--add-corpus", default=None,
        help="Round-19 incremental mode. Name of a corpus file under "
             "data/corpora/<name>.txt whose NCT ids will be fetched and "
             "added on top of --base-snapshot. Mutually exclusive with "
             "--add-trials.",
    )
    parser.add_argument(
        "--min-subgraph-success-rate", type=float, default=0.75,
        help="Round-20.5 silent-drop guard. Minimum fraction of "
             "ELIGIBLE trials that must produce a trial subgraph "
             "(default 0.75). The denominator excludes trials whose "
             "drop reason is a legitimate rejection "
             "(no_arms_filtered_by_diagnostic, no_conditions, "
             "no_primary_outcomes — these are non-therapeutic studies "
             "or fundamentally empty CT.gov records, not silent loss). "
             "Build aborts before extraction if the eligible-trial "
             "rate falls below this — catches the bug-indicating "
             "drops (no_indication, no_endpoint, "
             "no_arms_empty_arm_groups) that point to populator gaps. "
             "See data/dev/dropped_trial_subgraphs.jsonl for "
             "per-trial reasons.",
    )
    parser.add_argument(
        "--allow-partial-subgraphs", action="store_true",
        help="Override --min-subgraph-success-rate and continue building "
             "even when many trials silently failed to produce subgraphs. "
             "Use only when investigating the drop log itself.",
    )
    parser.add_argument(
        "--exclude-from-attribution", default="",
        help="Round-26 true-holdout flag. Comma-separated NCT ids whose "
             "subgraphs should be BUILT (steps 1-3 — fetch, populate, "
             "annotate) but whose evidence should NOT be folded into "
             "edge beliefs (step 4 attribute skipped). Use to make a "
             "true holdout: include the eval NCTs alongside the "
             "training corpus, then exclude them here so the resulting "
             "annotated.json has all subgraphs but only training "
             "attribution. Fixes the round-24 contamination bug where "
             "--add-trials re-ran attribution on holdouts.",
    )
    parser.add_argument(
        "--enrich-pubmed", action="store_true",
        help="Scale producer (the in-build 3rd call): for TERMINATED/WITHDRAWN "
             "trials with no posted AEs but >=1 reference PMID, fetch the linked "
             "abstract(s) via NCBI E-utilities and run a focused Anthropic call to "
             "extract structured safety (HR/CI/failure_causing), written to the "
             "<nct>_pubmed_safety.json the attribution step merges. Off by default "
             "(network + an extra LLM call for the subset). Grounds program-ending "
             "off-target toxicity (e.g. torcetrapib) the empty structured record "
             "never captured.",
    )
    parser.add_argument(
        "--bottom-up", action="store_true",
        help="(DEFAULT now; flag kept for back-compat, no-op) chains-first build: "
             "resolve each trial in isolation into trial-scoped nodes, then "
             "reassemble via the re-runnable node_merge projection. This is the "
             "only supported mode — append-only ingestion + retune any merge "
             "without a rebuild, and incremental --add-trials/--add-corpus appends "
             "new chains onto a base snapshot and re-runs the merge.",
    )
    parser.add_argument(
        "--top-down", action="store_true",
        help="Escape hatch (NOT recommended): the legacy overlap-first shared-store "
             "build. Kept only for the faithfulness comparison + tests. Bottom-up "
             "is the default and the production path.",
    )
    parser.add_argument(
        "--assemble", action=argparse.BooleanOptionalAction, default=True,
        help="Step 5: after attribution, run the v2 post-build geometry — fit "
             "boxes on all 7 chain node types, resolve the box-geometry is-a "
             "hierarchy (public SUBTYPE_OF edges), and materialize the per-(s,t) "
             "belief field. Boxes + field are PRIVATE artifacts written under "
             "EROOM_PRIVATE_ROOT (defaults to ~/.eroom/private; point it at the "
             "enterprise artifacts dir for canonical builds). ON by default — the "
             "(s,t) belief field is core to the product, so every build carries "
             "it in lockstep with the graph (it's post-hoc by construction and "
             "would otherwise drift stale). Pass --no-assemble for a fast debug "
             "build that skips the BioLORD embedding compute. Node-merge stays "
             "separate (assemble_v2 --merge).",
    )
    args = parser.parse_args()
    include_ncts = [n.strip() for n in args.include.split(",") if n.strip()]
    reannotate = [n.strip() for n in args.reannotate.split(",") if n.strip()]
    add_trials = [n.strip() for n in args.add_trials.split(",") if n.strip()]
    exclude_from_attribution = [
        n.strip() for n in args.exclude_from_attribution.split(",") if n.strip()
    ]

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    asyncio.run(main(
        condition=args.condition,
        max_trials=args.max_trials,
        include_terminated=args.include_terminated,
        keep_annotations=args.keep_annotations,
        reannotate=reannotate or None,
        concurrency=args.concurrency,
        area=args.area,
        corpus=args.corpus,
        include_ncts=include_ncts or None,
        min_classify_success_rate=args.min_classify_success_rate,
        allow_partial_classify=args.allow_partial_classify,
        base_snapshot=args.base_snapshot,
        add_trials=add_trials or None,
        add_corpus=args.add_corpus,
        min_subgraph_success_rate=args.min_subgraph_success_rate,
        allow_partial_subgraphs=args.allow_partial_subgraphs,
        exclude_from_attribution=exclude_from_attribution or None,
        assemble=args.assemble,
        enrich_pubmed=args.enrich_pubmed,
        bottom_up=not args.top_down,  # bottom-up is the default; --top-down opts out
    ))
