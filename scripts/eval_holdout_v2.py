"""Holdout v2 evaluation — lightweight chain resolution + clean filter.

Tests AUROC and binary accuracy of the trained graph's predictions on
fresh holdout trials, with three principles:

  1. **No graph mutation.** Each holdout trial gets resolved into a
     dict of chain node ids by reading from the cached extraction +
     trained graph. No populator, no deepcopy.
  2. **Target-anchored filter.** A trial is scored if its resolved
     `target_id` AND `indication_id` are in the set of nodes referenced
     by ≥1 training chain. Compound can be a novel string — what we
     test is "novel compound, familiar target."
  3. **Clean labels.** Success/failure only; ambiguous trials are
     dropped from binary scoring (reported separately in the funnel).

Usage:
    python -m scripts.eval_holdout_v2
    python -m scripts.eval_holdout_v2 --limit 5
    python -m scripts.eval_holdout_v2 --min-overlap 6
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

from src.graph.antibody_target_resolver import nl_target_to_gene
from src.graph.models import EdgeBeliefState, EdgeType
from src.graph.store import GraphStore
from src.ingestion.clinicaltrials import ClinicalTrialsClient
from src.prediction.path_query import predict_clinical_hypothesis

console = Console()

GRAPH_PATH = Path("data/exports/oncology_annotated.json")
CORPUS_PATH = Path("data/corpora/melanoma_145.txt")
ANN_DIR = Path("data/annotations")
CANONICALIZATION_CACHE = Path("data/cache/indication_canonicalizations.json")

CHAIN_FIELDS = (
    "compound_id", "target_id", "mechanism_id", "biology_id",
    "indication_id", "endpoint_id", "population_id",
)

# Common gene-symbol synonyms used by trial extractions. Maps the
# colloquial / brand name to the symbol the graph stores on TargetNodes.
_GENE_SYM_ALIASES: dict[str, str] = {
    "pd-1": "PDCD1", "pd1": "PDCD1",
    "pd-l1": "CD274", "pdl1": "CD274",
    "ctla-4": "CTLA4", "ctla4": "CTLA4",
    "lag-3": "LAG3", "lag3": "LAG3",
    "tim-3": "HAVCR2", "tim3": "HAVCR2",
    "tigit": "TIGIT",
    "her2": "ERBB2",
    "her-2": "ERBB2",
    "egfr": "EGFR",
    "vegfr2": "KDR",
    "mek": "MAP2K1", "mek1": "MAP2K1", "mek2": "MAP2K2",
    "erk": "MAPK1", "erk1": "MAPK3", "erk2": "MAPK1",
    "csf1r": "CSF1R", "fms": "CSF1R",
    "liv-1": "SLC39A6", "liv1": "SLC39A6",
    "ny-eso-1": "CTAG1B",
    "gp100": "PMEL",
    "ido": "IDO1", "ido1": "IDO1",
    "kit": "KIT", "c-kit": "KIT",
}


# ── Funnel-stage helpers ───────────────────────────────────────────────


def _holdout_ncts(corpus_path: Path, snapshot_path: Path) -> list[str]:
    snap = json.loads(snapshot_path.read_text())
    in_slice = set(snap.get("trial_subgraphs", {}).keys())
    corpus: list[str] = []
    for line in corpus_path.read_text().splitlines():
        clean = line.split("#", 1)[0].strip()
        if clean.startswith("NCT"):
            corpus.append(clean)
    return [n for n in corpus if n not in in_slice]


def _training_used_nodes(graph: GraphStore) -> set[str]:
    """Nodes referenced by ≥1 training chain. Excludes ghosts."""
    used: set[str] = set()
    for ts in graph.trial_subgraphs.values():
        if ts.parent_population_id:
            used.add(ts.parent_population_id)
        for arm in ts.arms:
            used.add(arm.regimen_compound_id)
            for cid in arm.compound_ids:
                used.add(cid)
        for c in ts.chains:
            for f in ("compound_id", "target_id", "mechanism_id",
                     "biology_id", "indication_id", "endpoint_id",
                     "subgroup_population_id"):
                v = getattr(c, f, None)
                if v and v != "UNKNOWN":
                    used.add(v)
    used.discard(None)
    return used


# ── Lightweight chain resolver (read-only) ─────────────────────────────


def _load_canonicalization_cache() -> dict[str, str]:
    """Map raw condition string → canonical indication slug."""
    if not CANONICALIZATION_CACHE.exists():
        return {}
    try:
        raw = json.loads(CANONICALIZATION_CACHE.read_text())
    except json.JSONDecodeError:
        return {}
    out: dict[str, str] = {}
    for k, v in raw.items():
        slug = v.split("|", 1)[0]
        if slug:
            out[k.lower()] = slug
    return out


def _lookup_compound(name_str: str, graph: GraphStore) -> str | None:
    """Try to find an InterventionNode whose id matches a normalized
    version of the compound name. Returns None for novel compounds."""
    if not name_str:
        return None
    # Strip parentheticals and stage info, keep first token cluster.
    cleaned = re.sub(r"\([^)]*\)", "", name_str).strip()
    cleaned = re.split(r"\s+\+\s+|\s+plus\s+|,", cleaned, maxsplit=1)[0].strip()
    slug = re.sub(r"[^a-z0-9]+", "_", cleaned.lower()).strip("_")
    if not slug:
        return None
    if slug in graph._graph:  # noqa: SLF001
        return slug
    return None


def _lookup_target(target_str: str, graph: GraphStore) -> str | None:
    """Map the extraction's claimed_target string to a TargetNode id.

    Handles combo-target strings with several separators: commas,
    semicolons, slashes, `+`, "and", "plus", "&". Returns the first
    piece that maps to a known TargetNode via gene-symbol alias.

    Round-24 Q3: ``nl_target_to_gene`` runs first so extraction strings
    like "amyloid beta", "IL-6 receptor", and "CD20" map to the gene
    symbol the graph stores. Without it, the audit's chain resolver
    returned UNKNOWN even when the AFFECTS edge existed in the graph.
    """
    if not target_str:
        return None
    pieces = re.split(r"[,;/+&]| and | plus ", target_str, flags=re.IGNORECASE)
    target_index = _target_gene_index(graph)
    for piece in pieces:
        piece_low = piece.strip().lower()
        if not piece_low:
            continue
        # Q3: try the NL → gene-symbol map first; falls back to the
        # legacy alias table and finally to the piece itself.
        sym = (
            nl_target_to_gene(piece_low)
            or _GENE_SYM_ALIASES.get(piece_low)
            or piece_low.upper()
        )
        if sym in target_index:
            return target_index[sym]
    return None


_TARGET_GENE_INDEX: dict[str, str] | None = None


def _target_gene_index(graph: GraphStore) -> dict[str, str]:
    """gene_symbol → TargetNode id, built once per process."""
    global _TARGET_GENE_INDEX
    if _TARGET_GENE_INDEX is not None:
        return _TARGET_GENE_INDEX
    idx: dict[str, str] = {}
    for n in graph._graph.nodes:  # noqa: SLF001
        nd = graph._graph.nodes[n]  # noqa: SLF001
        if nd.get("node_type") != "TargetNode":
            continue
        sym = nd.get("gene_symbol")
        if sym:
            idx[sym.upper()] = n
        # Also index by the node id (HGNC, CHEBI, ENSG) lowercased,
        # in case extraction returns a raw id.
        idx[n.upper()] = n
    _TARGET_GENE_INDEX = idx
    return idx


def _lookup_indication(
    conditions: list[str], graph: GraphStore,
    canon_cache: dict[str, str],
) -> str | None:
    """First condition that resolves to an IndicationNode."""
    for cond in conditions or []:
        slug = canon_cache.get(cond.lower())
        if slug and slug in graph._graph:  # noqa: SLF001
            return slug
        # Fallback name match.
        cond_low = cond.lower()
        for n in graph._graph.nodes:  # noqa: SLF001
            nd = graph._graph.nodes[n]  # noqa: SLF001
            if nd.get("node_type") != "IndicationNode":
                continue
            name = (nd.get("name") or "").lower()
            if name == cond_low:
                return n
    return None


def _walk_mechanism(target_id: str, graph: GraphStore) -> str | None:
    """Pick the target's best-evidenced modulates_via neighbor."""
    if not target_id or target_id == "UNKNOWN":
        return None
    g = graph._graph  # noqa: SLF001
    if target_id not in g:
        return None
    best_id, best_score = None, -1.0
    for _u, v, k, data in g.out_edges(target_id, keys=True, data=True):
        if k != EdgeType.MODULATES_VIA.value:
            continue
        try:
            belief = EdgeBeliefState.model_validate(data.get("belief") or {})
        except Exception:  # noqa: BLE001
            continue
        score = belief.evidence_strength
        if score > best_score:
            best_score = score
            best_id = v
    return best_id


def _walk_biology(
    mechanism_id: str, indication_id: str, graph: GraphStore,
) -> str | None:
    """Pick a biology node bridging mechanism → biology → indication."""
    if not mechanism_id or not indication_id:
        return None
    g = graph._graph  # noqa: SLF001
    if mechanism_id not in g or indication_id not in g:
        return None
    best_id, best_score = None, -1.0
    for _u, v, k, data in g.out_edges(mechanism_id, keys=True, data=True):
        if k != EdgeType.MECHANISM_AFFECTS.value:
            continue
        # Require the biology to actually feed our indication.
        if not g.has_edge(v, indication_id, key=EdgeType.BIOLOGY_DRIVES.value):
            continue
        try:
            belief = EdgeBeliefState.model_validate(data.get("belief") or {})
        except Exception:  # noqa: BLE001
            continue
        if belief.evidence_strength > best_score:
            best_score = belief.evidence_strength
            best_id = v
    return best_id


def _default_population(indication_id: str, graph: GraphStore) -> str | None:
    """Fallback `{indication}__unselected` population if it exists."""
    if not indication_id:
        return None
    pop_id = f"{indication_id}__unselected"
    if pop_id in graph._graph:  # noqa: SLF001
        return pop_id
    return None


def _resolve_population_for_trial(
    nct_id: str, extraction: dict, indication_id: str,
    graph: GraphStore,
) -> str | None:
    """Option X: query the SAME population edge the classifier wrote to.

    For in-sample trials whose TrialSubgraph is in the graph, read
    `parent_population_id` directly — that's the population the
    populator created and the classifier emitted onto.

    For holdout / fresh trials, derive a population_id from the
    extraction's subgroups via the populator's `extract_indication_qualifiers`
    canonicalizer + `PopulationNode.compose_id`. Falls back to
    `__unselected` if no subgroup info is available.

    Always validated against the trained graph — returns None if the
    derived population_id isn't a graph node.
    """
    # 1. In-sample path: TrialSubgraph already exists in the graph.
    try:
        ts = graph.get_trial_subgraph_by_id(nct_id)
        pop_id = ts.parent_population_id
        if pop_id and pop_id in graph._graph:  # noqa: SLF001
            return pop_id
    except KeyError:
        pass

    # 2. Holdout path: derive from extraction.subgroups using the same
    #    qualifier-extraction the populator uses on conditions.
    from src.graph.indication_taxonomy import extract_indication_qualifiers
    from src.graph.models import PopulationNode, SubgroupFeature

    qualifiers: list[SubgroupFeature] = []
    # Extraction's `context.indication` is the trial's primary condition
    # string; run it through the same qualifier extractor.
    cond_str = (extraction.get("context") or {}).get("indication", "")
    if cond_str:
        qualifiers.extend(extract_indication_qualifiers(cond_str))

    derived = PopulationNode.compose_id(indication_id, qualifiers)
    if derived in graph._graph:  # noqa: SLF001
        return derived

    # 3. Fallback: __unselected if present.
    return _default_population(indication_id, graph)


def resolve_chain(
    extraction: dict, conditions: list[str], graph: GraphStore,
    canon_cache: dict[str, str],
) -> dict:
    """Build a chain dict for one trial. No graph mutation.

    Endpoint is left UNKNOWN — `predict_clinical_hypothesis` auto-
    resolves it from indication via the strongest endpoint_captures
    edge, which is the desired behavior for novel-compound trials too.

    Round-16: population_id now resolves via `_resolve_population_for_trial`
    which queries the trial's actual parent_population_id (in-sample)
    or derives it from extraction.context.indication's qualifiers
    (holdout) — so the prediction queries the same population edge the
    classifier wrote evidence to.
    """
    th = extraction.get("therapeutic_hypothesis") or {}
    compound_str = th.get("compound") or ""
    target_str = th.get("claimed_target") or ""
    nct_id = extraction.get("nct_id") or ""

    compound_id = _lookup_compound(compound_str, graph)
    target_id = _lookup_target(target_str, graph)
    indication_id = _lookup_indication(conditions, graph, canon_cache)
    mechanism_id = _walk_mechanism(target_id, graph) if target_id else None
    biology_id = (
        _walk_biology(mechanism_id, indication_id, graph)
        if mechanism_id and indication_id else None
    )
    population_id = (
        _resolve_population_for_trial(nct_id, extraction, indication_id, graph)
        if indication_id else None
    )

    return {
        "compound_str": compound_str,
        "compound_id": compound_id,  # None for novel compounds
        "target_id": target_id or "UNKNOWN",
        "mechanism_id": mechanism_id or "UNKNOWN",
        "biology_id": biology_id or "UNKNOWN",
        "indication_id": indication_id or "UNKNOWN",
        "endpoint_id": "UNKNOWN",  # auto-resolved at predict time
        "population_id": population_id or "UNKNOWN",
    }


def _overlap_count(chain: dict, training_used: set[str]) -> int:
    """How many of the 7 chain nodes are in the training-used set.
    Counts compound_id (when resolved) plus the other 6."""
    n = 0
    for key in (
        "compound_id", "target_id", "mechanism_id", "biology_id",
        "indication_id", "endpoint_id", "population_id",
    ):
        v = chain.get(key)
        if v and v != "UNKNOWN" and v in training_used:
            n += 1
    return n


# ── Labels (three-bucket; binary scoring drops ambiguous) ──────────────


def _resolve_label(
    extraction: dict, classification: dict | None,
) -> str | None:
    """Resolve a trial's outcome label.

    Round-16 refinement (mechanistic-efficacy framing): we predict
    mechanistic success, so non-mechanistic failure modes are routed
    to 'ambiguous' and dropped from binary scoring. A trial that
    stopped for DLT, commercial reasons, was underpowered, or where
    the classifier had insufficient_information is not a fair test of
    the chain's prediction — those failures aren't predictable from
    the mechanistic chain.

    Returns 'success' / 'failure' / 'ambiguous' / None.
    """
    # Clean primary-endpoint readout is the strongest signal — trust it.
    met = (extraction.get("results") or {}).get("primary_endpoint_met")
    classification = classification or {}
    failure_modes = classification.get("failure_modes") or []
    primary_failure_mode = (
        failure_modes[0].get("mode") if failure_modes else ""
    )

    # Non-mechanistic failure modes get routed to ambiguous regardless
    # of trial_outcome — the chain wasn't given a fair test.
    _NON_MECHANISTIC_FAILURE_MODES = {
        "insufficient_information",   # extractor couldn't pull clean efficacy data
        "dose_limiting_toxicity",     # stopped for safety, no efficacy readout
        "commercial_not_scientific",  # sponsor decision, not chain fault
        "underpowered",               # trial design issue, not mechanism
    }

    if met is True:
        # Even a successful endpoint can be qualified by a non-mech
        # caveat; trust the explicit met=True.
        return "success"
    if met is False:
        # An explicit miss IS a failure, unless the classifier
        # attributes it to non-mechanistic causes.
        if primary_failure_mode in _NON_MECHANISTIC_FAILURE_MODES:
            return "ambiguous"
        return "failure"

    # met is None: fall back to classifier judgment.
    outcome = (classification.get("trial_outcome") or "").lower()
    if outcome == "success":
        return "success"
    if outcome == "failure":
        if primary_failure_mode in _NON_MECHANISTIC_FAILURE_MODES:
            return "ambiguous"
        return "failure"
    if outcome in ("ambiguous", "partial", "mixed", "inconclusive"):
        return "ambiguous"
    return None


# ── Metrics ────────────────────────────────────────────────────────────


def _auroc(probs: list[float], labels: list[int]) -> float:
    pos = [p for p, y in zip(probs, labels) if y == 1]
    neg = [p for p, y in zip(probs, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    correct = 0.0
    for pp in pos:
        for nn in neg:
            if pp > nn:
                correct += 1.0
            elif pp == nn:
                correct += 0.5
    return correct / (len(pos) * len(neg))


def _binary_accuracy(probs: list[float], labels: list[int], thr: float = 0.5):
    """Threshold-based binary accuracy: predict success if p >= thr."""
    if not probs:
        return float("nan"), 0, 0, 0, 0
    tp = sum(1 for p, y in zip(probs, labels) if p >= thr and y == 1)
    tn = sum(1 for p, y in zip(probs, labels) if p < thr and y == 0)
    fp = sum(1 for p, y in zip(probs, labels) if p >= thr and y == 0)
    fn = sum(1 for p, y in zip(probs, labels) if p < thr and y == 1)
    return (tp + tn) / len(probs), tp, tn, fp, fn


def _ascii_hist(values: list[float], bins: int = 10, width: int = 40) -> str:
    if not values:
        return "  (empty)"
    counts, edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
    max_c = max(counts) if max(counts) > 0 else 1
    out = []
    for i, c in enumerate(counts):
        bar = "█" * int(width * c / max_c)
        out.append(f"  [{edges[i]:.2f}–{edges[i+1]:.2f})  {c:3d} {bar}")
    return "\n".join(out)


# ── Main pipeline ──────────────────────────────────────────────────────


async def main_async(args: argparse.Namespace) -> int:
    console.rule("[bold]Holdout v2 eval (lightweight resolver)[/bold]")

    graph = GraphStore()
    graph.import_snapshot(str(args.graph))
    stats = graph.stats()
    console.print(
        f"Trained graph: {stats['node_count']} nodes, "
        f"{stats['edge_count']} edges, "
        f"{len(graph.trial_subgraphs)} training trial subgraphs"
    )

    training_used = _training_used_nodes(graph)
    console.print(
        f"Training-used nodes (referenced by ≥1 chain): "
        f"{len(training_used)} of {stats['node_count']}"
    )

    holdout = _holdout_ncts(args.corpus, args.graph)
    console.print(f"Holdout NCTs: {len(holdout)}")
    if args.limit:
        holdout = holdout[: args.limit]
        console.print(f"  (limited to {len(holdout)} for this run)")

    console.print("Fetching CT.gov records for holdout...")
    ct = ClinicalTrialsClient()
    sem = asyncio.Semaphore(8)
    conditions_map: dict[str, list[str]] = {}

    async def _fetch(nct: str) -> None:
        async with sem:
            try:
                rec = await ct.get_study(nct)
                conditions_map[nct] = (
                    list(getattr(rec, "conditions", []) or []) if rec else []
                )
            except Exception as exc:  # noqa: BLE001
                conditions_map[nct] = []
                console.print(f"  [warn] {nct} CT.gov fetch failed: {exc}")
    await asyncio.gather(*(_fetch(n) for n in holdout))
    console.print(
        f"  fetched {sum(1 for v in conditions_map.values() if v)}/{len(holdout)} "
        f"trial records with conditions"
    )

    canon_cache = _load_canonicalization_cache()
    console.print(
        f"  canonicalization cache: {len(canon_cache)} condition strings"
    )

    # ── Resolve chains (lightweight, read-only) ────────────────────────
    rows: list[dict] = []
    skip_resolve: list[tuple[str, str]] = []

    for nct in holdout:
        ext_path = ANN_DIR / f"{nct}_extraction.json"
        cls_path = ANN_DIR / f"{nct}_classification.json"
        if not ext_path.exists():
            skip_resolve.append((nct, "no extraction cached"))
            continue
        try:
            extraction = json.loads(ext_path.read_text())
        except json.JSONDecodeError as exc:
            skip_resolve.append((nct, f"extraction parse: {exc}"))
            continue
        classification = None
        if cls_path.exists():
            try:
                classification = json.loads(cls_path.read_text())
            except json.JSONDecodeError:
                pass
        label = _resolve_label(extraction, classification)
        conditions = conditions_map.get(nct, [])
        chain = resolve_chain(extraction, conditions, graph, canon_cache)
        overlap = _overlap_count(chain, training_used)
        rows.append({
            "nct": nct,
            "label": label,
            "chain": chain,
            "overlap": overlap,
            "predicted": False,
            "skip_reason": None,
            "p_success": None,
            "edge_contribs": None,
            "weakest": None,
        })

    # ── Filter + predict ──────────────────────────────────────────────
    console.print(
        f"[bold]Filter: target + indication MUST be in training-used set, "
        f"AND ≥{args.min_overlap}/7 total chain-node overlap. "
        f"Compound may be novel.[/bold]"
    )
    np.random.seed(42)  # determinism
    for r in rows:
        chain = r["chain"]
        if chain["target_id"] not in training_used:
            r["skip_reason"] = (
                f"target {chain['target_id']!r} not in training-used set"
            )
            continue
        if chain["indication_id"] not in training_used:
            r["skip_reason"] = (
                f"indication {chain['indication_id']!r} not in training-used set"
            )
            continue
        if r["overlap"] < args.min_overlap:
            r["skip_reason"] = (
                f"overlap {r['overlap']}/7 below threshold {args.min_overlap}"
            )
            continue
        # Build kwargs, dropping UNKNOWNs.
        kwargs = {}
        for k in ("target_id", "mechanism_id", "biology_id",
                  "endpoint_id", "population_id"):
            v = chain[k]
            if v and v != "UNKNOWN":
                kwargs[k] = v
        try:
            result = predict_clinical_hypothesis(
                graph,
                chain["compound_id"],  # may be None for novel
                chain["indication_id"],
                n_samples=5000,
                **kwargs,
            )
        except KeyError as exc:
            r["skip_reason"] = f"predict KeyError: {exc}"
            continue
        if not result.edge_contributions:
            r["skip_reason"] = "no evidenced edges in resolved chain"
            continue
        r["predicted"] = True
        r["p_success"] = result.overall_probability
        r["edge_contribs"] = result.edge_contributions
        r["n_edges_used"] = len(result.edge_contributions)
        r["weakest"] = (
            result.weakest_link.edge_type.value
            if result.weakest_link else "—"
        )
        r["compound_novel"] = chain["compound_id"] is None

    _report(rows, skip_resolve, training_used, args)
    return 0


def _report(rows, skip_resolve, training_used, args):
    console.rule("[bold]Funnel[/bold]")
    n_total = len(rows) + len(skip_resolve)
    n_resolved = len(rows)
    n_overlap = sum(1 for r in rows if r["overlap"] >= args.min_overlap)
    n_predicted = sum(1 for r in rows if r["predicted"])
    n_success = sum(1 for r in rows if r["predicted"] and r["label"] == "success")
    n_failure = sum(1 for r in rows if r["predicted"] and r["label"] == "failure")
    n_ambig = sum(1 for r in rows if r["predicted"] and r["label"] == "ambiguous")

    funnel = Table(show_header=False, box=None)
    funnel.add_column(""); funnel.add_column("n", justify="right")
    funnel.add_row("Total holdout", str(n_total))
    funnel.add_row("Chain resolved", str(n_resolved))
    funnel.add_row(f"Pass overlap ≥{args.min_overlap}/7", str(n_overlap))
    funnel.add_row("Predicted", str(n_predicted))
    funnel.add_row("  Label = success", str(n_success))
    funnel.add_row("  Label = failure", str(n_failure))
    funnel.add_row(
        "  Label = ambiguous (dropped from binary scoring)", str(n_ambig)
    )
    funnel.add_row(
        "  Label = null/unresolvable",
        str(n_predicted - n_success - n_failure - n_ambig),
    )
    console.print(funnel)

    # Skipped: combine pre-resolve failures + post-resolve filter rejects.
    all_skipped: list[tuple[str, str]] = list(skip_resolve)
    for r in rows:
        if not r["predicted"] and r["skip_reason"]:
            all_skipped.append((r["nct"], r["skip_reason"]))
    if all_skipped:
        console.print(f"\n[bold]Skipped ({len(all_skipped)})[/bold]")
        for nct, reason in all_skipped:
            console.print(f"  {nct}: {reason}")

    # Per-trial table.
    console.rule("[bold]Per-trial table[/bold]")
    print(
        f"\n{'NCT':<14}{'overlap':>9}{'novel?':>8}{'P(succ)':>10}"
        f"{'n_edges':>9}{'label':>13}  "
        f"{'compound':<28}{'target':<15}{'indication':<22}{'weakest'}"
    )
    print("-" * 160)
    for r in sorted(rows, key=lambda x: (
        not x["predicted"],
        -(x["p_success"] or -1),
    )):
        novel_marker = "★" if r.get("compound_novel") else " "
        c = r["chain"]
        if r["predicted"]:
            p = f"{r['p_success']:.3f}"
            ne = str(r["n_edges_used"])
            wk = r["weakest"]
            label_str = r["label"] or "—"
            correct = ""
            if r["label"] == "success":
                correct = " ✓" if r["p_success"] >= 0.5 else " ✗"
            elif r["label"] == "failure":
                correct = " ✓" if r["p_success"] < 0.5 else " ✗"
            label_str = f"{label_str}{correct}"
        else:
            p, ne, wk = "—", "—", "—"
            label_str = (r["label"] or "—") + " (skipped)"
        compound_display = c["compound_str"] or c.get("compound_id") or "—"
        print(
            f"{r['nct']:<14}{r['overlap']:>9}/7{novel_marker:>8}"
            f"{p:>10}{ne:>9}{label_str:>13}  "
            f"{compound_display:<28.28}{c['target_id']:<15.15}"
            f"{c['indication_id']:<22.22}{wk}"
        )
    print("\n(★ = compound is novel — not a node in the trained graph)")

    # Binary success/failure metrics.
    bin_rows = [
        r for r in rows
        if r["predicted"] and r["label"] in ("success", "failure")
    ]
    probs = [r["p_success"] for r in bin_rows]
    labels = [1 if r["label"] == "success" else 0 for r in bin_rows]

    console.rule("[bold]Metrics: success vs failure (ambiguous dropped)[/bold]")
    if bin_rows:
        auroc = _auroc(probs, labels)
        acc, tp, tn, fp, fn = _binary_accuracy(probs, labels, thr=0.5)
        pos = [p for p, y in zip(probs, labels) if y == 1]
        neg = [p for p, y in zip(probs, labels) if y == 0]
        n_pos, n_neg = len(pos), len(neg)
        console.print(
            f"n = {len(bin_rows)} (success={n_pos}, failure={n_neg})"
        )
        console.print(f"AUROC = {auroc:.3f}    (rank-based, threshold-free)")
        console.print(
            f"Binary accuracy @ P≥0.5 = {acc:.3f}    "
            f"(TP={tp}, TN={tn}, FP={fp}, FN={fn})"
        )
        if n_pos:
            console.print(f"Mean P | success = {np.mean(pos):.3f}")
        if n_neg:
            console.print(f"Mean P | failure = {np.mean(neg):.3f}")
        if pos:
            console.print("\nP(success) distribution — label=SUCCESS:")
            console.print(_ascii_hist(pos))
        if neg:
            console.print("\nP(success) distribution — label=FAILURE:")
            console.print(_ascii_hist(neg))
    else:
        console.print("No success/failure trials in predicted set.")

    # Ambiguous bucket (reported separately, not scored).
    ambig_rows = [r for r in rows if r["predicted"] and r["label"] == "ambiguous"]
    if ambig_rows:
        ambig_ps = [r["p_success"] for r in ambig_rows]
        console.rule("[bold]Ambiguous bucket (dropped from binary scoring)[/bold]")
        console.print(
            f"n = {len(ambig_rows)}, mean P = {np.mean(ambig_ps):.3f}, "
            f"median P = {np.median(ambig_ps):.3f}"
        )
        console.print(_ascii_hist(ambig_ps))

    if any(r["predicted"] for r in rows):
        console.rule("[bold]Bottleneck edge-type frequency[/bold]")
        counter = Counter(r["weakest"] for r in rows if r["predicted"])
        for etype, n in counter.most_common():
            console.print(f"  {etype:<22} {n}")

    # Per-failed-trial chain decomposition — what edge weights drove
    # the predictions on trials that actually failed.
    failed_rows = [
        r for r in rows
        if r["predicted"] and r["label"] == "failure"
    ]
    if failed_rows:
        console.rule(
            "[bold]Failed-trial decomposition — per-trial chain "
            "+ edge weights[/bold]"
        )
        # Sort by P desc so the most overconfident failures come first.
        for r in sorted(failed_rows, key=lambda x: -x["p_success"]):
            c = r["chain"]
            novel = "★ novel compound" if r.get("compound_novel") else ""
            print()
            print(
                f"── {r['nct']}   P(success)={r['p_success']:.3f}   "
                f"(label=FAILURE)   {novel}"
            )
            print(
                f"   chain: {c.get('compound_str') or c.get('compound_id') or 'UNKNOWN'} "
                f"→ {c['target_id']} → {c['mechanism_id']} → {c['biology_id']} "
                f"→ {c['indication_id']}"
            )
            print(
                f"   endpoint={c['endpoint_id']}   population={c['population_id']}"
            )
            for ec in r["edge_contribs"]:
                a = ec.belief.alpha
                b = ec.belief.beta
                ep = ec.belief.expected_probability
                n_eff = ec.belief.evidence_strength
                direction = "UP  ↑" if ep > 0.5 else ("DN  ↓" if ep < 0.5 else "==  •")
                print(
                    f"   [{direction}] {ec.edge_type.value:<22} "
                    f"{ec.source_id} → {ec.target_id}"
                )
                print(
                    f"        Beta({a:.2f}, {b:.2f})  E[p]={ep:.3f}  "
                    f"n_eff={n_eff:.2f}  bottleneck={ec.bottleneck_score:.3f}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=GRAPH_PATH)
    parser.add_argument("--corpus", type=Path, default=CORPUS_PATH)
    parser.add_argument(
        "--min-overlap", type=int, default=5,
        help="Min number of resolved chain nodes (of 7) that must "
             "appear in the training-used set.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap holdout NCTs processed (for smoke runs).",
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
