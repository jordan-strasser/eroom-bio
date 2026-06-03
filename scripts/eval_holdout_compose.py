"""Compose-and-scan holdout eval — predict test trials WITHOUT building them in.

The honest true-holdout: a test trial is NOT added to the graph. Its chain is
*composed* from known anchors and *scanned* for compatibility against the trained
graph — predicted ONLY if it lands, else reported "unknown" rather than
fabricating a number on a chain the corpus never established.

A holdout is, by definition, a NOVEL compound — so we DON'T require its compound
in the graph (that would defeat the point). We anchor on its TARGET: a new
anti-PD-1 should be predictable from what other PD-1 trials taught the graph.
`predict_clinical_hypothesis(compound_id=None, target_id=...)` natively supports
this ("novel compound, familiar target") — it skips the affects edge and predicts
the target-onward chain from the trained beliefs. Compound-specific binding and
off-target safety are deliberately NOT modeled: the question is "would an
anti-{target} in {indication} succeed, given what we've learned?"

Compatibility gate: scorable when the TARGET and INDICATION resolve to graph
nodes AND a path exists. Targets absent from the corpus (e.g. APP, VEGFA in a
small oncology corpus) are honest "unknown"s — the graph has no knowledge to
generalize from. The training graph stays pristine and the prediction uses only
what the OTHER trials established — which is what generalization actually means.

Three modes:

  1. **classic-5 direction** (default): the five case-study holdouts, predicted
     target-anchored, reporting direction vs literature + a landing funnel. With
     ``--field`` it ALSO reports the (s,t)-field-localized direction — but note
     the classic-5 are target-anchored with NO holdout subgraph in the graph, so
     there are no per-arm (s,t) anchors to localize against; the field column
     degrades to "—" (see the graceful-degradation note printed at the end).

  2. **classic-5 + field** (``--field <path>`` [``--bandwidth``]): as above, with
     scalar ✓/✗ AND field ✓/✗ columns.

  3. **AUROC over a corpus** (``--auroc --corpus <name>`` [``--field``]
     [``--min-overlap``]): iterate the corpus's trials, resolve each read-only
     from its cached extraction + the trained graph (NO mutation), apply the
     compatibility filter (target + indication in training-used nodes, ≥ overlap
     threshold), predict P(success) (scalar; + field if ``--field``), resolve a
     clean success/failure label (ambiguous dropped), and report a funnel, AUROC,
     binary accuracy, and a per-trial table. Works in-sample over cached train
     trials AND on a separate holdout corpus. Corpus trials that ARE in the graph
     carry their own per-arm (s,t), so the field localizes genuinely here.

Usage:
    python -m scripts.eval_holdout_compose --graph data/exports/mi_oc52_annotated.json
    python -m scripts.eval_holdout_compose --graph G --field F [--bandwidth 0.15]
    python -m scripts.eval_holdout_compose --graph G --auroc --corpus multi_indication_52_train
    python -m scripts.eval_holdout_compose --graph G --auroc --corpus C --field F --min-overlap 6
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from src.graph.antibody_target_resolver import nl_target_to_gene
from src.graph.models import EdgeBeliefState, EdgeType
from src.graph.store import GraphStore
from src.prediction.field_prediction import (
    build_st_desc_map,
    load_edge_fields,
    localized_chain_probability,
)
from src.prediction.path_query import predict_clinical_hypothesis

ANN_DIR = Path("data/annotations")
CORPORA_DIR = Path("data/corpora")
CANONICALIZATION_CACHE = Path("data/cache/indication_canonicalizations.json")

# Classic 5 case-study holdouts: label -> (compound, indication_slug,
# target_gene_symbol, literature_direction, nct). Indication is the slug the
# trial's condition canonicalizes to; target is the gene the novel compound
# engages. The NCT is the trial's own id — used to *attempt* a per-arm (s,t)
# desc-map; for these target-anchored holdouts the trial is not in the graph,
# so the map comes back empty and the field column degrades to "—".
HOLDOUTS: dict[str, tuple[str, str, str, str, str]] = {
    "nivolumab CheckMate-067": ("nivolumab", "melanoma", "PDCD1", "success", "NCT01844505"),
    "solanezumab EXPEDITION": ("solanezumab", "alzheimer_disease", "APP", "failure", "NCT01127633"),
    "bevacizumab AVANT": ("bevacizumab", "colorectal_cancer", "VEGFA", "failure", "NCT00112918"),
    "torcetrapib ILLUMINATE": ("torcetrapib", "cardiovascular_diseases", "CETP", "failure", "NCT00134264"),
    "selumetinib thyroid": ("selumetinib", "differentiated_thyroid_cancer", "MAP2K1", "success", "NCT00970359"),
}

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


# ── Shared graph helpers ───────────────────────────────────────────────


def _target_index(g: GraphStore) -> dict[str, str]:
    return {
        (g._graph.nodes[n].get("gene_symbol") or "").upper(): n  # noqa: SLF001
        for n in g._graph  # noqa: SLF001
        if g._graph.nodes[n].get("node_type") == "TargetNode"  # noqa: SLF001
    }


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


# ── Lightweight chain resolver (read-only, AUROC mode) ─────────────────


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
    """Try to find a compound node whose id matches a normalized version of the
    compound name. Returns None for novel compounds."""
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
        # Also index by the node id (HGNC, CHEBI, ENSG) uppercased,
        # in case extraction returns a raw id.
        idx[n.upper()] = n
    _TARGET_GENE_INDEX = idx
    return idx


def _lookup_target(target_str: str, graph: GraphStore) -> str | None:
    """Map the extraction's claimed_target string to a TargetNode id.

    Handles combo-target strings with several separators: commas,
    semicolons, slashes, `+`, "and", "plus", "&". Returns the first
    piece that maps to a known TargetNode via gene-symbol alias.

    ``nl_target_to_gene`` runs first so extraction strings like
    "amyloid beta", "IL-6 receptor", and "CD20" map to the gene symbol the
    graph stores; falls back to the legacy alias table and finally to the
    piece itself.
    """
    if not target_str:
        return None
    pieces = re.split(r"[,;/+&]| and | plus ", target_str, flags=re.IGNORECASE)
    target_index = _target_gene_index(graph)
    for piece in pieces:
        piece_low = piece.strip().lower()
        if not piece_low:
            continue
        sym = (
            nl_target_to_gene(piece_low)
            or _GENE_SYM_ALIASES.get(piece_low)
            or piece_low.upper()
        )
        if sym in target_index:
            return target_index[sym]
    return None


def _lookup_indication(
    conditions: list[str], graph: GraphStore,
    canon_cache: dict[str, str],
) -> str | None:
    """First condition that resolves to an IndicationNode."""
    for cond in conditions or []:
        if not cond:
            continue
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
    """Query the SAME population edge the classifier wrote to.

    For in-sample trials whose TrialSubgraph is in the graph, read
    `parent_population_id` directly. For holdout / fresh trials, derive a
    population_id from the extraction's condition string via the populator's
    `extract_indication_qualifiers` + `PopulationNode.compose_id`. Falls back to
    `__unselected`. Always validated against the trained graph — returns None if
    the derived population_id isn't a graph node.
    """
    # 1. In-sample path: TrialSubgraph already exists in the graph.
    try:
        ts = graph.get_trial_subgraph_by_id(nct_id)
        pop_id = ts.parent_population_id
        if pop_id and pop_id in graph._graph:  # noqa: SLF001
            return pop_id
    except KeyError:
        pass

    # 2. Holdout path: derive from extraction.context.indication using the same
    #    qualifier-extraction the populator uses on conditions.
    from src.graph.indication_taxonomy import extract_indication_qualifiers
    from src.graph.models import PopulationNode, SubgroupFeature

    qualifiers: list[SubgroupFeature] = []
    cond_str = (extraction.get("context") or {}).get("indication", "")
    if cond_str:
        qualifiers.extend(extract_indication_qualifiers(cond_str))

    derived = PopulationNode.compose_id(indication_id, qualifiers)
    if derived in graph._graph:  # noqa: SLF001
        return derived

    # 3. Fallback: __unselected if present.
    return _default_population(indication_id, graph)


def _trial_conditions(extraction: dict) -> list[str]:
    """Condition strings for a trial, sourced from the cached extraction (no
    network). Prefers an explicit conditions list; falls back to the single
    `context.indication` free-text string."""
    ctx = extraction.get("context") or {}
    conds = ctx.get("conditions")
    if conds:
        return list(conds)
    ind = ctx.get("indication")
    return [ind] if ind else []


def _insample_anchors(nct_id: str, graph: GraphStore) -> tuple[str | None, str | None]:
    """``(indication_id, target_id)`` from the trial's OWN in-graph subgraph, if
    it is in-sample. The build already canonicalized this trial's conditions and
    resolved its targets; for in-sample corpus trials we read those directly
    rather than re-deriving from the extraction's free text (whose strings don't
    match the CT.gov-keyed canonicalization cache). Picks the most common
    indication/target across the trial's chains. Returns ``(None, None)`` for
    out-of-sample holdouts (not in ``trial_subgraphs``) — those fall back to the
    extraction-text resolver. No mutation; this is the same "read what the build
    established" pattern `_resolve_population_for_trial` uses for population.
    """
    ts = graph.trial_subgraphs.get(nct_id)
    if not ts:
        return None, None
    inds = Counter(
        c.indication_id for c in ts.chains
        if c.indication_id and c.indication_id != "UNKNOWN"
    )
    tgts = Counter(
        c.target_id for c in ts.chains
        if c.target_id and c.target_id != "UNKNOWN"
    )
    ind = inds.most_common(1)[0][0] if inds else None
    tgt = tgts.most_common(1)[0][0] if tgts else None
    return ind, tgt


def resolve_chain(
    extraction: dict, conditions: list[str], graph: GraphStore,
    canon_cache: dict[str, str],
) -> dict:
    """Build a chain dict for one trial. No graph mutation.

    For an IN-SAMPLE trial (its subgraph is in the graph) the indication and
    target are read from the build's own resolution via `_insample_anchors` —
    the extraction's free-text condition string doesn't match the CT.gov-keyed
    canonicalization cache, so re-deriving would needlessly drop in-sample trials.
    OUT-OF-SAMPLE holdouts fall back to the extraction-text resolvers
    (`_lookup_indication`, `_lookup_target`). Endpoint is left UNKNOWN
    (`predict_clinical_hypothesis` auto-resolves it from indication). Population
    resolves via `_resolve_population_for_trial` (same in-sample-first pattern).
    """
    th = extraction.get("therapeutic_hypothesis") or {}
    compound_str = th.get("compound") or ""
    target_str = th.get("claimed_target") or ""
    nct_id = extraction.get("nct_id") or ""

    insample_ind, insample_tgt = _insample_anchors(nct_id, graph)

    compound_id = _lookup_compound(compound_str, graph)
    target_id = insample_tgt or _lookup_target(target_str, graph)
    indication_id = insample_ind or _lookup_indication(conditions, graph, canon_cache)
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
    for key in CHAIN_FIELDS:
        v = chain.get(key)
        if v and v != "UNKNOWN" and v in training_used:
            n += 1
    return n


# ── Labels (three-bucket; binary scoring drops ambiguous) ──────────────


def _resolve_label(
    extraction: dict, classification: dict | None,
) -> str | None:
    """Resolve a trial's outcome label.

    Mechanistic-efficacy framing: we predict mechanistic success, so non-
    mechanistic failure modes (DLT-only stop, commercial decision, underpowered,
    insufficient_information) route to 'ambiguous' and drop from binary scoring —
    those failures aren't predictable from the mechanistic chain.

    Returns 'success' / 'failure' / 'ambiguous' / None.
    """
    met = (extraction.get("results") or {}).get("primary_endpoint_met")
    classification = classification or {}
    failure_modes = classification.get("failure_modes") or []
    primary_failure_mode = (
        failure_modes[0].get("mode") if failure_modes else ""
    )

    _NON_MECHANISTIC_FAILURE_MODES = {
        "insufficient_information",   # extractor couldn't pull clean efficacy data
        "dose_limiting_toxicity",     # stopped for safety, no efficacy readout
        "commercial_not_scientific",  # sponsor decision, not chain fault
        "underpowered",               # trial design issue, not mechanism
    }

    if met is True:
        return "success"
    if met is False:
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


# ── Field localization (shared) ────────────────────────────────────────


def _localized_overall(
    g: GraphStore, nct: str, result, field_map, embed_fn, cache, bandwidth,
) -> tuple[float | None, bool]:
    """``(overall_localized, localizable)`` for one prediction result.

    Builds the trial's per-arm (s,t) desc-map and localizes the chain's
    efficacy, then folds in the same safety penalty as the scalar overall so the
    comparison is apples-to-apples. ``localizable`` is True only if at least one
    edge on the chain actually had a field anchor AND a per-arm (s,t) text to
    query — for target-anchored holdouts not in the graph the map is empty, so
    this returns ``(None, False)`` and the caller prints "—".
    """
    if field_map is None:
        return None, False
    st_map = build_st_desc_map(g, nct)
    _ps_eff, pl_eff, per_edge = localized_chain_probability(
        result.edge_contributions, field_map, st_map,
        embed_fn=embed_fn, bandwidth=bandwidth, embed_cache=cache,
    )
    localizable = any(e["is_localized"] for e in per_edge)
    if not localizable:
        return None, False
    overall = pl_eff * (1.0 - result.safety_penalty)
    return overall, True


# ── Mode 1/2: classic-5 direction (optionally + field) ─────────────────


def _run_classic5(args, g: GraphStore) -> int:
    tgt_idx = _target_index(g)

    field_map = embed_fn = None
    cache: dict = {}
    if args.field:
        from src.graph.biolord_embeddings import embed_text
        field_map = load_edge_fields(args.field)
        embed_fn = embed_text
        print(f"field: {len(field_map)} localized edges loaded from {args.field}")

    def present(nid: str) -> bool:
        return nid in g._graph  # noqa: SLF001

    print(f"compose-and-scan (target-anchored) holdout eval — graph: {args.graph}")
    if args.field:
        print(f"{'holdout':26} | landing      | scalar prediction                      "
              f"| dir(scalar) | P_field  | dir(field)")
        print("-" * 130)
    else:
        print(f"{'holdout':26} | landing      | prediction                                  | direction")
        print("-" * 112)

    scored = correct = 0
    field_localizable_count = 0
    for label, (drug, ind, gene, lit, nct) in HOLDOUTS.items():
        tid = tgt_idx.get(gene.upper())
        i_ok = present(ind)
        landing = f"tgt={'Y' if tid else 'n'} ind={'Y' if i_ok else 'n'}"
        pred, direction = "— (target absent — no knowledge)", "unknown"
        field_str, field_dir = "—", ""
        if tid and i_ok:
            try:
                r = predict_clinical_hypothesis(
                    g, None, ind, target_id=tid, n_samples=args.n_samples,
                )
                if r.edge_contributions:
                    overall = r.overall_probability
                    call = "success" if overall >= 0.5 else "failure"
                    ok = call == lit
                    scored += 1
                    correct += ok
                    path = "→".join(ec.target_id[:13] for ec in r.edge_contributions)
                    pred = f"P={overall:.3f} ({len(r.edge_contributions)}e: {path[:34]})"
                    direction = ("CORRECT" if ok else "WRONG") + f" [{call} vs {lit}]"
                    if field_map is not None:
                        f_overall, localizable = _localized_overall(
                            g, nct, r, field_map, embed_fn, cache, args.bandwidth,
                        )
                        if localizable and f_overall is not None:
                            field_localizable_count += 1
                            f_call = "success" if f_overall >= 0.5 else "failure"
                            field_str = f"P={f_overall:.3f}"
                            field_dir = ("✓" if f_call == lit else "✗") + f" [{f_call}]"
                        else:
                            field_str, field_dir = "—", "(not localizable)"
                else:
                    pred = "no path from target"
            except Exception as e:  # noqa: BLE001
                pred = f"err: {str(e)[:42]}"
        if args.field:
            print(f"{label:26} | {landing:12} | {pred:42.42} | {direction:11.11} "
                  f"| {field_str:8} | {field_dir}")
        else:
            print(f"{label:26} | {landing:12} | {pred:43} | {direction}")

    if args.field:
        print("-" * 130)
    else:
        print("-" * 112)
    rate = f"{correct}/{scored}" if scored else "0/0"
    print(f"scorable (target lands): {scored}/{len(HOLDOUTS)}    direction-correct (scalar): {rate}")
    print("\nNOTE: target-absent holdouts are HONEST unknowns — a small corpus simply has"
          "\nno knowledge of that biology to generalize from. The graph stays unmutated;"
          "\npredictions use only what the OTHER trials established on the shared chain.")
    if args.field:
        print(
            f"\nFIELD NOTE: the classic-5 are TARGET-ANCHORED holdouts — the trial is NOT"
            f"\nin the graph, so build_st_desc_map() finds no per-arm (s,t) anchors on the"
            f"\ncomposed chain. Field localization had something to localize for "
            f"{field_localizable_count}/{scored}"
            f"\nof the scored holdouts; the rest degrade to the scalar (shown as '—')."
            f"\n(s,t) localization is meaningful in --auroc corpus mode, where the trials"
            f"\nare in the graph and carry their own per-arm descriptions.")
    return 0


# ── Mode 3: AUROC over a corpus ────────────────────────────────────────


def _corpus_ncts(corpus_name: str) -> list[str]:
    path = CORPORA_DIR / (corpus_name if corpus_name.endswith(".txt") else f"{corpus_name}.txt")
    if not path.exists():
        raise FileNotFoundError(f"corpus not found: {path}")
    out: list[str] = []
    for line in path.read_text().splitlines():
        clean = line.split("#", 1)[0].strip()
        if clean.startswith("NCT"):
            out.append(clean)
    return out


def _run_auroc(args, g: GraphStore) -> int:
    print(f"AUROC corpus eval — graph: {args.graph}   corpus: {args.corpus}")
    stats = g.stats()
    print(f"trained graph: {stats['node_count']} nodes, {stats['edge_count']} edges, "
          f"{len(g.trial_subgraphs)} training trial subgraphs")

    training_used = _training_used_nodes(g)
    print(f"training-used nodes (referenced by >=1 chain): "
          f"{len(training_used)} of {stats['node_count']}")

    field_map = embed_fn = None
    cache: dict = {}
    if args.field:
        from src.graph.biolord_embeddings import embed_text
        field_map = load_edge_fields(args.field)
        embed_fn = embed_text
        print(f"field: {len(field_map)} localized edges loaded from {args.field}")

    ncts = _corpus_ncts(args.corpus)
    if args.limit:
        ncts = ncts[: args.limit]
    in_slice = sum(1 for n in ncts if n in g.trial_subgraphs)
    print(f"corpus NCTs: {len(ncts)}  ({in_slice} in-sample / "
          f"{len(ncts) - in_slice} out-of-sample)")
    canon_cache = _load_canonicalization_cache()
    print(f"canonicalization cache: {len(canon_cache)} condition strings\n")

    rows: list[dict] = []
    skip_resolve: list[tuple[str, str]] = []
    np.random.seed(42)  # determinism

    for nct in ncts:
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
        conditions = _trial_conditions(extraction)
        chain = resolve_chain(extraction, conditions, g, canon_cache)
        overlap = _overlap_count(chain, training_used)
        rows.append({
            "nct": nct, "label": label, "chain": chain, "overlap": overlap,
            "predicted": False, "skip_reason": None, "p_success": None,
            "p_field": None, "field_localizable": False,
            "edge_contribs": None, "weakest": None, "n_edges_used": 0,
            "compound_novel": chain["compound_id"] is None,
        })

    # Filter + predict.
    print(f"filter: target + indication MUST be in training-used set, AND "
          f">={args.min_overlap}/7 chain-node overlap. Compound may be novel.")
    for r in rows:
        chain = r["chain"]
        if chain["target_id"] not in training_used:
            r["skip_reason"] = f"target {chain['target_id']!r} not in training-used set"
            continue
        if chain["indication_id"] not in training_used:
            r["skip_reason"] = f"indication {chain['indication_id']!r} not in training-used set"
            continue
        if r["overlap"] < args.min_overlap:
            r["skip_reason"] = f"overlap {r['overlap']}/7 below threshold {args.min_overlap}"
            continue
        kwargs = {}
        for k in ("target_id", "mechanism_id", "biology_id", "endpoint_id", "population_id"):
            v = chain[k]
            if v and v != "UNKNOWN":
                kwargs[k] = v
        try:
            result = predict_clinical_hypothesis(
                g, chain["compound_id"], chain["indication_id"],
                n_samples=args.n_samples, **kwargs,
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
            result.weakest_link.edge_type.value if result.weakest_link else "—"
        )
        if field_map is not None:
            f_overall, localizable = _localized_overall(
                g, r["nct"], result, field_map, embed_fn, cache, args.bandwidth,
            )
            r["field_localizable"] = localizable
            r["p_field"] = f_overall if localizable else None

    _report_auroc(rows, skip_resolve, args, with_field=field_map is not None)
    return 0


def _report_auroc(rows, skip_resolve, args, *, with_field: bool) -> None:
    # Funnel.
    n_total = len(rows) + len(skip_resolve)
    n_resolved = len(rows)
    n_overlap = sum(1 for r in rows if r["overlap"] >= args.min_overlap)
    n_predicted = sum(1 for r in rows if r["predicted"])
    n_success = sum(1 for r in rows if r["predicted"] and r["label"] == "success")
    n_failure = sum(1 for r in rows if r["predicted"] and r["label"] == "failure")
    n_ambig = sum(1 for r in rows if r["predicted"] and r["label"] == "ambiguous")
    n_null = n_predicted - n_success - n_failure - n_ambig

    def _f(label: str, n: int) -> str:
        return f"  {label:<48}{n:>4}"

    print("\n── Funnel ──")
    print(_f("total corpus trials", n_total))
    print(_f("chain resolved", n_resolved))
    print(_f(f"pass overlap >={args.min_overlap}/7", n_overlap))
    print(_f("predicted (scorable)", n_predicted))
    print(_f("  label = success", n_success))
    print(_f("  label = failure", n_failure))
    print(_f("  label = ambiguous (dropped from scoring)", n_ambig))
    print(_f("  label = null/unresolvable", n_null))
    if with_field:
        n_field = sum(1 for r in rows if r["predicted"] and r["field_localizable"])
        print(_f("  field-localizable (>=1 anchored edge w/ (s,t))", n_field))

    all_skipped: list[tuple[str, str]] = list(skip_resolve)
    for r in rows:
        if not r["predicted"] and r["skip_reason"]:
            all_skipped.append((r["nct"], r["skip_reason"]))
    if all_skipped:
        print(f"\n── Skipped ({len(all_skipped)}) ──")
        for nct, reason in all_skipped:
            print(f"  {nct}: {reason}")

    # Per-trial table.
    print("\n── Per-trial table ──")
    fld_hdr = f"{'P_field':>9}" if with_field else ""
    print(f"\n{'NCT':<14}{'ovlp':>6}{'nv':>4}{'P(succ)':>10}{fld_hdr}"
          f"{'n_e':>5}  {'label':<18}{'compound':<26}{'target':<16}"
          f"{'indication':<24}{'weakest'}")
    print("-" * (132 + (9 if with_field else 0)))
    for r in sorted(rows, key=lambda x: (not x["predicted"], -(x["p_success"] or -1))):
        novel_marker = "*" if r.get("compound_novel") else " "
        c = r["chain"]
        if r["predicted"]:
            p = f"{r['p_success']:.3f}"
            ne = str(r["n_edges_used"])
            wk = r["weakest"]
            label_str = r["label"] or "—"
            correct = ""
            if r["label"] == "success":
                correct = " v" if r["p_success"] >= 0.5 else " x"
            elif r["label"] == "failure":
                correct = " v" if r["p_success"] < 0.5 else " x"
            label_str = f"{label_str}{correct}"
            fld = ""
            if with_field:
                fld = f"{r['p_field']:.3f}" if r["p_field"] is not None else "—"
                fld = f"{fld:>9}"
        else:
            p, ne, wk = "—", "—", "—"
            label_str = (r["label"] or "—") + " (skip)"
            fld = f"{'—':>9}" if with_field else ""
        compound_display = c["compound_str"] or c.get("compound_id") or "—"
        print(f"{r['nct']:<14}{str(r['overlap'])+'/7':>6}{novel_marker:>4}{p:>10}{fld}"
              f"{ne:>5}  {label_str:<18.18}{compound_display:<26.26}{c['target_id']:<16.16}"
              f"{c['indication_id']:<24.24}{wk}")
    print("\n(* = compound is novel — not a node in the trained graph)")

    # Binary success/failure metrics (scalar).
    bin_rows = [r for r in rows
                if r["predicted"] and r["label"] in ("success", "failure")]
    probs = [r["p_success"] for r in bin_rows]
    labels = [1 if r["label"] == "success" else 0 for r in bin_rows]

    print("\n── Metrics: success vs failure (ambiguous dropped) ──")
    if bin_rows:
        auroc = _auroc(probs, labels)
        acc, tp, tn, fp, fn = _binary_accuracy(probs, labels, thr=0.5)
        pos = [p for p, y in zip(probs, labels) if y == 1]
        neg = [p for p, y in zip(probs, labels) if y == 0]
        print(f"  n = {len(bin_rows)} (success={len(pos)}, failure={len(neg)})")
        print(f"  AUROC (scalar)            = {auroc:.3f}   (rank-based, threshold-free)")
        print(f"  binary accuracy @ P>=0.5  = {acc:.3f}   (TP={tp}, TN={tn}, FP={fp}, FN={fn})")
        if pos:
            print(f"  mean P | success          = {np.mean(pos):.3f}")
        if neg:
            print(f"  mean P | failure          = {np.mean(neg):.3f}")

        if with_field:
            # Field AUROC over the subset that actually localized.
            f_rows = [r for r in bin_rows if r["p_field"] is not None]
            if f_rows:
                f_probs = [r["p_field"] for r in f_rows]
                f_labels = [1 if r["label"] == "success" else 0 for r in f_rows]
                f_auroc = _auroc(f_probs, f_labels)
                f_acc, *_ = _binary_accuracy(f_probs, f_labels, thr=0.5)
                print(f"  AUROC (field, n={len(f_rows)})       = {f_auroc:.3f}   "
                      f"(only trials with >=1 localizable edge)")
                print(f"  binary accuracy (field)   = {f_acc:.3f}")
            else:
                print("  AUROC (field)             = —   (no scorable trial was localizable)")
    else:
        print("  no success/failure trials in predicted set.")

    if any(r["predicted"] for r in rows):
        print("\n── Bottleneck edge-type frequency ──")
        counter = Counter(r["weakest"] for r in rows if r["predicted"])
        for etype, n in counter.most_common():
            print(f"  {etype:<22} {n}")


# ── Entry point ────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--n-samples", type=int, default=2000)
    ap.add_argument("--field", default=None,
                    help="(s,t) belief-field snapshot; adds field-localized columns.")
    ap.add_argument("--bandwidth", type=float, default=None,
                    help="Override the field kernel bandwidth at query time.")
    ap.add_argument("--auroc", action="store_true",
                    help="Run AUROC-over-corpus mode (requires --corpus).")
    ap.add_argument("--corpus", default=None,
                    help="Corpus name (in data/corpora/) for --auroc mode.")
    ap.add_argument("--min-overlap", type=int, default=5,
                    help="Min resolved chain nodes (of 7) in the training-used set "
                         "(--auroc mode).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap corpus NCTs processed (--auroc smoke runs).")
    args = ap.parse_args()

    if args.auroc and not args.corpus:
        ap.error("--auroc requires --corpus")

    g = GraphStore()
    g.import_snapshot(args.graph)

    if args.auroc:
        return _run_auroc(args, g)
    return _run_classic5(args, g)


if __name__ == "__main__":
    raise SystemExit(main())
