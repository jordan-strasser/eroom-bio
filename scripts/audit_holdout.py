"""Deep audit of holdout predictions.

Loads the trained graph, scores every holdout NCT whose extraction +
classification are already cached on disk (NO LLM calls), and prints
six diagnostics:

  1. Per-trial table: NCT | P(success) | label | n_edges | weakest etype
     (sorted by P(success) desc)
  2. Top-5 highest-P FAILED trials — full edge decomposition
  3. Top-5 lowest-P SUCCEEDED trials — full edge decomposition
  4. Cross-contamination: edges whose evidence includes other holdout NCTs
  5. Indication leakage: edges with evidence from >1 indication
  6. P(success) histograms for label=0 vs label=1

Reproduces the holdout scoring path from scripts/eval_holdout.py without
any network or LLM calls — all 50 holdout trials are cached.

Usage:
    python -m scripts.audit_holdout
"""

from __future__ import annotations

import asyncio
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from src.graph.models import EdgeBeliefState, EdgeType, normalize_entity
from src.graph.store import GraphStore
from src.ingestion.clinicaltrials import ClinicalTrialsClient
from src.prediction.path_query import (
    _AUXILIARY_EDGES,
    _CAUSAL_CHAIN,
    _DEFAULT_BELIEF,
    _collect_modulation_edges,
    _resolve_chain_via_topology,
    _resolve_target_for_compound,
    _trust_weight,
    predict_clinical_hypothesis,
)


GRAPH_PATH = Path("data/exports/oncology_annotated.json")
CORPUS_PATH = Path("data/corpora/melanoma_145.txt")
ANN_DIR = Path("data/annotations")

NCT_RE = re.compile(r"^NCT\d+$")


# ── Helpers reproduced from eval_holdout.py ────────────────────────────


def _eval_normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def _resolve_label(extraction_dict: dict, classification_dict: dict | None) -> int | None:
    met = (extraction_dict.get("results") or {}).get("primary_endpoint_met")
    if met is True:
        return 1
    if met is False:
        return 0
    if classification_dict is None:
        return None
    outcome = (classification_dict.get("trial_outcome") or "").lower()
    if outcome == "success":
        return 1
    if outcome == "failure":
        return 0
    return None


def _holdout_ncts(corpus_path: Path, snapshot_path: Path) -> list[str]:
    snap = json.loads(snapshot_path.read_text())
    in_slice = set(snap.get("trial_subgraphs", {}).keys())
    corpus: list[str] = []
    for line in corpus_path.read_text().splitlines():
        clean = line.split("#", 1)[0].strip()
        if clean.startswith("NCT"):
            corpus.append(clean)
    return [n for n in corpus if n not in in_slice]


def _derive_hypothesis(
    extraction_dict: dict, conditions: list[str], graph: GraphStore,
) -> tuple[str, str] | None:
    """Pick (compound_id, indication_id). Conditions come from CT.gov."""
    candidate_names: list[str] = []
    for arm in (extraction_dict.get("arms") or []):
        for c in (arm.get("compounds") or []):
            if c and c.lower() != "placebo":
                candidate_names.append(c)
    th = extraction_dict.get("therapeutic_hypothesis") or {}
    if th.get("compound"):
        candidate_names.append(th["compound"])

    compound_id: str | None = None
    for name in candidate_names:
        try:
            slug = normalize_entity(name, "InterventionNode")
        except ValueError:
            continue
        try:
            graph.get_node(slug)
            compound_id = slug
            break
        except KeyError:
            pass
        key = (_eval_normalize(name), "compound")
        idx = getattr(graph, "_entity_index", None)
        if idx and key in idx:
            compound_id = idx[key]
            break
    if compound_id is None:
        return None

    if not conditions:
        return None
    indication_id: str | None = None
    for cond in conditions:
        try:
            slug = normalize_entity(cond, "IndicationNode")
        except ValueError:
            continue
        try:
            graph.get_node(slug)
            indication_id = slug
            break
        except KeyError:
            pass
        idx = getattr(graph, "_entity_index", None)
        if idx and (_eval_normalize(cond), "indication") in idx:
            indication_id = idx[(_eval_normalize(cond), "indication")]
            break
    if indication_id is None:
        return None

    return compound_id, indication_id


async def _fetch_conditions_map(ncts: list[str]) -> dict[str, list[str]]:
    """Async-fetch CT.gov conditions for each NCT. Returns {} on failure."""
    ct = ClinicalTrialsClient()
    out: dict[str, list[str]] = {}

    async def _one(n: str) -> None:
        try:
            rec = await ct.get_study(n)
            out[n] = list(getattr(rec, "conditions", []) or [])
        except Exception as exc:  # noqa: BLE001
            out[n] = []
            print(f"  [warn] {n}: CT.gov fetch failed ({exc})")

    # Modest concurrency to be polite to CT.gov.
    sem = asyncio.Semaphore(8)

    async def _gated(n: str) -> None:
        async with sem:
            await _one(n)

    await asyncio.gather(*(_gated(n) for n in ncts))
    return out


# ── Indication map: NCT → primary indication ───────────────────────────


def _build_nct_indication_map(graph: GraphStore) -> dict[str, str]:
    """Map NCT id → primary indication id for trained trials.

    Pulls from the snapshot's trial_subgraphs: parent_population_id is
    `{indication}__unselected` or `{indication}__{features}`, so strip
    after the double underscore to get the indication slug.
    """
    out: dict[str, str] = {}
    for trial_id, ts in graph.trial_subgraphs.items():
        pop = ts.parent_population_id
        ind = pop.split("__", 1)[0] if "__" in pop else pop
        out[trial_id] = ind
    return out


def _augment_with_holdout_indications(
    nct_to_ind: dict[str, str], holdout_rows: list[dict],
) -> None:
    for r in holdout_rows:
        if r.get("indication"):
            nct_to_ind[r["nct"]] = r["indication"]


# ── Edge introspection ─────────────────────────────────────────────────


def _collect_chain_edges_audit(
    graph: GraphStore, compound_id: str, indication_id: str,
):
    """Reproduce PredictionEngine._collect_edges without RNG sampling.

    Returns (list of (src, tgt, edge_type, belief), target_id, mech_id,
    biology_id) so we can both print per-edge contributions and the
    resolved chain identifiers used.
    """
    target_id = _resolve_target_for_compound(graph, compound_id)
    mechanism_id, biology_id = _resolve_chain_via_topology(
        graph, target_id, indication_id
    )

    chain_fields = {
        "compound_id": compound_id,
        "target_id": target_id,
        "mechanism_id": mechanism_id,
        "biology_id": biology_id,
        "indication_id": indication_id,
        "endpoint_id": "UNKNOWN",
        "subgroup_population_id": "UNKNOWN",
    }

    edges: list[tuple[str, str, EdgeType, EdgeBeliefState]] = []
    for src_field, tgt_field, edge_type in _CAUSAL_CHAIN + _AUXILIARY_EDGES:
        src_id = chain_fields[src_field]
        tgt_id = chain_fields[tgt_field]
        if src_id == "UNKNOWN" or tgt_id == "UNKNOWN":
            continue
        try:
            belief = graph.get_edge_belief(src_id, tgt_id, edge_type)
        except KeyError:
            belief = _DEFAULT_BELIEF
        edges.append((src_id, tgt_id, edge_type, belief))

    edges.extend(_collect_modulation_edges(graph, compound_id))
    return edges, target_id, mechanism_id, biology_id


def _classify_source(src_id: str) -> str:
    if NCT_RE.match(src_id):
        return "trial"
    if src_id.startswith("lincs:"):
        return "lincs"
    return "other"


def _nct_sources(belief: EdgeBeliefState) -> list[str]:
    return [
        ev.source_id for ev in belief.evidence
        if NCT_RE.match(ev.source_id)
    ]


# ── Histogram ──────────────────────────────────────────────────────────


def _ascii_histogram(values: list[float], bins: int = 10, width: int = 40) -> str:
    if not values:
        return "  (empty)"
    counts, edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
    max_c = max(counts) if max(counts) > 0 else 1
    out_lines = []
    for i, c in enumerate(counts):
        bar_len = int(width * c / max_c)
        bar = "█" * bar_len
        out_lines.append(
            f"  [{edges[i]:.2f}–{edges[i+1]:.2f})  {c:3d} {bar}"
        )
    return "\n".join(out_lines)


# ── Main ───────────────────────────────────────────────────────────────


def main() -> None:
    print("\n" + "=" * 80)
    print("HOLDOUT PREDICTION AUDIT")
    print("=" * 80)

    graph = GraphStore()
    graph.import_snapshot(str(GRAPH_PATH))
    stats = graph.stats()
    print(
        f"Graph: {stats['node_count']} nodes, {stats['edge_count']} edges, "
        f"{len(graph.trial_subgraphs)} trial subgraphs"
    )

    holdout = _holdout_ncts(CORPUS_PATH, GRAPH_PATH)
    print(f"Holdout NCTs: {len(holdout)}")
    print("Fetching conditions from CT.gov...")
    conditions_map = asyncio.run(_fetch_conditions_map(holdout))
    print(f"  fetched conditions for {sum(1 for v in conditions_map.values() if v)}/{len(holdout)}")

    nct_to_ind = _build_nct_indication_map(graph)

    rows: list[dict] = []
    skipped: list[tuple[str, str]] = []

    for nct in holdout:
        ext_path = ANN_DIR / f"{nct}_extraction.json"
        cls_path = ANN_DIR / f"{nct}_classification.json"
        if not (ext_path.exists() and cls_path.exists()):
            skipped.append((nct, "annotation not cached"))
            continue
        try:
            extraction = json.loads(ext_path.read_text())
        except json.JSONDecodeError as exc:
            skipped.append((nct, f"extraction JSON parse error: {exc}"))
            continue
        try:
            classification = json.loads(cls_path.read_text())
        except json.JSONDecodeError:
            classification = None

        label = _resolve_label(extraction, classification)
        if label is None:
            skipped.append((nct, "no resolvable label"))
            continue

        hyp = _derive_hypothesis(extraction, conditions_map.get(nct, []), graph)
        if hyp is None:
            skipped.append((nct, "compound/indication not in graph"))
            continue
        compound_id, indication_id = hyp

        try:
            np.random.seed(42)  # deterministic across runs
            result = predict_clinical_hypothesis(
                graph, compound_id, indication_id,
                endpoint_id=None, n_samples=5000,
            )
        except KeyError as exc:
            skipped.append((nct, f"predict KeyError: {exc}"))
            continue

        edges_info, t_id, m_id, b_id = _collect_chain_edges_audit(
            graph, compound_id, indication_id,
        )

        rows.append({
            "nct": nct,
            "label": label,
            "p_success": result.overall_probability,
            "ci_lo": result.ci_lower,
            "ci_hi": result.ci_upper,
            "compound": compound_id,
            "indication": indication_id,
            "target_resolved": t_id,
            "mechanism_resolved": m_id,
            "biology_resolved": b_id,
            "n_edges": len(edges_info),
            "weakest": (
                result.weakest_link.edge_type.value
                if result.weakest_link else "—"
            ),
            "weakest_full": (
                f"{result.weakest_link.edge_type.value}: "
                f"{result.weakest_link.source_id} → "
                f"{result.weakest_link.target_id}"
                if result.weakest_link else "—"
            ),
            "edges": edges_info,
            "edge_contribs": result.edge_contributions,
        })

    _augment_with_holdout_indications(nct_to_ind, rows)

    # ── 0. Skipped ────────────────────────────────────────────────────
    print(f"\nScored: {len(rows)}  |  Skipped: {len(skipped)}")
    if skipped:
        print("Skipped reasons:")
        for nct, reason in skipped:
            print(f"  {nct}: {reason}")

    if not rows:
        print("\nNo rows to audit.")
        return

    # ── 1. Per-trial table sorted by P desc ────────────────────────────
    print("\n" + "=" * 80)
    print("[1] PER-TRIAL TABLE — sorted by P(success) desc")
    print("=" * 80)
    print(
        f"{'NCT':<14}{'P(succ)':>9}{'label':>7}{'n_edges':>9}  "
        f"{'compound':<28}{'indication':<22}{'weakest_link'}"
    )
    print("-" * 130)
    for r in sorted(rows, key=lambda x: -x["p_success"]):
        marker = ""
        if r["p_success"] >= 0.6 and r["label"] == 0:
            marker = "  ← false-positive"
        elif r["p_success"] <= 0.4 and r["label"] == 1:
            marker = "  ← false-negative"
        print(
            f"{r['nct']:<14}{r['p_success']:>9.3f}{r['label']:>7d}"
            f"{r['n_edges']:>9d}  "
            f"{r['compound']:<28.28}{r['indication']:<22.22}"
            f"{r['weakest']}{marker}"
        )

    # ── 2 / 3. Edge decompositions ─────────────────────────────────────
    def _print_decomp(title: str, subset: list[dict]) -> None:
        print("\n" + "=" * 80)
        print(title)
        print("=" * 80)
        for r in subset:
            print(
                f"\n── {r['nct']}  P={r['p_success']:.3f}  label={r['label']}  "
                f"({r['compound']} → {r['indication']})"
            )
            print(
                f"   resolved chain: {r['compound']} → {r['target_resolved']} → "
                f"{r['mechanism_resolved']} → {r['biology_resolved']} → "
                f"{r['indication']}"
            )
            # Pair each engine EdgeContribution with our re-collected
            # edges to know the source NCTs on each.
            contribs_by_key = {
                (ec.source_id, ec.target_id, ec.edge_type.value): ec
                for ec in r["edge_contribs"]
            }
            for src, tgt, etype, belief in r["edges"]:
                key = (src, tgt, etype.value)
                ec = contribs_by_key.get(key)
                a, b = belief.alpha, belief.beta
                ep = belief.expected_probability
                ev_strength = belief.evidence_strength
                trust = _trust_weight(belief)
                sources = belief.evidence
                trial_srcs = [s.source_id for s in sources if NCT_RE.match(s.source_id)]
                lincs_srcs = sum(1 for s in sources if s.source_id.startswith("lincs:"))
                other_srcs = sum(
                    1 for s in sources
                    if not NCT_RE.match(s.source_id)
                    and not s.source_id.startswith("lincs:")
                )
                # An edge pulls UP if its E[p] > 0.5 (multiplicative geomean
                # contribution above the 0.5 prior baseline).
                direction = "UP  ↑" if ep > 0.5 else ("DN  ↓" if ep < 0.5 else "==  •")
                bottleneck = ec.bottleneck_score if ec else float("nan")
                print(
                    f"   [{direction}] {etype.value:<22} {src} → {tgt}"
                )
                print(
                    f"        Beta({a:.2f}, {b:.2f})  E[p]={ep:.3f}  "
                    f"n_eff={ev_strength:.2f}  trust={trust:.3f}  "
                    f"bottleneck={bottleneck:.3f}"
                )
                if trial_srcs:
                    counts = Counter(trial_srcs)
                    pretty = ", ".join(
                        f"{nct}{f'(×{n})' if n>1 else ''}"
                        for nct, n in counts.most_common(12)
                    )
                    if len(counts) > 12:
                        pretty += f", … (+{len(counts)-12} more)"
                    print(f"        trial sources ({len(trial_srcs)}): {pretty}")
                if lincs_srcs:
                    print(f"        LINCS records: {lincs_srcs}")
                if other_srcs:
                    print(f"        other (OT/structural): {other_srcs}")

    failed = [r for r in rows if r["label"] == 0]
    succeeded = [r for r in rows if r["label"] == 1]
    top_fp = sorted(failed, key=lambda x: -x["p_success"])[:5]
    bot_fn = sorted(succeeded, key=lambda x: x["p_success"])[:5]

    _print_decomp(
        "[2] TOP-5 HIGHEST-P TRIALS THAT FAILED — full edge decomposition",
        top_fp,
    )
    _print_decomp(
        "[3] TOP-5 LOWEST-P TRIALS THAT SUCCEEDED — full edge decomposition",
        bot_fn,
    )

    # ── 4. Cross-contamination: any edge cites another holdout NCT? ────
    print("\n" + "=" * 80)
    print("[4] CROSS-CONTAMINATION — test trial's edge cites OTHER holdout NCTs")
    print("=" * 80)
    holdout_set = set(holdout)
    any_leak = False
    for r in rows:
        leaks: list[tuple[str, str, str, list[str]]] = []
        for src, tgt, etype, belief in r["edges"]:
            other_holdout = sorted({
                ev.source_id for ev in belief.evidence
                if NCT_RE.match(ev.source_id)
                and ev.source_id in holdout_set
                and ev.source_id != r["nct"]
            })
            if other_holdout:
                leaks.append((src, tgt, etype.value, other_holdout))
        if leaks:
            any_leak = True
            print(f"\n  {r['nct']}  ({r['compound']} → {r['indication']}, "
                  f"P={r['p_success']:.3f}, label={r['label']})")
            for src, tgt, etype, others in leaks:
                pretty = ", ".join(others[:8])
                if len(others) > 8:
                    pretty += f", … (+{len(others)-8} more)"
                print(f"     {etype:<22} {src} → {tgt}   cites: {pretty}")
    if not any_leak:
        print("  No cross-contamination found.")

    # ── 5. Indication leakage ──────────────────────────────────────────
    print("\n" + "=" * 80)
    print("[5] INDICATION LEAKAGE — edges with evidence from >1 indication")
    print("=" * 80)
    # For every edge used by any test prediction, gather indications
    # implied by its trial-derived evidence source_ids.
    edge_to_ind_sources: dict[tuple[str, str, str], dict[str, list[str]]] = {}
    for r in rows:
        for src, tgt, etype, belief in r["edges"]:
            key = (src, tgt, etype.value)
            d = edge_to_ind_sources.setdefault(key, {})
            for ev in belief.evidence:
                if not NCT_RE.match(ev.source_id):
                    continue
                ind = nct_to_ind.get(ev.source_id, "<unknown>")
                d.setdefault(ind, []).append(ev.source_id)
    multi_ind = [
        (key, idx_map) for key, idx_map in edge_to_ind_sources.items()
        if len([i for i in idx_map.keys() if i != "<unknown>"]) >= 2
    ]
    if not multi_ind:
        print("  No edges with trial evidence spanning >1 indication.")
    else:
        print(
            f"  {len(multi_ind)} edges have trial evidence spanning ≥2 indications.\n"
        )
        # Sort by total evidence count, then etype
        multi_ind.sort(
            key=lambda kv: -sum(len(v) for v in kv[1].values())
        )
        for (src, tgt, etype), idx_map in multi_ind:
            total = sum(len(v) for v in idx_map.values())
            inds_str = ", ".join(
                f"{ind}(×{len(ncts)})"
                for ind, ncts in sorted(
                    idx_map.items(), key=lambda kv: -len(kv[1])
                )
            )
            print(f"  {etype:<22} {src} → {tgt}   total_trials={total}")
            print(f"     {inds_str}")

    # ── 6. Histograms ──────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("[6] P(success) HISTOGRAMS — by actual label")
    print("=" * 80)
    pos = [r["p_success"] for r in rows if r["label"] == 1]
    neg = [r["p_success"] for r in rows if r["label"] == 0]
    print(f"\nLabel = 1 (SUCCESS) — n={len(pos)},  "
          f"mean={np.mean(pos):.3f},  median={np.median(pos):.3f}")
    print(_ascii_histogram(pos))
    print(f"\nLabel = 0 (FAILURE) — n={len(neg)},  "
          f"mean={np.mean(neg):.3f},  median={np.median(neg):.3f}")
    print(_ascii_histogram(neg))

    # Quick summary metrics
    auroc_correct = 0.0
    for pp in pos:
        for nn in neg:
            if pp > nn:
                auroc_correct += 1.0
            elif pp == nn:
                auroc_correct += 0.5
    auroc = auroc_correct / (len(pos) * len(neg)) if pos and neg else float("nan")
    print(f"\nAUROC = {auroc:.3f}  (n_pos={len(pos)}, n_neg={len(neg)})")
    print(f"Mean P | label=1: {np.mean(pos):.3f}")
    print(f"Mean P | label=0: {np.mean(neg):.3f}")
    if np.mean(pos) < np.mean(neg):
        print("  ⚠ MEAN P IS LOWER FOR SUCCESSES — prediction is inverted")


if __name__ == "__main__":
    main()
