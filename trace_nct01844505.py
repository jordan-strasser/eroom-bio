"""End-to-end pipeline trace for NCT01844505 (CheckMate 017).

Prints every stage: raw CT.gov JSON, exact extraction prompt + response,
exact classification prompt + response, graph node mapping with canonical
ids, edge updates with before/after Beta values, and a prediction against
the graph BEFORE the trial is processed.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import anthropic
import httpx

from src.annotation.attributor import Attributor
from src.annotation.classifier import (
    Classifier,
    _format_classification_prompt,
    _load_prompt as _classifier_load_prompt,
)
from src.annotation.extractor import (
    Extractor,
    _call_messages_with_backoff,
    _format_trial_for_prompt,
    _load_prompt as _extractor_load_prompt,
)
from src.annotation.taxonomy import FailureClassification, FailureMode
from src.graph.models import (
    BiologyNode,
    CausalChain,
    CompoundNode,
    EdgeBeliefState,
    EdgeType,
    EndpointNode,
    EndpointType,
    GraphEdge,
    IndicationNode,
    MechanismCategory,
    MechanismNode,
    MechanismType,
    Modality,
    PopulationNode,
    RegulatoryStatus,
    TargetNode,
    TrialNode,
    TrialOutcome,
    TrialSubgraph,
    normalize_entity,
)
from src.graph.populate import (
    build_arms,
    build_trial_subgraph_from_extraction,
    classify_endpoint_deterministic,
)
from src.graph.store import GraphStore
from src.ingestion.clinicaltrials import ClinicalTrialsClient
from src.ingestion.opentargets import OpenTargetsClient
from src.prediction.path_query import PredictionEngine, predict_clinical_hypothesis

NCT_ID = "NCT01844505"
EXTRACTION_MODEL = "claude-sonnet-4-20250514"
CLASSIFICATION_MODEL = "claude-sonnet-4-20250514"

# Annotations dir gets touched by Extractor/Classifier; redirect to a
# scratch location so we always run live for this trace.
SCRATCH_ANNOTATIONS = Path("/tmp/eroom_trace_annotations")


def section(n: int, title: str) -> None:
    bar = "═" * 78
    print(f"\n{bar}\n  STEP {n}: {title}\n{bar}")


def subhead(s: str) -> None:
    print(f"\n── {s} ──")


def jdump(obj, *, max_chars: int | None = None) -> str:
    text = json.dumps(obj, indent=2, default=str)
    if max_chars and len(text) > max_chars:
        return text[:max_chars] + f"\n  …[truncated {len(text) - max_chars} chars]"
    return text


def belief_str(b: EdgeBeliefState) -> str:
    return (
        f"Beta(α={b.alpha:.3f}, β={b.beta:.3f}) "
        f"→ E[p]={b.expected_probability:.3f}, "
        f"strength={b.evidence_strength:.2f}, "
        f"n_evidence={len(b.evidence)}"
    )


async def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY not set")

    # Ensure no cached annotations short-circuit the live calls.
    SCRATCH_ANNOTATIONS.mkdir(parents=True, exist_ok=True)
    # Both Extractor and Classifier write to data/annotations by default;
    # monkey-patch their module-level constants to the scratch dir so we
    # always make a real API call for this trace and don't pollute the
    # user's data/annotations directory.
    import src.annotation.extractor as ex_mod
    import src.annotation.classifier as cl_mod
    ex_mod._ANNOTATIONS_DIR = SCRATCH_ANNOTATIONS
    cl_mod._ANNOTATIONS_DIR = SCRATCH_ANNOTATIONS

    client = anthropic.AsyncAnthropic(timeout=120.0)

    # ─────────────────────────────────────────────────────────────────────
    # STEP 1: Raw CT.gov data
    # ─────────────────────────────────────────────────────────────────────
    section(1, f"Raw ClinicalTrials.gov data for {NCT_ID}")
    url = f"https://clinicaltrials.gov/api/v2/studies/{NCT_ID}"
    print(f"GET {url}")
    async with httpx.AsyncClient(timeout=30.0) as http:
        resp = await http.get(url)
        resp.raise_for_status()
        raw_ct = resp.json()

    subhead("Top-level keys in raw response")
    print(list(raw_ct.keys()))

    subhead("Identification + status (excerpt)")
    proto = raw_ct["protocolSection"]
    print(jdump({
        "identificationModule": proto.get("identificationModule"),
        "statusModule": proto.get("statusModule"),
        "conditionsModule": proto.get("conditionsModule"),
        "armsInterventionsModule": proto.get("armsInterventionsModule"),
        "outcomesModule": {
            "primaryOutcomes": proto.get("outcomesModule", {}).get("primaryOutcomes"),
            "secondaryOutcomes_count": len(
                proto.get("outcomesModule", {}).get("secondaryOutcomes", []) or []
            ),
        },
        "designModule": {
            "phases": proto.get("designModule", {}).get("phases"),
            "enrollmentInfo": proto.get("designModule", {}).get("enrollmentInfo"),
        },
        "hasResults": raw_ct.get("hasResults"),
    }, max_chars=3500))

    subhead("Posted results section (size)")
    results_section = raw_ct.get("resultsSection") or {}
    print(f"  resultsSection keys: {list(results_section.keys())}")
    om = results_section.get("outcomeMeasuresModule", {}).get("outcomeMeasures", [])
    print(f"  outcome measures count: {len(om)}")
    if om:
        print(f"  first measure: title={om[0].get('title')!r}  type={om[0].get('type')}")
        analyses = om[0].get("analyses", []) or []
        if analyses:
            print(f"  first measure analysis[0]: {jdump(analyses[0])}")

    # Now run our adapter so we have a TrialRecord
    ct = ClinicalTrialsClient()
    trial = await ct.get_study(NCT_ID)
    subhead("Parsed TrialRecord (TrialRecord.model_dump excerpt)")
    parsed = trial.model_dump()
    print(jdump({
        **{k: v for k, v in parsed.items() if k != "results_summary"},
        "results_summary": "<<dict — see formatted prompt below>>",
    }))

    # ─────────────────────────────────────────────────────────────────────
    # STEP 2: Extraction — exact JSON sent and returned
    # ─────────────────────────────────────────────────────────────────────
    section(2, "Extraction — exact prompt sent to Claude, exact response")
    system_prompt_ext = _extractor_load_prompt("extraction_system.txt")
    user_prompt_ext = _format_trial_for_prompt(trial, abstract=None)

    subhead("System prompt (extraction_system.txt)")
    print(system_prompt_ext)

    subhead("User message (filled extraction_user.txt template)")
    print(user_prompt_ext)

    subhead("Exact request payload")
    extraction_request = {
        "model": EXTRACTION_MODEL,
        "max_tokens": 4096,
        "temperature": 0,
        "system": system_prompt_ext,
        "messages": [{"role": "user", "content": user_prompt_ext}],
    }
    print(jdump({
        "model": extraction_request["model"],
        "max_tokens": extraction_request["max_tokens"],
        "temperature": extraction_request["temperature"],
        "system": "<<see system prompt above>>",
        "messages": [{"role": "user", "content": "<<see user message above>>"}],
    }))

    subhead("Calling Anthropic API (extraction)…")
    response = await _call_messages_with_backoff(client, **extraction_request)
    raw_extraction_text = response.content[0].text.strip()

    subhead("Raw response text (Claude)")
    print(raw_extraction_text)

    # Parse and show structured object
    cleaned = raw_extraction_text
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()
    raw_extraction_json = json.loads(cleaned)
    subhead("Parsed extraction JSON")
    print(jdump(raw_extraction_json))

    # Hand the pre-parsed JSON to the Extractor so cache + reask logic still
    # runs correctly without making a second LLM call. We do this by
    # short-circuiting the `extract` cache: write the parsed JSON to the
    # scratch annotations dir, then call extractor.extract().
    extraction_cache_path = SCRATCH_ANNOTATIONS / f"{NCT_ID}_extraction.json"
    extraction_cache_path.write_text(json.dumps(raw_extraction_json, indent=2))
    extractor = Extractor(client)
    extraction = await extractor.extract(trial)

    subhead("TrialExtraction (model_dump)")
    print(jdump(extraction.model_dump()))

    # ─────────────────────────────────────────────────────────────────────
    # STEP 3: Classification — exact JSON sent and returned
    # ─────────────────────────────────────────────────────────────────────
    section(3, "Classification — exact prompt sent to Claude, exact response")
    system_prompt_cls = _classifier_load_prompt("classification_system.txt")
    user_prompt_cls = _format_classification_prompt(extraction)

    subhead("System prompt (classification_system.txt)")
    print(system_prompt_cls)

    subhead("User message (filled classification_user.txt template)")
    print(user_prompt_cls)

    subhead("Exact request payload")
    classification_request = {
        "model": CLASSIFICATION_MODEL,
        "max_tokens": 4096,
        "temperature": 0,
        "system": system_prompt_cls,
        "messages": [{"role": "user", "content": user_prompt_cls}],
    }
    print(jdump({
        "model": classification_request["model"],
        "max_tokens": classification_request["max_tokens"],
        "temperature": classification_request["temperature"],
        "system": "<<see system prompt above>>",
        "messages": [{"role": "user", "content": "<<see user message above>>"}],
    }))

    subhead("Calling Anthropic API (classification)…")
    response = await _call_messages_with_backoff(client, **classification_request)
    raw_classification_text = response.content[0].text.strip()

    subhead("Raw response text (Claude)")
    print(raw_classification_text)

    cleaned = raw_classification_text
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()
    raw_classification_json = json.loads(cleaned)
    subhead("Parsed classification JSON")
    print(jdump(raw_classification_json))

    # Cache and re-run via Classifier so we get the same FailureClassification
    # the pipeline would build (with _needs_review etc.)
    classification_cache_path = SCRATCH_ANNOTATIONS / f"{NCT_ID}_classification.json"
    classification_cache_path.write_text(json.dumps(raw_classification_json, indent=2))
    classifier = Classifier(client)
    classification = await classifier.classify(extraction)

    subhead("FailureClassification (model_dump)")
    print(jdump({
        **classification.model_dump(),
        "_needs_review": getattr(classification, "_needs_review", None),
        "_review_reason": getattr(classification, "_review_reason", None),
    }))

    # ─────────────────────────────────────────────────────────────────────
    # STEP 4: Build the prior graph + multi-arm × multi-subgroup × multi-endpoint
    # ─────────────────────────────────────────────────────────────────────
    section(4, "Graph: TrialNode + arms + subgroups + endpoints + chain-list TrialSubgraph")

    graph = GraphStore()

    # Indication = the trial's actual condition (no more made-up framing).
    indication_id = normalize_entity(
        trial.conditions[0] if trial.conditions else "unknown_indication",
        "IndicationNode",
    )
    mechanism_id = normalize_entity("checkpoint_blockade", "MechanismNode")
    biology_id = "R-HSA-389948"  # Reactome: PD-1 signaling

    # ── Resolve TargetNode IDs programmatically via Open Targets ──────────
    # Combo trials hit multiple targets; resolve each via OT.search_target
    # and add as TargetNodes. The arm→target map drives chain target_id.
    subhead("Resolving canonical TargetNode IDs via Open Targets + aliases via HGNC")
    from src.graph.hgnc_resolver import aliases_for, load as hgnc_load
    hgnc_load()  # ensure the alias dict is populated before TargetNode creation

    ot = OpenTargetsClient()
    target_for_symbol: dict[str, str] = {}
    for symbol in ("PD-1", "CTLA-4"):
        try:
            hit = await ot.search_target(symbol)
        except (KeyError, RuntimeError) as exc:
            print(f"  [warn] OT lookup failed for {symbol}: {exc}; skipping")
            continue
        ensembl_id = hit["target_id"]
        approved_symbol = hit.get("approved_symbol", symbol)
        target_for_symbol[symbol] = ensembl_id
        # Pull all known aliases for this gene from HGNC. Include the
        # query symbol itself ("PD-1") if it isn't already in the alias
        # list (HGNC might list "PD1" but not "PD-1" with the hyphen).
        aliases = aliases_for(approved_symbol)
        if symbol not in aliases and symbol != approved_symbol:
            aliases.insert(0, symbol)
        print(f"  {symbol:<10} → {ensembl_id} ({approved_symbol}: {hit.get('name')})  aliases={aliases}")
        try:
            graph.get_node(ensembl_id)
        except KeyError:
            graph.add_node(TargetNode(
                id=ensembl_id,
                name=hit.get("name", symbol),
                gene_symbol=approved_symbol,
                aliases=aliases,
            ))

    # Arm → primary target. Derived from each arm's compound_ids (which
    # depend on what the CT.gov arm-group labels slugify to). Mono arms
    # take the constituent compound's target; combo arms pick the first
    # constituent's target as a sentinel — binding evidence per-compound
    # routes through the chain-aware attributor, so combo→constituent
    # alignment via composed_of is what carries the combo's distinct
    # synergy beliefs downstream.
    _COMPOUND_TO_SYMBOL = {"nivolumab": "PD-1", "ipilimumab": "CTLA-4"}
    arms_for_targets = build_arms(trial)
    target_by_arm: dict[str, str] = {}
    for arm in arms_for_targets:
        primary = arm.compound_ids[0]
        symbol = _COMPOUND_TO_SYMBOL.get(primary)
        target_by_arm[arm.arm_id] = (
            target_for_symbol.get(symbol, "UNKNOWN") if symbol else "UNKNOWN"
        )
    print(f"  arm → primary target:")
    for aid, tid in target_by_arm.items():
        print(f"    {aid:>50s} → {tid}")

    # Constituent compound nodes — enrich with ChEMBL IDs + aliases via OT
    # so each compound carries downstream-useful identifiers (ChEMBL for
    # cross-database joins, aliases for matching brand / code names).
    subhead("Resolving CompoundNode ChEMBL IDs + aliases via Open Targets")
    for slug, query_name in (("nivolumab", "Nivolumab"), ("ipilimumab", "Ipilimumab")):
        try:
            drug_hit = await ot.search_drug(query_name)
        except (KeyError, RuntimeError) as exc:
            print(f"  [warn] OT drug lookup failed for {query_name}: {exc}; "
                  "creating CompoundNode without ChEMBL enrichment")
            graph.add_node(CompoundNode(id=slug, name=query_name, modality=Modality.ANTIBODY))
            continue
        print(f"  {query_name:<12} → ChEMBL={drug_hit['chembl_id']}  aliases={drug_hit['aliases'][:5]}{'...' if len(drug_hit['aliases']) > 5 else ''}")
        graph.add_node(CompoundNode(
            id=slug,
            name=drug_hit["name"],
            modality=Modality.ANTIBODY,
            chembl_id=drug_hit["chembl_id"],
            aliases=drug_hit["aliases"],
        ))
    graph.add_node(MechanismNode(
        id=mechanism_id, name="checkpoint blockade", mechanism_type=MechanismType.ANTAGONISM,
    ))
    graph.add_node(BiologyNode(id=biology_id, name="PD-1 signaling", pathway_ids=[biology_id]))
    graph.add_node(IndicationNode(
        id=indication_id, name=trial.conditions[0] if trial.conditions else indication_id,
    ))

    # ── EndpointNodes — one per primary outcome the trial reported ────────
    # Deterministic keyword match — no extra Anthropic call. CheckMate 067
    # has both PFS and OS as primary endpoints; both become EndpointNodes
    # and chains fan across them.
    subhead("Creating EndpointNodes from trial.primary_outcomes")
    endpoint_ids: dict[str, str] = {}
    for om in trial.primary_outcomes:
        ep_class = classify_endpoint_deterministic(om.measure)
        if ep_class == "other":
            print(f"  [skip] '{om.measure}' did not match any known EndpointClass")
            continue
        ep_id = normalize_entity(f"{ep_class}_{indication_id}", "EndpointNode")
        if ep_class in endpoint_ids:
            continue  # dedupe — the same class can repeat across multiple outcomes
        endpoint_ids[ep_class] = ep_id
        try:
            graph.get_node(ep_id)
        except KeyError:
            graph.add_node(EndpointNode(
                id=ep_id, name=f"{ep_class} ({trial.conditions[0] if trial.conditions else indication_id})",
                endpoint_type=EndpointType.PRIMARY,
                regulatory_status=RegulatoryStatus.ACCEPTED,
            ))
        print(f"  '{om.measure}' → class={ep_class}  node={ep_id}")

    if not endpoint_ids:
        raise RuntimeError(
            f"No EndpointNodes created for {NCT_ID} — "
            "deterministic classifier matched none of the primary outcomes"
        )

    print(f"  Backbone canonical IDs (shared across all chains):")
    for symbol, eid in target_for_symbol.items():
        print(f"    TargetNode      : {eid}    ({symbol})")
    print(f"    MechanismNode   : {mechanism_id}")
    print(f"    BiologyNode     : {biology_id}    (Reactome PD-1 signaling)")
    print(f"    IndicationNode  : {indication_id}")
    for cls, eid in endpoint_ids.items():
        print(f"    EndpointNode    : {eid}    ({cls})")

    # Seed prior edges. binds_to for each compound→target pair the trial
    # touches; one chain backbone shared across all chains.
    seed_edges: list[tuple[str, str, EdgeType, EdgeBeliefState]] = [
        (mechanism_id, biology_id, EdgeType.MECHANISM_AFFECTS, EdgeBeliefState(alpha=2.0, beta=1.0)),
        (biology_id, indication_id, EdgeType.BIOLOGY_DRIVES, EdgeBeliefState(alpha=3.0, beta=2.0)),
    ]
    for symbol, ensembl in target_for_symbol.items():
        compound = "nivolumab" if "1" in symbol else "ipilimumab"
        seed_edges.append((compound, ensembl, EdgeType.BINDS_TO, EdgeBeliefState(alpha=4.0, beta=1.0)))
        seed_edges.append((ensembl, mechanism_id, EdgeType.MODULATES_VIA, EdgeBeliefState(alpha=5.0, beta=1.0)))
    for ep_id in endpoint_ids.values():
        seed_edges.append((biology_id, ep_id, EdgeType.REFLECTS_BIOLOGY, EdgeBeliefState(alpha=2.5, beta=1.0)))
        seed_edges.append((ep_id, indication_id, EdgeType.ENDPOINT_CAPTURES, EdgeBeliefState(alpha=3.0, beta=1.0)))
    for src, tgt, et, belief in seed_edges:
        graph.add_edge(GraphEdge(source_id=src, target_id=tgt, edge_type=et, belief=belief))

    # Build the full trial subgraph from the extraction. This creates the
    # TrialNode, the synthesized combo CompoundNode (ipilimumab+nivolumab),
    # composed_of edges, per-subgroup PopulationNodes from extraction.subgroups,
    # and N arms × M subgroups × K endpoints CausalChains.
    trial_subgraph = build_trial_subgraph_from_extraction(
        graph, trial, extraction,
        target_by_arm=target_by_arm,
        mechanism_id=mechanism_id,
        biology_id=biology_id,
        indication_id=indication_id,
        endpoint_ids=endpoint_ids,
    )

    subhead("TrialNode + arms + chains")
    print(f"  TrialNode       : {NCT_ID} ({graph.get_node(NCT_ID)['name']})")
    print(f"  arms ({len(trial_subgraph.arms)}):")
    for a in trial_subgraph.arms:
        marker = " (combo, composed_of edges synthesized)" if a.is_combination else ""
        print(f"    {a.arm_id:>22s}  regimen={a.regimen_compound_id:<26s}  compounds={a.compound_ids}{marker}")
    composed = graph.get_edges_by_type(EdgeType.COMPOSED_OF)
    if composed:
        print(f"  composed_of edges:")
        for e in composed:
            print(f"    {e['source_id']} → {e['target_id']}")

    pop_nodes = sorted(n['id'] for n in graph.get_nodes_by_type("PopulationNode"))
    print(f"  PopulationNodes ({len(pop_nodes)}):")
    for pid in pop_nodes:
        node = graph.get_node(pid)
        feats = node.get("defining_features", [])
        feat_str = ", ".join(
            f"{f.get('axis')}/{f.get('key') or '-'}={f.get('level')}" for f in feats
        ) or "(none — parent enrollment population)"
        print(f"    {pid}: {feat_str}")

    n_subgroups = len(pop_nodes) - 1 if len(pop_nodes) > 1 else 1
    print(
        f"  chains ({len(trial_subgraph.chains)} = "
        f"{len(trial_subgraph.arms)} arms × {n_subgroups} subgroups × {len(endpoint_ids)} endpoints):"
    )
    for c in trial_subgraph.chains:
        es = f" es={c.effect_size}" if c.effect_size is not None else ""
        out = f" outcome={c.outcome.value}" if c.outcome.value != 'unknown' else ""
        ep_class = c.metadata.get("endpoint_class", "?")
        print(f"    arm={c.arm_id:>22s}  pop={c.subgroup_population_id:<48s}  ep={ep_class:<4s} → {c.endpoint_id}{es}{out}")

    # ─────────────────────────────────────────────────────────────────────
    # STEP 5: Prediction BEFORE attribution — one prediction per chain
    # ─────────────────────────────────────────────────────────────────────
    section(5, "Prediction BEFORE attribution (per chain)")
    engine = PredictionEngine(graph)
    prior_results: dict[tuple[str, str], "PredictionResult"] = {}  # type: ignore[name-defined]
    for c in trial_subgraph.chains:
        r = engine.predict(c, n_samples=10_000)
        prior_results[(c.arm_id, c.subgroup_population_id)] = r
        print(f"  arm={c.arm_id:>22s}  pop={c.subgroup_population_id:<48s}  P={r.overall_probability:.4f}  CI=[{r.ci_lower:.3f},{r.ci_upper:.3f}]  weakest={r.weakest_link.edge_type.value if r.weakest_link else 'n/a'}")

    # ─────────────────────────────────────────────────────────────────────
    # STEP 6: Attribution — chain-aware routing
    # ─────────────────────────────────────────────────────────────────────
    section(6, "Attribution — chain-aware routing of classifier edge updates")

    subhead("Classifier-suggested edge updates (raw)")
    raw_edges = getattr(classification, "_raw", {}).get("edges_to_update", [])
    print(jdump(raw_edges))

    # Snapshot before
    before: dict[tuple[str, str, str], EdgeBeliefState] = {}
    for src, tgt, et, _ in seed_edges:
        before[(src, tgt, et.value)] = graph.get_edge_belief(src, tgt, et)

    attributor = Attributor(graph)
    updates = attributor.attribute(classification, trial_subgraph)

    subhead(f"Applied edge updates ({len(updates)})")
    for u in updates:
        print(f"  {u.edge_type.value:22s} {u.source_id} → {u.target_id}")
        print(f"    direction={u.evidence.direction.value}  magnitude={u.evidence.magnitude:.3f}  source_type={u.evidence.source_type.value}")
        print(f"    BEFORE: {belief_str(u.pre_update_belief)}")
        print(f"    AFTER : {belief_str(u.post_update_belief)}")
        change = u.post_update_belief.expected_probability - u.pre_update_belief.expected_probability
        print(f"    ΔP    = {change:+.4f}")

    subhead("Edge belief deltas across the prior subgraph")
    for (src, tgt, et_value), pre in before.items():
        try:
            post = graph.get_edge_belief(src, tgt, EdgeType(et_value))
        except KeyError:
            continue
        tag = "UPDATED" if (post.alpha, post.beta) != (pre.alpha, pre.beta) else "unchanged"
        print(f"  [{tag}] {et_value:22s} {src} → {tgt}")
        print(f"     before: {belief_str(pre)}")
        print(f"     after : {belief_str(post)}")

    # Surface anything that didn't route — the original CTLA-4-onto-PD-1 bug
    # would now appear here instead of silently corrupting the wrong edge.
    unrouted_log = Path("data/dev/unrouted_attribution_updates.jsonl")
    if unrouted_log.exists():
        recent = [
            json.loads(line) for line in unrouted_log.read_text().splitlines()
            if NCT_ID in line
        ]
        if recent:
            subhead(f"Unrouted updates ({len(recent)}) — logged to {unrouted_log}")
            for r in recent[-10:]:
                print(f"  {r.get('source_entity')} → {r.get('target_entity')}  ({r.get('reason')})")

    # ─────────────────────────────────────────────────────────────────────
    # STEP 7: Prediction AFTER attribution
    # ─────────────────────────────────────────────────────────────────────
    section(7, "Prediction AFTER attribution (per chain, before → after)")
    for c in trial_subgraph.chains:
        post = engine.predict(c, n_samples=10_000)
        prior = prior_results[(c.arm_id, c.subgroup_population_id)]
        delta = post.overall_probability - prior.overall_probability
        print(f"  arm={c.arm_id:>22s}  pop={c.subgroup_population_id:<48s}  before={prior.overall_probability:.4f}  after={post.overall_probability:.4f}  Δ={delta:+.4f}")


if __name__ == "__main__":
    asyncio.run(main())
