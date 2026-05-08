"""
Eroom Bio—Annotation Pipeline Smoke Test (Sprints 6-8)
Tests the full pipeline: real trial -> extraction -> classification -> edge attribution
Requires ANTHROPIC_API_KEY env var.

Usage: python smoke_test_annotation.py
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
WARN = "\033[93m!\033[0m"
results = []


def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append((name, condition))
    print(f"  {status} {name}" + (f"—{detail}" if detail else ""))
    return condition


def section(title):
    print(f"\n\033[1m{'='*60}\033[0m")
    print(f"\033[1m  {title}\033[0m")
    print(f"\033[1m{'='*60}\033[0m")


# ============================================================
# SETUP
# ============================================================
section("0. Setup")

api_key = os.environ.get("ANTHROPIC_API_KEY")
check("ANTHROPIC_API_KEY set", api_key is not None,
      "required for extraction + classification")
if not api_key:
    print("\n  Set ANTHROPIC_API_KEY and re-run.")
    sys.exit(1)

try:
    import anthropic
    client = anthropic.AsyncAnthropic(api_key=api_key)
    check("Anthropic SDK imports", True)
except ImportError as e:
    check("Anthropic SDK imports", False, str(e))
    sys.exit(1)

try:
    from src.graph.store import GraphStore
    from src.graph.models import (
        EdgeBeliefState, EvidenceRecord, GraphEdge, EdgeType,
        EvidenceType, CompoundNode, TargetNode, IndicationNode,
        EndpointNode, TrialSubgraph, Modality, EndpointType,
        RegulatoryStatus,
    )
    from src.ingestion.clinicaltrials import ClinicalTrialsClient
    from src.annotation.extractor import Extractor
    from src.annotation.classifier import Classifier
    from src.annotation.attributor import Attributor
    from src.annotation.taxonomy import FailureMode, FAILURE_MODE_RULES
    check("All modules import", True)
except ImportError as e:
    check("All modules import", False, str(e))
    sys.exit(1)


# ============================================================
# TEST 1: Taxonomy sanity
# ============================================================
section("1. Failure Taxonomy")

check("All 13 failure modes defined",
      len(FailureMode) >= 13,
      f"found {len(FailureMode)}")

check("All modes have rules",
      all(mode in FAILURE_MODE_RULES for mode in FailureMode),
      "every FailureMode has an EdgeUpdateRule")

# Check the critical design patterns
no_engage = FAILURE_MODE_RULES.get(FailureMode.NO_TARGET_ENGAGEMENT)
if no_engage:
    has_binds = EdgeType.BINDS_TO in no_engage.edges_to_weaken
    check("NO_TARGET_ENGAGEMENT weakens binds_to", has_binds)

tgt_bio = FAILURE_MODE_RULES.get(FailureMode.TARGET_ENGAGED_BIOLOGY_NOT_MOVED)
if tgt_bio:
    weakens_mech = EdgeType.MECHANISM_AFFECTS in tgt_bio.edges_to_weaken
    check("TARGET_ENGAGED_BIO_NOT_MOVED weakens mechanism_affects", weakens_mech)
    strengthens_bind = EdgeType.BINDS_TO in tgt_bio.edges_to_strengthen
    check("TARGET_ENGAGED_BIO_NOT_MOVED strengthens binds_to", strengthens_bind)

wrong_time = FAILURE_MODE_RULES.get(FailureMode.WRONG_TIMEFRAME)
if wrong_time:
    no_mech_edges = len(wrong_time.edges_to_weaken) == 0 and len(wrong_time.edges_to_strengthen) == 0
    check("WRONG_TIMEFRAME touches zero mechanistic edges", no_mech_edges)


# ============================================================
# TEST 2: Extract a successful trial
# ============================================================
section("2. Extract Successful Trial (Nivolumab, NCT01721772)")

async def test_extraction_success():
    ct_client = ClinicalTrialsClient()
    extractor = Extractor(client)

    trial = await ct_client.get_study("NCT01721772")
    check("Fetched trial", trial is not None, trial.title[:50] + "...")
    check("Trial has results", trial.has_results)

    extraction = await extractor.extract(trial)
    check("Extraction returned", extraction is not None)

    # TrialExtraction has flat fields: compound_name, target_name, primary_endpoint, etc.
    check("Has compound", bool(extraction.compound_name),
          extraction.compound_name[:40] if extraction.compound_name else "")
    check("Has target", bool(extraction.target_name),
          extraction.target_name[:40] if extraction.target_name else "")
    check("Has endpoint", bool(extraction.primary_endpoint),
          extraction.primary_endpoint[:40] if extraction.primary_endpoint else "")

    check("Endpoint met is not None", extraction.primary_endpoint_met is not None,
          f"met={extraction.primary_endpoint_met}")
    check("Has effect size", extraction.effect_size is not None,
          str(extraction.effect_size)[:50])
    check("Has p-value", extraction.p_value is not None,
          f"p={extraction.p_value}")

    return extraction

extraction_success = asyncio.run(test_extraction_success())


# ============================================================
# TEST 3: Extract a failed trial
# ============================================================
section("3. Extract Failed Trial (find one)")

async def test_extraction_failure():
    ct_client = ClinicalTrialsClient()
    extractor = Extractor(client)

    # Search for terminated Phase 3 oncology trials with results
    trials = await ct_client.search(
        condition="cancer",
        phase="PHASE3",
        status="TERMINATED",
        has_results=True,
        max_results=5
    )

    if not trials:
        # Fallback: try completed trials (some completed but failed endpoints)
        trials = await ct_client.search(
            condition="lung cancer",
            phase="PHASE3",
            has_results=True,
            max_results=5
        )

    check("Found trials to test", len(trials) > 0,
          f"got {len(trials)}")

    if not trials:
        return None

    trial = trials[0]
    print(f"\n  Testing: {trial.nct_id}—{trial.title[:60]}...")

    extraction = await extractor.extract(trial)
    check("Failed trial extraction works", extraction is not None)
    check("Has compound name", bool(extraction.compound_name))

    return extraction, trial

failed_result = asyncio.run(test_extraction_failure())


# ============================================================
# TEST 4: Classify the successful trial
# ============================================================
section("4. Classify Successful Trial")

async def test_classification_success():
    classifier = Classifier(client)
    classification = await classifier.classify(extraction_success)

    check("Classification returned", classification is not None)

    # FailureClassification has: primary_failure_mode, confidence, reasoning
    # The _raw dict has: trial_outcome, failure_modes, edges_to_update, etc.
    raw = getattr(classification, "_raw", {})
    trial_outcome = raw.get("trial_outcome", "unknown")
    check("Has trial outcome", trial_outcome != "unknown",
          f"outcome={trial_outcome}")
    check("Outcome is success",
          trial_outcome in ("success", "partial"),
          f"got: {trial_outcome}")
    check("Has reasoning",
          classification.reasoning is not None and len(classification.reasoning) > 50,
          f"{len(classification.reasoning)} chars")
    check("Has confidence",
          classification.confidence is not None,
          f"confidence={classification.confidence}")

    raw_edges = raw.get("edges_to_update", [])
    if raw_edges:
        print(f"\n  Edge updates suggested: {len(raw_edges)}")
        for eu in raw_edges[:3]:
            print(f"    {eu.get('edge_type')}: {eu.get('direction')} "
                  f"({eu.get('source_entity')} → {eu.get('target_entity')})")

    return classification

classification_success = asyncio.run(test_classification_success())


# ============================================================
# TEST 5: Classify the failed trial
# ============================================================
section("5. Classify Failed Trial")

async def test_classification_failure():
    if failed_result is None:
        check("Failed trial available", False, "no failed trial found in test 3")
        return None

    extraction, trial = failed_result
    classifier = Classifier(client)
    classification = await classifier.classify(extraction)

    check("Classification returned", classification is not None)

    raw = getattr(classification, "_raw", {})
    raw_modes = raw.get("failure_modes", [])
    check("Has failure modes",
          len(raw_modes) > 0,
          f"found {len(raw_modes)} modes")

    if raw_modes:
        top_mode = raw_modes[0]
        check("Top mode has confidence",
              top_mode.get("confidence") is not None,
              f"mode={top_mode.get('mode')}, confidence={top_mode.get('confidence')}")
        check("Top mode has evidence string",
              top_mode.get("evidence") is not None and len(top_mode.get("evidence", "")) > 10)

        print(f"\n  Top failure mode: {top_mode.get('mode')}")
        print(f"  Confidence: {top_mode.get('confidence')}")
        print(f"  Evidence: {str(top_mode.get('evidence', ''))[:100]}...")

    raw_edges = raw.get("edges_to_update", [])
    check("Has edge updates",
          len(raw_edges) > 0,
          f"{len(raw_edges)} edge updates")

    if raw_edges:
        print(f"\n  Edge updates:")
        for eu in raw_edges[:5]:
            print(f"    {eu.get('direction')} {eu.get('edge_type')}: "
                  f"{eu.get('source_entity')} → {eu.get('target_entity')} "
                  f"(magnitude={eu.get('magnitude')})")

    needs_review = getattr(classification, "_needs_review", None)
    check("Expert review flag set",
          needs_review is not None,
          f"needs_review={needs_review}")

    return classification

classification_failure = asyncio.run(test_classification_failure())


# ============================================================
# TEST 6: Edge attribution on real graph
# ============================================================
section("6. Edge Attribution (Graph Updates)")

async def test_attribution():
    # Build a small graph with relevant nodes
    store = GraphStore()

    # Add nodes that match the nivolumab trial
    store.add_node(CompoundNode(id="DRUG:nivolumab", name="Nivolumab",
                                modality=Modality.ANTIBODY))
    store.add_node(TargetNode(id="TARGET:PD1", name="PD-1",
                              gene_symbol="PDCD1"))
    store.add_node(IndicationNode(id="DISEASE:melanoma",
                                  name="Melanoma"))
    store.add_node(EndpointNode(id="ENDPOINT:os",
                                name="Overall Survival",
                                endpoint_type=EndpointType.PRIMARY,
                                regulatory_status=RegulatoryStatus.ACCEPTED))

    # Add edges with uniform priors
    for src, tgt, etype in [
        ("DRUG:nivolumab", "TARGET:PD1", EdgeType.BINDS_TO),
        ("TARGET:PD1", "DISEASE:melanoma", EdgeType.BIOLOGY_DRIVES),
        ("ENDPOINT:os", "DISEASE:melanoma", EdgeType.ENDPOINT_CAPTURES),
    ]:
        store.add_edge(GraphEdge(
            source_id=src, target_id=tgt, edge_type=etype,
            belief=EdgeBeliefState(alpha=1.0, beta=1.0)
        ))

    # Create a trial subgraph
    trial_sub = TrialSubgraph(
        trial_id="NCT01721772",
        compound_id="DRUG:nivolumab",
        target_id="TARGET:PD1",
        mechanism_id="MECH:checkpoint_blockade",
        biology_id="BIO:antitumor_immunity",
        indication_id="DISEASE:melanoma",
        endpoint_id="ENDPOINT:os",
        population_id="POP:untreated_melanoma",
        outcome="success",
        phase="3",
    )

    # Get pre-update beliefs
    pre_binds = store.get_edge_belief("DRUG:nivolumab", "TARGET:PD1",
                                       EdgeType.BINDS_TO)
    print(f"\n  Pre-update binds_to: Beta({pre_binds.alpha}, {pre_binds.beta})"
          f" -> P={pre_binds.expected_probability:.3f}")

    # Run attribution
    attributor = Attributor(store)
    updates = attributor.attribute(classification_success, trial_sub)

    check("Attribution produced updates",
          updates is not None and len(updates) > 0,
          f"{len(updates)} updates")

    # Summarize updates
    if updates:
        summary = attributor.apply_updates(updates)
        check("Updates applied", summary is not None)

        # Check that beliefs actually changed
        post_binds = store.get_edge_belief("DRUG:nivolumab", "TARGET:PD1",
                                            EdgeType.BINDS_TO)
        print(f"  Post-update binds_to: Beta({post_binds.alpha:.1f}, "
              f"{post_binds.beta:.1f})"
              f" -> P={post_binds.expected_probability:.3f}")

        belief_changed = (post_binds.alpha != pre_binds.alpha or
                         post_binds.beta != pre_binds.beta)
        check("Beliefs actually changed", belief_changed,
              f"alpha: {pre_binds.alpha} -> {post_binds.alpha:.1f}")

        if belief_changed:
            direction = "strengthened" if post_binds.expected_probability > pre_binds.expected_probability else "weakened"
            check(f"Successful trial {direction} the edge",
                  post_binds.expected_probability > pre_binds.expected_probability,
                  f"P: {pre_binds.expected_probability:.3f} -> {post_binds.expected_probability:.3f}")

        # Print all updated edges
        print(f"\n  All edge updates applied:")
        for u in updates:
            print(f"    {u.edge_type.value}: {u.source_id} → {u.target_id}")
            pre_p = u.pre_update_belief.expected_probability
            post_p = u.post_update_belief.expected_probability
            print(f"      P: {pre_p:.3f} → {post_p:.3f}")

try:
    asyncio.run(test_attribution())
except Exception as e:
    check("Attribution pipeline", False, str(e))
    import traceback
    traceback.print_exc()


# ============================================================
# TEST 7: Full pipeline end-to-end
# ============================================================
section("7. Full Pipeline: NCT ID → Graph Updates")

async def test_full_pipeline():
    ct_client = ClinicalTrialsClient()
    extractor = Extractor(client)
    classifier = Classifier(client)
    store = GraphStore()

    # Populate minimal graph
    store.add_node(CompoundNode(id="DRUG:pembrolizumab",
                                name="Pembrolizumab", modality=Modality.ANTIBODY))
    store.add_node(TargetNode(id="TARGET:PD1", name="PD-1",
                              gene_symbol="PDCD1"))
    store.add_node(IndicationNode(id="DISEASE:nsclc",
                                  name="Non-Small Cell Lung Cancer"))
    store.add_node(EndpointNode(id="ENDPOINT:os",
                                name="Overall Survival",
                                endpoint_type=EndpointType.PRIMARY,
                                regulatory_status=RegulatoryStatus.ACCEPTED))

    for src, tgt, etype in [
        ("DRUG:pembrolizumab", "TARGET:PD1", EdgeType.BINDS_TO),
        ("TARGET:PD1", "DISEASE:nsclc", EdgeType.BIOLOGY_DRIVES),
        ("ENDPOINT:os", "DISEASE:nsclc", EdgeType.ENDPOINT_CAPTURES),
    ]:
        store.add_edge(GraphEdge(
            source_id=src, target_id=tgt, edge_type=etype,
            belief=EdgeBeliefState(alpha=1.0, beta=1.0)
        ))

    # Fetch a real trial
    trial = await ct_client.get_study("NCT02142738")  # KEYNOTE-010
    check("Fetched KEYNOTE-010", trial is not None,
          trial.title[:50] + "...")

    # Extract
    extraction = await extractor.extract(trial)
    check("Extracted", extraction is not None)

    # Classify
    classification = await classifier.classify(extraction)
    check("Classified", classification is not None)
    raw = getattr(classification, "_raw", {})
    trial_outcome = raw.get("trial_outcome", "unknown")
    print(f"    outcome={trial_outcome}")

    # Attribute
    trial_sub = TrialSubgraph(
        trial_id="NCT02142738",
        compound_id="DRUG:pembrolizumab",
        target_id="TARGET:PD1",
        mechanism_id="MECH:checkpoint_blockade",
        biology_id="BIO:antitumor_immunity",
        indication_id="DISEASE:nsclc",
        endpoint_id="ENDPOINT:os",
        population_id="POP:pdl1_positive_nsclc",
        outcome=trial_outcome,
        phase="3",
    )

    attributor = Attributor(store)
    updates = attributor.attribute(classification, trial_sub)
    check("Attributed", len(updates) > 0, f"{len(updates)} edge updates")

    if updates:
        attributor.apply_updates(updates)

    # Final graph state
    stats = store.stats()
    print(f"\n  Final graph state:")
    for k, v in stats.items():
        print(f"    {k}: {v}")

    binds = store.get_edge_belief("DRUG:pembrolizumab", "TARGET:PD1",
                                   EdgeType.BINDS_TO)
    print(f"\n  pembrolizumab -> PD-1 (binds_to):")
    print(f"    Beta({binds.alpha:.1f}, {binds.beta:.1f})")
    print(f"    P = {binds.expected_probability:.3f}")
    print(f"    Evidence count = {len(binds.evidence)}")

asyncio.run(test_full_pipeline())


# ============================================================
# SUMMARY
# ============================================================
section("SUMMARY")

passed = sum(1 for _, ok in results if ok)
failed = sum(1 for _, ok in results if not ok)
total = len(results)

print(f"\n  {PASS} Passed: {passed}/{total}")
if failed:
    print(f"  {FAIL} Failed: {failed}/{total}")
    print(f"\n  Failed tests:")
    for name, ok in results:
        if not ok:
            print(f"    {FAIL} {name}")

print(f"\n{'='*60}")
if failed == 0:
    print(f"  \033[92mAnnotation pipeline fully operational.\033[0m")
    print(f"  Real trials → structured extraction → failure classification")
    print(f"  → edge attribution → graph updates. It works.")
    print(f"  Ready for Sprint 9 (Bayesian inference engine).")
elif failed <= 3:
    print(f"  \033[93mMostly working. Fix failures before Sprint 9.\033[0m")
else:
    print(f"  \033[91mSignificant issues in the annotation pipeline.\033[0m")
print(f"{'='*60}\n")
