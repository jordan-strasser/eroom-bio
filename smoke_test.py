"""
Eroom Bio—Smoke Test Suite
Run this after Sprints 1-5 to verify the system actually works
with real data, not just mocked unit tests.

Usage: python smoke_test.py
"""

import asyncio
import json
import sys
import os
from datetime import datetime, date
from pathlib import Path

# Add src to path
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
# TEST 1: Can you import everything?
# ============================================================
section("1. Import Check")

try:
    from graph.models import (
        CompoundNode, TargetNode, MechanismNode, BiologyNode,
        BiomarkerNode, PopulationNode, EndpointNode, IndicationNode,
        GraphEdge, EdgeBeliefState, EvidenceRecord, TrialSubgraph,
        EdgeType, EvidenceType
    )
    check("All models import", True)
except ImportError as e:
    check("All models import", False, str(e))

try:
    from graph.store import GraphStore
    check("GraphStore imports", True)
except ImportError as e:
    check("GraphStore imports", False, str(e))

try:
    from ingestion.clinicaltrials import ClinicalTrialsClient, TrialRecord
    check("ClinicalTrials client imports", True)
except ImportError as e:
    check("ClinicalTrials client imports", False, str(e))

try:
    from ingestion.opentargets import OpenTargetsClient
    check("Open Targets client imports", True)
except ImportError as e:
    check("Open Targets client imports", False, str(e))


# ============================================================
# TEST 2: Do the models actually validate?
# ============================================================
section("2. Model Validation")

try:
    compound = CompoundNode(
        id="DRUG:imatinib",
        name="Imatinib",
        modality="small_molecule",
    )
    check("CompoundNode creates", True, f"id={compound.id}")
except Exception as e:
    check("CompoundNode creates", False, str(e))

try:
    target = TargetNode(
        id="UNIPROT:P00519",
        name="BCR-ABL",
        gene_symbol="ABL1",
    )
    check("TargetNode creates", True, f"gene={target.gene_symbol}")
except Exception as e:
    check("TargetNode creates", False, str(e))

try:
    belief = EdgeBeliefState(alpha=18.0, beta=3.0)
    prob = belief.expected_probability
    check("EdgeBeliefState math works", True,
          f"Beta(18,3) -> P={prob:.3f}")
    check("Expected probability correct",
          abs(prob - 18/21) < 0.001,
          f"expected {18/21:.3f}, got {prob:.3f}")
except Exception as e:
    check("EdgeBeliefState math works", False, str(e))

# Test validation catches bad data
try:
    bad = EdgeBeliefState(alpha=-1.0, beta=1.0)
    check("Rejects negative alpha", False, "should have raised error")
except Exception:
    check("Rejects negative alpha", True)

try:
    bad_ev = EvidenceRecord(
        source_id="test",
        source_type=list(EvidenceType)[0],
        quality_score=1.5,  # should fail: must be 0-1
        direction="supporting",
        magnitude=1.0,
        timestamp=datetime.now(),
    )
    check("Rejects quality_score > 1", False, "should have raised error")
except Exception:
    check("Rejects quality_score > 1", True)


# ============================================================
# TEST 3: Does the graph store work end-to-end?
# ============================================================
section("3. Graph Store Operations")

try:
    store = GraphStore()
    check("GraphStore creates", True)
except Exception as e:
    check("GraphStore creates", False, str(e))

try:
    compound = CompoundNode(id="DRUG:imatinib", name="Imatinib", modality="small_molecule")
    target = TargetNode(id="UNIPROT:P00519", name="BCR-ABL", gene_symbol="ABL1")
    indication = IndicationNode(id="DISEASE:cml", name="Chronic Myeloid Leukemia")

    store.add_node(compound)
    store.add_node(target)
    store.add_node(indication)

    retrieved = store.get_node("DRUG:imatinib")
    check("Add + retrieve node", retrieved is not None,
          f"got {type(retrieved)}")
except Exception as e:
    check("Add + retrieve node", False, str(e))

try:
    evidence = EvidenceRecord(
        source_id="NCT00006343",
        source_type=EvidenceType.CLINICAL_PHASE3,
        quality_score=0.95,
        direction="supporting",
        magnitude=0.9,
        timestamp=datetime.now(),
    )

    edge = GraphEdge(
        source_id="DRUG:imatinib",
        target_id="UNIPROT:P00519",
        edge_type=EdgeType.BINDS_TO if hasattr(EdgeType, 'BINDS_TO')
                  else list(EdgeType)[0],
        belief=EdgeBeliefState(alpha=1.0, beta=1.0),
    )
    store.add_edge(edge)
    check("Add edge", True)

    # Update with evidence
    edge_type = EdgeType.BINDS_TO if hasattr(EdgeType, 'BINDS_TO') else list(EdgeType)[0]
    updated = store.update_edge_belief(
        "DRUG:imatinib", "UNIPROT:P00519",
        edge_type, evidence
    )
    check("Update edge belief", updated is not None,
          f"P={updated.expected_probability:.3f}")
    check("Alpha increased after supporting evidence",
          updated.alpha > 1.0,
          f"alpha={updated.alpha:.2f}")
except Exception as e:
    check("Edge operations", False, str(e))

# Snapshot roundtrip
try:
    tmp_path = "/tmp/eroom_smoke_test.json"
    store.export_snapshot(tmp_path)
    check("Export snapshot", os.path.exists(tmp_path),
          f"size={os.path.getsize(tmp_path)} bytes")

    store2 = GraphStore()
    store2.import_snapshot(tmp_path)
    node2 = store2.get_node("DRUG:imatinib")
    check("Import snapshot + retrieve",
          node2 is not None, "roundtrip works")
    os.remove(tmp_path)
except Exception as e:
    check("Snapshot roundtrip", False, str(e))

# Stats
try:
    stats = store.stats()
    check("Stats returns data", isinstance(stats, dict),
          f"keys: {list(stats.keys())[:5]}")
except Exception as e:
    check("Stats", False, str(e))


# ============================================================
# TEST 4: Can you hit the real ClinicalTrials.gov API?
# ============================================================
section("4. ClinicalTrials.gov API (live)")

async def test_ctgov():
    try:
        client = ClinicalTrialsClient()
        study = await client.get_study("NCT02813785")
        check("Fetch single study", study is not None,
              f"title: {study.title[:60]}...")
        check("NCT ID correct", study.nct_id == "NCT02813785")
        check("Has phase", study.phase is not None, f"phase={study.phase}")
        return True
    except Exception as e:
        check("Fetch single study", False, str(e))
        return False

async def test_ctgov_search():
    try:
        client = ClinicalTrialsClient()
        trials = await client.search(
            condition="melanoma",
            phase="PHASE3",
            has_results=True,
            max_results=5
        )
        check("Search returns results", len(trials) > 0,
              f"got {len(trials)} trials")
        if trials:
            t = trials[0]
            check("Trial has conditions", len(t.conditions) > 0,
                  f"conditions: {t.conditions[:2]}")
            check("Trial has interventions",
                  t.interventions is not None)
    except Exception as e:
        check("Search", False, str(e))

try:
    asyncio.run(test_ctgov())
    asyncio.run(test_ctgov_search())
except Exception as e:
    check("ClinicalTrials.gov API", False, f"async error: {e}")


# ============================================================
# TEST 5: Can you hit the real Open Targets API?
# ============================================================
section("5. Open Targets API (live)")

async def test_opentargets():
    try:
        client = OpenTargetsClient()

        # Search for BRAF
        target = await client.search_target("BRAF")
        check("Search target (BRAF)", target is not None,
              f"got: {target}")

        # Get associations for melanoma
        assocs = await client.get_disease_associations("EFO_0000389")
        check("Disease associations", len(assocs) > 0,
              f"got {len(assocs)} target-disease pairs for melanoma")
        return True
    except Exception as e:
        check("Open Targets API", False, str(e))
        return False

try:
    asyncio.run(test_opentargets())
except Exception as e:
    check("Open Targets API", False, f"async error: {e}")


# ============================================================
# TEST 6: Can you actually populate a graph from real data?
# ============================================================
section("6. End-to-End Graph Population")

async def test_population():
    try:
        from graph.populate import PopulationPipeline

        store = GraphStore()
        pipeline = PopulationPipeline(store)

        # Try populating with just 10 trials to keep it fast
        await pipeline.populate_oncology(max_trials=10)

        stats = store.stats()
        check("Graph has nodes after population",
              stats.get("node_count", 0) > 0,
              f"nodes={stats.get('node_count')}, edges={stats.get('edge_count')}")

        print(f"\n  Graph stats after population:")
        for k, v in stats.items():
            print(f"    {k}: {v}")

        return True
    except ImportError:
        check("PopulationPipeline importable", False,
              "src/graph/populate.py may not exist yet")
        return False
    except Exception as e:
        check("Graph population", False, str(e))
        return False

try:
    asyncio.run(test_population())
except Exception as e:
    check("Graph population", False, f"async error: {e}")


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
    print(f"  \033[92mAll checks passed. Sprints 1-5 are solid.\033[0m")
    print(f"  Ready for Sprint 6 (failure taxonomy + annotation).")
elif failed <= 3:
    print(f"  \033[93mMostly working. Fix the failures above before Sprint 6.\033[0m")
else:
    print(f"  \033[91mSignificant issues. Debug before moving forward.\033[0m")
print(f"{'='*60}\n")
