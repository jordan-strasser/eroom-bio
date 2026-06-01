"""End-to-end validation: does the PubMed enrichment ground torcetrapib's
off-target toxicity through the REAL attribution + prediction pipeline?

Loads the ceiling graph (torcetrapib in-sample, currently 0 causes_ae edges),
enriches its extraction via the committed PubMed cache + the build-side
``maybe_enrich_by_nct`` hook, runs ``attribute_adverse_events`` (creating real
causes_ae edges), and predicts before/after. The MedDRA normalization for the 2
AE terms is the only LLM call (existing pipeline; cached after first run).

    python -m scripts.validate_torcetrapib_enrichment
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import anthropic

from src.annotation.attributor import Attributor
from src.annotation.extractor import _parse_extraction_response
from src.annotation.meddra import MeddraCache
from src.annotation.pubmed_safety import maybe_enrich_by_nct
from src.annotation.taxonomy import TrialExtraction
from src.graph.models import EdgeType
from src.graph.store import GraphStore
from src.prediction.path_query import predict_clinical_hypothesis

NCT = "NCT00134264"


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graph", default="data/exports/mi_v2_ceiling_annotated.json")
    args = ap.parse_args()

    g = GraphStore()
    g.import_snapshot(args.graph)
    ts = g.trial_subgraphs.get(NCT)
    if not ts or not ts.chains:
        print(f"torcetrapib {NCT} not in {args.graph}")
        return 1
    ch = next((c for c in ts.chains if "torcetrapib" in (c.compound_id or "").lower()),
              ts.chains[0])
    print(f"chain: {ch.compound_id} -> ... -> {ch.indication_id}\n")

    def predict():
        return predict_clinical_hypothesis(g, ch.compound_id, ch.indication_id, n_samples=4000)

    def n_causes_ae():
        return len(list(g.get_neighboring_edges(ch.compound_id, edge_types=[EdgeType.CAUSES_AE])))

    r0 = predict()
    print(f"BEFORE  causes_ae={n_causes_ae()}  efficacy={r0.efficacy_probability:.3f}  "
          f"safety_penalty={r0.safety_penalty:.3f}  overall={r0.overall_probability:.3f}")

    # Enrich the extraction exactly as attribution now does, then attribute.
    ext_path = Path("data/annotations") / f"{NCT}_extraction.json"
    extraction = (_parse_extraction_response(json.loads(ext_path.read_text()), NCT)
                  if ext_path.exists() else TrialExtraction(trial_id=NCT))
    extraction = maybe_enrich_by_nct(extraction, NCT)
    print(f"\nenriched extraction AEs: {[a.term for a in extraction.adverse_events]}")

    client = anthropic.AsyncAnthropic(timeout=60.0)
    ae_updates = await Attributor(g).attribute_adverse_events(
        ts, extraction, client=client, meddra_cache=MeddraCache(),
    )
    print(f"attribute_adverse_events -> {len(ae_updates)} causes_ae updates\n")

    r1 = predict()
    print(f"AFTER   causes_ae={n_causes_ae()}  efficacy={r1.efficacy_probability:.3f}  "
          f"safety_penalty={r1.safety_penalty:.3f}  overall={r1.overall_probability:.3f}")
    verdict = "FAILURE ✓ (direction-correct)" if r1.overall_probability < 0.5 else "success ✗ (still wrong)"
    print(f"\ntorcetrapib (literature outcome = FAILURE): "
          f"{r0.overall_probability:.3f} -> {r1.overall_probability:.3f}  {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
