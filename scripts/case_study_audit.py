"""Round-18 case-study audit against the multi_indication graph.

Loads the 5 well-known drug-development case studies from
`audit/cross-trial-learning/trial_ids.json` + `key_trials.md`, runs
`predict_clinical_hypothesis` against the multi-indication snapshot
for each, and prints a side-by-side decomposition vs the
literature-documented failure mode. Output also written to
`audit/cross-trial-learning/results.md`.

The case studies test five different failure / success mechanisms:

  1. Nivolumab in melanoma (CheckMate-067)  — clean SUCCESS
  2. Solanezumab in Alzheimer's              — biology→endpoint flat
  3. Bevacizumab in adjuvant colon (AVANT)   — wrong population
  4. Torcetrapib (ILLUMINATE)                — compound-specific toxicity
  5. Selumetinib in thyroid                  — right population selection

For each: the chain's full edge decomposition, the system's identified
weakest_link / bottleneck, and a 'matches / partial / disagrees'
verdict against the literature failure mode.

Usage:
    python -m scripts.case_study_audit
    python -m scripts.case_study_audit --graph data/exports/multi_indication_annotated.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import numpy as np

from scripts.eval_holdout_v2 import (
    _load_canonicalization_cache,
    _training_used_nodes,
    resolve_chain,
)
from src.graph.store import GraphStore
from src.ingestion.clinicaltrials import ClinicalTrialsClient
from src.prediction.path_query import predict_clinical_hypothesis

GRAPH_PATH = Path("data/exports/multi_indication_annotated.json")
TRIAL_IDS_JSON = Path("audit/cross-trial-learning/trial_ids.json")
RESULTS_MD = Path("audit/cross-trial-learning/results.md")


# Per-case-study expected bottleneck edge + literature failure-mode tag.
# Used for the side-by-side "matches / partial / disagrees" verdict.
_EXPECTED: dict[str, dict[str, str]] = {
    "nivolumab_melanoma_checkmate_067": {
        "literature_outcome": "success",
        "expected_bottleneck": "none (clean success)",
        "literature_mode": (
            "Positive control. All chain edges should be strong: "
            "compound→target (anti-PD-1 binds PDCD1), "
            "target→mechanism (checkpoint blockade), "
            "biology→indication (PD-1 signaling drives melanoma immune "
            "evasion). System should predict HIGH P(success); if any "
            "edge is weak, the chain is miscalibrated."
        ),
    },
    "solanezumab_alzheimer": {
        "literature_outcome": "failure",
        "expected_bottleneck": "biology_drives",
        "literature_mode": (
            "biology→endpoint flat. Drug bound amyloid-beta and cleared "
            "plaques (mechanism worked, biomarker moved), but cognition "
            "did not improve. The amyloid hypothesis link from biology "
            "to clinical disease is broken — system should flag "
            "biology_drives (R-HSA-amyloid → alzheimer_disease) as "
            "weakest. Three Phase III failures (EXPEDITION 1, 2, 3)."
        ),
    },
    "bevacizumab_adjuvant_colon_avant": {
        "literature_outcome": "failure",
        "expected_bottleneck": "responds_differently",
        "literature_mode": (
            "Wrong population. Bevacizumab + oxaliplatin works in "
            "metastatic colorectal cancer but fails in the adjuvant "
            "setting (minimal residual disease post-surgery). Same "
            "compound, same target (VEGF), same disease, different "
            "clinical context. System should show strong upstream "
            "edges (affects, modulates_via, mechanism_affects) and "
            "flag responds_differently as the weakest link."
        ),
    },
    "torcetrapib_illuminate": {
        "literature_outcome": "failure",
        "expected_bottleneck": "causes_ae",
        "literature_mode": (
            "Compound-specific toxicity. Torcetrapib worked "
            "mechanistically (raised HDL dramatically via CETP "
            "inhibition) but caused excess deaths in ILLUMINATE via "
            "off-target hypertension. Compound-specific safety, not "
            "mechanism failure. System should NOT flag any chain edge "
            "as weak; expects safety_risks to surface compound-specific "
            "AE evidence."
        ),
    },
    "selumetinib_thyroid": {
        "literature_outcome": "success",
        "expected_bottleneck": "responds_differently (subgroup-specific)",
        "literature_mode": (
            "Right population selection. MEK inhibitor failed in "
            "unselected NSCLC but succeeded in BRAF-mutant thyroid "
            "(Ho 2013). The mechanism works, but only in the right "
            "molecular subtype. System's responds_differently edge for "
            "the trial's BRAF-mutant / RAI-refractory population "
            "should pull P(success) up, contrasted with hypothetical "
            "unselected populations."
        ),
    },
}


async def _audit_one(
    name: str,
    nct_id: str,
    story: str,
    expected: dict[str, str],
    graph: GraphStore,
    canon_cache: dict[str, str],
    ct: ClinicalTrialsClient,
) -> dict:
    """Run prediction + literature comparison for one case study."""
    # Pull trial record (eligibility criteria + conditions for resolver)
    trial = await ct.get_study(nct_id)
    if trial is None:
        return {
            "name": name, "nct": nct_id, "story": story,
            "error": "CT.gov fetch returned None",
        }

    # Try to load cached extraction so the resolver has therapeutic_hypothesis
    ann_dir = Path("data/annotations")
    ext_path = ann_dir / f"{nct_id}_extraction.json"
    if ext_path.exists():
        extraction = json.loads(ext_path.read_text())
    else:
        extraction = {"nct_id": nct_id}

    chain = resolve_chain(extraction, list(trial.conditions), graph, canon_cache)

    # Build predict kwargs.
    kwargs: dict = {}
    for k in ("target_id", "mechanism_id", "biology_id",
              "endpoint_id", "population_id"):
        v = chain.get(k)
        if v and v != "UNKNOWN":
            kwargs[k] = v

    indication_id = chain["indication_id"]
    if indication_id == "UNKNOWN" or indication_id not in graph._graph:  # noqa: SLF001
        return {
            "name": name, "nct": nct_id, "story": story,
            "chain": chain,
            "error": f"indication {indication_id!r} not in graph",
        }

    np.random.seed(42)
    result = predict_clinical_hypothesis(
        graph,
        chain.get("compound_id"),
        indication_id,
        n_samples=5000,
        **kwargs,
    )

    # Identify the system's bottleneck and verdict.
    weakest_etype = (
        result.weakest_link.edge_type.value
        if result.weakest_link else "(no edges)"
    )
    expected_bot = expected["expected_bottleneck"]
    # Round-20: a meaningful safety_penalty means safety is dragging
    # P(success) down on its own. For safety-driven failure cases
    # (torcetrapib-class — chain works, drug kills), this turns a
    # would-be PARTIAL/DISAGREE into a MATCH-VIA-SAFETY since the
    # system DID surface the failure cause, just via the AE channel
    # instead of the mechanistic-chain channel.
    safety_significant = result.safety_penalty > 0.10

    if expected_bot == "none (clean success)":
        verdict = (
            "MATCH" if result.overall_probability >= 0.6
            else "PARTIAL" if result.overall_probability >= 0.5
            else "DISAGREE"
        )
    elif "causes_ae" in expected_bot:
        # Safety-driven failure expected. MATCH-VIA-SAFETY when the
        # safety_penalty dragged overall_probability low enough to
        # predict the failure direction; PARTIAL when safety surfaced
        # but didn't pull below 0.5; DISAGREE when neither.
        if safety_significant and result.overall_probability < 0.5:
            verdict = "MATCH-VIA-SAFETY"
        elif safety_significant:
            verdict = "PARTIAL"
        elif result.overall_probability >= 0.5:
            verdict = "DISAGREE"
        else:
            verdict = "PARTIAL"
    else:
        # Mechanistic-chain bottleneck expected. MATCH when the chain
        # bottleneck matches the literature; PARTIAL when the system
        # got the direction right via safety even though the chain
        # bottleneck didn't match.
        if weakest_etype in expected_bot:
            verdict = "MATCH"
        elif safety_significant and result.overall_probability < 0.5:
            verdict = "MATCH-VIA-SAFETY"
        elif result.overall_probability < 0.5:
            verdict = "PARTIAL"
        else:
            verdict = "DISAGREE"

    return {
        "name": name,
        "nct": nct_id,
        "story": story,
        "literature_outcome": expected["literature_outcome"],
        "expected_bottleneck": expected_bot,
        "literature_mode": expected["literature_mode"],
        "chain": chain,
        # Round-20: report all three numbers so safety vs efficacy
        # drag is visible in the audit output.
        "p_success": result.overall_probability,
        "efficacy_probability": result.efficacy_probability,
        "safety_penalty": result.safety_penalty,
        "ci_lower": result.ci_lower,
        "ci_upper": result.ci_upper,
        "edge_contribs": result.edge_contributions,
        "weakest_etype": weakest_etype,
        "weakest_link": (
            f"{result.weakest_link.edge_type.value}: "
            f"{result.weakest_link.source_id} → "
            f"{result.weakest_link.target_id}"
            if result.weakest_link else "(none)"
        ),
        "safety_risks": result.safety_risks,
        "verdict": verdict,
    }


def _format_md(rows: list[dict]) -> str:
    """Render the audit rows as a comparison markdown file."""
    out: list[str] = []
    out.append("# Round-18 case-study audit — multi-indication graph\n")
    out.append(
        "Five well-known drug-development stories tested against the\n"
        "`multi_indication` graph (melanoma_145 + alzheimers + colorectal\n"
        "+ atherosclerosis + hypercholesterolemia + NSCLC + thyroid). For\n"
        "each: full edge decomposition + the system's identified\n"
        "weakest_link + a verdict against the literature failure mode\n"
        "documented in `key_trials.md`.\n\n"
    )
    for r in rows:
        out.append("\n---\n")
        out.append(f"## {r['name']}\n")
        out.append(f"**NCT**: `{r['nct']}`\n")
        out.append(f"**Story**: {r['story']}\n\n")

        if "error" in r:
            out.append(f"**Error**: {r['error']}\n")
            if "chain" in r:
                out.append("\nResolved chain (despite error):\n")
                out.append("```\n")
                for k, v in r["chain"].items():
                    out.append(f"  {k}: {v}\n")
                out.append("```\n")
            continue

        out.append(f"### Literature failure mode (expected)\n")
        out.append(f"- **Outcome**: {r['literature_outcome']}\n")
        out.append(f"- **Expected bottleneck**: `{r['expected_bottleneck']}`\n")
        out.append(f"- {r['literature_mode']}\n\n")

        out.append(f"### System prediction\n")
        eff = r.get("efficacy_probability")
        pen = r.get("safety_penalty")
        if eff is not None and pen is not None:
            out.append(
                f"- **P(success) = {r['p_success']:.3f}** "
                f"= efficacy {eff:.3f} × (1 − safety {pen:.3f}) "
                f"(95% CI [{r['ci_lower']:.3f}, {r['ci_upper']:.3f}])\n"
            )
        else:
            out.append(
                f"- **P(success) = {r['p_success']:.3f}** "
                f"(95% CI [{r['ci_lower']:.3f}, {r['ci_upper']:.3f}])\n"
            )
        out.append(f"- **Weakest link**: `{r['weakest_link']}`\n")
        out.append(f"- **Verdict**: {r['verdict']}\n\n")

        c = r["chain"]
        out.append("### Resolved chain\n")
        out.append("```\n")
        out.append(
            f"  compound:    {c.get('compound_str') or c.get('compound_id') or '(unknown)'}\n"
        )
        for k in ("target_id", "mechanism_id", "biology_id",
                 "indication_id", "endpoint_id", "population_id"):
            out.append(f"  {k:<12} {c.get(k, '?')}\n")
        out.append("```\n\n")

        out.append("### Edge decomposition\n")
        out.append("```\n")
        for ec in r["edge_contribs"]:
            a = ec.belief.alpha
            b = ec.belief.beta
            ep = ec.belief.expected_probability
            n_eff = ec.belief.evidence_strength
            direction = "UP  ↑" if ep > 0.5 else ("DN  ↓" if ep < 0.5 else "==  •")
            out.append(
                f"  [{direction}] {ec.edge_type.value:<22} {ec.source_id} → {ec.target_id}\n"
                f"       Beta({a:.2f}, {b:.2f})  E[p]={ep:.3f}  "
                f"n_eff={n_eff:.2f}  bottleneck={ec.bottleneck_score:.3f}\n"
            )
        out.append("```\n\n")

        if r.get("safety_risks"):
            out.append("### Safety risks (folded into P(success) via safety_penalty)\n")
            out.append("```\n")
            for sr in r["safety_risks"][:8]:
                out.append(
                    f"  {sr.ae_name:<30}  source={sr.source}  "
                    f"belief={sr.belief_probability:.3f}  "
                    f"n_eff={sr.evidence_strength:.1f}\n"
                )
            out.append("```\n\n")

    out.append("\n---\n## Summary\n\n")
    out.append(
        "| Case study | NCT | P(success) | Efficacy | Safety drag | "
        "Weakest link | Verdict |\n"
    )
    out.append("|---|---|---:|---:|---:|---|---|\n")
    for r in rows:
        if "error" in r:
            out.append(
                f"| {r['name']} | `{r['nct']}` | — | — | — | — | "
                f"error: {r['error']} |\n"
            )
        else:
            eff = r.get("efficacy_probability")
            pen = r.get("safety_penalty")
            eff_cell = f"{eff:.3f}" if eff is not None else "—"
            pen_cell = f"{pen:.3f}" if pen is not None else "—"
            out.append(
                f"| {r['name']} | `{r['nct']}` | "
                f"{r['p_success']:.3f} | {eff_cell} | {pen_cell} | "
                f"`{r['weakest_etype']}` | "
                f"**{r['verdict']}** |\n"
            )
    return "".join(out)


async def main(args: argparse.Namespace) -> int:
    graph = GraphStore()
    graph.import_snapshot(str(args.graph))
    stats = graph.stats()
    print(
        f"Loaded {args.graph}: {stats['node_count']} nodes, "
        f"{stats['edge_count']} edges, "
        f"{len(graph.trial_subgraphs)} trial subgraphs"
    )

    canon_cache = _load_canonicalization_cache()
    case_studies = json.loads(TRIAL_IDS_JSON.read_text())
    ct = ClinicalTrialsClient()

    rows: list[dict] = []
    for name, entry in case_studies.items():
        nct = entry["nct_id"]
        story = entry["story"]
        expected = _EXPECTED.get(name) or {
            "literature_outcome": "unknown",
            "expected_bottleneck": "unknown",
            "literature_mode": "(not documented)",
        }
        print(f"\n>> {name}  ({nct})")
        try:
            row = await _audit_one(name, nct, story, expected, graph, canon_cache, ct)
        except Exception as exc:  # noqa: BLE001
            row = {
                "name": name, "nct": nct, "story": story,
                "error": f"exception: {exc.__class__.__name__}: {exc}",
            }
        rows.append(row)
        if "error" in row:
            print(f"   ERROR: {row['error']}")
        else:
            print(
                f"   P(success)={row['p_success']:.3f}   "
                f"weakest={row['weakest_etype']}   "
                f"verdict={row['verdict']}"
            )

    md = _format_md(rows)
    RESULTS_MD.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_MD.write_text(md)
    print(f"\n  Wrote {RESULTS_MD}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=GRAPH_PATH)
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(args)))
