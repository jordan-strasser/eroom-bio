"""Round-25 migration: in-place conversion of pre-round-25 snapshots.

Takes a snapshot where populator priors were hand-set as Beta(α, β)
without evidence records and converts each curated-source edge to the
new arch (Beta(1, 1) prior + DATABASE_CURATED EvidenceRecord). Trial-
attribution records (which the OLD attributor wrote during step 4) are
preserved and replayed on top of the new prior, so the posterior reflects
both curated and trial evidence under round-25 semantics.

This is faster than a full rebuild and produces a snapshot that's
directly comparable to the round-24 clean-holdout baseline.

Run:
  .venv/bin/python scripts/migrate_snapshot_to_round25.py \\
    --in  data/exports/multi_indication_52_initial.json \\
    --out data/exports/multi_indication_52_v25_initial.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from src.graph.curated_evidence import (
    belief_from_curated_record,
    make_curated_record,
)
from src.graph.models import (
    EdgeBeliefState,
    EvidenceRecord,
    EvidenceType,
)
from src.graph.store import GraphStore
from src.inference.beliefs import (
    SupportBucket,
    apply_virtual_evidence,
    effective_n_for_evidence,
    p_obs_for_bucket,
)


# Source-tag → (bucket, evidence-type, notes_prefix) for known populator
# curated edges. Sources not listed here either pass through unchanged
# (genuine trial-attribution edges) or get reset to Beta(1, 1) (heuristic
# seeds).
#
# Each entry's EvidenceType matches what populate.py emits live for that
# source tag — see EVIDENCE_TYPE_N_EFF in src/inference/beliefs.py for
# the n_eff values picked from source character.
_CURATED_SOURCE_BUCKETS: dict[str, tuple[SupportBucket, EvidenceType, str]] = {
    "opentargets": (
        SupportBucket.STRONG_SUPPORT, EvidenceType.DATABASE_OT_DIRECT,
        "OT drug-target binding",
    ),
    "chembl_rest_fallback": (
        SupportBucket.STRONG_SUPPORT, EvidenceType.DATABASE_CHEMBL,
        "ChEMBL action_type",
    ),
    "mab_fallback": (
        SupportBucket.STRONG_SUPPORT, EvidenceType.DATABASE_MAB_TABLE,
        "mAb-curated-table",
    ),
    "trial_inference": (
        SupportBucket.MODERATE_SUPPORT, EvidenceType.DATABASE_CHEMBL,
        "trial → mechanism inference",
    ),
    "reactome_target_lookup": (
        SupportBucket.MODERATE_SUPPORT, EvidenceType.DATABASE_REACTOME_GO,
        "Reactome target lookup",
    ),
    "quickgo_target_lookup": (
        SupportBucket.MODERATE_SUPPORT, EvidenceType.DATABASE_REACTOME_GO,
        "GO term lookup",
    ),
    "go_augmentation": (
        SupportBucket.MODERATE_SUPPORT, EvidenceType.DATABASE_REACTOME_GO,
        "GO pathway augmentation",
    ),
    "trial_biology_fallback": (
        SupportBucket.WEAK_SUPPORT, EvidenceType.DATABASE_FALLBACK,
        "synthesized biology fallback",
    ),
    "combo_inherit": (
        SupportBucket.MODERATE_SUPPORT, EvidenceType.DATABASE_FALLBACK,
        "combo-arm inherited AFFECTS",
    ),
    "cross_reference": (
        SupportBucket.WEAK_SUPPORT, EvidenceType.DATABASE_CROSS_REFERENCE,
        "name-match cross-reference",
    ),
    "vaccine_component_targets": (
        SupportBucket.STRONG_SUPPORT, EvidenceType.DATABASE_MAB_TABLE,
        "vaccine-component curated target",
    ),
    "cell_therapy_heuristic": (
        SupportBucket.STRONG_SUPPORT, EvidenceType.DATABASE_MAB_TABLE,
        "cell-therapy pattern match",
    ),
    "endpoint_type_prior": (
        SupportBucket.MODERATE_SUPPORT, EvidenceType.DATABASE_ENDPOINT_PRIOR,
        "endpoint-class prior",
    ),
    "lincs": (
        SupportBucket.STRONG_SUPPORT, EvidenceType.DATABASE_LINCS,
        "LINCS L1000 perturbation signature",
    ),
    "indication_taxonomy": (
        SupportBucket.MODERATE_SUPPORT, EvidenceType.DATABASE_INDICATION_TAXONOMY,
        "indication-taxonomy structural",
    ),
}

# Heuristic seed sources: their hand-set Beta(α, β) was a placeholder,
# never real evidence. Migration RESETS them to Beta(1, 1).
_HEURISTIC_SEED_SOURCES = {
    "default_unselected",
    "indication_qualifiers",
    "llm_eligibility",
    "regex_plus_llm_eligibility",
    "extraction_subgroup",
    "combo_synthesis",  # COMPOSED_OF edges
}


def _replay_evidence(
    seed: EdgeBeliefState, evidence: list[EvidenceRecord],
) -> EdgeBeliefState:
    """Apply each EvidenceRecord to ``seed`` via the conjugate update.

    Used to fold preexisting trial-attribution evidence back onto a
    freshly-constructed round-25 prior, so the migrated snapshot's
    posterior is equivalent to "rebuilt the populator + re-ran
    attribution against the same training corpus."
    """
    belief = seed.model_copy()
    for rec in evidence:
        try:
            bucket = SupportBucket(rec.support)
        except ValueError:
            continue
        n_eff = effective_n_for_evidence(rec.source_type, rec.quality_score)
        p_obs = p_obs_for_bucket(bucket)
        belief = apply_virtual_evidence(belief, n_eff=n_eff, p_obs=p_obs)
    belief.evidence = list(evidence)
    return belief


def _migrate_edge(
    source_id: str,
    target_id: str,
    edge_data: dict,
) -> EdgeBeliefState:
    """Return the migrated EdgeBeliefState for one edge.

    Three cases:
      1. metadata.source is in _CURATED_SOURCE_BUCKETS: emit a synthetic
         curated record matching that source, apply it to Beta(1, 1),
         then replay any pre-existing trial-attribution records.
      2. metadata.source is in _HEURISTIC_SEED_SOURCES: reset to
         Beta(1, 1) and replay only trial-attribution records (drops
         the hand-set seed prior).
      3. Unknown source: leave belief unchanged (already-correct, e.g.
         pure trial-attributed edges from the attributor).
    """
    raw_belief = EdgeBeliefState.model_validate(edge_data["belief"])
    metadata = edge_data.get("metadata") or {}
    source_tag = metadata.get("source") or ""

    if source_tag in _CURATED_SOURCE_BUCKETS:
        bucket, source_type, notes_prefix = _CURATED_SOURCE_BUCKETS[source_tag]
        # Safeguard: only emit a curated record when the original
        # belief actually had a populator-set prior bump (alpha+beta>2).
        # An edge tagged "lincs" with Beta(1, 1) didn't have a real
        # prior — it was a placeholder MECHANISM_AFFECTS edge created
        # by _ensure_edge for graph wiring. Treat as passthrough so we
        # don't accidentally inflate beliefs that started uninformed.
        has_populator_prior = (raw_belief.alpha + raw_belief.beta) > 2.0
        if not has_populator_prior:
            return raw_belief

        # Filter the existing evidence: keep only records whose
        # source_type is NOT one of the new DATABASE_* tiers (defensive
        # — these shouldn't exist in a pre-round-25 snapshot).
        new_database_types = {
            EvidenceType.DATABASE_OT_DIRECT,
            EvidenceType.DATABASE_CHEMBL,
            EvidenceType.DATABASE_MAB_TABLE,
            EvidenceType.DATABASE_OT_ASSOCIATION,
            EvidenceType.DATABASE_REACTOME_GO,
            EvidenceType.DATABASE_LINCS,
            EvidenceType.DATABASE_ENDPOINT_PRIOR,
            EvidenceType.DATABASE_INDICATION_TAXONOMY,
            EvidenceType.DATABASE_CROSS_REFERENCE,
            EvidenceType.DATABASE_FALLBACK,
        }
        trial_evidence = [
            r for r in raw_belief.evidence
            if r.source_type not in new_database_types
        ]
        record = make_curated_record(
            source_id=f"{source_tag}:{source_id}__{target_id}",
            bucket=bucket,
            source_type=source_type,
            notes=f"{notes_prefix} (migrated from round-25 rewrite)",
        )
        new_seed = belief_from_curated_record(record)
        return _replay_evidence(new_seed, [record] + trial_evidence)

    if source_tag in _HEURISTIC_SEED_SOURCES:
        # Replay only the trial-attribution evidence (preserve any
        # classifier-emitted records); start from Beta(1, 1) so the
        # heuristic seed is dropped.
        new_database_types = {
            EvidenceType.DATABASE_OT_DIRECT,
            EvidenceType.DATABASE_CHEMBL,
            EvidenceType.DATABASE_MAB_TABLE,
            EvidenceType.DATABASE_OT_ASSOCIATION,
            EvidenceType.DATABASE_REACTOME_GO,
            EvidenceType.DATABASE_LINCS,
            EvidenceType.DATABASE_ENDPOINT_PRIOR,
            EvidenceType.DATABASE_INDICATION_TAXONOMY,
            EvidenceType.DATABASE_CROSS_REFERENCE,
            EvidenceType.DATABASE_FALLBACK,
        }
        trial_evidence = [
            r for r in raw_belief.evidence
            if r.source_type not in new_database_types
        ]
        return _replay_evidence(EdgeBeliefState(), trial_evidence)

    # Unknown source — most likely a pure trial-attribution edge added
    # by the attributor (no populator source tag). Leave it unchanged.
    return raw_belief


def migrate_snapshot(in_path: Path, out_path: Path) -> dict:
    """Read in_path, migrate every edge, write out_path. Return stats."""
    graph = GraphStore()
    graph.import_snapshot(str(in_path))

    stats: Counter[str] = Counter()
    g = graph._graph  # noqa: SLF001
    for u, v, key, data in list(g.edges(keys=True, data=True)):
        source_tag = (data.get("metadata") or {}).get("source") or "<none>"
        migrated = _migrate_edge(u, v, data)
        data["belief"] = migrated.model_dump(mode="json")

        # Classify for stats reporting.
        if source_tag in _CURATED_SOURCE_BUCKETS:
            stats["curated_migrated"] += 1
        elif source_tag in _HEURISTIC_SEED_SOURCES:
            stats["heuristic_seed_reset"] += 1
        else:
            stats["passthrough"] += 1
        stats[f"source:{source_tag}"] += 1

    graph.export_snapshot(str(out_path))
    return dict(stats)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--in", dest="in_path", type=Path, required=True,
        help="Pre-round-25 snapshot JSON to read.",
    )
    parser.add_argument(
        "--out", dest="out_path", type=Path, required=True,
        help="Migrated round-25 snapshot JSON to write.",
    )
    args = parser.parse_args()

    if not args.in_path.exists():
        print(f"missing input: {args.in_path}")
        return 1
    stats = migrate_snapshot(args.in_path, args.out_path)
    print(f"Migrated {args.in_path} → {args.out_path}\n")
    print("Edge migration counts:")
    for k in ("curated_migrated", "heuristic_seed_reset", "passthrough"):
        print(f"  {k}: {stats.get(k, 0)}")
    print("\nPer-source breakdown:")
    for k, v in sorted(stats.items()):
        if k.startswith("source:"):
            print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
