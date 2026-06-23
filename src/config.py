"""The one baked algorithm config for Eroom Bio.

This module is the single source of truth for every modeling choice that used to
be an ``EROOM_*`` environment flag. The values below are the **frontier config**
that produced the headline oncology result (holdout AUROC ~0.700, in-sample
~0.768 on ``onco_scale_500``): routing on, biology→GO ontology keying on,
native direction on, informed prior on, safety DLT-gate on, weakest-link softmin
aggregation. There are no production flags anymore — one build path, one predict
path, one eval path all read from here.

Why a frozen object instead of flags: the audit (docs/dev/reports/REPO_AS_BUILT.md)
found flag sprawl and dead/unproven branches behind default-off env vars. Baking
the proven config makes the whole algorithm legible in one place and removes the
combinatorial surface of "which flags were set when this number was produced".

Deployment/path config (snapshot location, the open/private boundary root) is NOT
here — those remain ordinary env vars (EROOM_GRAPH_SNAPSHOT, EROOM_PRIVATE_ROOT)
because they are not algorithm choices. Likewise the eval ``--seed`` is a
reproducibility knob, not an algorithm parameter, so it lives on the eval CLI.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AlgorithmConfig:
    """Immutable, baked configuration for the prediction algorithm."""

    # ── Belief formation (attribution) ────────────────────────────────────
    # Route each training update by failure class (competing-risks censoring +
    # principled responsibility) instead of one class-blind update. This is the
    # A3/A4 "reason routing" that cut contamination 92%→31% and is the reason
    # the beliefs are clean. (was EROOM_ROUTING)
    routing: bool = True

    # Per-edge failure attribution mode. "explain_away" splits a failure's
    # down-weight across chain edges by (1 - E[p]) so well-evidenced curated
    # edges self-protect. The symmetric_* experiment variants are deleted.
    # (was EROOM_EDGE_ATTR; the alternatives are gone)
    edge_attr: str = "explain_away"

    # ── Substrate (node identity) ─────────────────────────────────────────
    # Key biology nodes by curated GO-BP id (bio:GO:xxxx) instead of a content
    # hash of the description, so the same biology pools across trials. The gate
    # is the BioLORD cosine floor for accepting a GO mapping. (was
    # EROOM_BIO_ONTOLOGY / EROOM_BIO_ONTOLOGY_GATE)
    bio_ontology: bool = True
    bio_ontology_gate: float = 0.60

    # Native modulation direction (inhibit/activate) from ChEMBL action_type,
    # used at query time to subtract opposite-direction evidence off a shared
    # edge. A near-no-op correctness fix on oncology. (was EROOM_DIRECTION)
    direction: bool = True

    # Cosine threshold for the BioLORD description-similarity merge tier (the
    # one geometric merge tier still used, on BiologyNodes). (was
    # EROOM_MERGE_COSINE)
    merge_cosine: float = 0.85

    # ── Prediction ────────────────────────────────────────────────────────
    # LOAD-BEARING. Under-evidenced / unobserved chain edges defer to a weak
    # prior centered on a plausible base rate (~0.75) instead of the Beta(1,1)
    # coin-flip, so a sparse edge doesn't spuriously become the weakest link.
    # On a sparse graph this prior carries a lot of the prediction — keep it in
    # view when reasoning about why a number is what it is. (was
    # EROOM_INFORMED_PRIOR / EROOM_PRIOR_MEAN / EROOM_PRIOR_STRENGTH)
    informed_prior: bool = True
    prior_mean: float = 0.75       # Beta(a0,b0) mean for an unobserved edge
    prior_strength: float = 2.0    # pseudo-observations of that weak prior

    # Chain aggregation = weakest-link softmin over per-edge Beta samples
    # (efficacy + measurement edges pooled). Temperature → 0 approaches hard
    # min. The geomean/product/min/harmonic variants are deleted; softmin is
    # the sole path. (was EROOM_AGG / EROOM_SOFTMIN_T)
    softmin_t: float = 0.10

    # Safety enters as overall = efficacy × (1 − safety_penalty). The DLT gate
    # penalizes failure-causing toxicity (failure_causing_fraction) rather than
    # mere AE occurrence; tolerated AEs contribute nothing (floor 0.0). Penalty
    # is the MAX over per-AE contributions, capped. (was EROOM_SAFETY_DLT_GATE;
    # floor/cap were already constants)
    safety_dlt_gate: bool = True
    safety_dlt_floor: float = 0.0
    safety_penalty_cap: float = 0.60

    # Hierarchical pooling parent concentration for the indication/population
    # backoff prior. (was EROOM_POOL_PRIOR_STRENGTH)
    pool_prior_strength: float = 20.0


# The single instance every module imports.
CONFIG = AlgorithmConfig()


# ──────────────────────────────────────────────────────────────────────────
# LOAD-BEARING REFERENCE — evidence-weight (n_eff) tiers.
#
# These are NOT redefined here (the canonical table is EVIDENCE_TYPE_N_EFF in
# src/inference/beliefs.py, which the Beta-Binomial update reads directly), but
# they are documented here because, together with the informed prior above,
# they carry much of the prediction on a sparse graph: a Phase-3 readout moves a
# belief ~15× as hard as a single in-vitro assay. Stronger evidence = bigger
# n_eff = more pseudo-observations per conjugate update.
#
#   CLINICAL_PHASE3        15.0     DATABASE_OT_DIRECT       12.0
#   CLINICAL_PHASE2         6.0     DATABASE_CHEMBL          10.0
#   CLINICAL_PHASE1         2.0     DATABASE_MAB_TABLE       10.0
#   GENETIC_MR             10.0     DATABASE_OT_ASSOCIATION   2.0
#   GENETIC_GWAS            4.0     DATABASE_ENDPOINT_PRIOR   2.0
#   PRECLINICAL_IN_VIVO     2.0     DATABASE_REACTOME_GO      1.5
#   PRECLINICAL_IN_VITRO    1.0     DATABASE_LINCS            1.0
#   DATABASE_LLM_INFERENCE  3.0     DATABASE_INDICATION_TAX   1.0
#   COMPUTATIONAL           0.3     DATABASE_FALLBACK         0.5
#   LITERATURE              0.2     DATABASE_CROSS_REFERENCE  0.3
# ──────────────────────────────────────────────────────────────────────────
