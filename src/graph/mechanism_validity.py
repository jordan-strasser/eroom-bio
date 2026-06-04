"""Mechanism-validity filter — a MechanismNode must denote a cellular ACTION
or process, not a category / grouping / disease / therapeutic-collection.

Reactome/GO resolution occasionally lands a target on an entry that isn't a
mechanism of action at all: a *therapeutic collection* ("Potential therapeutics
for SARS" — a drug bucket under the SARS-CoV disease module), a *disease module*
("Defective CYP17A1 causes AH5"), a *pathogen-lifecycle* pathway ("Binding and
entry of HIV virion"), a *receptor-family grouping* ("Hormone ligand-binding
receptors"), a pharmacokinetics entry ("Aspirin ADME"), or the bare ``other``
placeholder. These should never have become MechanismNodes.

The gate is judged by the Reactome/GO **NAME** (the node's identity), not the
terse 2-word description (which can mask the real identity — e.g. the SARS node
described itself as "kinase inhibition"). This is the deterministic, non-LLM
counterpart to a classifier-side validity flag: it over-matches nothing valid
(the curated D-set and pathogen tokens are specific), so off-target hits and
sub-mechanisms — real biology we deliberately keep — are untouched.

The pattern tiers (collection / disease / pathogen / placeholder) generalize to
new trials on append/scale; the receptor-grouping set is curated (those names
are heterogeneous and a loose pattern would catch valid "X bind Y" actions).
"""

from __future__ import annotations

import re

# A — therapeutic collections (drug buckets; no cellular action)
_THERAPEUTIC_COLLECTION = re.compile(r"potential therapeutics|therapeutics for", re.I)

# B — disease / deficiency causation modules (name a pathology, not an action)
_DISEASE_MODULE = re.compile(
    r"defective\b.*\bcauses\b|deficiency\s*-|loss of function", re.I
)

# C — pathogen / infection-lifecycle pathways (Reactome infectious-disease branch;
# clinically irrelevant to an onco/CVD corpus). Tokens are specific enough that no
# valid human therapeutic mechanism collides.
_PATHOGEN_TOKENS = (
    "hiv virion", "listeria", "leishmania", "sars-cov", "vrna synthesis",
)

# D — receptor / transporter / channel family GROUPINGS + pharmacokinetics (ADME).
# Curated by exact name: these denote a class of entities or a disposition route,
# not an action. Kept explicit so action-naming siblings ("Chemokine receptors
# bind chemokines", "Ion transport by P-type ATPases") are never caught.
_GROUPING_OR_PK_NAMES = frozenset({
    "class b/2 (secretin family receptors)",
    "hormone ligand-binding receptors",
    "peptide ligand-binding receptors",
    "prostanoid ligand receptors",
    "stimuli-sensing channels",
    "cation-coupled chloride cotransporters",
    "aspirin adme",
})


def invalid_mechanism_tier(name: str) -> str | None:
    """Return the drop-tier label if ``name`` is NOT a cellular action/process,
    else ``None``. Judged on the Reactome/GO name."""
    if not name:
        return None
    n = name.strip().lower()
    if n == "other":
        return "E_placeholder"
    if _THERAPEUTIC_COLLECTION.search(n):
        return "A_therapeutic_collection"
    if _DISEASE_MODULE.search(n):
        return "B_disease_module"
    if any(tok in n for tok in _PATHOGEN_TOKENS):
        return "C_pathogen"
    if n in _GROUPING_OR_PK_NAMES:
        return "D_grouping_or_PK"
    return None


def is_invalid_mechanism(name: str, ontology_id: str | None = None) -> bool:
    """True when a mechanism NAME denotes a category/grouping/disease/collection
    /placeholder rather than a cellular action or process."""
    return invalid_mechanism_tier(name) is not None


def prune_invalid_mechanisms(graph) -> dict:
    """Drop MechanismNodes that aren't cellular actions + their chain variants.

    Removes each flagged node (NetworkX drops its incident ``modulates_via`` /
    ``mechanism_affects`` edges) and every chain in ``trial_subgraphs`` whose
    ``mechanism_id`` points at one — so a re-attribution from the pruned snapshot
    never credits a trial's outcome through a non-mechanism. Trials with other
    valid chains stay; a trial whose only chains went through dropped mechanisms
    is left chain-less (and should be re-attributed against, not silently kept).

    Idempotent: re-running over an already-clean graph is a no-op. Returns stats
    (counts + per-tier breakdown + dropped ids) for logging.
    """
    g = graph._graph  # noqa: SLF001 — store-internal NetworkX handle, by design
    by_tier: dict[str, list[str]] = {}
    invalid_ids: set[str] = set()
    for nid, data in g.nodes(data=True):
        if data.get("node_type") != "MechanismNode":
            continue
        tier = invalid_mechanism_tier(data.get("name", ""))
        if tier:
            invalid_ids.add(nid)
            by_tier.setdefault(tier, []).append(data.get("name", nid))

    chains_dropped = 0
    trials_affected: set[str] = set()
    trials_emptied: set[str] = set()
    for tid, ts in graph.trial_subgraphs.items():
        before = len(ts.chains)
        kept = [ch for ch in ts.chains if ch.mechanism_id not in invalid_ids]
        if len(kept) != before:
            chains_dropped += before - len(kept)
            trials_affected.add(tid)
            ts.chains = kept
            if not kept:
                trials_emptied.add(tid)

    for nid in invalid_ids:
        if g.has_node(nid):
            g.remove_node(nid)

    return {
        "nodes_dropped": len(invalid_ids),
        "chains_dropped": chains_dropped,
        "trials_affected": len(trials_affected),
        "trials_emptied": sorted(trials_emptied),
        "by_tier": {t: len(v) for t, v in sorted(by_tier.items())},
        "dropped_ids": sorted(invalid_ids),
    }
