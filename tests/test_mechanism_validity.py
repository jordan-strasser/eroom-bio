"""Tests for the mechanism-validity filter (a MechanismNode must denote a
cellular action/process, not a category/disease/collection/placeholder)."""

from __future__ import annotations

import pytest

from src.graph.mechanism_validity import (
    invalid_mechanism_tier,
    is_invalid_mechanism,
    prune_invalid_mechanisms,
)

# The non-mechanisms found in the n=252 corpus, by tier.
INVALID = [
    ("Potential therapeutics for SARS", "A_therapeutic_collection"),
    ("Defective ACTH causes obesity and POMCD", "B_disease_module"),
    ("Defective CYP17A1 causes AH5", "B_disease_module"),
    ("Defective visual phototransduction due to STRA6 loss of function", "B_disease_module"),
    ("TLR3 deficiency - HSE", "B_disease_module"),
    ("UNC93B1 deficiency - HSE", "B_disease_module"),
    ("Binding and entry of HIV virion", "C_pathogen"),
    ("InlB-mediated entry of Listeria monocytogenes into host cell", "C_pathogen"),
    ("Purinergic signaling in leishmaniasis infection", "C_pathogen"),
    ("SARS-CoV-1 activates/modulates innate immune responses", "C_pathogen"),
    ("vRNA Synthesis", "C_pathogen"),
    ("Hormone ligand-binding receptors", "D_grouping_or_PK"),
    ("Stimuli-sensing channels", "D_grouping_or_PK"),
    ("Class B/2 (Secretin family receptors)", "D_grouping_or_PK"),
    ("Aspirin ADME", "D_grouping_or_PK"),
    ("other", "E_placeholder"),
    # F — Reactome residual "Other X" buckets (over-pooling hubs: distinct
    # targets collapse in, then fan out to unrelated biology). Leading "Other".
    ("Other interleukin signaling", "F_residual_bucket"),
    ("Other semaphorin interactions", "F_residual_bucket"),
]

# Real cellular actions/processes that MUST survive — including off-target,
# aberrant-cancer-signaling, and coarse-but-real action categories.
VALID = [
    "Co-inhibition by PD-1",
    "RUNX1 and FOXP3 control the development of regulatory T lymphocytes (Tregs)",
    "Constitutive Signaling by Aberrant PI3K in Cancer",  # aberrant signaling = real action
    "Signaling by moderate kinase activity BRAF mutants",
    "Drug-mediated inhibition of MET activation",
    "Chemokine receptors bind chemokines",                # "bind" = action, not a grouping
    "TNFs bind their physiological receptors",
    "Ion transport by P-type ATPases",                    # transport = action
    "angiogenesis",
    "enzyme inhibition",   # coarse fallback category, but names an action → keep
    "kinase inhibition",
    "receptor antagonism",
    "ISG15 antiviral mechanism",  # host antiviral process (not pathogen-lifecycle) → keep
    # "other" mid-name (NOT a leading residual bucket) — a real pathway, keep.
    "APC/C:Cdh1 mediated degradation of Cdc20 and other APC/C:Cdh1 targeted "
    "proteins in late mitosis/early G1",
]


@pytest.mark.parametrize("name,tier", INVALID)
def test_invalid_names_flagged_with_tier(name, tier):
    assert invalid_mechanism_tier(name) == tier
    assert is_invalid_mechanism(name) is True


@pytest.mark.parametrize("name", VALID)
def test_valid_actions_survive(name):
    assert invalid_mechanism_tier(name) is None
    assert is_invalid_mechanism(name) is False


def test_prune_drops_nodes_chains_and_edges():
    from src.graph.models import (
        CausalChain, EdgeBeliefState, EdgeType, GraphEdge, TrialOutcome,
        TrialSubgraph,
    )
    from src.graph.store import GraphStore

    g = GraphStore()
    g._graph.add_node("SARS", node_type="MechanismNode",
                      name="Potential therapeutics for SARS", ontology_id="R-HSA-9679191")
    g._graph.add_node("PD1", node_type="MechanismNode",
                      name="Co-inhibition by PD-1", ontology_id="R-HSA-389948")
    for mid in ("SARS", "PD1"):
        g.add_edge(GraphEdge(source_id="TGT", target_id=mid,
                             edge_type=EdgeType.MODULATES_VIA, belief=EdgeBeliefState()))
        g.add_edge(GraphEdge(source_id=mid, target_id="BIO",
                             edge_type=EdgeType.MECHANISM_AFFECTS, belief=EdgeBeliefState()))

    def _chain(mid):
        return CausalChain(arm_id="A", compound_id="C", subgroup_population_id="P",
                           target_id="TGT", mechanism_id=mid, biology_id="BIO",
                           indication_id="IND", endpoint_id="EP", outcome=TrialOutcome.SUCCESS)
    g.trial_subgraphs["NCT1"] = TrialSubgraph(
        trial_id="NCT1", parent_population_id="P",
        chains=[_chain("SARS"), _chain("PD1")])
    g.trial_subgraphs["NCT2"] = TrialSubgraph(
        trial_id="NCT2", parent_population_id="P", chains=[_chain("SARS")])

    stats = prune_invalid_mechanisms(g)
    assert stats["nodes_dropped"] == 1
    assert stats["chains_dropped"] == 2
    assert stats["trials_affected"] == 2
    assert stats["trials_emptied"] == ["NCT2"]  # only-SARS trial left chain-less
    assert not g._graph.has_node("SARS")
    assert g._graph.has_node("PD1")
    # SARS's incident edges are gone; PD1's survive
    assert g._graph.number_of_edges() == 2
    # PD1 chain survives in NCT1
    assert [c.mechanism_id for c in g.trial_subgraphs["NCT1"].chains] == ["PD1"]
    # idempotent
    assert prune_invalid_mechanisms(g)["nodes_dropped"] == 0
