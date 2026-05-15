"""Tests for src/graph/pathway_ranker.py.

The re-ranker is the fix for round 7 finding #2 / round 5 Finding C: Reactome's
default top-1 is often disease-context-inappropriate (CRBN → SARS therapeutics,
VEGFA → platelet degranulation, IL2RA → RAF/MAP kinase cascade). Pathway data
in these tests is drawn from the live Reactome cache under
``data/cache/lincs/reactome/`` so the regression assertions mirror reality.
"""

from src.graph.pathway_ranker import rerank_pathways
from src.ingestion.lincs import ReactomePathway


def _p(stable_id: str, display_name: str) -> ReactomePathway:
    return ReactomePathway(stable_id=stable_id, display_name=display_name)


# ── Core behavior ────────────────────────────────────────────────────────


def test_empty_context_preserves_input_order():
    pathways = [_p("R-1", "Alpha"), _p("R-2", "Beta")]
    ranked = rerank_pathways(pathways)
    assert [p.stable_id for p in ranked] == ["R-1", "R-2"]


def test_empty_input_returns_empty():
    assert rerank_pathways([], mechanism_name="kinase_inhibition") == []


def test_zero_overlap_preserves_input_order():
    pathways = [_p("R-1", "Alpha pathway"), _p("R-2", "Beta cascade")]
    ranked = rerank_pathways(
        pathways,
        mechanism_name="checkpoint_blockade",
        indication_name="melanoma",
        gene_symbol="PDCD1",
    )
    assert [p.stable_id for p in ranked] == ["R-1", "R-2"]


def test_stable_sort_on_ties():
    """When multiple pathways tie on overlap score, original order survives."""
    pathways = [
        _p("R-1", "Signaling by VEGF"),
        _p("R-2", "VEGF binds VEGFR"),
        _p("R-3", "Unrelated pathway"),
    ]
    ranked = rerank_pathways(pathways, gene_symbol="VEGFA")
    # R-1 and R-2 both have "vegf" overlap with "vegfa"; R-3 has none.
    assert ranked[-1].stable_id == "R-3"
    assert [p.stable_id for p in ranked[:2]] == ["R-1", "R-2"]


def test_substring_gene_symbol_overlap():
    """Gene-name prefixes (vegf ⊂ vegfa) should count as overlap."""
    pathways = [
        _p("R-1", "Unrelated thing"),
        _p("R-2", "Signaling by VEGF"),
    ]
    ranked = rerank_pathways(pathways, gene_symbol="VEGFA")
    assert ranked[0].stable_id == "R-2"


def test_short_tokens_ignored():
    """Tokens under 4 chars are coincidental noise and should not score."""
    pathways = [
        _p("R-1", "An of by"),  # all stopwords / short
        _p("R-2", "Receptor signaling"),
    ]
    # context "il2" is 3 chars → ignored; "receptor" 8 chars → matches
    ranked = rerank_pathways(
        pathways,
        mechanism_name="receptor_agonism",
        gene_symbol="il2",
    )
    assert ranked[0].stable_id == "R-2"


# ── Regression: round-7 worst-offender cases ─────────────────────────────


def test_vegfa_no_longer_picks_platelet_degranulation():
    """Real VEGFA pathway list from data/cache/lincs/reactome/VEGFA.json.
    Top-1 must shift off "Platelet degranulation" to a VEGF-named pathway.
    """
    pathways = [
        _p("R-HSA-114608", "Platelet degranulation"),
        _p("R-HSA-1234158", "Regulation of gene expression by Hypoxia-inducible Factor"),
        _p("R-HSA-194138", "Signaling by VEGF"),
        _p("R-HSA-194313", "VEGF ligand-receptor interactions"),
        _p("R-HSA-195399", "VEGF binds to VEGFR leading to receptor dimerization"),
        _p("R-HSA-4420097", "VEGFA-VEGFR2 Pathway"),
        _p("R-HSA-5218921", "VEGFR2 mediated cell proliferation"),
        _p("R-HSA-6785807", "Interleukin-4 and Interleukin-13 signaling"),
        _p("R-HSA-8866910", "TFAP2 (AP-2) family regulates transcription of growth factors and their receptors"),
        _p("R-HSA-9679191", "Potential therapeutics for SARS"),
    ]
    ranked = rerank_pathways(
        pathways,
        mechanism_name="angiogenesis_inhibition",
        indication_name="melanoma",
        gene_symbol="VEGFA",
    )
    assert ranked[0].stable_id != "R-HSA-114608", (
        "Platelet degranulation should not be top-1 for VEGFA + angiogenesis"
    )
    assert "vegf" in ranked[0].display_name.lower()


def test_il2ra_no_longer_picks_raf_mapk():
    """Real IL2RA pathway list. Top-1 must shift off RAF/MAP kinase cascade."""
    pathways = [
        _p("R-HSA-5673001", "RAF/MAP kinase cascade"),
        _p("R-HSA-8877330", "RUNX1 and FOXP3 control the development of regulatory T lymphocytes (Tregs)"),
        _p("R-HSA-9020558", "Interleukin-2 signaling"),
        _p("R-HSA-912526", "Interleukin receptor SHC signaling"),
    ]
    ranked = rerank_pathways(
        pathways,
        mechanism_name="receptor_agonism",
        indication_name="melanoma",
        gene_symbol="IL2RA",
    )
    assert ranked[0].stable_id != "R-HSA-5673001", (
        "RAF/MAP kinase cascade should not be top-1 for IL2RA + receptor_agonism"
    )


def test_metadata_order_reflects_reranking():
    """The reordered list, when serialized to pathway_ids, must put the
    context-relevant pathway before the off-context one. This is what makes
    BiologyNode.pathway_ids carry meaningful ranking for cross-indication
    queries against alternates.
    """
    pathways = [
        _p("R-OFF", "Platelet degranulation"),
        _p("R-ON", "Signaling by VEGF"),
    ]
    ranked = rerank_pathways(pathways, gene_symbol="VEGFA")
    pathway_ids = [p.stable_id for p in ranked]
    assert pathway_ids == ["R-ON", "R-OFF"]
