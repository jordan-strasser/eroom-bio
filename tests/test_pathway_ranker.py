"""Tests for src/graph/pathway_ranker.py.

The re-ranker is the fix for round 7 finding #2 / round 5 Finding C: Reactome's
default top-1 is often disease-context-inappropriate (CRBN → SARS therapeutics,
VEGFA → platelet degranulation, IL2RA → RAF/MAP kinase cascade). Pathway data
in these tests is drawn from the live Reactome cache under
``data/cache/lincs/reactome/`` so the regression assertions mirror reality.
"""

from src.graph.pathway_ranker import (
    relevance_floor,
    rerank_pathways,
    semantic_relevance_floor,
)
from src.ingestion.lincs import ReactomePathway


def _p(stable_id: str, display_name: str) -> ReactomePathway:
    return ReactomePathway(stable_id=stable_id, display_name=display_name)


# TUBB4B's actual 5 GO biological-process terms (UniProt P68371) — the taxane
# noise cluster. The on-mechanism terms are mitotic cell cycle + microtubule-
# based process; the rest are real-but-off-context leaves.
_TUBB4B_GO = [
    _p("GO:0030317", "flagellated sperm motility"),
    _p("GO:0042267", "natural killer cell mediated cytotoxicity"),
    _p("GO:0000278", "mitotic cell cycle"),
    _p("GO:0007010", "cytoskeleton organization"),
    _p("GO:0007017", "microtubule-based process"),
]


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
    """Real IL2RA pathway list. With receptor_agonism vocab expansion
    (includes "interleukin"), Interleukin-2 signaling should now win directly.
    """
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


def test_pdpk1_kinase_inhibition_finds_akt_pathway():
    """Real PDPK1 pathway list from data/cache/lincs/reactome/PDPK1.json. The
    canonical PDK1 biology is PIP3/AKT signaling; Reactome's default top-1
    is "GPVI-mediated activation cascade" (platelet-specific, wrong context).
    The mechanism vocabulary expansion (kinase_inhibition → akt, pi3k, ...)
    should pull an AKT-named pathway to the top.
    """
    pathways = [
        _p("R-HSA-114604", "GPVI-mediated activation cascade"),
        _p("R-HSA-1257604", "PIP3 activates AKT signaling"),
        _p("R-HSA-165158", "Activation of AKT2"),
        _p("R-HSA-202424", "Downstream TCR signaling"),
        _p("R-HSA-389357", "CD28 dependent PI3K/Akt signaling"),
        _p("R-HSA-5218921", "VEGFR2 mediated cell proliferation"),
        _p("R-HSA-5674400", "Constitutive Signaling by AKT1 E17K in Cancer"),
    ]
    ranked = rerank_pathways(
        pathways,
        mechanism_name="kinase_inhibition",
        indication_name="melanoma",
        gene_symbol="PDPK1",
    )
    assert ranked[0].stable_id != "R-HSA-114604", (
        "GPVI-mediated activation cascade should not be top-1 for PDPK1 + kinase_inhibition"
    )
    # Top pick should mention AKT or PI3K — the canonical PDPK1 biology.
    top_name = ranked[0].display_name.lower()
    assert "akt" in top_name or "pi3k" in top_name


def test_protein_degradation_vocab_finds_ubiquitin_pathway():
    """The CRBN gene itself has no rerankable pathways (Reactome only has the
    SARS one), but the mechanism vocab itself should still work end-to-end:
    given pathways that include ubiquitin/proteasome biology, the vocab
    should pull them ahead of generic alternates.
    """
    pathways = [
        _p("R-1", "Generic cellular response"),
        _p("R-2", "Ubiquitin-mediated proteolysis"),
        _p("R-3", "Cell cycle regulation"),
    ]
    ranked = rerank_pathways(
        pathways,
        mechanism_name="protein_degradation",
        indication_name="multiple_myeloma",
    )
    assert ranked[0].stable_id == "R-2"


def test_unknown_mechanism_does_not_crash():
    """An off-vocab mechanism slug yields no expansion and the ranker falls
    back to slug tokenization. No crash, no fabricated matches."""
    pathways = [_p("R-1", "Alpha"), _p("R-2", "Beta")]
    ranked = rerank_pathways(
        pathways,
        mechanism_name="some_made_up_mechanism",
    )
    assert [p.stable_id for p in ranked] == ["R-1", "R-2"]


def test_short_canonical_name_beats_verbose_specific_name():
    """The pathway-name-length normalization keeps short canonical names
    competitive against verbose specific ones that incidentally contain
    more context tokens via name length alone.

    Concrete case from the slice rebuild: VEGFA + angiogenesis_inhibition +
    melanoma. Reactome's "Signaling by VEGF" (2 tokens, 1 match → 50%)
    should beat GO:0038033's 14-token specific term (2 matches → ~15%).
    """
    candidates = [
        _p("R-HSA-194138", "Signaling by VEGF"),
        _p(
            "GO:0038033",
            "positive regulation of endothelial cell chemotaxis by "
            "VEGF-activated vascular endothelial growth factor receptor "
            "signaling pathway",
        ),
    ]
    ranked = rerank_pathways(
        candidates,
        mechanism_name="angiogenesis_inhibition",
        indication_name="melanoma",
        gene_symbol="VEGFA",
    )
    assert ranked[0].stable_id == "R-HSA-194138"


def test_score_is_fractional_in_unit_interval():
    """score_candidate must return a value in [0, 1] after normalization
    so consumers (the populator) can rely on consistent ranges across
    pathway-name lengths."""
    from src.graph.pathway_ranker import score_candidate

    short = _p("R-1", "Signaling by VEGF")
    long = _p("R-2", "positive regulation of endothelial cell chemotaxis by VEGF receptor signaling")
    s_short = score_candidate(short, mechanism_name="angiogenesis_inhibition", gene_symbol="VEGFA")
    s_long = score_candidate(long, mechanism_name="angiogenesis_inhibition", gene_symbol="VEGFA")
    assert 0.0 <= s_long <= s_short <= 1.0


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


# ── Relevance floor (GO-augmentation noise prune) ────────────────────────


def test_relevance_floor_drops_off_context_go_leaves_via_description():
    """With the extracted action description as context, the floor keeps the
    on-mechanism GO terms and drops the off-context tubulin leaves."""
    kept = relevance_floor(
        _TUBB4B_GO,
        mechanism_name="microtubule stabilization mitotic spindle arrest",
        gene_symbol="TUBB4B",
    )
    names = {p.display_name for p in kept}
    assert "mitotic cell cycle" in names
    assert "microtubule-based process" in names
    assert "flagellated sperm motility" not in names
    assert "natural killer cell mediated cytotoxicity" not in names
    assert "cytoskeleton organization" not in names


def test_relevance_floor_uses_microtubule_binding_vocab():
    """Even with only the drug-class slug (no description), the new
    microtubule_binding vocab lets the on-mechanism terms clear the floor."""
    kept = relevance_floor(
        _TUBB4B_GO, mechanism_name="microtubule_binding", gene_symbol="TUBB4B"
    )
    names = {p.display_name for p in kept}
    assert "microtubule-based process" in names
    assert "mitotic cell cycle" in names  # via {mitotic} in the vocab
    assert "flagellated sperm motility" not in names


def test_relevance_floor_falls_back_to_top1_when_no_signal():
    """No context signal → everything ties at 0.0 → keep exactly the top-1
    (never drop the chain), never the whole noisy set."""
    kept = relevance_floor(_TUBB4B_GO, mechanism_name="other")
    assert len(kept) == 1


def test_relevance_floor_empty_input():
    assert relevance_floor([], mechanism_name="microtubule_binding") == []


# ── Semantic relevance floor (BioLORD-backed GO prune) ───────────────────


def _fake_embed(mapping):
    """Build an embed_fn returning the mapped vector per text (default orthogonal
    low-similarity vector), so floor behavior is deterministic in tests."""
    def _embed(texts):
        return [mapping.get(t, [0.0, 0.0, 1.0]) for t in texts]
    return _embed


def test_semantic_floor_keeps_on_mechanism_drops_leaves():
    # context aligns with the two on-mechanism terms; leaves are orthogonal.
    embed = _fake_embed({
        "antimitotic microtubule": [1.0, 0.0, 0.0],
        "mitotic cell cycle": [0.98, 0.10, 0.0],
        "microtubule-based process": [0.95, 0.20, 0.0],
        "flagellated sperm motility": [0.0, 1.0, 0.0],
        "natural killer cell mediated cytotoxicity": [0.1, 0.97, 0.0],
    })
    kept = semantic_relevance_floor(_TUBB4B_GO, "antimitotic microtubule", embed)
    names = {p.display_name for p in kept}
    assert "mitotic cell cycle" in names
    assert "microtubule-based process" in names
    assert "flagellated sperm motility" not in names
    assert "natural killer cell mediated cytotoxicity" not in names


def test_semantic_floor_top1_fallback_when_all_orthogonal():
    embed = _fake_embed({"unrelated context": [1.0, 0.0, 0.0]})  # all terms → default orthogonal
    kept = semantic_relevance_floor(_TUBB4B_GO, "unrelated context", embed)
    assert len(kept) == 1  # top-1 fallback, never the whole noisy set


def test_semantic_floor_empty_context_keeps_all():
    kept = semantic_relevance_floor(_TUBB4B_GO, "", _fake_embed({}))
    assert len(kept) == len(_TUBB4B_GO)  # no signal → don't prune
