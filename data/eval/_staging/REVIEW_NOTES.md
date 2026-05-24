# Biology Gold Set — Review Notes

Generated: 2026-05-24
Total pairs: 353

## Label Distribution

- **merge**: 49
- **parent_of**: 64
- **child_of**: 52
- **sibling**: 156
- **unrelated**: 32

## Flagged needs_human_review: 1
  - agrees_with_hint==False: 0
  - Phase-4 disagreements: 1
  - low-confidence: 0

## Source Distribution (counting both source_a and source_b)

- **S1_extraction**: 587
- **S2_reactome**: 113
- **S3_go**: 6

## Candidate Strategy Distribution

- **cross_domain**: 24
- **ontology_parent**: 3
- **ontology_sibling**: 50
- **random**: 26
- **same_trial_mech_bio**: 120
- **topk_neighbor**: 130

## Top Pairs for Human Review

> Priority: agrees_with_hint==False, Phase-4 disagreements, then calibration samples.

### Pair 1: `68021e32-f5c4-4667-860c-537d927005e6`
**Flags**: PHASE4_DISAGREE
- **Label**: `merge` | **Confidence**: high | **Strategy**: topk_neighbor
- **Hint**: None | **Agrees with hint**: None
- **Text A**: _monoclonal antibody targeting amyloid beta plaques_
- **Text B**: _monoclonal antibody targeting amyloid plaques for clearance_
- **Justification**: Both texts describe the same therapeutic approach of using monoclonal antibodies to target amyloid plaques (with 'amyloid beta' being the standard specification of the plaques involved in Alzheimer's disease), differing only in the explicit mention of the clearance mechanism in Text B.
- **Phase-4 second label**: `sibling`

### Pair 2: `05332dfc-806a-4ac6-b8ca-09004933934e`
- **Label**: `merge` | **Confidence**: high | **Strategy**: topk_neighbor
- **Hint**: None | **Agrees with hint**: None
- **Text A**: _DNA synthesis inhibition leading to tumor cell death_
- **Text B**: _DNA synthesis disruption and tumor cell death_
- **Justification**: Both texts describe the same biological outcome where disruption/inhibition of DNA synthesis results in tumor cell death, differing only in word choice (inhibition vs disruption).

### Pair 3: `1775566b-6fe7-4afb-9230-0137e9ac4f62`
- **Label**: `merge` | **Confidence**: high | **Strategy**: topk_neighbor
- **Hint**: None | **Agrees with hint**: None
- **Text A**: _BRAF inhibition combined with MEK inhibition to prevent resistance bypass_
- **Text B**: _MEK inhibition prevents resistance bypass that occurs with BRAF inhibition alone_
- **Justification**: Both texts describe the same biological concept: combining BRAF and MEK inhibition to overcome resistance that develops when using BRAF inhibition as monotherapy, just expressed with different syntactic emphasis.

### Pair 4: `66fc3a28-cca9-4c08-b1d9-227a77199673`
- **Label**: `merge` | **Confidence**: high | **Strategy**: topk_neighbor
- **Hint**: None | **Agrees with hint**: None
- **Text A**: _MEK inhibition prevents reactivation of MAPK pathway that occurs with BRAF inhibition alone_
- **Text B**: _MEK inhibition prevents reactivation of MAPK signaling downstream of BRAF blockade_
- **Justification**: Both texts describe the same mechanism: MEK inhibition blocking MAPK pathway reactivation that would otherwise occur when BRAF is inhibited, differing only in terminology (pathway vs signaling, alone vs blockade).

### Pair 5: `a026bb7c-f304-4c5d-84bf-2e459dc17947`
- **Label**: `parent_of` | **Confidence**: high | **Strategy**: topk_neighbor
- **Hint**: None | **Agrees with hint**: None
- **Text A**: _dual mTORC1/mTORC2 inhibition in differentiated thyroid cancer_
- **Text B**: _dual mTORC1/mTORC2 inhibition in BRAF wildtype differentiated thyroid cancer_
- **Justification**: Text A describes dual mTORC1/mTORC2 inhibition in differentiated thyroid cancer broadly, while Text B specifies this same mechanism in the more restrictive context of BRAF wildtype differentiated thyroid cancer, making Text B a specific case of Text A.

### Pair 6: `355d283b-990d-4928-bb0d-97deb502fce6`
- **Label**: `parent_of` | **Confidence**: high | **Strategy**: topk_neighbor
- **Hint**: None | **Agrees with hint**: None
- **Text A**: _cholesterol metabolism modulation_
- **Text B**: _cholesterol metabolism modulation in hypercholesterolemia_
- **Justification**: Text A describes a general biological process (cholesterol metabolism modulation) while Text B describes the same process applied specifically to the disease context of hypercholesterolemia, making Text A the broader parent concept.

### Pair 7: `1258658d-5d53-450e-a21a-874cefff60cc`
- **Label**: `parent_of` | **Confidence**: high | **Strategy**: ontology_parent
- **Hint**: parent_of | **Agrees with hint**: True
- **Text A**: _The dissolution of the nuclear membrane marks the beginning of the prometaphase. Kinetochores are created when proteins attach to the centromeres. Microtubules then attach at the kinetochores, and the_
- **Text B**: _The resolution of sister chromatids in mitotic prometaphase involves removal of cohesin complexes from chromosomal arms, with preservation of cohesion at centromeres (Losada et al. 1998, Hauf et al. 2_
- **Justification**: Reactome curator-annotated parent-child; text_a is the parent pathway.

### Pair 8: `d9a91fcc-a506-4af9-993e-1d1aa574c37b`
- **Label**: `child_of` | **Confidence**: high | **Strategy**: same_trial_mech_bio
- **Hint**: None | **Agrees with hint**: None
- **Text A**: _dual cholesterol reduction via intestinal absorption blockade and hepatic synthesis inhibition_
- **Text B**: _cholesterol homeostasis regulation_
- **Justification**: Text A describes a specific dual mechanism for reducing cholesterol (intestinal absorption blockade + hepatic synthesis inhibition), which is a particular strategy within the broader concept of cholesterol homeostasis regulation in Text B.

### Pair 9: `cf28cc13-5ea3-4824-bdc2-927ef10277c8`
- **Label**: `child_of` | **Confidence**: high | **Strategy**: same_trial_mech_bio
- **Hint**: None | **Agrees with hint**: None
- **Text A**: _triglyceride reduction and anti-inflammatory effects_
- **Text B**: _cardiovascular risk reduction_
- **Justification**: Triglyceride reduction and anti-inflammatory effects are specific mechanisms that contribute to the broader outcome of cardiovascular risk reduction, making Text A a more specific manifestation of the general concept in Text B.

### Pair 10: `057f1612-e190-407c-b6a4-4a0fb90a36eb`
- **Label**: `child_of` | **Confidence**: high | **Strategy**: same_trial_mech_bio
- **Hint**: None | **Agrees with hint**: None
- **Text A**: _microsomal triglyceride transfer protein inhibition_
- **Text B**: _lipid metabolism modulation_
- **Justification**: Microsomal triglyceride transfer protein (MTP) inhibition is a specific mechanism that modulates lipid metabolism, making it a narrower/more specific example of the broader concept of lipid metabolism modulation.

### Pair 11: `c3d04a98-a88a-4e24-af4b-1b3f94126505`
- **Label**: `sibling` | **Confidence**: high | **Strategy**: same_trial_mech_bio
- **Hint**: None | **Agrees with hint**: None
- **Text A**: _VEGFR2 antagonism blocking angiogenesis_
- **Text B**: _tumor angiogenesis inhibition_
- **Justification**: Both describe inhibition of angiogenesis at the same conceptual level, but through different mechanisms—one specifies the molecular target (VEGFR2 antagonism) while the other specifies the biological context (tumor), making them distinct approaches to the same parent concept of angiogenesis inhibition.

### Pair 12: `27eecdb2-c514-4577-a5e2-6a6aa8362b1a`
- **Label**: `sibling` | **Confidence**: high | **Strategy**: ontology_sibling
- **Hint**: sibling | **Agrees with hint**: True
- **Text A**: _The signal from unattached kinetochores is amplified through a Mad2 inhibitory signal that is propagated by the binding of Mad1 to the kinetochore, the association of Mad2 with Mad1, the conversion of_
- **Text B**: _The degradation of cyclin B1, which appears to occur at the mitotic spindle, is delayed until the metaphase /anaphase transition by the spindle assembly checkpoint and is required in order for sister _
- **Justification**: Reactome curator-annotated siblings; both are children of the same parent pathway.

### Pair 13: `4075aed5-767a-4267-a9b0-9b8bad298cbf`
- **Label**: `sibling` | **Confidence**: high | **Strategy**: topk_neighbor
- **Hint**: None | **Agrees with hint**: None
- **Text A**: _dual mTORC1/mTORC2 inhibition in BRAF wildtype anaplastic thyroid cancer_
- **Text B**: _dual mTORC1/mTORC2 inhibition in BRAF wildtype differentiated thyroid cancer_
- **Justification**: Both describe dual mTORC1/mTORC2 inhibition in BRAF wildtype thyroid cancer, but differ at the same granularity level by cancer subtype (anaplastic vs. differentiated), representing distinct disease contexts under a shared parent concept.

### Pair 14: `15b24664-225a-4b19-a3a2-b1bccb0becba`
- **Label**: `unrelated` | **Confidence**: high | **Strategy**: cross_domain
- **Hint**: None | **Agrees with hint**: None
- **Text A**: _dual VEGFR2/EGFR/RET inhibition combined with proteasome inhibition_
- **Text B**: _microsomal triglyceride transfer protein inhibition_
- **Justification**: Text A describes a multi-targeted kinase inhibition strategy combined with proteasome inhibition for cancer therapy, while Text B describes lipid metabolism modulation through a single protein target, with no overlapping mechanisms or biological pathways.

### Pair 15: `556c85f9-df6a-4c8a-8b1f-e8496701d7f7`
- **Label**: `unrelated` | **Confidence**: high | **Strategy**: same_trial_mech_bio
- **Hint**: None | **Agrees with hint**: None
- **Text A**: _cholesterol synthesis inhibition_
- **Text B**: _amyloid beta metabolism modulation_
- **Justification**: Cholesterol synthesis inhibition and amyloid beta metabolism modulation are distinct biological processes affecting different molecular pathways with no direct hierarchical or semantic relationship, though they may have indirect connections in neurodegenerative disease contexts.

### Pair 16: `b6fe1b60-0fe7-468f-8c58-2f2e97509ab7`
- **Label**: `unrelated` | **Confidence**: high | **Strategy**: cross_domain
- **Hint**: None | **Agrees with hint**: None
- **Text A**: _CTLA-4 checkpoint inhibition_
- **Text B**: _surgical rearrangement of digestive tract to reduce caloric absorption and alter incretin hormone signaling_
- **Justification**: CTLA-4 checkpoint inhibition is an immunotherapy mechanism targeting T cell regulation, while surgical rearrangement of the digestive tract is a metabolic intervention for weight management; they operate in entirely different biological systems and pathways with no hierarchical or mechanistic relationship.
