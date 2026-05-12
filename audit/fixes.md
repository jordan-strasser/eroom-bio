Here's what I noticed, organized by where the breakage happens in the
  pipeline. The most consequential bugs are at the JSON → node-mapping step, not
   in the raw extraction.                                                       
                          
  1. Combo arms collapse to a single target, losing the other mechanism entirely
                                                                                
  This is the most damaging pattern, repeated across all four trials I looked   
  at:                                                                           
                                                                                
  ┌─────────────┬─────────────────────────┬─────────────────────────────────┐   
  │    Trial    │ Extraction says target  │    Every chain's target_id      │
  │             │            =            │           actually is           │   
  ├─────────────┼─────────────────────────┼─────────────────────────────────┤
  │ NCT01844505 │ "PD-1 and CTLA-4"       │ ENSG00000188389 (PD-1) —        │
  │             │                         │ ipilimumab arm too              │   
  ├─────────────┼─────────────────────────┼─────────────────────────────────┤
  │ NCT01950390 │ "CTLA-4 and VEGF"       │ ENSG00000112715 (VEGFA) —       │   
  │             │                         │ ipilimumab arm too              │
  ├─────────────┼─────────────────────────┼─────────────────────────────────┤
  │ NCT03484923 │ "PD-1, LAG-3, c-MET,    │ ENSG00000188389 (PD-1) — all 4  │
  │             │ IL-1β, CDK4/6"          │ combo arms                      │   
  ├─────────────┼─────────────────────────┼─────────────────────────────────┤
  │ NCT03618641 │ "TLR9 + PD-1"           │ UNKNOWN (codename cmp_001 lost) │   
  └─────────────┴─────────────────────────┴─────────────────────────────────┘   
   
  In NCT01844505 the ipilimumab-only arm (Chain 3, 6, 9, 12) is labeled         
  target_id=PD-1, mechanism=checkpoint_blockade — but ipilimumab binds CTLA-4,
  not PD-1. Same in NCT01950390: the ipi-only chain is labeled target_id=VEGFA, 
  mechanism=angiogenesis_inhibition. The populator is picking ONE target per
  (compound, indication) pair and applying it to every chain regardless of which
   arm.
  2. Classifier is also missing the second target                               
   
  For NCT01844505, the classifier emits only binds_to: nivolumab → PD-1 and     
  binds_to: ipilimumab+nivolumab → PD-1. There's no ipilimumab → CTLA-4 edge
  update at all. The synthesized combo ipilimumab+nivolumab is also being       
  credited with binding PD-1 directly, which is a category error — the combo
  should decompose into constituent binds_to edges via composed_of.
  3. Subgroups attach to arms they don't apply to
  The PD-L1 (cd274) subgroup chains in NCT01844505 are built for all 3 arms,    
  including the ipi-only arm. But PD-L1 stratification is mechanistically
  meaningful only for nivolumab (PD-1 blockade) — the ipi monotherapy's response
   shouldn't be conditioned on PD-L1 status. The graph now has 4 ipi-only ×
  PD-L1 chains pointing the wrong mechanism at PD-L1 subgroups.
  4. Indication slug fragmentation breaks endpoint matching                     
   
  NCT01950390's chain has:                                                      
  - indication_id = stage_iiic_cutaneous_melanoma_ajcc_v7
  - endpoint_id = OS_unresectable_melanoma                                      
                                          
  The endpoint id is anchored to a different indication slug than the trial's   
  own indication. The classifier even tried to fix this with an                 
  endpoint_captures: OS_unresectable_melanoma →                                 
  stage_iiic_cutaneous_melanoma_ajcc_v7 update — bridging two slugs that should 
  have been the same node. CT.gov's free-text condition strings are fragmenting
  one disease into many indication nodes. NCT01844505 uses
  unresectable_or_metastatic_melanoma; NCT01950390 uses
  stage_iiic_cutaneous_melanoma_ajcc_v7; the extremes file shows yet another
  (stage_iii_melanoma, melanoma, lymph_node_cancer). Cross-trial evidence is
  being scattered across what should be one node.
  5. PD-L1 threshold collapse is lossy and inconsistent                         
   
  The extraction maps PD-L1 expression levels like this:                        
                  
  ┌────────────────┬────────────┬─────────────────────────────┐                 
  │ Raw descriptor │ Slug level │ Resulting population suffix │
  ├────────────────┼────────────┼─────────────────────────────┤                 
  │ PD-L1 < 1%     │ low        │ cd274_low                   │
  ├────────────────┼────────────┼─────────────────────────────┤
  │ PD-L1 ≥ 1%     │ positive   │ cd274_positive              │
  ├────────────────┼────────────┼─────────────────────────────┤                 
  │ PD-L1 < 5%     │ low        │ cd274_low                   │
  ├────────────────┼────────────┼─────────────────────────────┤                 
  │ PD-L1 ≥ 5%     │ high       │ cd274_high                  │
  ├────────────────┼────────────┼─────────────────────────────┤                 
  │ PD-L1 < 10%    │ low        │ cd274_low                   │
  ├────────────────┼────────────┼─────────────────────────────┤                 
  │ PD-L1 ≥ 10%    │ high       │ cd274_high                  │
  └────────────────┴────────────┴─────────────────────────────┘
  Three different < thresholds collapse to one node; the ≥ side splits into two 
  nodes (1% → positive, 5%/10% → high). The vocabulary axis is doing two
  different jobs at once — direction (low/positive) and magnitude (high) — and  
  the cutoffs are inconsistent.
  6. NCT01950390 (failed trial) contributed ZERO efficacy edge updates          
   
  22 edge updates on this trial, but every single one is causes_ae. The         
  classifier emitted exactly one edges_to_update entry (an endpoint_captures
  with ambiguous support), and that one had the slug-mismatch problem in #4, so 
  it likely got dropped via _log_unrouted. Net result: a failed Phase 2 produced
   no efficacy evidence — the BEFORE/AFTER for predict barely moves (0.7990 →
  0.7988). The system literally cannot learn from this failure. This is what's
  pushing it into the "worst" bucket: the trial fails but the graph never sees
  the contradicting signal.
  7. "Unspecified adverse event" is a meaningless AE node                       
   
  NCT01950390 extracted term: "Grade 3-5 adverse events" (the summary row, 58%  
  vs 39%). The MedDRA normalizer coerced this into AE:unspecified_adverse_event
  rather than rejecting it. Ipilimumab now has a Beta(85.90, 6.70) → P=0.93     
  probability of causing "unspecified adverse event". This edge is being
  reinforced by many trials but doesn't mean anything — the extractor shouldn't
  include the summary row, or the normalizer should reject meta-terms.
  8. Spelling inconsistency: hemolytic vs haemolytic, anaemia vs anemia         
   
  For the same trial, "Haemolytic anaemia" → AE:hemolytic_anemia (Americanized) 
  while "Anaemia" → AE:anaemia (British preserved). The MedDRA normalizer is
  non-deterministic across British/American variants — same evidence stream is  
  split across two nodes for the same AE term depending on which spelling the
  trial used.
  9. Bevacizumab gets moderate_support for AEs from n≈2 patients                
   
  NCT01950390's combo arm (only) has 1.2% incidence for gait disturbance, sudden
   death, erythema multiforme, pruritus, skin ulceration. The _ae_support_bucket
   threshold (RR ≥ 2 with the 0.5pp control-rate floor) labels all of these     
  moderate_support for bevacizumab. With n=169 patients in the trial split
  across two arms, 1.2% = ~1 patient. That's noise being graded as moderate
  evidence on the bevacizumab → AE edge — and these AEs are equally compatible
  with being ipi-attributable in a combo. Threshold needs an absolute-count
  gate.
  ---
  Where to fix first, in priority order:
                                                                                
  1. #1 + #2 — Per-arm target/mechanism resolution. The populator and classifier
   both need to route by arm.compound_ids rather than picking one canonical     
  target per trial. This is the single biggest source of wrong evidence
  accumulation. (Architecture-locked at v0.1.0 per CLAUDE.md — but the          
  extraction/classification prompts and populate logic aren't locked.)
  2. #4 — Indication slug canonicalization. Map CT.gov conditions through a
  normalizer (MeSH or EFO) so stage_iiic_cutaneous_melanoma_ajcc_v7 and         
  unresectable_or_metastatic_melanoma resolve to one melanoma node. Endpoint ids
   will then share too.                                                         
  3. #6 — Classifier should emit efficacy edges for failures. The prompt is
  letting "insufficient information" suppress all edge updates, which throws    
  away the failure signal. A failed trial should at least weaken
  endpoint_captures and/or biology_drives even without biomarker data.          
  4. #7, #8 — MedDRA normalizer pre-filter. Reject "Grade N adverse events"
  summary rows; canonicalize British/American spellings (lowercase + a small    
  synonym table).
  5. #5 — PD-L1 (and biomarker) vocabulary. Pick one axis: either a single      
  threshold cut (positive/negative at 1%) or a numeric value carried as         
  metadata. The current 3-bucket non-monotonic mapping is worse than either
  alternative.

6. Adding Real biology nodes:
The biology layer needs real pathway data, not mechanism__indication slugs. Fix the biology resolution:

When resolving a trial's biology_id, do NOT create a slug fallback. Instead:

1. Look up the trial's resolved target in the LINCS-populated edges. If target ENSG00000188389 (PD-1) has mechanism_affects edges to Reactome pathway nodes, use those pathway nodes as the biology_id.

2. If multiple pathways exist for the same target-mechanism pair, create chains for each — they're different biological hypotheses.

3. If no LINCS pathway exists for this target, THEN use OpenTargets to look up the target's associated pathways via the Reactome API. Cache results.

4. Only use the mechanism__indication slug as a last resort, and tag it in metadata as "unresolved_biology" so we can track how many chains are running on placeholder biology.

After this fix, a checkpoint blockade trial should map to actual pathway nodes like PD-1 signaling, T-cell receptor signaling, or adaptive immune response — not "checkpoint_blockade__melanoma". These pathway nodes are shared across indications, which is what enables cross-disease learning.

Show me the biology_id resolution for NCT01844505 after this fix. I want to see real Reactome pathway names.                                                                  
                  

