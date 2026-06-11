# Cross-indication transfer (leave-one-indication-out) — findings

Source: `data/exports/phasec_n250_annotated.json` (phasec_n250), n_samples=2000.

Hold out EVERY trial touching indication Y from attribution; predict Y's trials from the SHARED upstream edges other indications populated. AUROC > 0.5 = transfer.

- LOIO-evaluable indications: **8** (≥4 both-class trials)
- pooled held-out trials: **56** (succ 38 / fail 18)
- **POOLED LOIO AUROC = 0.468** (chance 0.5; random-fold holdout 0.534)
- **MEAN within-indication AUROC = 0.530** (strictest — removes base-rate)

## per-indication AUROC
| indication | AUROC | n |
|---|---|---|
| colon_cancer | 1.000 | 4 |
| crohn_disease | 0.833 | 5 |
| cardiovascular_disease | 0.556 | 10 |
| alzheimer_disease | 0.500 | 4 |
| rheumatoid_arthritis | 0.350 | 14 |
| diabetes | 0.333 | 10 |
| lupus_nephritis | 0.333 | 5 |
| breast_cancer | 0.333 | 4 |
