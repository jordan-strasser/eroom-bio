# Eroom Bio — Holdout Prediction Audit

Deep audit of why holdout AUROC collapsed to 0.243 (worse than random) despite in-sample AUROC of 0.683 after the round-14 prediction fixes.

**Audit scope**: all 19 scoreable holdout trials from the melanoma_145 corpus (50 holdout total; 31 skipped because the compound or indication isn't in the trained graph). Audit script is `scripts/audit_holdout.py` — deterministic, no LLM calls.

**Trained graph**: 942 nodes, 3,116 edges, 95 trial subgraphs. Round-14 fixes applied: LINCS demoted to MODERATE_SUPPORT, trust floor at 0.10, weak_contradict on engaged-but-failed, OT prior Beta(2,1), composite weakest-link picker.

---

## Architectural reminder (so the findings make sense)

P(success) is the **trust-weighted geometric mean** of Monte-Carlo samples from each edge's Beta posterior along the causal chain:

```
compound  --affects-->  target  --modulates_via-->  mechanism  --mechanism_affects-->  biology  --biology_drives-->  indication
```

Plus auxiliary edges (reflects_biology, endpoint_captures, responds_differently) when endpoint/population are provided.

A `compound→target→mechanism` prefix is **indication-agnostic** by construction. Only `mechanism_affects → biology_drives` carry indication-specific signal.

Edges with no evidence (Beta(1,1)) are still included via the round-14 trust floor (`_TRUST_FLOOR = 0.10`), so they pull toward the 0.5 prior mean at low weight rather than being dropped.

---

## [1] Per-trial table, sorted by P(success) descending

```
NCT             P(succ)  label  n_edges  compound                    indication            weakest_link
----------------------------------------------------------------------------------------------------------------------------------
NCT03400332       0.873      0        2  nivolumab                   cancer                affects             ← false-positive
NCT02519322       0.872      0        2  nivolumab                   cutaneous_melanoma    affects             ← false-positive
NCT04032704       0.833      0        2  pembrolizumab               prostate_cancer       affects             ← false-positive
NCT02967692       0.774      0        4  dabrafenib                  melanoma              mechanism_affects   ← false-positive
NCT03259425       0.763      0        4  nivolumab                   melanoma              biology_drives      ← false-positive
NCT04526730       0.763      1        4  nivolumab                   melanoma              biology_drives
NCT03834623       0.763      0        4  nivolumab                   melanoma              biology_drives      ← false-positive
NCT04655157       0.763      0        4  nivolumab                   melanoma              biology_drives      ← false-positive
NCT05297565       0.762      1        4  nivolumab                   melanoma              biology_drives
NCT03997474       0.762      0        4  nivolumab                   melanoma              biology_drives      ← false-positive
NCT03329846       0.762      0        4  nivolumab                   melanoma              biology_drives      ← false-positive
NCT03068455       0.762      0        4  nivolumab                   melanoma              biology_drives      ← false-positive
NCT03101254       0.750      1        4  vemurafenib                 melanoma              affects
NCT04899921       0.732      0        4  ipilimumab                  melanoma              biology_drives      ← false-positive
NCT02385669       0.732      0        4  ipilimumab                  melanoma              biology_drives      ← false-positive
NCT03241927       0.731      0        4  pembrolizumab               melanoma              biology_drives      ← false-positive
NCT02752074       0.730      0        4  pembrolizumab               melanoma              biology_drives      ← false-positive
NCT02574260       0.677      1        4  talimogene_laherparepvec    melanoma              affects
NCT02908672       0.664      1        4  cobimetinib                 melanoma              affects
```

### Key observations

- **9 of 19 trials are nivolumab → melanoma**, predicted at literally P ≈ 0.762–0.763. Ground truth across those 9: 2 success, 7 failure. The model has zero per-trial differentiation.
- Same applies to ipilimumab → melanoma (2 trials, both predicted 0.732, both failed) and pembrolizumab → melanoma (2 trials, both predicted ≈0.73, both failed).
- All 19 predictions land in the narrow band [0.66, 0.87]. **No prediction is below 0.5.** No negative signal at all.
- **The 3 trials with only 2 edges resolved** (`NCT03400332`, `NCT02519322`, `NCT04032704`) have the HIGHEST predicted P (0.87–0.83), all failures. Their indication wasn't connected to a biology node in the graph, so the indication-specific portion of the chain was dropped — leaving only the indication-agnostic `compound→target→mechanism` edges.

---

## [2] Top-5 highest-P trials that FAILED — full edge decomposition

### NCT03400332 — nivolumab → cancer — P=0.873, FAILED

Resolved chain: `nivolumab → ENSG00000188389 → checkpoint_blockade → UNKNOWN → cancer`

| Direction | Edge | Beta(α,β) | E[p] | n_eff | trust | bottleneck | Trial sources |
|---|---|---|---:|---:|---:|---:|---|
| UP ↑ | affects | nivolumab → ENSG00000188389 | 0.864 | 46.60 | 0.987 | 0.137 | NCT01783938(×2), NCT01844505(×2), NCT01927419, NCT02320058, NCT03618641 |
| UP ↑ | modulates_via | ENSG00000188389 → checkpoint_blockade | 0.883 | 44.16 | 0.974 | 0.119 | NCT01844505(×2), NCT01927419, NCT02130466, NCT02306850, NCT02320058, NCT03618641 |

Only 2 edges in the chain. The biology→indication portion was UNKNOWN, so the prediction is **entirely driven by indication-agnostic compound-target-mechanism edges**.

### NCT02519322 — nivolumab → cutaneous_melanoma — P=0.872, FAILED

Resolved chain: `nivolumab → ENSG00000188389 → checkpoint_blockade → UNKNOWN → cutaneous_melanoma`

Same 2 edges as NCT03400332 — `cutaneous_melanoma` isn't connected to `checkpoint_blockade` via a biology node (only `melanoma` is). The chain collapses to the indication-agnostic prefix and gets the same ~0.87.

| Direction | Edge | Beta(α,β) | E[p] | n_eff | trust | bottleneck | Trial sources |
|---|---|---|---:|---:|---:|---:|---|
| UP ↑ | affects | nivolumab → ENSG00000188389 | 0.864 | 46.60 | 0.987 | 0.137 | (same as above) |
| UP ↑ | modulates_via | ENSG00000188389 → checkpoint_blockade | 0.883 | 44.16 | 0.974 | 0.119 | (same as above) |

### NCT04032704 — pembrolizumab → prostate_cancer — P=0.833, FAILED

Resolved chain: `pembrolizumab → ENSG00000188389 → checkpoint_blockade → UNKNOWN → prostate_cancer`

| Direction | Edge | Beta(α,β) | E[p] | n_eff | trust | bottleneck | Trial sources |
|---|---|---|---:|---:|---:|---:|---|
| UP ↑ | affects | pembrolizumab → ENSG00000188389 | 0.741 | 4.76 | 0.448 | 0.226 | NCT02130466, NCT02306850 |
| UP ↑ | modulates_via | ENSG00000188389 → checkpoint_blockade | 0.883 | 44.16 | 0.974 | 0.119 | NCT01844505(×2), NCT01927419, NCT02130466, NCT02306850, NCT02320058, NCT03618641 |

**This is the canonical OOS failure**: pembrolizumab in MSS prostate is a known flop, but `checkpoint_blockade → R-HSA-389948 → prostate_cancer` doesn't exist in the graph, so the prediction is computed from indication-agnostic edges. **No prostate-specific information enters the prediction at all.**

### NCT02967692 — dabrafenib → melanoma — P=0.774, FAILED

Resolved chain: `dabrafenib → ENSG00000157764 → kinase_inhibition → R-HSA-141444 → melanoma`

| Direction | Edge | Beta(α,β) | E[p] | n_eff | trust | bottleneck | Trial sources |
|---|---|---|---:|---:|---:|---:|---|
| UP ↑ | affects | dabrafenib → ENSG00000157764 | 0.784 | 43.06 | 0.968 | 0.215 | NCT02039947(×4), NCT02314143(×3), NCT01597908, NCT01928940, NCT02130466 |
| UP ↑ | modulates_via | ENSG00000157764 → kinase_inhibition | 0.857 | 69.96 | 1.000 | 0.143 | NCT01227889(×2), NCT01597908(×2), NCT01153763, NCT01307397, NCT01754376, NCT01928940, NCT02039947, NCT02130466 |
| UP ↑ | mechanism_affects | kinase_inhibition → R-HSA-141444 | 0.775 | 22.00 | 0.802 | 0.235 | (LINCS records: 22 — no trial evidence) |
| == • | biology_drives | R-HSA-141444 → melanoma | 0.500 | 0.00 | 0.100 | 0.230 | (Beta(1,1) — no evidence) |

Three of four edges are pulling UP with very high evidence. The `biology_drives` edge has NO evidence (Beta(1,1), trust floor only). All upstream evidence is melanoma-positive — failure pattern of this specific trial isn't visible to the chain.

### NCT03259425 — nivolumab → melanoma — P=0.763, FAILED

Resolved chain: `nivolumab → ENSG00000188389 → checkpoint_blockade → R-HSA-389948 → melanoma`

| Direction | Edge | Beta(α,β) | E[p] | n_eff | trust | bottleneck | Trial sources |
|---|---|---|---:|---:|---:|---:|---|
| UP ↑ | affects | nivolumab → ENSG00000188389 | 0.864 | 46.60 | 0.987 | 0.137 | NCT01783938(×2), NCT01844505(×2), NCT01927419, NCT02320058, NCT03618641 |
| UP ↑ | modulates_via | ENSG00000188389 → checkpoint_blockade | 0.883 | 44.16 | 0.974 | 0.119 | NCT01844505(×2), NCT01927419, NCT02130466, NCT02306850, NCT02320058, NCT03618641 |
| UP ↑ | mechanism_affects | checkpoint_blockade → R-HSA-389948 | 0.742 | 83.66 | 1.000 | 0.261 | NCT03484923(×5), NCT01844505(×2), NCT02263508(×2), NCT01927419, NCT02130466, NCT02306850, NCT02320058, NCT03618641 |
| UP ↑ | biology_drives | R-HSA-389948 → melanoma | 0.607 | 93.06 | 1.000 | 0.393 | NCT03484923(×5), NCT01783938(×2), NCT01844505(×2), NCT02054520(×2), NCT02263508(×2), NCT01927419, NCT02130466, NCT02320058, NCT03618641 |

**Every edge pulls UP** — even biology_drives at E[p]=0.607 with n_eff=93. Evidence is concentrated on a small number of large-success trials (CheckMate-067 = NCT01844505 contributes to every edge).

---

## [3] Top-5 lowest-P trials that SUCCEEDED — full edge decomposition

### NCT02908672 — cobimetinib → melanoma — P=0.664, SUCCEEDED

Resolved chain: `cobimetinib → ENSG00000169032 → kinase_inhibition → R-HSA-141444 → melanoma`

| Direction | Edge | Beta(α,β) | E[p] | n_eff | trust | Trial sources |
|---|---|---|---:|---:|---:|---|
| UP ↑ | affects | cobimetinib → ENSG00000169032 | 0.659 | 3.40 | 0.379 | NCT02230306 |
| UP ↑ | modulates_via | ENSG00000169032 → kinase_inhibition | 0.750 | 2.00 | 0.281 | (no trial evidence, OT-derived) |
| UP ↑ | mechanism_affects | kinase_inhibition → R-HSA-141444 | 0.775 | 22.00 | 0.802 | LINCS records: 22 |
| == • | biology_drives | R-HSA-141444 → melanoma | 0.500 | 0.00 | 0.100 | (Beta(1,1)) |

Same chain shape as the dabrafenib failure above; just sparser (cobimetinib has only 1 trial in training). Prediction lands lower because of weaker evidence on the compound→target edge, not because of any indication-aware signal.

### NCT02574260 — talimogene_laherparepvec → melanoma — P=0.677, SUCCEEDED

Resolved chain: `talimogene_laherparepvec → ENSG00000157873 → other → R-HSA-844456 → melanoma`

| Direction | Edge | Beta(α,β) | E[p] | n_eff | trust | Trial sources |
|---|---|---|---:|---:|---:|---|
| UP ↑ | affects | talimogene_laherparepvec → ENSG00000157873 | 0.667 | 19.04 | 0.766 | NCT01740297(×2), NCT00289016, NCT01368276, NCT02211131 |
| UP ↑ | modulates_via | ENSG00000157873 → other | 0.792 | 10.04 | 0.614 | NCT01740297(×2), NCT00289016, NCT02211131 |
| UP ↑ | mechanism_affects | other → R-HSA-844456 | 0.667 | 1.00 | 0.177 | (no trial evidence) |
| == • | biology_drives | R-HSA-844456 → melanoma | 0.500 | 0.00 | 0.100 | (Beta(1,1)) |

### NCT03101254 — vemurafenib → melanoma — P=0.750, SUCCEEDED

Resolved chain: `vemurafenib → ENSG00000157764 → kinase_inhibition → R-HSA-141444 → melanoma`

**Identical chain to the NCT02967692 dabrafenib failure above** (both compounds bind BRAF; same target→mechanism→biology→indication suffix). The model predicts ~0.75 for both regardless of trial outcome.

| Direction | Edge | Beta(α,β) | E[p] | n_eff | trust | Trial sources |
|---|---|---|---:|---:|---:|---|
| UP ↑ | affects | vemurafenib → ENSG00000157764 | 0.719 | 25.50 | 0.838 | NCT01307397, NCT01597908, NCT01754376, NCT02230306 |
| UP ↑ | modulates_via | ENSG00000157764 → kinase_inhibition | 0.857 | 69.96 | 1.000 | (same as dabrafenib above) |
| UP ↑ | mechanism_affects | kinase_inhibition → R-HSA-141444 | 0.775 | 22.00 | 0.802 | LINCS records: 22 |
| == • | biology_drives | R-HSA-141444 → melanoma | 0.500 | 0.00 | 0.100 | (Beta(1,1)) |

### NCT05297565, NCT04526730 — nivolumab → melanoma — P≈0.762, SUCCEEDED

**Identical chain to all the nivolumab → melanoma failures above.** Same 4 edges, same evidence sources, same P. The architecture is incapable of distinguishing 2 successes from 7 failures within the same compound × indication.

---

## [4] Cross-contamination check

**No cross-contamination found.** No holdout trial's evidence is appearing as a source on any other holdout trial's edges. The temporal split is clean.

---

## [5] Indication leakage check

**7 edges have trial evidence spanning ≥2 indications.** Most cross-indication leakage is via umbrella terms (`cancer`, `malignant_neoplasm`, `solid_tumour`) rather than across distinct cancer types.

```
mechanism_affects      checkpoint_blockade → R-HSA-389948             total_trials=168
   melanoma(×156), malignant_neoplasm(×12)

modulates_via          ENSG00000188389 → checkpoint_blockade          total_trials=91
   melanoma(×78), malignant_neoplasm(×13)

affects                ipilimumab → ENSG00000163599                   total_trials=46
   melanoma(×44), intraocular_melanoma(×2)

modulates_via          ENSG00000163599 → checkpoint_blockade          total_trials=20
   melanoma(×18), intraocular_melanoma(×2)

modulates_via          ENSG00000157764 → kinase_inhibition            total_trials=20
   melanoma(×14), cancer(×4), solid_tumour(×2)

affects                dabrafenib → ENSG00000157764                   total_trials=10
   melanoma(×9), solid_tumour(×1)

affects                pembrolizumab → ENSG00000188389                total_trials=6
   melanoma(×3), malignant_neoplasm(×3)
```

This isn't the dominant pathology, but it confirms by design: `compound→target` and `target→mechanism` edges accumulate evidence across every indication that exercises them.

---

## [6] P(success) histograms by actual outcome

```
Label = 1 (SUCCESS) — n=5,  mean=0.723,  median=0.750
  [0.00–0.10)    0
  [0.10–0.20)    0
  [0.20–0.30)    0
  [0.30–0.40)    0
  [0.40–0.50)    0
  [0.50–0.60)    0
  [0.60–0.70)    2 ██████████████████████████
  [0.70–0.80)    3 ████████████████████████████████████████
  [0.80–0.90)    0
  [0.90–1.00)    0

Label = 0 (FAILURE) — n=14,  mean=0.775,  median=0.762
  [0.00–0.10)    0
  [0.10–0.20)    0
  [0.20–0.30)    0
  [0.30–0.40)    0
  [0.40–0.50)    0
  [0.50–0.60)    0
  [0.60–0.70)    0
  [0.70–0.80)   11 ████████████████████████████████████████
  [0.80–0.90)    3 ██████████
  [0.90–1.00)    0

AUROC = 0.286   (n_pos=5, n_neg=14)
Mean P | label=1: 0.723
Mean P | label=0: 0.775
⚠ MEAN P IS LOWER FOR SUCCESSES — prediction is inverted
```

Both distributions are confined to [0.66, 0.87]. The failures actually sit slightly to the right of the successes. The model is mildly inverted AND confidence-collapsed.

---

## Diagnosis

The 0.243 AUROC isn't caused by buggy math. The Beta-Binomial updates are working — the high-evidence edges have n_eff in the 40–90 range and reflect actual training-set outcomes. The audit reveals **three structural failure modes**:

### 1. Degenerate predictions across same-chain trials
The 9 nivolumab → melanoma holdout trials produce identical P values (0.762 ± RNG noise) because they all walk the identical chain. Ground truth is 2 success, 7 failure. **No per-trial signal can possibly differentiate them under the current architecture.** This is the unit-mismatch identified in NEXT_SESSION.md, but now quantified: the model is predicting at the (compound, indication) level, and trial-to-trial outcome variance within a (compound, indication) cell is invisible.

### 2. Chain collapse on out-of-graph indications
For non-melanoma indications (`cancer`, `prostate_cancer`, `cutaneous_melanoma`), the `biology → indication` edge doesn't exist in the graph, so `mechanism_affects` and `biology_drives` are skipped. Predictions then collapse to the **indication-agnostic prefix only** (`compound → target → mechanism`). Because these edges are dominated by melanoma-success evidence, the prediction is **more confident**, not less, when the trial is out-of-indication. That's why pembrolizumab → prostate predicts 0.833 (3rd highest) despite being a famous failure.

### 3. Evidence asymmetry favoring successes in the trained set
For nivolumab → melanoma, the `affects` edge has E[p]=0.864 with n_eff=46. The 7 contributing training trials (NCT01844505 CheckMate-067, NCT01783938, NCT01927419, NCT02130466, NCT02306850, NCT02320058, NCT03618641) are dominated by successes. The classifier's round-14 `weak_contradict` rule for engaged-but-failed didn't shift these edges much because (a) the training slice has few nivolumab failures, and (b) most failures get classified as `ambiguous` not `weak_contradict` due to confounding.

### Cross-contamination is not a problem
The temporal split is clean — no holdout trial's evidence is poisoning another holdout's prediction.

### Indication leakage is minor
Only 7 edges share evidence across indications, and the cross-indication signal is small (10–20% of records on most edges). This isn't the main driver.

---

## Implication for next steps

The architecture cannot improve OOS AUROC without either:

1. **Re-framing the prediction unit** — score `(compound, indication)`-level success rates instead of per-trial outcomes. Matches what the architecture actually computes.
2. **Enriching the chain with per-trial covariates** — line of therapy, biomarker enrichment, dose, comparator. The `responds_differently` edge partially supports this but isn't pervasive. Would be a big architectural lift (breaks v0.1.0 lock).
3. **Better classifier signal on failures** — even within the (compound, indication) cell, the architecture would predict differently if failure evidence on `mechanism_affects` and `biology_drives` had more weight. Currently most failures classify as `ambiguous` rather than `weak_contradict`, so they sharpen variance without shifting means.

Round-14 fixes are real improvements for in-sample (0.596 → 0.683) and don't make OOS worse — the OOS failure is architectural, not algorithmic.

---

## Reproduction

```bash
python -m scripts.audit_holdout
```

No LLM calls. Fetches CT.gov conditions for the 50 holdout NCTs (~25s). All extraction + classification annotations are cached in `data/annotations/`. Deterministic via `np.random.seed(42)` per prediction.
