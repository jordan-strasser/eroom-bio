# Prediction Benchmark, Calibration & Tuning — design doc

Status: **design** (not yet built). Owner decision pending on sequencing. Companion to `SCALING.md`. Cold-start: linked from `NEXT_SESSION.md`.

## Why this exists
External pushback: "an LLM / XGBoost / logistic regression may match or beat the structured graph on prediction while implicitly capturing cross-trial accumulation." We should *want* to know. This doc specifies a fair comparator **and** resolves two deeper questions it forces: what accuracy *target* is even right, and how to tune without cheating.

**Honest grounding (n=500 true-holdout, K=5, n=129, ~71% successes):** scalar holdout AUROC **0.568**, binary accuracy **0.566** — *below* the 0.713 "always predict success" base rate. Field holdout **0.633** (mild marginal-leak upper bound). So the first baselines to beat are **base rates**, not XGBoost; and the accuracy/AUROC gap + under-prediction (FN=37) says the model is **mis-calibrated (pessimistic)**, not necessarily weak.

## Q1 — What accuracy target is correct? (the bedrock)
Interpretability / counterfactuals / "new biology" are only worth anything on **accurate** edges — accuracy IS the bedrock. But the bedrock metric is **edge calibration**, not chain-AUROC→1:
- A counterfactual ("anti-PD-1 in a new indication, BRAF-mutant population") *composes* per-edge Beta beliefs. It is trustworthy iff (a) each edge is calibrated (a 0.7 edge holds ~70%) and (b) the softmin composition is sound. Both are **directly measurable** — reliability diagrams, Brier, ECE — independent of the noisy chain outcome.
- **AUROC→1 is not achievable** for trial success: the outcome is multi-causal (dose, population, powering, safety, commercial, luck), mostly orthogonal to mechanism. The graph predicts the *mechanistic component*; feature-rich approval models top out ~0.7–0.85 *with* non-mechanistic features. So ~0.7 in-sample for a pure-mechanism model is plausibly near the ceiling. The real questions: **how much variance is mechanistic at all** (ceiling analysis), and **are edges calibrated + beating base rates** (the achievable, counterfactual-enabling target).
- **Ceiling analysis:** train XGBoost on (i) mechanistic fields only vs (ii) all fields incl. non-mechanistic (phase, enrollment, sponsor, year). If (ii) ≫ (i), the ceiling for ANY mechanism model is low — and that's the honest cap to report, not a failure of the graph.

## Q2 — What to train baselines on (the three-level ladder)
Not one variable — three experiments, each isolating a different claim. Run all three; **(b) is the primary fair head-to-head.**

| Level | Inputs | Question | If a baseline wins |
|---|---|---|---|
| (a) raw **design** text | eligibility/arms/endpoints/mechanism prose (NEVER results) | does extraction+structure beat raw text? | extraction isn't earning its cost |
| **(b) extracted fields** ⭐ | compound/target/mechanism/biology/indication/endpoint/population (the graph's own inputs, minus outcome) | does typed structure + Bayesian accumulation + softmin beat a *flat* model on identical features? | structure helps interpretability, not accuracy |
| (c) graph features | node/edge Beta features, path features | is the softmin math leaving accuracy on the table vs an ML head? | the *prediction math* is the weak link, not the graph |

(b) is the dispute made testable: the graph is built from exactly these fields, so a flat model on them isolates the value of structure.

## Methods
- **Base rates (floors, must clear):** marginal, per-indication, per-phase, per-target-class.
- **Logistic regression / XGBoost** on (b) extracted fields (and as ablations on (a) text-embeddings and (c) graph features).
- **LLM** — see contamination warning below; in-context (K training trials as examples) or fine-tuned, on (a) design text.
- **The graph** — scalar + field, from `eval_holdout_kfold`.

## Controls that decide validity (more important than model choice)
1. **Outcome leakage:** every method sees **design-time inputs only** — no results section (it contains the answer). The graph already does this.
2. **LLM pretraining contamination (the hardest):** most of the corpus is old, published trials the LLM likely *memorized*. A zero-shot LLM recalls, not predicts. Clean LLM test = trials whose outcome **postdates the model cutoff**, or a strict held-out fine-tune; otherwise report the LLM number as a contaminated ceiling.
3. **Identical label + folds:** reuse `eval_holdout_kfold`'s scorable set, fold assignment, and `_resolve_label` (with ambiguous-drop). Same held-out trials → **paired** tests (DeLong / McNemar) + CIs, which you need at n≈129 (0.57 vs 0.61 is not distinguishable unpaired).
4. **IID vs structured holdout — report both.** Random IID favors flat models (interpolation). The discriminating test for the thesis is **leave-one-target-out / leave-one-indication-out**: flat models fall back to base rates on unseen structure; the graph should compose via shared mechanisms (the bevacizumab-lands-via-VEGFA pattern). If the graph wins anywhere, it's here.

## Tuning discipline (so we don't cheat)
- **Nested CV** (two loops): the OUTER loop holds out a test fold and never uses it for any decision; the INNER loop runs a *second* CV inside the outer-train data to pick hyperparameters (N_eff tiers, field bandwidth, softmin temperature, safety weights), then the chosen config is scored once on the untouched outer-test fold. Inner loop = "which hyperparameters"; outer loop = "how good is the *tuned* procedure" — the test fold selects nothing. NEVER tune on the headline holdout (the round-25/28 anti-pattern: `[[project_round28_sicko_mode]]`, `[[project_round25_evidence_records]]`). Cheaper pragmatic alt at our small-n + expensive re-attribution: a fixed train/validation/test split (tune on val, report on test) or a light inner grid.
- **Tune for calibration first, AUROC second.** The 0.566<0.71 gap is likely a calibration/threshold artifact; a Platt/isotonic recalibration of predictions + revisiting the pessimism in the failure-`contradict` p_obs / softmin may lift accuracy AND make counterfactuals trustworthy — the bedrock from Q1.
- **All methods tuned the same way.** Comparing a tuned graph to a default XGBoost (or vice-versa) is invalid; tune every method on the same validation folds, report all on the same test fold.

## Metrics
AUROC **+ PR-AUC** (class imbalance) **+ Brier + ECE/reliability bins** (calibration = the bedrock) **+ lift over base rate** + the structured-holdout (leave-X-out) AUROCs. Report DeLong CIs and paired tests, not bare point estimates.

## Sequencing recommendation (resolves "tune vs compare vs scale")
1. **Build the harness** (`scripts/eval_baselines.py` + nested-CV + calibration metrics, reusing `eval_holdout_kfold` folds/labels). This single harness serves tuning, the ceiling analysis, AND the ML comparison.
2. **One disciplined calibration/tuning pass** on n=500 via nested CV. Expect modest AUROC gains (low ceiling) but meaningful calibration gains.
3. **ML comparison** inside the same harness (tuned-vs-tuned, shared folds, base-rate floors, IID + leave-X-out).
4. **Do NOT gate scaling on maxing the n=500 AUROC.** Optimal knobs shift with corpus density (redundancy/conflict dynamics change), and corpus-quality fixes (target-inference, population/endpoint enrichment — `[[project_target_inference_diagnostic_gate]]`, `[[project_population_endpoint_enrichment]]`) likely move the ceiling more than knob-tuning. Re-tune per scaling rung.

## What "winning" means (honest positioning)
Two outcomes, both valuable: (i) if a baseline matches/beats IID AUROC, the graph's value is its *other* axes — interpretability, compositional counterfactuals, novel-compound/indication generalization, mechanism discovery (none of which a flat model does); (ii) if the graph wins, it should be on **structured generalization** + **calibration**, where composition beats interpolation. Don't concede that AUROC is the only scoreboard — but put the graph on it honestly, and lead the bedrock with **edge calibration**, since that's what makes every other claim load-bearing.

## Files (when built)
- `scripts/eval_baselines.py` — base rates / LogReg / XGBoost / LLM on (a)/(b)/(c), reusing `eval_holdout_kfold` folds+labels; paired CIs.
- nested-CV + calibration (Brier/ECE/reliability) added to the holdout harness.
- start cheapest: base-rate floors + LogReg-on-fields (the floors that matter most) for a number first.
