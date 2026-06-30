# How edge weights are assigned per trial

*Technical onboarding. Code-grounded walkthrough of belief formation: how one
clinical trial turns into concrete `Beta(α, β)` updates on the knowledge-graph
edges. Plain-language companion: [`docs/ORIENTATION.md`](ORIENTATION.md). This
doc is the precise, file-cited version.*

> Scope: this is the **training / belief-formation** path (how a trial *writes*
> edge beliefs). The **prediction** path (how those beliefs are *read* and
> composed into P(success)) is a separate concern — see the
> "Where this meets prediction" section at the end.

---

## 0. The one-line answer

Every edge carries a `Beta(α, β)` belief. A trial updates an edge by a
**Beta-Binomial conjugate step**:

```
α  +=  n_eff · p_obs
β  +=  n_eff · (1 − p_obs)
```

(`src/inference/beliefs.py::apply_virtual_evidence`, line 472)

So "assigning an edge weight" decomposes into two numbers per edge per trial:

- **`n_eff`** — *how much* this trial's evidence counts (a virtual sample size).
- **`p_obs`** — *which direction and how far* (probability the edge "held",
  ∈ [0, 1]).

Everything below is how the pipeline computes those two numbers for each edge.

---

## 1. The pipeline that produces the inputs

A trial flows through three annotation stages before any belief moves
(`scripts/build_graph.py`: fetch → populate → annotate → attribute):

1. **Extraction** (`src/annotation/extractor.py`) — LLM reads the trial and
   emits structured `results_by_chain`: per-arm outcomes, the enrollment
   `sample_size`, effect sizes, p-values.
2. **Classification** (`src/annotation/classifier.py`) — LLM assigns a
   `FailureClassification`: a `primary_failure_mode` (one of 13), a
   `confidence` ∈ [0,1], and a coarse `operational_failure` bit
   (`src/annotation/taxonomy.py::FailureClassification`, line 644).
3. **Attribution** (`src/annotation/attributor.py::Attributor.attribute`,
   line 673) — the step that actually writes beliefs. The rest of this doc
   lives here.

The key architectural decision (line 681): **a single trial cannot pinpoint
which edge of its chain failed** — that would be premature falsification. So the
backbone is *not* attributed by name-matching a classifier-named edge. Instead
the trial's per-arm **outcome conditions the whole chain by edge id**, and
cross-trial overlap triangulates the responsible edge over many trials. *Trial
failure ≠ mechanistic falsification.*

---

## 2. The Beta belief object

Each edge stores `EdgeBeliefState(alpha, beta, evidence=[...])`. The prior is
`Beta(1, 1)` (uniform). `evidence_strength ≈ (α + β − 2)` — an edge still at
`Beta(1,1)` has zero learned evidence and is **dropped** by the predictor.

The reliability estimate is `E[p] = α / (α + β)`; confidence grows as `α + β`
grows. `α` accumulates "held / worked" mass; `β` accumulates "failed" mass.

---

## 3. Computing `n_eff` — how much the trial counts

`n_eff` for a backbone edge is built up in
`_condition_chain_on_outcomes` (line 783) as:

```
w_base = base_n · f_N · gate_weight          (line 866)
n_eff_i = w_base · (per-edge share)          (lines 1013 / 1132)
```

### 3a. `base_n` — evidence-class constant × LLM confidence

```
base_n = effective_n_for_evidence(evidence_type, quality)
       = EVIDENCE_TYPE_N_EFF[evidence_type] · quality       (beliefs.py:245)
```

- `evidence_type` comes from the trial **phase**
  (`_PHASE_TO_EVIDENCE`, attributor.py:105): Phase 3/4 → `CLINICAL_PHASE3`,
  Phase 2 → `CLINICAL_PHASE2`, Phase 1 → `CLINICAL_PHASE1`.
- `EVIDENCE_TYPE_N_EFF` (beliefs.py:100) is the pseudocount table. The clinical
  tiers that the outcome path uses:

  | evidence type | n_eff |
  |---|---:|
  | `CLINICAL_PHASE3` | 15.0 |
  | `CLINICAL_PHASE2` | 6.0 |
  | `CLINICAL_PHASE1` | 2.0 |

  (The database tiers — OT-direct 12, ChEMBL 10, Reactome/GO 1.5, etc. — are for
  the *ingestion-time* prior edges the populator lays down, not the per-trial
  outcome update. Same table, different callers.)
- `quality = min(classification.confidence, 1.0)` (line 860) — the LLM
  classifier's self-reported confidence discounts the weight.

### 3b. `f_N` — saturating √N population multiplier

```
f_N = _precision_multiplier(extraction.sample_size)          (beliefs.py:210)
```

Concave in N (sqrt), anchored at N=350 (≈ corpus median enrollment) → mult 1.0,
floored 0.5, ceiled 2.5. A 1400-patient trial counts ~2× a 350-patient one, not
4×. Called **directly** in the outcome path (always on), independent of any
flag.

### 3c. `gate_weight` — the operational down-weight

```
gate_weight = classification.gate_weight                     (taxonomy.py:667)
            = 0.2 if operational_failure is True else 1.0     (taxonomy.py:626)
```

If the classifier is *positive* the failure was trial-conduct (recruitment
collapse, manufacturing, funding termination), the whole trial barely perturbs
the mechanism beliefs (×0.2). Conservative by design — only down-weights on a
positive operational signal, never a guess. (Note: under routing, an
operational *failure* is censored entirely — see §5 — so this gate mostly
matters for operational trials that still carry a usable outcome.)

### 3d. the per-edge share

The last factor depends on outcome + routing (§4–§5). Either:
- **SUCCESS**: share = 1.0 — every edge gets the *full* `w_base`.
- **Routed EFFICACY/MEASUREMENT failure**: each edge gets the full `w_base`
  too, but the α/β *split* encodes the blame (responsibility, §5b).
- **Unknown-branch failure**: `w_base` is *split* across edges by the
  explaining-away fractions `w_i` (§5c), so total chain mass = `w_base`.

---

## 4. Computing `p_obs` — direction and magnitude

`p_obs` is set by the **per-arm outcome**, resolved by
`_aggregate_arm_outcomes` (line 313): it reads `extraction.results_by_chain`,
maps each result's arm to a graph arm id (direct `group_id` match, else
compound-set fallback), and yields `{arm_id: TrialOutcome}`. A trial-level
fallback applies when no per-arm outcome resolves (line 849).

The baked p_obs constants (attributor.py:137, reusing the `BUCKET_TO_P_OBS`
scale in beliefs.py:65):

| outcome | p_obs | meaning |
|---|---:|---|
| SUCCESS | **0.80** | the whole conjunctive path operated |
| FAILURE | **0.20** | modest contradict (something broke — but what?) |
| PARTIAL | **0.35** | weaker / more ambiguous contradict |
| safety-survival credit | **0.0** | "the safety gate did not fire" (all mass → β) |

The `0.05 / 0.95` floors at the bucket extremes (beliefs.py:65) keep any single
record from driving a posterior to logical certainty.

---

## 5. The per-trial control flow (the core loop)

`_condition_chain_on_outcomes` iterates every chain (arm × subgroup × endpoint
fan-out). For each chain with a known, non-UNKNOWN outcome:

1. Collect the chain's **live backbone edges** — those that exist in the graph
   (`_chain_backbone_edges`, line 193): AFFECTS, MODULATES_VIA,
   MECHANISM_AFFECTS, BIOLOGY_DRIVES, and (when an endpoint is present)
   REFLECTS_BIOLOGY, ENDPOINT_CAPTURES. A `(edge_type, src, tgt, arm_id)` dedup
   set guarantees each edge is conditioned once per arm.
2. Branch on **outcome** and (for failures) the **routed failure reason**.

### 5a. SUCCESS — never routed (line 948)

The whole path operated, so **every** backbone edge gets a support update at the
*full* trial weight:

```
n_eff_i = w_base ,  p_obs = 0.80
```

Plus a **safety-survival credit** (§6b): the trial reached readout without a
halt, so its safety gates "did not fire."

### 5b. EFFICACY / MEASUREMENT failure — routed responsibility (line 969 → `_apply_responsibility_update`, line 1075)

This is the A4 "principled responsibility" update. The reason router maps the
failure mode to a branch (`routing_branch_for`, taxonomy.py:98;
`FAILURE_MODE_BRANCH`):

- `NO_TARGET_ENGAGEMENT`, `TARGET_ENGAGED_BIOLOGY_NOT_MOVED` → **EFFICACY**
- `BIOLOGY_MOVED_ENDPOINT_FLAT`, `WRONG_TIMEFRAME`, `HIGH_PLACEBO_RESPONSE`,
  `WRONG_POPULATION`, `EFFICACY_IN_SUBGROUP_ONLY` → **MEASUREMENT**

Both branches blame *within the must-hold backbone*. Let `r_a = E[p_a]` be each
live edge's current posterior mean and `M = ∏_a r_a` the joint reliability. For
each edge:

```
ρ_a   = (1 − r_a) / (1 − M)        # failure responsibility (β share)
p_obs = (r_a − M) / (1 − M)        # survival credit        (α share)
n_eff = w_base                     # full mass on EACH edge
```

(`_apply_responsibility_update`, lines 1099–1134.) Since `ρ_a + p_obs = 1`,
each edge receives exactly `w_base` total mass — but a **high-reliability edge**
(`r_a → 1`) gets `p_obs → 1` (credited for surviving), while a
**low-reliability edge** absorbs the blame (β grows with `1 − r_a`). A curated
`binds_to` with `α ≫ β` self-protects; the uncertain causal edge takes the hit.
Guard: if `1 − M < 1e-6` (all-reliable chain) the failure is uninformative —
skip (line 1118).

> Note the contrast with §5c: the routed path puts **full `w_base` on every
> edge** (the split is in p_obs); the unknown path **splits `w_base` across
> edges** (the split is in n_eff, all toward contradict). Routed failures are
> therefore higher-information.

### 5c. UNKNOWN failure — explaining-away fallback (line 981 → lines 991–1053)

`MULTIPLE_FACTORS` (or any unmapped mode) falls through to the legacy heuristic
spread. Per edge, unnormalized weak-weight `u_i = 1 − E[p_i]`, normalized
`w_i = u_i / Σ u_j` (uniform `1/L` if all `u_i = 0`):

```
n_eff_i = w_base · w_i ,  p_obs = 0.20  (FAILURE)  or  0.35 (PARTIAL)
```

The failure mass is split toward the currently-uncertain edges, so no single
trial collapses an edge and curated edges absorb ≈0.

### 5d. SAFETY / OPERATIONAL failure — censor (line 955)

```
continue   # ZERO virtual evidence to the efficacy+measurement backbone
```

A safety death never revealed whether the biology would have worked; an
operational stop never tested the chain. Neither a downvote nor an upvote. The
edges are deliberately **not** marked applied, so a *different* arm of the same
trial with a real outcome can still condition a shared edge. (Safety still moves
the AE edges separately — §6.)

### 5e. CT.gov stop-reason override (line 884 → taxonomy.py:148)

The LLM classifier reads results text, not the structured CT.gov
`overallStatus` / `whyStopped`. So an early stop (TERMINATED / WITHDRAWN /
SUSPENDED) for accrual / funding / toxicity can be misrouted to EFFICACY and
wrongly downvote the biology. A deterministic override consults the cached
CT.gov status and reroutes to OPERATIONAL/SAFETY (censor) — **unless** the
why-stopped text contains efficacy/futility language (`"futility"`, `"did not
meet"`, …), which means the stop *is* a real efficacy signal and the classifier
is left alone. No-op when the status cache is absent (preserves
reproducibility).

---

## 6. The safety branch — a different axis

Efficacy/measurement ask "did the chain hold?"; safety asks "did this adverse
event occur?" — updated on AE edges (`causes_ae` off the compound,
`target_associated_ae` off the target), **not** the backbone.

### 6a. AE occurrence

Incidence rates from the extraction move the AE edges' `α` toward "fires" via
the `attribute_adverse_events` path (`src/inference/ae_propagation.py`). The
target-level safety belief is then **pooled across every compound that hits the
target** — which is why safety is the densest, most reliable signal in the
graph.

### 6b. Safety-gate survival credit (line 1055 → `_credit_safety_survival`, line 1177)

A trial that *reached readout* (any successful arm, or an efficacy/measurement
death — it ran and missed) is good news for safety: its safety gates did **not**
halt it. Each existing `causes_ae` edge off a treatment compound gets
`β += w_base` (`p_obs = 0.0`, all mass to "did-not-fire"). Safety deaths (gate
fired → handled by the occurrence path) and operational stops get no survival
credit.

---

## 7. Worked example

A **Phase 3** trial, `confidence = 0.9`, `sample_size = 350`, a valid (non-
operational) test, chain edges currently at `r = [0.87, 0.78, 0.64]`.

```
base_n     = EVIDENCE_TYPE_N_EFF[PHASE3] · quality = 15 · 0.9 = 13.5
f_N        = _precision_multiplier(350) = 1.0     (at the anchor)
gate_weight= 1.0                                  (valid test)
w_base     = 13.5 · 1.0 · 1.0 = 13.5
```

**If SUCCESS** — every edge: `n_eff = 13.5`, `p_obs = 0.80`
→ `α += 10.8`, `β += 2.7` on each. (Plus survival credit on the AE edges.)

**If EFFICACY failure** — `M = 0.87·0.78·0.64 = 0.434`, `1 − M = 0.566`:

| edge | r_a | p_obs = (r_a−M)/(1−M) | α += w·p_obs | β += w·(1−p_obs) |
|---|---:|---:|---:|---:|
| affects | 0.87 | 0.770 | 10.4 | 3.1 |
| modulates_via | 0.78 | 0.611 | 8.2 | 5.3 |
| mechanism_affects | 0.64 | 0.364 | 4.9 | 8.6 |

The reliable `affects` edge is mostly *credited* (it survived); the weak
`mechanism_affects` edge absorbs the *blame*. Each still receives `13.5` total
mass.

---

## 8. Persistence & faithful replay

`update_edge_belief` (`src/graph/store.py`, line 155) does the I/O: read belief
→ `apply_virtual_evidence(n_eff_override, p_obs_override)` → append the
`EvidenceRecord` to the edge's replay log. Crucially it **persists the exact
applied weights**:

```
evidence.applied_n_eff = n_eff
evidence.applied_p_obs = p_obs        (store.py:216)
```

so any re-derivation — node-merge belief replay, the (s,t)-field materializer,
LOO self-exclusion — reconstructs the scalar faithfully via
`beliefs.applied_weights` (line 256) instead of recomputing a *nominal* n_eff
that would ignore the explaining-away / responsibility split (the "Bug-B"
family). This is what makes the honest holdout exact.

---

## 9. Where this meets prediction

The per-trial updates above produce the **scalar marginal** `Beta(α, β)` on each
edge — the conjugate pool of every trial's records. The default predictor
(`predict_clinical_hypothesis` → `PredictionEngine._collect_edges` →
`get_edge_belief`) reads **that scalar** and samples `rng.beta(α, β)`. It
composes two quantities: an **efficacy** softmin over all backbone edges
(efficacy + measurement edges pooled) × a **safety** factor `(1 − penalty)` from
the AE edges. The "measurement" routing branch shapes *which edges learn from a
failure* (this doc, §5b); it is **not** a separate composed branch at prediction
time.

The (s,t) belief **field** keeps the same evidence additionally indexed by
source/target description, so a query can localize instead of using the pooled
marginal — but only the separate `field_prediction.py` path reads it; the
default predictor does not.

---

## File map

| concern | file:symbol |
|---|---|
| conjugate step `α+=n·p`, `β+=n·(1−p)` | `src/inference/beliefs.py::apply_virtual_evidence` |
| n_eff table + `effective_n_for_evidence` | `src/inference/beliefs.py` (EVIDENCE_TYPE_N_EFF) |
| p_obs buckets | `src/inference/beliefs.py` (BUCKET_TO_P_OBS) |
| per-trial orchestration | `src/annotation/attributor.py::Attributor.attribute` |
| outcome → whole-chain conditioning | `…::_condition_chain_on_outcomes` |
| routed responsibility (A4) | `…::_apply_responsibility_update` |
| safety-survival credit (A3) | `…::_credit_safety_survival` |
| per-arm outcome resolution | `…::_aggregate_arm_outcomes` |
| failure-mode → routing branch | `src/annotation/taxonomy.py::FAILURE_MODE_BRANCH` |
| CT.gov stop-reason override | `src/annotation/taxonomy.py::stop_reason_override` |
| gate weight | `src/annotation/taxonomy.py::gate_weight_for` |
| persist applied weights / replay | `src/graph/store.py::update_edge_belief`, `beliefs.applied_weights` |

*Routing (A3/A4) is baked ON (`src/config.py::CONFIG.routing = True`). Some
inline comments still say "EROOM_ROUTING default OFF" — that flag was removed in
the config-bake; the behavior is now always-on. Verify against `src/config.py`
if in doubt.*
