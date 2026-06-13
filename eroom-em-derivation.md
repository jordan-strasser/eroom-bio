# eroom.bio — EM/VBEM for the noisy-AND × detection × safety-OR chain model

Implementation-ready derivation. The model factors trial success into three independent events — a true effect **exists**, it gets **detected**, and the trial **survives safety** — then uses EM to attribute each failure to the right latent cause and feed soft counts back to per-edge Beta posteriors. Failure-reason metadata, where present, collapses the latent attribution into a routed (and partially censored) update.

---

## 1. Notation

For trial $t$ with subgraph $C_t$, partition its parameters into three classes:

- **Must-hold reliabilities** $r_a$, $a \in M_t$. These are the conjunctive "good-path" parameters — the efficacy spine `intervention→target→mechanism→biology`, the collapsed **triangle factor** $v$ (one parameter for the biology–endpoint–indication validity), and the **population** edge `indication→population`. Each $r_a = P(\text{link/factor holds})$, with $r_a \sim \mathrm{Beta}(\alpha_a, \beta_a)$.
- **Safety gates** $q_k$, $k \in G_t$ (the AE→intervention, AE→target edges). $q_k = P(\text{this AE pathway triggers a halt})$, with $q_k \sim \mathrm{Beta}(a_k, b_k)$. Note these are **failure** probabilities, so in Beta terms a "fired" event increments $a_k$.
- **Leak** $\ell$: one small global scalar, $\ell = P(\text{no unmodeled factor kills an otherwise-perfect trial})$ — absorbs powering, enrollment, operational, strategic kills. Endpoint/population/safety are NOT in here; they have explicit edges.

Shorthands per trial:
$$\Phi_t = \prod_{a \in M_t} r_a \quad(\text{true detectable effect exists}),\qquad s_t = \prod_{k \in G_t}(1 - q_k)\quad(\text{survives safety}).$$

In all expectations below, plug in current posterior means $\bar r_a = \frac{\alpha_a}{\alpha_a+\beta_a}$, $\bar q_k = \frac{a_k}{a_k+b_k}$. (Using posterior means inside the responsibilities makes this a mean-field VBEM; drop to the mode and it's EM-for-MAP.)

---

## 2. Generative model (the latent-variable form)

Independent latent Bernoullis per trial:
$$f_a \sim \mathrm{Bernoulli}(r_a)\ \ (a\in M_t),\qquad g_k \sim \mathrm{Bernoulli}(q_k)\ \ (k\in G_t),\qquad e \sim \mathrm{Bernoulli}(\ell).$$

Observed success is the noisy-AND with embedded safety-OR:
$$y_t = \Big(\prod_{a\in M_t} f_a\Big)\,\Big(\prod_{k\in G_t}(1-g_k)\Big)\, e.$$

So
$$\pi_t \equiv P(y_t = 1) = \Phi_t \, s_t \, \ell.$$

The $\{f_a, g_k, e\}$ are the EM latents. **A success is fully observed** ($y_t=1 \Rightarrow$ every $f_a=1$, every $g_k=0$, $e=1$) and needs no inference. **A failure is the only place EM works.**

---

## 3. E-step — responsibilities given a failure

### 3.1 The core result (unrouted, no failure-reason metadata)

For any single latent, given $y_t = 0$, the posterior that *it* was a cause is its own bad-event probability normalized by the total failure probability. Derivation for a must-hold $a$: $P(\text{fail}\mid f_a{=}0)=1$ and $P(f_a{=}0)=1-r_a$, so

$$\rho^{\text{fail}}_a \equiv P(f_a = 0 \mid y_t = 0) = \frac{(1-r_a)\cdot 1}{1-\pi_t} = \frac{1 - r_a}{1 - \pi_t}.$$

The identical argument gives, for safety gates and leak:

$$\rho^{\text{kill}}_k = P(g_k = 1 \mid y_t = 0) = \frac{q_k}{1-\pi_t},\qquad \rho^{\text{leak}} = P(e=0\mid y_t=0)=\frac{1-\ell}{1-\pi_t}.$$

> **One formula:** every latent's failure responsibility $= \dfrac{P(\text{its bad event})}{1-\pi_t}$. Reliable links ($r_a\to1$) and safe gates ($q_k\to0$) collect ~no blame; the unreliable/dangerous ones absorb it. These are marginal expectations, not a partition — several latents can co-fail, so they need not sum to 1.

For the **partial success credit** a must-hold still earns on a failed trial (it may have held even though something else broke):
$$P(f_a = 1 \mid y_t = 0) = \frac{r_a - \pi_t}{1 - \pi_t}.$$

### 3.2 Routed E-step (when the failure reason is observed)

This is where most of the information per trial comes from, and where competing-risks **censoring** matters: a trial killed on safety never reveals whether its biology would have worked, so it must contribute **nothing** to the efficacy edges — not a success, not a failure. The reason field tells you which branch failed, which branches *survived* (→ success-style counts), and which are *censored*.

| Observed reason | Safety gates $q_k$ | Must-holds $r_a$ (spine + triangle + pop) | Leak |
|---|---|---|---|
| **Success** ($y=1$) | did-not-fire → $b_k\!+\!=\!1$ | held → $\alpha_a\!+\!=\!1$ | ok |
| **Safety death** | fired: split by $\rho^{\text{kill}}_k$ (below) | **censored — no update** | censored |
| **Efficacy / futility** (ran to readout, missed) | survived → $b_k\!+\!=\!1$ | blame within $M_t$: denom $1-\Phi_t$ | ok |
| **Strategic / business / enrollment** | censored | censored | (optional leak update) |
| **Unknown** (no metadata) | $\rho^{\text{kill}}_k=\frac{q_k}{1-\pi_t}$ | full responsibilities (3.1) | $\frac{1-\ell}{1-\pi_t}$ |

Two routed specifics:

- **Safety death** ⇒ condition on "≥1 gate fired." Exact per-gate responsibility:
$$\rho^{\text{kill}}_k = \frac{q_k}{1 - \prod_{k'\in G_t}(1-q_{k'})} = \frac{q_k}{1 - s_t}.$$
Gate $k$ gets $\rho^{\text{kill}}_k$ toward $a_k$ and $(1-\rho^{\text{kill}}_k)$ toward $b_k$. Efficacy edges get **nothing** (censored).

- **Efficacy / futility death** ⇒ we know all gates survived ($b_k\mathrel{+}=1$ — efficacy failures are *informative about safety in the good direction*) and leak was ok. The miss is inside $M_t$, so blame splits across spine vs. detection automatically, with the **branch-local** denominator:
$$\rho^{\text{fail}}_a = \frac{1-r_a}{1-\Phi_t},\qquad P(f_a{=}1\mid\cdot) = \frac{r_a-\Phi_t}{1-\Phi_t}\quad (a\in M_t).$$
The spine-vs-triangle-vs-population split falls out of the relative $(1-r_a)$ with no extra machinery — this is exactly the "did the biology fail or did the endpoint fail to capture it" decomposition.

---

## 4. M-step — Beta soft-count updates

Reset to priors each iteration, then accumulate expected counts over all trials where the parameter appears **and is not censored**.

**Must-hold reliability** $r_a$:
$$\alpha_a \leftarrow \alpha_a^0 + \!\!\sum_{t\,\ni\, a,\ \text{not censored}}\!\! \Big[\mathbb{1}(y_t{=}1) + \mathbb{1}(y_t{=}0)\,P(f_a{=}1\mid\cdot)\Big]$$
$$\beta_a \leftarrow \beta_a^0 + \!\!\sum_{t\,\ni\, a,\ \text{not censored}}\!\! \mathbb{1}(y_t{=}0)\,\rho^{\text{fail}}_{a,t}$$

**Safety gate** $q_k$:
$$a_k \leftarrow a_k^0 + \sum_{t\,\ni\,k}\big[\mathbb{1}(\text{safety death})\,\rho^{\text{kill}}_{k,t} + \mathbb{1}(\text{unknown fail})\tfrac{q_k}{1-\pi_t}\big]$$
$$b_k \leftarrow b_k^0 + \sum_{t\,\ni\,k}\big[\mathbb{1}(\text{survived: }y_t{=}1\text{ or efficacy death}) + \mathbb{1}(\text{safety death})(1-\rho^{\text{kill}}_{k,t}) + \mathbb{1}(\text{unknown fail})(1-\tfrac{q_k}{1-\pi_t})\big]$$

**Leak** $\ell$: start by **fixing it** (e.g. 0.85–0.95) so it doesn't soak up signal. Only free it once residual failures cluster in trials whose modeled path looks healthy; then update it as the global mean of $\rho^{\text{leak}}$ over uninformative/unknown failures.

The denominator in each responsibility uses the branch-appropriate all-good prob ($1-\pi_t$ unrouted, $1-\Phi_t$ for efficacy deaths, $1-s_t$ for safety deaths), per §3.

---

## 5. The loop

```
init  α_a,β_a,a_k,b_k ← class-specific priors;  ℓ ← 0.9 (fixed)
repeat until Δparams < tol  (or expected complete-data LL plateaus):
    # E-step
    for each trial t:
        r̄ = {α/(α+β)};  q̄ = {a/(a+b)}
        Φ_t = ∏ r̄_a;  s_t = ∏(1-q̄_k);  π_t = Φ_t·s_t·ℓ
        determine branch from failure-reason (or 'unknown')
        compute per-latent responsibilities + censoring mask (§3.2 table)
    # M-step
    reset α,β,a,b ← priors
    accumulate expected counts from all non-censored (trial,param) pairs (§4)
predict:  π̂(new chain) = ∏ r̄_a · ∏(1-q̄_k) · ℓ
          report the factored risk  (Φ̂, D̂=v̄·w̄, ŝ)  alongside π̂
```

Monotonicity: standard EM/VBEM lower-bound guarantees hold provided censoring is missing-at-random within branch (safe assumption if reason coding isn't correlated with edge identity beyond the branch it names).

---

## 6. Prediction & the actually-useful output

At test time, $\hat\pi = \hat\Phi\cdot\hat D\cdot\hat s\cdot\ell$ is the success probability, but the **factored decomposition is the product**: telling a user "this program's dominant risk is *detection* (endpoint won't capture the biology), not the biology itself" or "the target carries a transferable on-target tox liability seen in 4 prior programs" is the decision-support value that binary AUC never captures. Expose $\hat\Phi, \hat D, \hat s$ and the top blamed edges, not just $\hat\pi$.

For **unseen edges**, back off to a class-level (hierarchical) prior rather than a flat global one — partial pooling by edge class, and within class by context (disease/modality) where data allows.

---

## 7. Correctness notes / failure modes of the method itself

- **Censoring is the crux.** The single highest-value correction over the current code is competing-risks routing: safety deaths must not touch efficacy edges, business kills must not touch any edge. Get this wrong and you reintroduce the exact poisoning that flattens the posteriors.
- **Identifiability rides on reuse.** With many must-holds per trial and one outcome bit, individual $r_a$ are weakly identified unless edges recur across trials (the P6 sparsity check) **and** failures are routed (P4). EM will contentedly return flat, uninformative posteriors on sparse data — the math is necessary, not sufficient. Validate on synthetic data with known edge reliabilities first.
- **Triangle as one factor** removes the loop entirely, so the product factorization is exact; no junction-tree/loopy-BP needed.
- **Plug-in responsibilities** (posterior means inside the E-step) make this generalized/variational EM — fine in practice, slightly optimistic uncertainty.
- **Class-specific priors:** efficacy-spine links can be mildly optimistic (clinic-stage = pre-filtered); detection/endpoint priors more skeptical; AE gates low base rate with heavy tails for known-liability target families.

---

## 8. Recommended validation before trusting any real number

Build a synthetic generator with known $r_a, q_k, \ell$ and known shared edges across trials. Confirm EM recovers the planted reliabilities and that routed-failure data recovers them with far fewer trials than unrouted. Only then point it at the corpus. This is the cheapest way to separate "the model is wrong" from "the data can't identify it."
