# eroom.bio — orientation map

*Read this first. Plain-language state of the project + glossary. Drop into `docs/ORIENTATION.md` or fold into `CLAUDE.md`.*

## What this is (no jargon)
eroom.bio reads clinical trials and learns which biological bets tend to succeed or fail, so trials can be designed with better odds. Long-term aim: bend the ~90% trial failure rate by accumulating transferable knowledge of human physiology across trials. How: each trial is broken into **one connected causal chain** — **drug → target → mechanism → biology → endpoint → indication → population**, plus **adverse-event** links off the drug and target. That one chain's edges split into **three functional branches** (efficacy, measurement, safety — see glossary): one chain by topology, three branches by function. Each link gets a belief (how reliable it is). Beliefs pool across trials that share the same link. **The system learns by comparing trials that share parts.**

## The one idea that governs everything: reuse
A link is only learnable when several trials share it. **Reuse = how many trials touch the same node/edge.** If every trial is unique, there's nothing to compare and nothing transfers — beliefs revert to a default guess. Low reuse is the project's single binding constraint, confirmed from every angle.

## Glossary (the lexicon, plainly)
- **reuse / effective reuse** — how much evidence informs a node/edge: direct (same node in N trials) + borrowed (from similar nodes). Low = learns nothing.
- **node merging** — deciding two trials mean the same thing (same target/biology) so their evidence pools. Done by a shared id (gene id, pathway id, ontology term).
- **branch / edge class** — the three kinds of link, each a segment of the one chain: **efficacy** (drug→target→mechanism→biology: does a real effect exist), **measurement** (biology→endpoint→indication→population: does the trial detect it), **safety** (the adverse-event side-links: does it harm). They share nodes (biology is the seam; drug/target feed safety) but are scored and updated separately because they fail for different reasons.
- **per-branch / per-target recovery** — how faithfully the system learns the true reliabilities, measured separately per branch and per drug target.
- **routing / censoring (the "A34" work)** — only update the link a failure actually implicates (a safety death must not downvote the biology). Uses the trial's stated failure reason.
- **responsibility** — how much blame a failed trial puts on each link: its own unreliability ÷ total failure probability.
- **manifold / domain geometry** — placing entities in a space where "close" means "behaves similarly": **chemical structure** for compounds, **pathway membership** for targets — so a node borrows evidence from similar neighbors, not only identical ones.
- **alignment** — whether "close in the geometry" really means "shares outcomes." Domain geometry (structure, pathway) **passed**; text geometry (BioLORD embeddings) **failed**.
- **on/off-target decomposition** — for an observed side effect: is it intrinsic to the target (any drug hitting it causes it → on-target) or specific to the chemistry (escapable by changing the molecule → off-target)?
- **Pillar A / B / C** — A: fix the inference (done — routing/censoring). B: fix the representation so reuse can rise (done — biology→ontology, domain manifolds). C: get reuse-dense data (**the remaining unlock**).

## How a new trial updates the beliefs
Each link is a **tally** — a Beta(α, β) counter where α is "held/worked" evidence and β is "failed" evidence; the reliability estimate is α/(α+β), and confidence grows as the totals grow. A new trial nudges the tallies, but **which** links it touches depends on the outcome and (for failures) the stated reason — that routing is what keeps the three branches independent:
- **Success** → every efficacy and measurement link in the chain gets an upvote (the whole chain held); every safety link gets "didn't fire" evidence (the trial reached readout without a halt).
- **Efficacy / measurement failure** → the downvote is *split by responsibility*: each must-hold link's share ∝ its own unreliability, so a well-established link barely moves and the weak link absorbs most of the blame. Safety links still get "survived" credit.
- **Safety failure** → only the implicated adverse-event link moves (toward "fires"); efficacy and measurement are **left untouched** — the trial never revealed whether the biology would have worked.
- **Operational / commercial stop** → nothing biological updates; the science was never tested.

Two asymmetries are the whole point. A **success is clean** evidence (everything held); a **failure is ambiguous** (something broke — but what?), so failures are *routed* to the right branch and *weighted* by responsibility instead of smeared across every link. And **safety updates on a different axis** from efficacy — not "did the chain hold" but "did this adverse event occur": a new trial updates the compound's AE tallies directly, and the target-level safety belief is then pooled from every compound that hits that target. That pooling across compounds is the cross-trial sharing that makes safety the densest, most reliable signal in the graph.


- **Validated:** the inference method is correct; domain geometry is reliability-aligned; the pipeline catches its own data leakage (so the numbers are trustworthy).
- **Ships now:** (1) the per-target **safety signal**; (2) the **on/off-target decomposition** as an attribution over *observed* data (validated on EGFR→rash and INSR→hypoglycemia).
- **Gated on more data:** predicting risk for *novel/unseen* compounds & targets, and binary trial-success prediction — both blocked by low reuse at n≈472, **not** by method.
- **Binding constraint:** reuse-per-edge. ~500 trials across 5 diseases barely overlap → little to compare. Fix = a concentrated, reuse-dense corpus.

## What's next
1. *(optional, last inference squeeze)* **B2/B3** — hierarchical pooling so sparse links borrow from class/context siblings.
2. *(the real unlock)* **Pillar C** — acquire reuse-dense data: many trials hitting the **same targets/pathways**.
3. **Ship** the two validated surfaces (safety + decomposition).

## Key result docs (in repo)
`A34_RESULTS.md`, `SYNTH_REPORT.md`, `MERGE_POOLING_MAP.md`, `B1_DECISION.md`, `B1_BUILD_RESULTS.md`, `SAFETY_MANIFOLD_ALIGNMENT.md`, `SAFETY_MANIFOLD_RESULTS.md`.

## Flags (both default OFF; old behavior preserved)
- `EROOM_ROUTING` — reason-routed failure updates (A34).
- `EROOM_BIO_ONTOLOGY` — biology ids from GO-BP (B1).
- `EROOM_SAFETY_MANIFOLD` — domain-manifold AE borrowing (predictive borrowing disabled pending more data; decomposition + exact-id ship).
