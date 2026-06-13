# Branch note — safety manifold

**Branch:** `arch/triangulation-edge-weights`  ·  **Status:** measured, not merged

## What we tried
The graph already learns a real on-target safety signal by exact identity: when
several drugs that hit the *same target* report the same kind of toxicity, the graph
records a stable "this target carries this liability" belief (the most consistent
cross-trial signal we have). The open question: can we go further and let a drug
*borrow* a safety expectation from its **neighbors** — other drugs with a similar
chemical structure, or other targets in the same biological pathway — so that (a) the
safety signal gets stronger, (b) a brand-new drug or target inherits a calibrated
liability before it has any trials of its own, and (c) we can tell whether a side
effect comes from the *target* (mechanism-intrinsic, unavoidable) or from the
*chemistry* (scaffold-specific, potentially engineered out).

We built two "manifolds" (similarity spaces): chemical structure (Morgan/ECFP4
fingerprints from ChEMBL SMILES) and target pathway co-membership (Reactome/GO, already
in the graph).

## What actually happened (honest result)
1. **The geometry is real.** In both spaces, drugs/targets that are *near each other*
   genuinely share more adverse events than distant ones — and this is statistically
   significant (not chance). This is the precondition the earlier text-embedding field
   failed. So the idea is not crazy: the right kind of similarity does track toxicity.

2. **The on-target vs off-target decomposition works** — and it's the genuinely useful
   output. Given a drug's *observed* side effects, it correctly labels each one. It
   nails textbook cases: EGFR-inhibitor rash routes "on-target, shared across
   structurally different drugs that hit EGFR"; insulin's hypoglycemia routes
   "on-target." It also separates a drug's ubiquitous chemo toxicity (baseline) from its
   compound-specific quirks (idiosyncratic). This is a per-program liability profile a
   user could act on.

3. **But the headline promise — a brand-new drug inheriting a useful safety prediction
   from neighbors — does not hold up at our current data size.** When we test it
   honestly (hide a drug completely, predict its side effects from neighbors only, and
   forbid neighbors that shared a trial with it), the prediction is no better than just
   guessing the population average. The encouraging numbers we first saw turned out to
   be two artifacts: a base-rate illusion (common side effects are common) and trial
   leakage (combo-arm trials assign the same side effects to several drugs at once).
   Strip those out and the borrowing adds nothing yet.

## Why, and what would change it
Same wall the rest of the project keeps hitting: there simply aren't enough trials per
shared structure/pathway for neighbor-borrowing to transfer. It's a data-density
problem, not a broken idea — the similarity spaces are correctly aligned, there just
aren't enough trial-independent neighbors per drug at ~470 trials. More trials
concentrated on shared targets/scaffolds (a safety-enriched data pull) would let the
same harness re-run and likely flip the result.

## Decision
- **Ship:** the existing exact-identity on-target safety signal (untouched, still
  works), plus the on/off-target **decomposition** as an interpretation of a program's
  *observed* side effects.
- **Do not enable:** the predictive neighbor-borrowing. It stays behind the
  `EROOM_SAFETY_MANIFOLD` switch (off), with the honest measurement on record, to be
  re-tested when the corpus grows. No risk to anything currently shipping.

Full detail: `SAFETY_MANIFOLD_ALIGNMENT.md` (the go/no-go gate) and
`SAFETY_MANIFOLD_RESULTS.md` (the scoreboard + examples). Re-runnable harness in
`scratch/safety_manifold/`.
