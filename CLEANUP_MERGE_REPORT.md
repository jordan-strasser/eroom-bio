# Cleanup + merge — Phase-1 gate STOP report

**Status: STOPPED at the Phase-1 safety gate. Nothing was changed.** No reset, no
recommit, no merge, no `.gitignore` edit, no push. The repo is exactly as found
(working tree: only untracked `MERGE_PLAN.md`, `CLEANUP_MERGE_REPORT.md`,
`docs/ORIENTATION.md`; HEAD still `c094f96`; all `origin/*` untouched).

**Why stopped:** the rules say *"If any large blob is found in already-pushed history
(at or below 1773c0f): STOP and report… do not do it in this pass,"* and the Phase-1
gate says *"if any large blob is in pushed history → STOP, report it, recommend a
separate filter-repo decision."* The blob census found exactly one such blob.

---

## Phase 1 — Measurements (all assumptions re-confirmed)

### Refs (match MERGE_PLAN.md — no drift)

| ref | sha |
|---|---|
| main | `240264d` |
| arch/triangulation-edge-weights (local) | `c094f96` |
| **origin/arch/triangulation-edge-weights** | **`1773c0f`** ✓ |
| fix/st-field-faithfulness | `44d16de` |
| fix/endpoint-deorphan-punchlist | `5b1ad14` |

### Ancestry (all confirmed)

`main ⊂ fix/endpoint-deorphan ⊂ fix/st-field ⊂ arch` — and `main ⊂ arch`
(fast-forwardable). Linear stack, as MERGE_PLAN.md described.

### Un-pushed local commits (the only ones eligible for rewrite)

```
c094f96  feat(safety): domain-manifold AE borrowing + on/off-target decomposition …
2eddfcc  feat(inference): A3/A4 reason-routed EM + B1 substrate measurement
```
These two (range `1773c0f..arch`) are the only commits not on origin. Confirmed.

### Blob census — every blob >50 MB in history, classified

| size | classification | blob | path |
|---:|---|---|---|
| 543 MB | un-pushed (arc D) | `d1ef2d7f` | data/exports/onco_scale_500_enr_premerge.json |
| 526 MB | un-pushed (arc D) | `61964d11` | data/exports/multi_500_premerge.json |
| 335 MB | un-pushed (arc D) | `8233fee4` | data/exports/onco_scale_500_premerge.json |
| 285 MB | un-pushed (arc D) | `abe9fe09` | data/exports/phasec_n250_premerge.json |
| 234 MB | un-pushed (arc D) | `52f83141` | data/exports/onco_scale_252_premerge.json |
| 124 MB | un-pushed (arc D) | `382ddc4c` | data/exports/neff100_premerge.json |
| 123 MB | un-pushed (arc D) | `db5cea96` | data/exports/phasec_n100_premerge.json |
| 64 MB | un-pushed (arc D) | `491d64f2` | data/exports/onco_scale_500_enr_annotated.json |
| **61 MB** | **PUSHED (≤1773c0f)** | **`f6177ae4`** | **data/exports/onco_scale_500_annotated.json** |
| 54 MB | un-pushed (arc D) | `a7f48685` | data/exports/phaseb_n50b_premerge.json |
| 54 MB | un-pushed (arc D) | `36f983cc` | data/exports/phaseb_n50_premerge.json |
| 52 MB | un-pushed (arc D) | `9e163560` | data/exports/rebuild_n50_p4_premerge.json |
| 52 MB | un-pushed (arc D) | `3528b5b6` | data/exports/rebuild_n50_premerge.json |
| 52 MB | un-pushed (arc D) | `e78740a8` | data/exports/rebuild_n50_p5_premerge.json |

- **The strip target (~2.5 GB across 13 blobs) is entirely un-pushed** — confirmed only
  in `1773c0f..arch` (the two local commits). The named blobs (`multi_500_premerge.json`
  526 MB, `onco_scale_500_enr_premerge.json` 543 MB, all `*premerge*.json`) are all
  un-pushed. **This part of the gate passes.**
- **One large blob is in PUSHED history:** `data/exports/onco_scale_500_annotated.json`
  (61 MB, `f6177ae4`). Exact-match reachability: present in `1773c0f` (1×), `origin/main`
  (1×), **absent** from the un-pushed range (0×). Introduced by commit
  **`0e2d52a "data: publish the n=500 public graph snapshot"`**, which is an ancestor of
  `main`, `1773c0f`, and every active branch.

---

## The gate analysis (and why this one is not "bloat")

`onco_scale_500_annotated.json` is **not** part of arc D's `add -A` sweep. It is a
**deliberately published public artifact** — the commit message is literally "publish
the n=500 public graph snapshot," and under the owner's **confirmed** boundary thesis
("data is all public-source, so the public-data belief-state stays public") this
snapshot is *supposed* to be public and tracked. It is the flagship open-core
belief-state snapshot, not accidental bloat.

So this is a different situation than the gate was written to catch. The gate was
written to catch the case where *the ~2 GB I'm about to strip* turns out to be pushed
(which would force filter-repo). That did **not** happen — the 2.5 GB strip target is
all un-pushed and safe. What surfaced instead is a separate, smaller, **intentional**
publication sitting in pushed history.

### Two facts that matter for the human decision

1. **My planned cleanup never touches this blob.** A soft-reset to `1773c0f` + clean
   recommit rewrites only the two un-pushed commits; everything ≤`1773c0f` (including
   `onco_scale_500_annotated.json`) stays byte-for-byte. So the un-pushed cleanup is
   safe *with respect to pushed history* regardless of this blob.
2. **Removing it would contradict the confirmed decision.** Filter-repo-ing it would
   delete a deliberately-published public snapshot from history and require a
   force-push to `main` + every branch — undoing the "publish the n=500 public
   snapshot" act the owner chose. **Recommendation: do NOT filter-repo it.** It is the
   product, published on purpose.

---

## What I did NOT do (honoring the gate)

- Did **not** soft-reset, recommit, or rewrite `2eddfcc`/`c094f96`.
- Did **not** edit `.gitignore`.
- Did **not** fast-forward `main`.
- Did **not** push or force-push anything; `origin/*` untouched.

All Phase-1 operations were read-only git inspection.

---

## Decision needed from the human

The gate stopped me. Two independent calls:

1. **The pushed 61 MB `onco_scale_500_annotated.json`** — recommended action: **leave it**
   (intentional public publication, consistent with the confirmed thesis). If you
   instead want it out of history, that is a *separate* `git filter-repo` + force-push
   pass across `main` and all branches — not this pass.

2. **The un-pushed 2.5 GB cleanup (Phases 2–7)** — recommended action: **proceed.** It
   is safe (touches only the two un-pushed commits, never pushed history), and it is
   the actual goal. I paused only because the literal gate said STOP on *any* pushed
   large blob. On your go-ahead I will execute exactly:
   - **Phase 2 — `.gitignore`** add: `data/exports/*premerge*.json`,
     `data/exports/*_annotated.json` over a size threshold (proposed: ignore the
     listed oversized exports; keep small public snapshots tracked — to be stated
     precisely), `data/dev/`, `data/cache/` (already ignored), `data/viz/`. Keep
     `data/corpora/` id lists and result `*.md` docs tracked.
   - **Phase 3** — `git reset --soft 1773c0f`; verify the premerge blobs are now
     untracked + gitignored; selective (never `add -A`) clean commits:
     `feat(inference): A34 routing/censoring (EROOM_ROUTING)`,
     `feat(graph): B1 biology→GO-BP ids (EROOM_BIO_ONTOLOGY)`,
     `feat(safety): domain-manifold borrowing + decomposition (EROOM_SAFETY_MANIFOLD)`,
     `chore: result docs + .gitignore + branch-notes`.
   - **Phase 4** — leave `archive/round-4-sub-chains` untouched.
   - **Phase 5** — `git checkout main && git merge --ff-only arch` (local only).
   - **Phase 6** — verify: no tracked blob >50 MB except the intentional
     `onco_scale_500_annotated.json` (or, if you want, I can also untrack it going
     forward via `git rm --cached` **without** rewriting history — it stays in old
     history but leaves the live tree); flags default-OFF byte-identical; full test
     suite (~1394); `check_public_snapshots.py`; clean status.
   - **Phase 7** — STOP, print push commands, do not push.

**Tell me:** proceed with the un-pushed cleanup (Phases 2–7) as above — yes/no — and
whether to leave the 61 MB published snapshot as-is (recommended) or also `git rm
--cached` it (live-tree only, no history rewrite).
