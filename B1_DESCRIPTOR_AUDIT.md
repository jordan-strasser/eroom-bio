# B1 Phase 0 — BiologyNode descriptor provenance audit

**Read-only measurement. Gates Phase 1's interpretation.** Question: is the 71%
biology-singleton rate partly a *descriptor artifact* — the Tier-3 BioLORD merge
comparing a wrong/truncated/inconsistent string — in which case the cheap fix is
"compare the right string," tested before any ontology work?

All numbers on `data/exports/multi_500_annotated.json` (n=472 in-graph trials,
212 BiologyNodes). Probe: `scratch/diagnostics/b1_descriptor_audit.py`.

---

## The four descriptor consumers, traced to the byte

| # | consumer | file:line | exact string it sees (BiologyNode) |
|---|---|---|---|
| a | **id hash** `bio:<sha1>` | `populate.py:650-651` | `" ".join(description.strip().lower().split())` → sha1[:12] |
| b | **SapBERT `name_id`** (Tier-2) | `node_merge.py:118-120`, used `:261-269` | the same phrase, lowercased + punctuation→space (`anti-inflammatory`→`anti inflammatory`) |
| c | **BioLORD Tier-3** (cosine ≥ 0.85) | `node_merge.py:134` (`_node_text`), `:293-299`; `MergeConfig.biolord_threshold` `:80` | `node["description"] or node["name"]`, `.strip()` (original case) |
| d | **(s,t) field** | `field_prediction.build_st_desc_map:82`, `:116-119` | per-chain `biology_description` (pre-merge trial phrasing) |

The BioLORD cache (`biolord_embeddings.py:110-114`) keys on `lower()+collapse-ws`,
so Tier-3's case difference vs the id hash is **normalized away before cosine**.

## Finding 1 — the four consumers see the SAME string (no wrong-string artifact)

Measured over all 212 BiologyNodes:

| identity check | result |
|---|---:|
| `id == sha1(norm(node.description))` | **212/212** |
| `id == sha1(norm(node.name))` | **212/212** |
| `norm(description) == norm(name)` | **212/212** |

`name == description` for every biology node, and the id is the hash of exactly
that string. So **(a) id-hash, (b) name_id, and (c) Tier-3 `_node_text` are all the
same phrase** (modulo case/punctuation that the cache normalizer erases). The field
(d) reads the *pre-merge per-chain* phrasing: identical to the node for 151/212
nodes; a chain is longer than the node for 22; the node is longer for 35 — but it
is the same register of short phrase, not a richer claim.

> **Verdict on the artifact hypothesis: REFUTED.** Tier-3 is comparing the correct,
> consistent biological claim — there is no truncated/templated stub standing in for
> a richer string, and no inconsistent string across the merge tiers. The cheap fix
> the task hypothesized ("compare the right string / re-normalize descriptors") is
> **ruled out**: it would change nothing, because all consumers already compare the
> same normalized phrase. The 71% singleton rate is a *real semantic* fact, not a
> normalization bug — so Phase 1 (is it collapsible?) is the right next gate.

## Finding 2 — the descriptor is impoverished, not rich

The string every consumer compares is a **4–5 word functional-outcome phrase**, not
a rich mechanistic claim:

| string | median | mean | p90 | max |
|---|---:|---:|---:|---:|
| node.description (words) | 5 | 5.0 | 8 | 12 |
| node.description (chars) | 47 | 47 | 67 | 97 |
| per-chain biology_description (words) | 4 | 4.5 | 7 | 12 |

Representative: `blood pressure reduction`, `mitotic arrest`, `protein synthesis
disruption`, `renal inflammation reduction`, `LDL-C reduction and atherosclerosis
inhibition`, `glycemic control via incretin enhancement`. These are *drug-effect /
physiological-outcome* phrases, frequently compound (two concepts joined by "and"),
not single curated biological processes. This is the substantive reason paraphrases
fork into separate hashes — `renal vasodilation and natriuresis` vs `renal perfusion
improvement and natriuresis` are one biology, two hashes — and it is exactly what an
ontology id (a single canonical process term) is supposed to fix.

## Finding 3 — format/session drift is mild and is NOT a confounder

- **No version/created-at/schema metadata** on biology nodes (only `source` =
  `trial_biology_description` and `merged_from`), so per-session grouping isn't
  recoverable — but the format is homogeneous enough that it doesn't need to be.
- **No markdown / multi-sentence structure** in biology descriptions (the `#`/`**`
  cache keys belong to *other* node types — indications). 0% of biology descriptions
  carry markdown.
- Only drift is **case**: 84% of node descriptions all-lowercase; 16% of per-chain
  strings start uppercase (`HER2-...`). This is erased by the cache normalizer
  (id, name_id, BioLORD all see lowercase), so it does not fork identities.

## Finding 4 (flag for Phase 3) — the existing merge already pools distinct chains

For 22/212 nodes, the post-merge node pools per-chain descriptions that are *not*
paraphrases of the winner — e.g. node `DNA-damage-induced apoptosis` carries chains
`suppression of lymphocyte proliferation` and `tumor vascularization suppression`;
node `immune suppression` pools `immune system reset and tolerance restoration`.
This is a **pre-existing context-collapse signal** (Tier-2/Tier-3 already over-pool
some distinct biology). It sets a baseline the Phase-3 context-collapse guard must
beat, not regress.

---

## What Phase 0 hands to Phase 1

1. The singleton rate is **not** rescuable by descriptor normalization — the merge
   already compares the right, consistent string. So the only two live levers remain
   the two the task names: a **looser similarity bar** (measured in
   `b1_phase1_cosine.py`) or a **controlled-vocabulary id** (measured in
   `b1_phase1_ontology.py`).
2. Because the descriptor is a short *functional-outcome* phrase (often compound and
   direction-laden, e.g. "...inhibition"), raw-description cosine will conflate
   direction and concept; a process-identity ontology term (which drops direction to
   metadata) is the representation most likely to collapse true paraphrases without
   merging opposite-direction biology. Phase 1 measures whether it actually does.
