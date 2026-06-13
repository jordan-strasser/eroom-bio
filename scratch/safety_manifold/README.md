# Safety manifold harness

Read-only measurement of cross-node AE borrowing over two domain manifolds
(compound chemical structure, target Reactome/GO pathway co-membership). Tests
whether domain geometry can manufacture the cross-node safety transfer the
BioLORD text field couldn't. Deliverable docs at repo root:
`SAFETY_MANIFOLD_ALIGNMENT.md` (Phases 0–1, the gate) and
`SAFETY_MANIFOLD_RESULTS.md` (Phases 2–4 + decomposition).

## Run
```bash
.venv/bin/python scratch/safety_manifold/phase1_alignment.py     # alignment gate
.venv/bin/python scratch/safety_manifold/phase2_reuse_peredge.py # effective reuse
.venv/bin/python scratch/safety_manifold/phase3_validate.py      # decomposition + known cases
.venv/bin/python scratch/safety_manifold/phase4_novel.py         # honest novel-entity test
```
Default snapshot: `data/exports/multi_500_annotated.json`. Captured output in
`RESULTS_LOG.txt`.

## Modules
- `geometry.py` — loads a snapshot; builds Morgan/ECFP4 fps (RDKit, SMILES from
  `data/cache/chembl_smiles.json`), target pathway sets, and the AE substrate.
- `borrow.py` — `SafetyManifold`: Nadaraya–Watson kernel over each manifold; exact-id
  anchor + kernel-weighted neighbor evidence. Tuned defaults locked in the class.
- `phase3_decompose.py` — `Decomposer`: SOC-rollup, background-corrected on/off-target
  attribution (noisy-OR responsibility).

## Bottom line
Both manifolds pass the alignment gate (geometry predicts AE-sharing,
permutation-significant). The decomposition validates on known on-target class
effects (EGFR→rash, insulin→hypoglycemia). But **honest trial-disjoint novel-entity
liability transfer does not beat the base-rate prior at n=472** for either manifold
(per-SOC AUROC ≤ 0.5; the naive signal is trial co-occurrence leakage). Ship exact-id
+ the decomposition (in-sample attribution); hold predictive borrowing behind a flag,
OFF, until corpus reuse grows.

## Dependencies
`rdkit` (installed from pythonhosted). SMILES resolved once from ChEMBL by `chembl_id`
and cached; no other network needed to re-run.
