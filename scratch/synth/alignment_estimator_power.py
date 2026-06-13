"""Control for Step 3: can the geometry_alignment estimator even SEE alignment
at the corpus reuse rate?

Step 3 measured real BioLORD alignment ≈ 0.05 via a kNN reliability-prediction
corr on edges with ≥2 host trials. But at reuse ~2 the edge reliabilities are
themselves near-prior noise (SYNTH: individual reliabilities unrecoverable below
reuse ~8), which would ATTENUATE any geometry↔reliability correlation regardless
of the true alignment. This control plants a KNOWN alignment, recovers the
reliabilities at a given reuse, and runs the SAME estimator on the recovered
(noisy) reliabilities — so we can see what the estimator reads back vs truth.

If at reuse 1.24 the estimator reads ~0 even for planted alignment 0.9, then
Step 3's real ≈0.05 is uninformative (measurement-floored), and the honest
statement is "alignment is unmeasurable at this substrate," not "the geometry is
misaligned." Either way the actionable conclusion is the same — embedding-based
sharing has no exploitable signal at reuse 1.24 — but this separates the cause.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate as G  # noqa: E402
import recover as R  # noqa: E402
import geometry as GEO  # noqa: E402
from generate import Corpus  # noqa: E402


def _knn_align(emb: np.ndarray, rel: np.ndarray, k: int = 5) -> float:
    """LOO kNN reliability-prediction corr — the Step-3 estimator."""
    n = len(rel)
    if n < k + 2 or rel.std() < 1e-9:
        return float("nan")
    E = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9, None)
    cos = E @ E.T
    np.fill_diagonal(cos, -1.0)
    preds = np.empty(n)
    for i in range(n):
        nn = np.argsort(-cos[i])[:k]
        preds[i] = rel[nn].mean()
    return float(np.corrcoef(preds, rel)[0, 1]) if preds.std() > 1e-9 else float("nan")


def run(planted_aligns, reuses, seeds=8, n_trials=500):
    print(f"{'reuse':>7}{'planted':>9}{'est(TRUE r)':>13}{'est(recovered r)':>18}"
          f"{'#obs nodes':>12}")
    out = {}
    for reuse in reuses:
        for pa in planted_aligns:
            est_true, est_rec, nobs = [], [], []
            for seed in range(seeds):
                geo = GEO.make_geo_ground_truth(
                    seed, pa, n_trials=n_trials, reuse_rate=reuse, reason_noise=0.0)
                gt = geo.gt
                train = Corpus(gt=gt, trials=G.sample_trials(
                    gt, n_trials, seed=seed + 10_000))
                res = R.em_recover(train, "b")
                rbar = res["alpha"] / (res["alpha"] + res["beta"])
                obs = train.must_incidence() > 0
                # mirror Step 3: only edges with ≥2 host trials are "estimable"
                obs2 = train.must_incidence() >= 2
                if obs2.sum() >= 8:
                    est_true.append(_knn_align(geo.T[obs2], gt.r_must[obs2]))
                    est_rec.append(_knn_align(geo.T[obs2], rbar[obs2]))
                    nobs.append(int(obs2.sum()))
            mt = np.nanmean(est_true) if est_true else float("nan")
            mr = np.nanmean(est_rec) if est_rec else float("nan")
            mn = float(np.mean(nobs)) if nobs else 0.0
            out[(reuse, pa)] = (mt, mr, mn)
            print(f"{reuse:>7}{pa:>9}{mt:>13.3f}{mr:>18.3f}{mn:>12.0f}")
    return out


if __name__ == "__main__":
    print("=== Step-3 estimator power control (kNN reliability-prediction corr) ===")
    print("est(TRUE r): estimator on planted reliabilities (its ceiling at this n)")
    print("est(recovered r): estimator on EM-recovered reliabilities (what Step 3 sees)\n")
    run(planted_aligns=[0.0, 0.5, 0.9], reuses=[1.24, 2.0, 8.0, 64.0], seeds=8)
