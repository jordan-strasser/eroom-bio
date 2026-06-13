"""B1 Step 2 — identity vs similarity arbiter on PLANTED biology geometry.

Net-new under scratch/synth/. Layers a geometry plant + two sharing
representations onto the §4-§5 routed EM (recover.em_recover), so we can ask the
B1 question on KNOWN truth: at the corpus reuse rate, does similarity-sharing
(the kernel-smoothed (s,t) field) clear the recovery bar that identity-sharing
(discrete ontology canonicalization) clears — and at what geometry_alignment does
each stop helping (where smoothing/merging becomes context collapse)?

THE PLANT (extends generate.make_ground_truth).
  - K latent biology classes. Each must-hold edge a is assigned a class; its
    BIOLOGY-side embedding T_a and MECHANISM-side embedding S_a are unit vectors
    near the class centroid (controllable within-class `scatter`).
  - geometry_alignment ∈ [0,1] = how well embedding neighborhood predicts
    reliability. r_a is the Beta(6,2)-quantile of Φ(z_a) with
    z_a = align·z_class[c(a)] + √(1−align²)·ε_a. At align=1 every edge in a class
    shares one reliability (proximity ⇒ identical r); at align=0 reliability is
    independent of the embedding (proximity uninformative). The Beta(6,2) marginal
    is held fixed across alignment, so only the geometry↔reliability COUPLING
    moves. Safety gates carry NO geometry (the real safety layer keys on target
    identity, already merged — Step-1/Step-3 note).

REPRESENTATIONS (routing = mode "b" for all; "Routing stays on" per the spec):
  - none              : per-node routed EM (the current substrate, reuse≈1.24).
  - discrete-canonical: KMeans-cluster the biology embeddings into nodes, pool
                        each cluster's trials, run the routed EM on merged nodes
                        (IDENTITY sharing — the Reactome/mechanism pattern).
  - field-perdim      : per-node routed EM, then cross-node kernel smoothing with
                        the REAL field's per-endpoint additive factorization
                        K = exp((cos S + cos T − 2)/bw)  (SIMILARITY sharing).
  - field-fullkernel  : same, but a JOINT cosine over the concatenated [S;T]
                        embedding K = exp((cos[S;T] − 1)/bw) — tests whether the
                        per-dim/per-endpoint independence assumption costs recovery.

Reports the reuse × alignment recovery surface (composite corr(π̂,π_true), the
SYNTH_REPORT's sharpest signal) + effective-reuse distributions + α*.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import beta as beta_dist, norm as norm_dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate as G  # noqa: E402
import recover as R  # noqa: E402
from generate import GroundTruth, Corpus, Trial  # noqa: E402

_EPS = 1e-9


# ---------------------------------------------------------------------------
# geometry plant
# ---------------------------------------------------------------------------
def _unit(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(n, _EPS, None)


class GeoGT:
    """A GroundTruth with planted per-must-hold geometry (S, T embeddings + class)."""

    def __init__(self, gt: GroundTruth, S: np.ndarray, T: np.ndarray,
                 cls: np.ndarray, alignment: float, n_classes: int):
        self.gt = gt
        self.S = S            # (n_must, d) mechanism-side embedding
        self.T = T            # (n_must, d) biology-side embedding
        self.cls = cls        # (n_must,) true latent class
        self.alignment = alignment
        self.n_classes = n_classes


def make_geo_ground_truth(seed: int, alignment: float, *, n_classes: int = 10,
                          emb_dim: int = 12, scatter: float = 0.35,
                          plant_a: float = 6.0, plant_b: float = 2.0,
                          **kw) -> GeoGT:
    """Plant pools/gates (via generate) then OVERRIDE must-hold reliabilities with
    the geometry model and attach (S, T) embeddings."""
    gt = G.make_ground_truth(seed, **kw)
    rng = np.random.default_rng(seed + 777)
    n, d, K = gt.n_must, emb_dim, n_classes

    Tc = _unit(rng.normal(size=(K, d)))
    Sc = _unit(rng.normal(size=(K, d)))
    cls = rng.integers(K, size=n)
    T = _unit(Tc[cls] + scatter * rng.normal(size=(n, d)))
    S = _unit(Sc[cls] + scatter * rng.normal(size=(n, d)))

    z_class = rng.normal(size=K)
    eps = rng.normal(size=n)
    a = float(np.clip(alignment, 0.0, 1.0))
    z = a * z_class[cls] + np.sqrt(max(1.0 - a * a, 0.0)) * eps
    # Beta(plant_a, plant_b) marginal, ordered by z (probability-integral transform)
    u = norm_dist.cdf(z)
    r = beta_dist.ppf(np.clip(u, 1e-6, 1 - 1e-6), plant_a, plant_b)
    gt.r_must = np.clip(r, 0.02, 0.98)
    return GeoGT(gt, S, T, cls, a, K)


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------
def _cos_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return _unit(A) @ _unit(B).T


def field_kernel(geo: GeoGT, kind: str, bandwidth: float) -> np.ndarray:
    """(n_must, n_must) cross-node kernel weight matrix.

    perdim : exp((cos(S_a,S_b) + cos(T_a,T_b) − 2)/bw)  — the real field's
             per-endpoint additive factorization.
    full   : exp((cos([S_a;T_a],[S_b;T_b]) − 1)/bw)     — joint cosine.
    """
    if kind == "perdim":
        cs = _cos_matrix(geo.S, geo.S)
        ct = _cos_matrix(geo.T, geo.T)
        return np.exp((cs + ct - 2.0) / bandwidth)
    if kind == "full":
        cat = np.concatenate([geo.S, geo.T], axis=1)
        cc = _cos_matrix(cat, cat)
        return np.exp((cc - 1.0) / bandwidth)
    raise ValueError(f"unknown kernel {kind!r}")


# ---------------------------------------------------------------------------
# recovery representations  (routed EM, mode "b", on all)
# ---------------------------------------------------------------------------
def _predict_scores(gt: GroundTruth, rbar_eff: np.ndarray, qbar_eff: np.ndarray,
                    leak: float, test_trials):
    scores = np.empty(len(test_trials))
    ys = np.empty(len(test_trials), dtype=int)
    for i, t in enumerate(test_trials):
        scores[i] = (np.prod(rbar_eff[t.must_idx]) *
                     np.prod(1.0 - qbar_eff[t.gate_idx]) * leak)
        ys[i] = t.y
    return ys, scores


def _class_backoff(gt: GroundTruth, rbar: np.ndarray, obs_m: np.ndarray) -> np.ndarray:
    """Back unobserved edges off to the recovered class mean (mirrors recover.predict)."""
    out = rbar.copy()
    for cls in ("spine", "triangle", "pop"):
        m = gt.must_class == cls
        seen = m & obs_m
        bo = rbar[seen].mean() if seen.any() else R.PRIOR_MUST_MEAN
        out[m & ~obs_m] = bo
    return out


def recover_none(corpus: Corpus, geo: GeoGT):
    """Per-node routed EM, class-mean backoff for unseen (the current substrate)."""
    res = R.em_recover(corpus, "b")
    rbar, qbar = R.recovered_means(res)
    inc_m, inc_g = corpus.must_incidence(), corpus.gate_incidence()
    obs_m, obs_g = inc_m > 0, inc_g > 0
    rbar_eff = _class_backoff(corpus.gt, rbar, obs_m)
    qbar_eff = qbar.copy()
    if (~obs_g).any():
        qbar_eff[~obs_g] = qbar[obs_g].mean() if obs_g.any() else R.PRIOR_GATE_MEAN
    eff_reuse = inc_m.astype(float)  # discrete trials/node
    return dict(rbar=rbar, rbar_eff=rbar_eff, qbar_eff=qbar_eff,
                leak=res["leak"], eff_reuse=eff_reuse, obs_m=obs_m, obs_g=obs_g)


def recover_discrete_canonical(corpus: Corpus, geo: GeoGT, n_clusters: int):
    """KMeans-cluster biology embeddings -> merge nodes -> pool trials -> routed EM."""
    from sklearn.cluster import KMeans
    gt = corpus.gt
    km = KMeans(n_clusters=n_clusters, n_init=4, random_state=0)
    labels = km.fit_predict(geo.T)  # cluster on the biology-side embedding
    # Build a merged corpus: each trial's must_idx -> cluster ids (dedup within trial).
    mgt = GroundTruth(
        r_must=np.zeros(n_clusters), must_class=np.array(["spine"] * n_clusters),
        n_spine=n_clusters, n_tri=0, n_pop=0, q_gate=gt.q_gate, leak=gt.leak,
        cfg=dict(gt.cfg),
    )
    mtrials = []
    for t in corpus.trials:
        mi = np.unique(labels[t.must_idx])
        mtrials.append(Trial(must_idx=mi, gate_idx=t.gate_idx, y=t.y,
                             reason=t.reason, reason_obs=t.reason_obs))
    mcorpus = Corpus(gt=mgt, trials=mtrials)
    res = R.em_recover(mcorpus, "b")
    rbar_c, qbar = R.recovered_means(res)
    inc_c = mcorpus.must_incidence()
    obs_c = inc_c > 0
    # cluster-mean backoff for unobserved clusters
    bo = rbar_c[obs_c].mean() if obs_c.any() else R.PRIOR_MUST_MEAN
    rbar_c_eff = rbar_c.copy()
    rbar_c_eff[~obs_c] = bo
    # map clusters back onto the per-node index space
    rbar_eff = rbar_c_eff[labels]
    inc_g = corpus.gate_incidence()
    obs_g = inc_g > 0
    qbar_eff = qbar.copy()
    if (~obs_g).any():
        qbar_eff[~obs_g] = qbar[obs_g].mean() if obs_g.any() else R.PRIOR_GATE_MEAN
    # effective reuse of a node = #trials in its cluster
    eff_reuse = inc_c[labels].astype(float)
    return dict(rbar_eff=rbar_eff, qbar_eff=qbar_eff, leak=res["leak"],
                eff_reuse=eff_reuse, obs_g=obs_g, labels=labels)


def recover_field(corpus: Corpus, geo: GeoGT, kind: str, bandwidth: float):
    """Per-node routed EM, then cross-node kernel-smoothed posteriors (similarity)."""
    res = R.em_recover(corpus, "b")
    gt = corpus.gt
    inc_m, inc_g = corpus.must_incidence(), corpus.gate_incidence()
    obs_m, obs_g = inc_m > 0, inc_g > 0
    alpha, beta = res["alpha"], res["beta"]
    # own soft-count mass beyond the prior (so the kernel borrows EVIDENCE, not prior)
    s = R.PRIOR_MUST_MEAN  # prior mean
    ps = 2.0               # prior strength to anchor far-from-neighbor queries
    own_a = np.clip(alpha - R.PRIOR_MUST_MEAN * 2.0, 0.0, None)
    own_b = np.clip(beta - (1.0 - R.PRIOR_MUST_MEAN) * 2.0, 0.0, None)
    own_a[~obs_m] = 0.0
    own_b[~obs_m] = 0.0
    W = field_kernel(geo, kind, bandwidth)
    np.fill_diagonal(W, 1.0)
    # borrow only from OBSERVED neighbors
    Wm = W[:, obs_m]
    a_sm = ps * s + Wm @ own_a[obs_m]
    b_sm = ps * (1.0 - s) + Wm @ own_b[obs_m]
    rbar_eff = a_sm / (a_sm + b_sm)
    qbar = res["a"] / (res["a"] + res["b"])
    qbar_eff = qbar.copy()
    if (~obs_g).any():
        qbar_eff[~obs_g] = qbar[obs_g].mean() if obs_g.any() else R.PRIOR_GATE_MEAN
    # effective reuse = kernel-weighted count of neighbor trials (incl self)
    eff_reuse = W @ inc_m.astype(float)
    return dict(rbar_eff=rbar_eff, qbar_eff=qbar_eff, leak=res["leak"],
                eff_reuse=eff_reuse, obs_m=obs_m, obs_g=obs_g)


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def _score(geo: GeoGT, rep: dict, test_trials):
    gt = geo.gt
    ys, scores = _predict_scores(gt, rep["rbar_eff"], rep["qbar_eff"],
                                 rep["leak"], test_trials)
    au = R.auroc(ys, scores)
    pi_true = np.array([np.prod(gt.r_must[t.must_idx]) *
                        np.prod(1.0 - gt.q_gate[t.gate_idx]) * gt.leak
                        for t in test_trials])
    pi_corr = (float(np.corrcoef(scores, pi_true)[0, 1])
               if np.std(scores) > 1e-9 else 0.0)
    er = rep["eff_reuse"]
    return dict(
        auroc=au, pi_corr=pi_corr,
        eff_reuse_median=float(np.median(er)),
        eff_reuse_p90=float(np.percentile(er, 90)),
        eff_reuse_ge8=float(np.mean(er >= 8.0)),
        eff_reuse_mean=float(np.mean(er)),
    )


def run_geo_cell(seed, reuse, alignment, n_trials, n_test, n_classes,
                 scatter, bandwidth):
    geo = make_geo_ground_truth(
        seed, alignment, n_classes=n_classes, scatter=scatter,
        n_trials=n_trials, reuse_rate=reuse, reason_noise=0.0,
    )
    gt = geo.gt
    train = Corpus(gt=gt, trials=G.sample_trials(gt, n_trials, seed=seed + 10_000))
    test = G.sample_trials(gt, n_test, seed=seed + 20_000)
    reps = {
        "none": recover_none(train, geo),
        "discrete": recover_discrete_canonical(train, geo, n_clusters=n_classes),
        "field_perdim": recover_field(train, geo, "perdim", bandwidth),
        "field_full": recover_field(train, geo, "full", bandwidth),
    }
    out = {name: _score(geo, rep, test) for name, rep in reps.items()}
    out["_oracle"] = R.oracle_auroc(gt, test)
    return out


def aggregate(cells, names, metric):
    agg = {}
    for nm in names:
        vals = np.array([c[nm][metric] for c in cells
                         if not np.isnan(c[nm][metric])])
        agg[nm] = (float(vals.mean()) if vals.size else float("nan"),
                   float(vals.std()) if vals.size else float("nan"))
    return agg


REP_NAMES = ["none", "discrete", "field_perdim", "field_full"]


def main():
    ap = argparse.ArgumentParser(description="B1 geometry arbiter (identity vs similarity).")
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--n-trials", type=int, default=500)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--n-classes", type=int, default=10)
    ap.add_argument("--scatter", type=float, default=0.35)
    ap.add_argument("--bandwidth", type=float, default=0.25)
    ap.add_argument("--reuses", type=float, nargs="*",
                    default=[1.24, 2.0, 4.0, 8.0])
    ap.add_argument("--alignments", type=float, nargs="*",
                    default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--out", default="scratch/synth/geometry_sweep.json")
    args = ap.parse_args()

    grid = {}
    for reuse in args.reuses:
        for align in args.alignments:
            cells = [run_geo_cell(seed, reuse, align, args.n_trials, args.n_test,
                                  args.n_classes, args.scatter, args.bandwidth)
                     for seed in range(args.seeds)]
            key = f"{reuse}|{align}"
            grid[key] = {
                "pi_corr": aggregate(cells, REP_NAMES, "pi_corr"),
                "auroc": aggregate(cells, REP_NAMES, "auroc"),
                "eff_reuse_median": aggregate(cells, REP_NAMES, "eff_reuse_median"),
                "eff_reuse_ge8": aggregate(cells, REP_NAMES, "eff_reuse_ge8"),
                "eff_reuse_mean": aggregate(cells, REP_NAMES, "eff_reuse_mean"),
                "oracle": float(np.mean([c["_oracle"] for c in cells])),
            }
            print(f"reuse={reuse:<5} align={align:<4} "
                  f"pi_corr none={grid[key]['pi_corr']['none'][0]:.3f} "
                  f"discrete={grid[key]['pi_corr']['discrete'][0]:.3f} "
                  f"f_perdim={grid[key]['pi_corr']['field_perdim'][0]:.3f} "
                  f"f_full={grid[key]['pi_corr']['field_full'][0]:.3f}",
                  flush=True)

    blob = {"config": vars(args), "grid": grid}

    # ---- report: composite recovery surface (corr π̂,π_true) ----
    print("\n=== composite recovery corr(π̂,π_true) — reuse × alignment ===")
    for rep in REP_NAMES:
        print(f"\n-- {rep} --")
        print("reuse\\align " + "".join(f"{a:>8}" for a in args.alignments))
        for reuse in args.reuses:
            row = f"{reuse:>10} "
            for align in args.alignments:
                row += f"{grid[f'{reuse}|{align}']['pi_corr'][rep][0]:>8.3f}"
            print(row + ("   <- corpus" if reuse == 1.24 else ""))

    # ---- crossover: discrete vs field is a CROSSOVER in alignment ----
    # field degrades gracefully (gentle smoothing); discrete is high-variance
    # (full pooling) — great at high alignment, context-collapse at low. So
    # field WINS below a crossover α and discrete wins above it. α_cross = the
    # smallest alignment where discrete overtakes the best field. The decision
    # rule (Step 3): real alignment < α_cross → field; > α_cross → discrete.
    print("\n=== winner by alignment (composite corr) at corpus reuse ===")
    corpus_reuse = 1.24 if 1.24 in args.reuses else args.reuses[0]
    alphas = sorted(args.alignments)
    print(f"{'align':>6}{'none':>10}{'discrete':>10}{'field*':>10}{'winner':>14}")
    alpha_cross = None
    for align in alphas:
        g = grid[f"{corpus_reuse}|{align}"]["pi_corr"]
        none_v = g["none"][0]
        disc_v = g["discrete"][0]
        field_v = max(g["field_perdim"][0], g["field_full"][0])
        best = max(("none", none_v), ("discrete", disc_v), ("field", field_v),
                   key=lambda kv: kv[1])
        if disc_v > field_v + 1e-6 and alpha_cross is None:
            alpha_cross = align
        print(f"{align:>6}{none_v:>10.3f}{disc_v:>10.3f}{field_v:>10.3f}"
              f"{best[0]:>14}")
    print(f"\n  α_cross (discrete overtakes best field) at reuse {corpus_reuse}"
          f" = {alpha_cross}")
    print("  → real alignment BELOW α_cross ⇒ field; ABOVE ⇒ discrete "
          "(see Step 3 for the real estimate)")

    # ---- effective reuse (% nodes ≥ 8) at corpus reuse ----
    print(f"\n=== effective reuse: %% nodes ≥ 8 at reuse {corpus_reuse} ===")
    print("align " + "".join(f"{r:>16}" for r in REP_NAMES))
    for align in args.alignments:
        g = grid[f"{corpus_reuse}|{align}"]
        row = f"{align:>5} "
        for rep in REP_NAMES:
            row += f"{100*g['eff_reuse_ge8'][rep][0]:>14.0f}%%"
        print(row)

    with open(args.out, "w") as fh:
        json.dump(blob, fh, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
