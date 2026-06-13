"""P2.3 (score distribution / chain-length / calibration) + P7 in-sample AUROC
+ P9.3 partial (safety_penalty vs outcome).

Runs the DEPLOYED predictor (predict_clinical_hypothesis) in-sample over the
corpus. In-sample is leaky for AUROC but exactly right for the SHAPE questions
P2.3 asks (does score track chain length? is it calibrated? is it collapsed?).
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict

import numpy as np

import scratch.diagnostics._common as C
from scripts.eval_holdout_compose import (
    _corpus_ncts, _load_canonicalization_cache, _overlap_count, _resolve_label,
    _training_used_nodes, _trial_conditions, resolve_chain,
)
from src.prediction.path_query import predict_clinical_hypothesis

G = sys.argv[1] if len(sys.argv) > 1 else "data/exports/multi_500_annotated.json"
CORPUS = sys.argv[2] if len(sys.argv) > 2 else "multi_500"
NS = int(sys.argv[3]) if len(sys.argv) > 3 else 800
g = C.load_graph(G)
training_used = _training_used_nodes(g)
canon = _load_canonicalization_cache()

rows = []  # dict per scored trial
for nct in _corpus_ncts(CORPUS):
    ext, cls = C.load_annotation(nct)
    if ext is None:
        continue
    lab = _resolve_label(ext, cls)
    if lab not in ("success", "failure"):
        continue
    chain = resolve_chain(ext, _trial_conditions(ext), g, canon)
    if chain["target_id"] not in training_used or chain["indication_id"] not in training_used:
        continue
    if _overlap_count(chain, training_used) < 5:
        continue
    kwargs = {k: chain[k] for k in
              ("target_id", "mechanism_id", "biology_id", "endpoint_id", "population_id")
              if chain[k] and chain[k] != "UNKNOWN"}
    try:
        r = predict_clinical_hypothesis(g, chain["compound_id"], chain["indication_id"],
                                        n_samples=NS, **kwargs)
    except KeyError:
        continue
    if not r.edge_contributions:
        continue
    eclasses = Counter(C.EDGE_CLASS.get(ec.edge_type.value, "?") for ec in r.edge_contributions)
    rows.append(dict(
        nct=nct, y=1 if lab == "success" else 0,
        overall=r.overall_probability, efficacy=r.efficacy_probability,
        safety=r.safety_penalty, n_edges=len(r.edge_contributions),
        weakest=(r.weakest_link.edge_type.value if r.weakest_link else "—"),
        n_eff_edges=eclasses.get(C.EFFICACY, 0), n_meas=eclasses.get(C.MEASUREMENT, 0),
    ))

n = len(rows)
y = np.array([r["y"] for r in rows])
ov = np.array([r["overall"] for r in rows])
eff = np.array([r["efficacy"] for r in rows])
ne = np.array([r["n_edges"] for r in rows])
print(f"graph: {G}  corpus: {CORPUS}  n_samples={NS}")
print(f"scored trials: {n}  (succ={int(y.sum())}, fail={int(n-y.sum())}, base={y.mean():.3f})")


def auroc(p, yy):
    pos = p[yy == 1]; neg = p[yy == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5*(pos[:, None] == neg[None, :]).mean())


print("\n========== P2.3 — SCORE DISTRIBUTION ==========")
print(f"  overall P(success): mean={ov.mean():.3f} sd={ov.std():.3f} "
      f"min={ov.min():.3f} p25={np.percentile(ov,25):.3f} med={np.median(ov):.3f} "
      f"p75={np.percentile(ov,75):.3f} max={ov.max():.3f}")
print(f"  efficacy-only     : mean={eff.mean():.3f} sd={eff.std():.3f}")
print(f"  mean overall | success={ov[y==1].mean():.3f}  | failure={ov[y==0].mean():.3f}  "
      f"Δ={ov[y==1].mean()-ov[y==0].mean():+.3f}")
print(f"  in-sample AUROC (overall)  = {auroc(ov,y):.3f}   (LEAKY — upper bound)")
print(f"  in-sample AUROC (efficacy) = {auroc(eff,y):.3f}")

print("\n========== P2.3 — SCORE vs CHAIN LENGTH vs OUTCOME ==========")
print(f"  corr(overall, n_edges) = {np.corrcoef(ov, ne)[0,1]:+.3f}")
print(f"  corr(overall, outcome) = {np.corrcoef(ov, y)[0,1]:+.3f}")
print(f"  corr(n_edges, outcome) = {np.corrcoef(ne, y)[0,1]:+.3f}")
print(f"  chain length: mean={ne.mean():.1f} sd={ne.std():.1f} "
      f"min={ne.min()} med={int(np.median(ne))} max={ne.max()}")
# does score track length more than outcome? compare |corr|
print(f"  |corr(score,length)|={abs(np.corrcoef(ov,ne)[0,1]):.3f} vs "
      f"|corr(score,outcome)|={abs(np.corrcoef(ov,y)[0,1]):.3f}")

print("\n========== P2.3 — CALIBRATION (binned) ==========")
print(f"  {'bin':<12}{'count':>7}{'mean_pred':>11}{'obs_freq':>10}")
edges_bin = np.linspace(0, 1, 11)
for i in range(10):
    lo, hi = edges_bin[i], edges_bin[i+1]
    m = (ov >= lo) & (ov < hi if i < 9 else ov <= hi)
    if m.sum() == 0:
        continue
    print(f"  [{lo:.1f},{hi:.1f}){'':<3}{int(m.sum()):>7}{ov[m].mean():>11.3f}{y[m].mean():>10.3f}")

print("\n========== weakest-link edge-type frequency ==========")
for et, c in Counter(r["weakest"] for r in rows).most_common():
    print(f"  {et:24} {c}  class={C.EDGE_CLASS.get(et,'?')}")

print("\n========== P9.3 (partial) — safety_penalty vs outcome ==========")
sp = np.array([r["safety"] for r in rows])
nz = sp > 0
print(f"  trials with safety_penalty>0: {int(nz.sum())}/{n}")
if nz.sum() > 3:
    print(f"  mean safety_penalty | success={sp[y==1].mean():.3f}  | failure={sp[y==0].mean():.3f}")
    print(f"  AUROC of safety_penalty ALONE predicting FAILURE = "
          f"{auroc(sp, 1-y):.3f}  (does AE drag flag failures?)")
print("\nNOTE: DLT/safety-driven failures are routed to 'ambiguous' by _resolve_label,"
      "\nso they are NOT in this scored set — a dedicated safety-failure AUROC (P9.3)"
      "\nis not computable here (only 3 DLT trials in the whole corpus).")
