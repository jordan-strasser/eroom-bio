"""P1.3 (edge posterior vs host-trial base rate) + P4 (failure-mode routing).

Builds a per-trial label/failure-mode map from cached annotations, then:
  P1.3 — Pearson r between each edge's E[p] and the base success rate of the
         trials whose outcome touched it, by class.
  P4   — coverage of failure modes + how often the failure REASON implicates a
         class OTHER than the efficacy/measurement backbone the attributor
         always downvotes.
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict

import numpy as np

import scratch.diagnostics._common as C
from scripts.eval_holdout_compose import _resolve_label

G = sys.argv[1] if len(sys.argv) > 1 else "data/exports/multi_500_annotated.json"
g = C.load_graph(G)

# ── per-trial signals ───────────────────────────────────────────────────────
label: dict[str, int | None] = {}        # binary mechanistic success/fail
trial_outcome: dict[str, str] = {}       # classifier raw success/failure/partial
fmode: dict[str, str] = {}
operational: dict[str, object] = {}
for nct in g.trial_subgraphs:
    ext, cls = C.load_annotation(nct)
    if ext is None:
        continue
    lab = _resolve_label(ext, cls)
    label[nct] = (1 if lab == "success" else 0 if lab == "failure" else None)
    trial_outcome[nct] = ((cls or {}).get("trial_outcome") or "").lower()
    fmode[nct] = C.primary_failure_mode(cls)
    op = (cls or {}).get("operational_failure")
    operational[nct] = op if isinstance(op, bool) else None

binlab = {k: v for k, v in label.items() if v is not None}
print(f"graph: {G}")
print(f"trials with mechanistic binary label: {len(binlab)} "
      f"(succ={sum(binlab.values())}, fail={len(binlab)-sum(binlab.values())}, "
      f"base={np.mean(list(binlab.values())):.3f})")

# ── P1.3: edge posterior vs base success rate of its host trials ────────────
print("\n========== P1.3 — edge E[p] vs host-trial base success rate ==========")
rows_by_class = defaultdict(list)   # cls -> [(E[p], host_base_rate, n_hosts)]
for u, v, key, b in C.iter_edges(g):
    if b is None or b.evidence_strength <= 0:
        continue
    cls = C.EDGE_CLASS.get(key, "?")
    if cls not in (C.EFFICACY, C.MEASUREMENT, C.SAFETY):
        continue
    hosts = [n for n in set(C.trial_evidence_ncts(b)) if n in binlab]
    if not hosts:
        continue
    host_rate = np.mean([binlab[n] for n in hosts])
    rows_by_class[cls].append((b.expected_probability, host_rate, len(hosts)))


def pearson(xs, ys):
    if len(xs) < 3:
        return float("nan")
    return float(np.corrcoef(xs, ys)[0, 1])


for cls in (C.EFFICACY, C.MEASUREMENT, C.SAFETY):
    rows = rows_by_class.get(cls, [])
    if not rows:
        print(f"  {cls:12} no host-labeled edges")
        continue
    ep = [r[0] for r in rows]
    hr = [r[1] for r in rows]
    r_all = pearson(ep, hr)
    # restrict to edges with >=2 host trials (where 'echo' is meaningful)
    multi = [r for r in rows if r[2] >= 2]
    r_multi = pearson([r[0] for r in multi], [r[1] for r in multi]) if len(multi) >= 3 else float("nan")
    print(f"  {cls:12} n_edges={len(rows):5}  Pearson(E[p], host_base_rate) r={r_all:+.3f}"
          f"   | >=2 hosts: n={len(multi)} r={r_multi:+.3f}")
# the punchline: a 1-host efficacy edge's E[p] is a deterministic function of
# that host's outcome (success->0.80-ish, failure->down). So r should be ~1 for
# singletons — quantify how much of the edge population is singleton-driven.
single_eff = [r for r in rows_by_class[C.EFFICACY] if r[2] == 1]
print(f"\n  efficacy edges with exactly 1 host trial: {len(single_eff)}/"
      f"{len(rows_by_class[C.EFFICACY])} "
      f"({100*len(single_eff)/max(len(rows_by_class[C.EFFICACY]),1):.0f}%) "
      f"— for these E[p] is a pure function of that one trial's outcome")
# mean E[p] of single-host edges split by host outcome
succ_ep = [r[0] for r in single_eff if r[1] == 1.0]
fail_ep = [r[0] for r in single_eff if r[1] == 0.0]
if succ_ep and fail_ep:
    print(f"  single-host efficacy E[p]:  host=success mean={np.mean(succ_ep):.3f}  "
          f"host=failure mean={np.mean(fail_ep):.3f}  Δ={np.mean(succ_ep)-np.mean(fail_ep):+.3f}")

# ── P4: failure-mode coverage + routing mismatch ────────────────────────────
print("\n========== P4 — failure-mode coverage + routing ==========")
# Consider trials the classifier called a failure (trial_outcome) OR mechanistic
# binary failure. The attributor conditions on trial_outcome, so use that.
fail_ncts = [n for n in g.trial_subgraphs if trial_outcome.get(n) == "failure"]
print(f"trials with trial_outcome=='failure': {len(fail_ncts)}")
fm_counts = Counter(fmode.get(n, "") for n in fail_ncts)
print("\nprimary failure mode distribution among failures (→ implicated class):")
for fm, n in fm_counts.most_common():
    icls = C.FAILMODE_TO_CLASS.get(fm, "?")
    print(f"  {fm or '(none)':34} {n:4}  implicates={icls}")

# The attributor downvotes the EFFICACY+MEASUREMENT backbone on EVERY failure
# (full weight unless operational_failure==True gates to 0.2). So a failure
# whose implicated class is NOT efficacy/measurement is a MIS-ROUTED downvote.
mis = [n for n in fail_ncts
       if C.FAILMODE_TO_CLASS.get(fmode.get(n, ""), "?")
       not in (C.EFFICACY, C.MEASUREMENT)]
print(f"\nfailures whose REASON is NOT efficacy/measurement (safety / operational /"
      f" business / none) yet still downvote the efficacy+measurement spine:")
print(f"  {len(mis)}/{len(fail_ncts)} = {100*len(mis)/max(len(fail_ncts),1):.0f}% of failures")
# of those, how many had the operational gate actually fire?
mis_op = [n for n in mis if operational.get(n) is True]
should_gate = [n for n in fail_ncts
               if C.FAILMODE_TO_CLASS.get(fmode.get(n, ""), "?") in ("operational", "business")]
gate_fired = [n for n in should_gate if operational.get(n) is True]
print(f"\noperational gate (operational_failure==True → 0.2 weight):")
print(f"  failures flagged operational_failure==True: "
      f"{sum(1 for n in fail_ncts if operational.get(n) is True)}/{len(fail_ncts)}")
print(f"  failures whose MODE is operational/business: {len(should_gate)} "
      f"— of these, gate actually fired: {len(gate_fired)} "
      f"({100*len(gate_fired)/max(len(should_gate),1):.0f}%)")
# safety (DLT) failures: full-weight downvote of the (possibly-fine) biology spine
dlt = [n for n in fail_ncts if fmode.get(n) == "dose_limiting_toxicity"]
dlt_gated = [n for n in dlt if operational.get(n) is True]
print(f"\nDLT (safety) failures: {len(dlt)} — these are 'valid tests' (gate stays "
      f"1.0 for {len(dlt)-len(dlt_gated)}/{len(dlt)}), so they downvote the efficacy"
      f" spine at FULL weight while the real cause is toxicity")
