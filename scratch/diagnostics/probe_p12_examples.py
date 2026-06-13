"""P1.2 — concrete per-trial mismatch table.

For a sample of FAILED trials spanning failure modes, show: the failure reason,
the edge class it IMPLICATES, whether the operational gate fired, and how many
efficacy-spine / measurement backbone edges in the trial's subgraph got the
failure downvote anyway (the attributor conditions ALL of them).
"""
from __future__ import annotations

from collections import Counter

import scratch.diagnostics._common as C
from src.annotation.attributor import _chain_backbone_edges

g = C.load_graph("data/exports/multi_500_annotated.json")

# index trials by failure mode so we can sample across classes
by_mode = {}
for nct, ts in g.trial_subgraphs.items():
    ext, cls = C.load_annotation(nct)
    if cls is None or (cls.get("trial_outcome") or "").lower() != "failure":
        continue
    fm = C.primary_failure_mode(cls)
    by_mode.setdefault(fm, []).append((nct, ts, cls))

# pick up to 2 per mode, prioritising non-efficacy modes
order = ["dose_limiting_toxicity", "commercial_not_scientific", "underpowered",
         "manufacturing_or_delivery", "biology_moved_endpoint_flat", "wrong_population",
         "high_placebo_response", "insufficient_information",
         "target_engaged_biology_not_moved", "no_target_engagement"]

print(f"{'NCT':<13}{'failure_mode':<32}{'implic.':<12}{'op_gate':<8}"
      f"{'#eff_edges':>11}{'#meas_edges':>12}")
print("-" * 88)
for fm in order:
    for nct, ts, cls in (by_mode.get(fm, [])[:2]):
        op = cls.get("operational_failure")
        gate = "0.2" if op is True else "1.0"
        eff = meas = 0
        seen = set()
        for ch in ts.chains:
            for s, t, et in _chain_backbone_edges(ch):
                key = (et.value, s, t)
                if key in seen:
                    continue
                seen.add(key)
                cl = C.EDGE_CLASS.get(et.value)
                if cl == C.EFFICACY:
                    eff += 1
                elif cl == C.MEASUREMENT:
                    meas += 1
        implic = C.FAILMODE_TO_CLASS.get(fm, "?")
        print(f"{nct:<13}{fm:<32}{implic:<12}{gate:<8}{eff:>11}{meas:>12}")
print("\nNOTE: every one of these failed trials downvoted ALL #eff_edges efficacy-spine")
print("edges (explaining-away split) regardless of whether the failure implicated them.")
print("AE edges are NOT in this backbone — a DLT death never downvotes the AE edge here;")
print("it downvotes the (possibly-correct) mechanism/biology spine instead.")
