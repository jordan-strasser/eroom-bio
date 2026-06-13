"""P5 reliability PROXIES (no paid re-runs).

Re-running the LLM on 15 trials would cost real API spend and only one cached
run exists per trial, so cross-run vote agreement isn't measurable. Instead we
bound reliability from internal consistency of the two INDEPENDENT LLM-derived
outcome signals about each trial:
  - extraction.results.primary_endpoint_met  (the extractor)
  - classification.trial_outcome             (the classifier)
plus classifier self-reported confidence, needs_review rate, and the
'insufficient_information' rate (the LLM declaring it couldn't extract).
"""
from __future__ import annotations

from collections import Counter

import numpy as np

import scratch.diagnostics._common as C

corpus = C.load_corpus("multi_500")
conf = []
need_review = 0
outcome_ct = Counter()
fmode_ct = Counter()
n = 0
agree = disagree = comparable = 0
for nct in corpus:
    ext, cls = C.load_annotation(nct)
    if ext is None or cls is None:
        continue
    n += 1
    c = cls.get("confidence_overall")
    if isinstance(c, (int, float)):
        conf.append(float(c))
    if cls.get("needs_expert_review"):
        need_review += 1
    to = (cls.get("trial_outcome") or "").lower()
    outcome_ct[to] += 1
    fmode_ct[C.primary_failure_mode(cls)] += 1
    met = (ext.get("results") or {}).get("primary_endpoint_met")
    if met is not None and to in ("success", "failure"):
        comparable += 1
        pred = "success" if met else "failure"
        if pred == to:
            agree += 1
        else:
            disagree += 1

conf = np.array(conf)
print(f"annotated trials: {n}")
print(f"\nclassifier confidence_overall: mean={conf.mean():.3f} sd={conf.std():.3f} "
      f"p25={np.percentile(conf,25):.3f} med={np.median(conf):.3f} p75={np.percentile(conf,75):.3f}")
print(f"  confidence < 0.5: {(conf<0.5).mean()*100:.0f}%   < 0.7: {(conf<0.7).mean()*100:.0f}%")
print(f"needs_expert_review: {need_review}/{n} ({100*need_review/n:.0f}%)")

print(f"\ntrial_outcome distribution:")
for k, v in outcome_ct.most_common():
    print(f"  {k or '(blank)':12} {v:4} ({100*v/n:.0f}%)")

print(f"\ninternal consistency — extractor primary_endpoint_met vs classifier trial_outcome:")
print(f"  comparable (both definitive): {comparable}")
print(f"  AGREE: {agree} ({100*agree/max(comparable,1):.0f}%)   "
      f"DISAGREE: {disagree} ({100*disagree/max(comparable,1):.0f}%)")
print(f"  → {100*disagree/max(comparable,1):.0f}% of trials where the two LLM passes give "
      f"OPPOSITE binary outcomes")

print(f"\ninsufficient_information (LLM couldn't extract clean efficacy) as primary mode:")
ii = fmode_ct.get("insufficient_information", 0)
print(f"  {ii}/{n} ({100*ii/n:.0f}%) of all annotated trials")
