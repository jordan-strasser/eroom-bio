"""Phase 3 — B1 scoreboard: baseline (content-hash biology) vs B1 (GO-BP biology).

Both annotated snapshots come from the SAME multi_500_initial.json + the same
annotations; the only difference is the biology id scheme (flag). Metrics per the
task's Option-2 scoreboard:

  (1) effective reuse on identity  — biology node + biology-incident edge reuse
  (2) per-branch recovery          — efficacy/measurement/safety edge reuse
  (3) context-collapse guard       — outcome/indication heterogeneity of merged biology
  (5) safety invariance            — within-target target_associated_ae E[p] SD ~0.048
"""
import statistics as st
from collections import defaultdict

import numpy as np

import scratch.diagnostics._common as C

BASE = "data/exports/b1_baseline_annotated.json"
B1 = "data/exports/b1_ontology_annotated.json"


def node_trials_and_ctx(g):
    """biology_id -> (set host NCTs, [outcomes], set indications)."""
    nt = defaultdict(set)
    outc = defaultdict(list)
    inds = defaultdict(set)
    for nct, ts in g.trial_subgraphs.items():
        for ch in ts.chains:
            bid = getattr(ch, "biology_id", None)
            if not bid or bid == "UNKNOWN":
                continue
            nt[bid].add(nct)
            o = getattr(ch, "outcome", None)
            if o:
                outc[bid].append(str(o))
            ind = getattr(ch, "indication_id", None)
            if ind:
                inds[bid].add(ind)
    return nt, outc, inds


def reuse_hist(vals, label):
    vals = sorted(vals)
    n = len(vals) or 1
    s1 = sum(1 for v in vals if v <= 1)
    g4 = sum(1 for v in vals if v >= 4)
    g8 = sum(1 for v in vals if v >= 8)
    print(f"  {label:<16} n={len(vals):>4} median={st.median(vals) if vals else 0:>4} "
          f"mean={np.mean(vals) if vals else 0:.2f} %singleton={100*s1/n:>4.1f} "
          f"%>=4={100*g4/n:>4.1f} %>=8={100*g8/n:>4.1f} max={max(vals) if vals else 0}")


def main():
    gb = C.load_graph(BASE)
    g1 = C.load_graph(B1)

    print("=" * 78)
    print("(1) BIOLOGY NODE TRIAL-REUSE — headline scoreboard (target: %>=8 -> 18.7%)")
    print("=" * 78)
    for name, g in (("baseline", gb), ("B1-GO", g1)):
        nt, _, _ = node_trials_and_ctx(g)
        bio_ids = [n for n in g._graph.nodes if g._graph.nodes[n].get("node_type") == "BiologyNode"]
        vals = [len(nt.get(b, set())) for b in bio_ids]
        reuse_hist(vals, name)

    print("\n" + "=" * 78)
    print("(1b) BIOLOGY-INCIDENT EDGE REUSE (#distinct host NCTs), by edge type")
    print("=" * 78)
    inc_types = ["mechanism_affects", "biology_drives", "reflects_biology"]
    for et in inc_types:
        row = []
        for name, g in (("baseline", gb), ("B1-GO", g1)):
            vals = []
            for u, v, key, b in C.iter_edges(g):
                if key == et and b is not None:
                    vals.append(len(set(C.trial_evidence_ncts(b))))
            if vals:
                mean = np.mean(vals)
                ge2 = 100 * sum(1 for x in vals if x >= 2) / len(vals)
                ge8 = 100 * sum(1 for x in vals if x >= 8) / len(vals)
                row.append(f"{name}: n={len(vals)} mean={mean:.2f} %>=2={ge2:.0f} %>=8={ge8:.0f}")
        print(f"  {et:20} | " + "   ||   ".join(row))

    print("\n" + "=" * 78)
    print("(2) PER-BRANCH EDGE-CLASS REUSE (mean trials/edge, %>=2)")
    print("=" * 78)
    for cls in (C.EFFICACY, C.MEASUREMENT, C.SAFETY):
        row = []
        for name, g in (("baseline", gb), ("B1-GO", g1)):
            vals = []
            for u, v, key, b in C.iter_edges(g):
                if C.EDGE_CLASS.get(key) == cls and b is not None:
                    vals.append(len(set(C.trial_evidence_ncts(b))))
            if vals:
                row.append(f"{name}: n={len(vals)} mean={np.mean(vals):.2f} "
                           f"%>=2={100*sum(1 for x in vals if x>=2)/len(vals):.0f}")
        print(f"  {cls:12} | " + "   ||   ".join(row))

    print("\n" + "=" * 78)
    print("(3) CONTEXT-COLLAPSE GUARD — heterogeneity of biology nodes (reuse>=2)")
    print("   (does merging pool mixed OUTCOMES / multiple INDICATIONS under one node?)")
    print("=" * 78)
    # resolve a real binary label per trial (chain.outcome is UNKNOWN pre-scoring)
    from scripts.eval_holdout_compose import _resolve_label
    trial_label = {}
    for nct in set(gb.trial_subgraphs) | set(g1.trial_subgraphs):
        ext, clsj = C.load_annotation(nct)
        if ext is None:
            continue
        lab = _resolve_label(ext, clsj)
        if lab in ("success", "failure"):
            trial_label[nct] = lab
    print(f"  (resolved binary labels for {len(trial_label)} trials)")
    for name, g in (("baseline", gb), ("B1-GO", g1)):
        nt, outc, inds = node_trials_and_ctx(g)
        bio_ids = [n for n in g._graph.nodes if g._graph.nodes[n].get("node_type") == "BiologyNode"]
        multi = [b for b in bio_ids if len(nt.get(b, set())) >= 2]
        n_ind = [len(inds.get(b, set())) for b in multi]

        def labels_for(b):
            return [trial_label[t] for t in nt.get(b, set()) if t in trial_label]

        def mixed(b):
            ls = set(labels_for(b))
            return "success" in ls and "failure" in ls
        mixed_ct = sum(1 for b in multi if mixed(b))
        # mean per-node label entropy (0 = pure, 1 = max mixed)
        ents = []
        for b in multi:
            ls = labels_for(b)
            if len(ls) >= 2:
                p = ls.count("success") / len(ls)
                if 0 < p < 1:
                    ents.append(-(p * np.log2(p) + (1 - p) * np.log2(1 - p)))
                else:
                    ents.append(0.0)
        print(f"  {name:9} | reuse>=2 nodes={len(multi):4}  "
              f"mean #indications/node={np.mean(n_ind) if n_ind else 0:.2f}  "
              f"max={max(n_ind) if n_ind else 0}  "
              f"%pooling mixed succ+fail={100*mixed_ct/max(1,len(multi)):.0f}%  "
              f"mean outcome-entropy={np.mean(ents) if ents else 0:.3f}")

    # focus: the GO-merged nodes only (B1 nodes whose id is bio:GO:*)
    nt1, outc1, inds1 = node_trials_and_ctx(g1)
    go_nodes = [n for n in g1._graph.nodes
                if g1._graph.nodes[n].get("node_type") == "BiologyNode"
                and (str(g1._graph.nodes[n].get("ontology_id", "")).startswith("bio:GO:")
                     or str(n).startswith("bio:GO:"))]
    go_multi = [b for b in go_nodes if len(nt1.get(b, set())) >= 2]
    print(f"\n  [B1 GO-keyed nodes] total={len(go_nodes)} reuse>=2={len(go_multi)}; "
          f"mean #indications={np.mean([len(inds1.get(b,set())) for b in go_multi]) if go_multi else 0:.2f}")
    # show the highest-indication-spread GO nodes (candidate context collapse)
    sp = sorted(go_multi, key=lambda b: -len(inds1.get(b, set())))[:8]
    for b in sp:
        nd = g1._graph.nodes[b]
        os_ = Counter(outc1.get(b, []))
        print(f"    {b} '{(nd.get('name') or '')[:42]}' trials={len(nt1.get(b,set()))} "
              f"inds={len(inds1.get(b,set()))} outcomes={dict(os_)}")

    print("\n" + "=" * 78)
    print("(5) SAFETY INVARIANCE — within-target target_associated_ae E[p] SD (~0.048)")
    print("=" * 78)
    for name, g in (("baseline", gb), ("B1-GO", g1)):
        tgt = defaultdict(list)
        for u, v, key, b in C.iter_edges(g):
            if key == "target_associated_ae" and b is not None and b.evidence_strength > 0:
                tgt[u].append(b.expected_probability)
        sds = [np.std(vs) for vs in tgt.values() if len(vs) >= 3]
        print(f"  {name:9} | targets(>=3 AE)={len(sds):3}  mean within-target SD="
              f"{np.mean(sds) if sds else float('nan'):.4f}")


if __name__ == "__main__":
    from collections import Counter  # noqa: E402
    globals()["Counter"] = Counter
    main()
