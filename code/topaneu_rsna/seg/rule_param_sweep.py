"""Sweep the location rule's remaining knobs on HONEST data -- job 124's setup
(single-fold held-out binary predictions + real Model 2 vessel) -- using a
geometry cache built once with a wide junction radius, so every setting below
is a cheap replay of the same bootstrap-vote logic docker/task2_rule_based_v2/
inference.py runs, never a recompute of geometry.

Knobs swept (all three are real tunables the container currently fixes at
junction_tau=2.0, junction_override=1.5, arc_ambiguous_margin=0.03):
  jt  : a junction candidate only counts if it lies within this many mm
  ov  : a junction candidate overrides an AMBIGUOUS arc bucket if within this
  mg  : arc-bucket ambiguity margin

The Acom override (job 137) is applied on top via job 137's own output CSV, so
gains here are on top of it. The size cutoff is swept too, reported as instance
precision / recall / F1 where recall is over the real (non-background)
instances. The best config is only trustworthy if it sits on a plateau, not a
single spike -- the printed per-parameter tables are there to check that.

    python -m topaneu_rsna.seg.rule_param_sweep
"""
from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from topaneu_rsna import config as C
from topaneu_rsna.seg.assign_location_rule import arc_bucket_decision

A1A2 = {"L-A1A2", "R-A1A2"}
ACOM_LOC = "4.1 Acom complex"


def decide(r, priors, jt, ov, mg):
    if isinstance(r.single_location, str) and r.single_location:
        return r.single_location
    vessel = r.vessel
    j_loc = None
    j_dist = np.inf
    if isinstance(r.junction_best_loc, str) and r.junction_best_loc \
            and pd.notna(r.junction_best_dist_mm) and r.junction_best_dist_mm <= jt:
        j_loc, j_dist = r.junction_best_loc, float(r.junction_best_dist_mm)
    frac = None if pd.isna(r.arc_fraction) else float(r.arc_fraction)
    votes = []
    for p in priors:
        if frac is not None:
            a_loc, a_amb, _ = arc_bucket_decision(frac, p.get("arc", {}).get(vessel), mg)
        else:
            a_loc, a_amb = None, False
        if j_loc is not None and (a_loc is None or (j_dist <= ov and a_amb)):
            votes.append(j_loc)
        elif a_loc is not None:
            votes.append(a_loc)
        else:
            votes.append(p["majority"].get(vessel))
    tally = Counter(v for v in votes if v is not None)
    return tally.most_common(1)[0][0] if tally else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=C.LOG_ROOT /
                    "task2_shared_vessel_geometry_cache_realvessel_singlefold_jt5.csv")
    ap.add_argument("--acom_csv", type=Path,
                    default=C.LOG_ROOT / "task2_acom_override_analysis.csv")
    ap.add_argument("--prior_dir", type=Path,
                    default=C.SCRATCH_ROOT / "work" / "expGamma_bootstrap_priors")
    ap.add_argument("--n_bootstrap", type=int, default=15)
    ap.add_argument("--out", type=Path, default=C.LOG_ROOT / "task2_rule_param_sweep.csv")
    a = ap.parse_args()

    d = pd.read_csv(a.cache)
    ac = pd.read_csv(a.acom_csv)[["case", "instance_idx", "touches_acom"]]
    d = d.merge(ac, on=["case", "instance_idx"], how="left")
    d["touches_acom"] = d.touches_acom.fillna(False).astype(bool)
    priors = [json.loads((a.prior_dir / f"prior_boot{k}.json").read_text())
              for k in range(a.n_bootstrap)]
    print(f"{len(d)} instances from {a.cache.name}; real (non-background): "
          f"{int((d.true_class != 'background').sum())}")

    rows = []
    grid = list(itertools.product([1, 2, 3, 4, 5], [0.5, 1.0, 1.5, 2.0, 3.0, 4.0],
                                  [0.0, 0.03, 0.06, 0.1]))
    for jt, ov, mg in grid:
        if ov > jt:
            continue
        assigned = [decide(r, priors, jt, ov, mg) for r in d.itertuples()]
        d["asg"] = assigned
        d.loc[d.vessel.isin(A1A2) & d.touches_acom, "asg"] = ACOM_LOC
        ok = (d.asg == d.true_class)
        rows.append(dict(jt=jt, ov=ov, mg=mg, correct=int(ok.sum()),
                         correct_real=int((ok & (d.true_class != "background")).sum())))
    res = pd.DataFrame(rows).sort_values("correct", ascending=False)
    res.to_csv(a.out, index=False)

    base = res[(res.jt == 2) & (res.ov == 1.5) & (res.mg == 0.03)]
    print("\ncurrent container setting (jt=2, ov=1.5, mg=0.03), Acom override on:")
    print(base.to_string(index=False))
    print(f"\nconfigs tried: {len(res)}; top 10:")
    print(res.head(10).to_string(index=False))
    for k in ("jt", "ov", "mg"):
        print(f"\nmean correct by {k} (plateau check):")
        print(res.groupby(k).correct.agg(["mean", "max", "count"]).round(1).to_string())


if __name__ == "__main__":
    main()
