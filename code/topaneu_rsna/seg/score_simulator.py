"""Leaderboard-style score simulator on honest single-fold data (job 124's
detections + real vessels, with job 137's Acom override applied).

The challenge scores each of the 52 location classes separately, then averages
(see evaluate_location.py, which reproduces those formulas): per case and class,
TP = GT has the class and the prediction overlaps it, FN = GT has it and it is
missed, FP = GT lacks it but it is predicted, TN otherwise; classes with no
defined value (e.g. never present and never predicted) drop out of the mean.
A class the pipeline only hallucinates therefore contributes precision 0, while
a class it never emits costs nothing -- so precision-oriented post-hoc policies
can move the averaged score by more than an instance-level view suggests.

Overlap is approximated by "the instance's majority GT class equals the class it
was assigned" (the same true_class column job 124 already records). Policies are
compared on mean precision / recall / F1 / MCC over classes. Class-abstention
policies are fit on one random half of the cases and scored on the other half
(20 random splits) so they are not judged on the data that chose them.

    python -m topaneu_rsna.seg.score_simulator
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from topaneu_rsna import config as C
from topaneu_rsna.utils import io as uio
from topaneu_rsna.seg.assign_location_rule import load_binary_pred_paths

A1A2 = {"L-A1A2", "R-A1A2"}
ACOM = "4.1 Acom complex"


def class_metrics(pred_sets, ovl_sets, gt_sets, cases, classes):
    n = len(classes)
    tp = np.zeros(n); fp = np.zeros(n); fn = np.zeros(n); tn = np.zeros(n)
    for case in cases:
        P = pred_sets.get(case, set()); O = ovl_sets.get(case, set()); G = gt_sets[case]
        for i, c in enumerate(classes):
            g, p, o = c in G, c in P, c in O
            if g and o:
                tp[i] += 1
            elif g:
                fn[i] += 1
            elif p:
                fp[i] += 1
            else:
                tn[i] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1 = 2 * tp / (2 * tp + fp + fn)
        mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return dict(precision=np.nanmean(prec), recall=np.nanmean(rec),
                f1=np.nanmean(f1), mcc=np.nanmean(mcc))


def sets_from(kept: pd.DataFrame):
    pred, ovl = {}, {}
    for r in kept.itertuples():
        if not isinstance(r.asg, str) or not r.asg:
            continue
        pred.setdefault(r.case, set()).add(r.asg)
        if r.asg == r.true_class:
            ovl.setdefault(r.case, set()).add(r.asg)
    return pred, ovl


def score(kept, gt_sets, cases, classes):
    pred, ovl = sets_from(kept)
    return class_metrics(pred, ovl, gt_sets, cases, classes)


def fmt(name, m):
    return (f"{name:<34} P={m['precision']:.3f} R={m['recall']:.3f} "
            f"F1={m['f1']:.3f} MCC={m['mcc']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acom_csv", type=Path,
                    default=C.LOG_ROOT / "task2_acom_override_analysis.csv")
    ap.add_argument("--gt_cache", type=Path, default=C.LOG_ROOT / "task2_gt_class_sets.csv")
    ap.add_argument("--modality", choices=["all", "ct", "mr"], default="all")
    a = ap.parse_args()

    spec = C.load_labels()
    classes = list(spec.locations)

    bin_paths = load_binary_pred_paths(C.DS_ANEURYSM, C.TRAINER_LOC, C.PLANS_RESENC,
                                       [0, 1, 2, 3, 4])
    cases = sorted(bin_paths)
    if a.gt_cache.exists():
        g = pd.read_csv(a.gt_cache)
        gt_sets = {r.case: set(str(r.classes).split(";")) - {"", "nan"} for r in g.itertuples()}
    else:
        gt_sets, rows = {}, []
        for case in cases:
            p = C.LOCATION_MASKS / f"{case}{C.LABEL_SUFFIX}"
            vals = set(np.unique(uio.read(p)[0]).tolist()) - {0} if p.exists() else set()
            gt_sets[case] = {classes[v - 1] for v in vals if 1 <= v <= len(classes)}
            rows.append(dict(case=case, classes=";".join(sorted(gt_sets[case]))))
        pd.DataFrame(rows).to_csv(a.gt_cache, index=False)
    cases = [c for c in cases if c in gt_sets]
    if a.modality != "all":
        tag = "_ct_" if a.modality == "ct" else "_mr_"
        cases = [c for c in cases if tag in c]
        print(f"modality filter: {a.modality.upper()}")
    print(f"{len(cases)} cases; GT locations per case: "
          f"mean {np.mean([len(gt_sets[c]) for c in cases]):.2f}")

    m = pd.read_csv(a.acom_csv)
    m["touches_acom"] = m.touches_acom.astype(bool)
    m["asg"] = m.assigned
    sel = m.host_vessel.isin(A1A2) & m.touches_acom
    m.loc[sel, "asg"] = ACOM
    m = m[m.case.isin(cases)]

    print("\n== absolute size cutoff (voxels), Acom override on ==")
    for thr in [0, 13, 50, 100, 150, 200, 300]:
        print(fmt(f"size>={thr}", score(m[m["size"] >= thr], gt_sets, cases, classes)))

    print("\n== relative size: keep instances >= f x largest in case (and size>=50) ==")
    mx = m.groupby("case")["size"].transform("max")
    for f in [0.05, 0.1, 0.2, 0.3, 0.5]:
        k = m[(m["size"] >= 50) & (m["size"] >= f * mx)]
        print(fmt(f"rel>={f}", score(k, gt_sets, cases, classes)))

    print("\n== top-k largest instances per case (size>=50) ==")
    base = m[m["size"] >= 50].copy()
    base["rank"] = base.groupby("case")["size"].rank(ascending=False, method="first")
    for k in [1, 2, 3, 4]:
        print(fmt(f"top{k}", score(base[base["rank"] <= k], gt_sets, cases, classes)))

    print("\n== class abstention (fit on half the cases, scored on the other half; 20 splits) ==")
    kept0 = m[m["size"] >= 100]
    rng = np.random.default_rng(0)
    for tau, min_n in itertools.product([0.2, 0.3, 0.4], [3, 5]):
        d = {k: [] for k in ("precision", "recall", "f1", "mcc")}
        n_ab = []
        for _ in range(20):
            perm = rng.permutation(len(cases)); h = len(cases) // 2
            A = {cases[i] for i in perm[:h]}; B = [cases[i] for i in perm[h:]]
            tr = kept0[kept0.case.isin(A)]
            n = tr.groupby("asg").size()
            ok = tr[tr.asg == tr.true_class].groupby("asg").size().reindex(n.index).fillna(0)
            ab = set(n.index[(n >= min_n) & (ok / n < tau)])
            n_ab.append(len(ab))
            te = kept0[kept0.case.isin(B)]
            b0 = score(te, gt_sets, B, classes)
            b1 = score(te[~te.asg.isin(ab)], gt_sets, B, classes)
            for k in d:
                d[k].append(b1[k] - b0[k])
        print(f"tau={tau} min_n={min_n}: mean classes abstained {np.mean(n_ab):.1f}; "
              + " ".join(f"d{k}={np.mean(v):+.3f}" for k, v in d.items()))


if __name__ == "__main__":
    main()
