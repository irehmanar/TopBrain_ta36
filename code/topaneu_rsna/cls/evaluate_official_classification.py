"""
Official-metric evaluation of the RSNA-derived ROI classifier (Task 1: 52-way
multi-label location presence) trained in jobs 07/08 (cls/train.py).

Why this is a SEPARATE script from seg/evaluate_official.py rather than a
call into it: that script's official_score() derives "which locations are
present" from a segmentation mask's own unique nonzero voxel values (`pred_locs
= set(np.unique(pred)) - {0}`), because the real challenge submission for Task
2 is a voxel mask, not a probability vector. This classifier never produces a
spatial mask at all -- forcing its per-case probabilities through that API
would mean synthesizing a fake mask just so `np.unique()` reads back the right
classes, which would also silently produce a Dice/VOLSIM/HD95 number (since
official_score always computes them) that is spatially meaningless -- near-zero
for every true positive whose synthesized voxels don't happen to sit on the
real lesion, not a fair "N/A". This script instead computes ONLY the official
evaluator's classification block (Precision/Recall/F1/MCC, presence-based
TP/FP/FN/TN, summed across cases then averaged per class) directly from
probabilities -- the exact same formulas as seg/evaluate_official.py's
official_score(), lines computing tp/fp/fn/tn/precision/recall/f1/mcc, just
fed from a probability threshold instead of a mask's unique values. Dice/
VOLSIM/HD95 are correctly reported as not applicable, not faked.

Honesty of the evaluation set: cls/predict.py's production CSV always
ensembles the best 4-of-5 folds on EVERY case, which means most cases were
seen during training by 3 of those 4 models -- not a fair generalization
estimate. This script instead pools each fold's own oof.npz (saved by
cls/train.py at that fold's own best epoch: genuinely held-out predictions,
from a model that never trained on that case), matching this pipeline's own
established "pool k folds' own held-out predictions" convention used
everywhere else (Dataset304's load_binary_pred_paths, the location
classifier's LOCO evaluation in Experiment 2). No GPU/inference needed here at
all -- oof.npz already has everything.

Threshold: at the default 0.5, this classifier's own first run showed
precision 0.66 but recall 0.035 (F1 0.051) -- far more conservative than the
rule's 0.36/0.33. config.py's ClsConfig documents that the location head was
trained with only w_loc=0.1 (vs. w_sphere=1.0), explicitly tuned to maximize
AUC, not calibrated for a hard 0.5 cutoff -- so 0.5 may simply be the wrong
operating point, not evidence the underlying ranking is bad. --thresholds
sweeps a range of global cutoffs (applied uniformly, not per-class -- a
per-class-tuned threshold on this small a pooled sample would overfit) to see
where the honest Precision/Recall/F1/MCC tradeoff actually sits, and reports
the F1-maximizing and MCC-maximizing threshold found.

    python -m topaneu_rsna.cls.evaluate_official_classification
    python -m topaneu_rsna.cls.evaluate_official_classification \\
        --thresholds 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from topaneu_rsna import config as C


def pool_oof(results_dir: Path, n_loc: int):
    y_by_case, p_by_case = {}, {}
    n_folds_found = 0
    for fold_dir in sorted(results_dir.glob("fold_*")):
        oof_p = fold_dir / "oof.npz"
        if not oof_p.exists():
            print(f"[warn] {fold_dir.name}: no oof.npz (no best-epoch checkpoint saved yet?)")
            continue
        n_folds_found += 1
        d = np.load(oof_p, allow_pickle=True)
        y, p, cases = d["y"], d["p"], d["cases"]
        for i, case in enumerate(cases):
            case = str(case)
            if case in y_by_case:
                print(f"[warn] case {case} appears in more than one fold's oof.npz "
                     f"-- keeping the first, this shouldn't happen with a clean split")
                continue
            y_by_case[case] = y[i]
            p_by_case[case] = p[i]
    return y_by_case, p_by_case, n_folds_found


def score_at_threshold(Y: np.ndarray, Pprob: np.ndarray, threshold: float, locations: list[str]):
    """Y, Pprob: (n_cases, n_loc) boolean ground truth / float32 probabilities.
    Returns (per_class: dict[name -> metrics], class_avg: dict[metric -> float]),
    same presence-based TP/FP/FN/TN -> Precision/Recall/F1/MCC formulas as
    seg/evaluate_official.py's official_score()."""
    P = Pprob >= threshold
    tp = (P & Y).sum(0); fp = (P & ~Y).sum(0)
    fn = (~P & Y).sum(0); tn = (~P & ~Y).sum(0)

    with np.errstate(invalid="ignore", divide="ignore"):
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * tp / (2 * tp + fp + fn)
        mcc = (tp * tn - fp * fn) / np.sqrt(
            (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    per_class = {}
    for i, loc in enumerate(locations):
        per_class[loc] = dict(precision=float(precision[i]), recall=float(recall[i]),
                              f1=float(f1[i]), mcc=float(mcc[i]),
                              tp=int(tp[i]), fp=int(fp[i]), fn=int(fn[i]), tn=int(tn[i]))

    class_avg = {k: float(np.nanmean([per_class[loc][k] for loc in locations]))
                for k in ("precision", "recall", "f1", "mcc")}
    return per_class, class_avg


def write_per_class_csv(per_class: dict, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "precision", "recall", "f1", "mcc", "tp", "fp", "fn", "tn"])
        for loc, m in per_class.items():
            w.writerow([loc, m["precision"], m["recall"], m["f1"], m["mcc"],
                       m["tp"], m["fp"], m["fn"], m["tn"]])
    print(f"per-class detail written to {out_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=Path, default=C.CLS_RESULTS_DIR)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="single fixed threshold (used unless --thresholds is given)")
    ap.add_argument("--thresholds", type=float, nargs="+", default=None,
                    help="sweep these thresholds instead of a single fixed one, e.g. "
                         "--thresholds 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5")
    ap.add_argument("--out_csv", type=Path, default=None)
    a = ap.parse_args()

    spec = C.load_labels()
    y_by_case, p_by_case, n_folds_found = pool_oof(a.results_dir, spec.n_loc)
    cases = sorted(y_by_case)
    print(f"{n_folds_found} fold(s) with a saved oof.npz, {len(cases)} cases pooled "
         f"as genuinely held-out (never trained on by the model that predicted them)")
    if n_folds_found < C.CLS.n_folds:
        print(f"[warn] only {n_folds_found}/{C.CLS.n_folds} folds have been trained -- "
             f"this is a PARTIAL evaluation, not the full cohort")
    if not cases:
        raise SystemExit("no oof.npz files found -- train at least one fold first "
                         "(jobs/02_classification_rsna/07_train_cls.sbatch)")

    Y = np.stack([y_by_case[c] for c in cases]) > 0.5          # ground truth presence
    Pprob = np.stack([p_by_case[c] for c in cases])            # (n_cases, n_loc) probabilities
    print("dice/volsim/hd95: N/A -- this model never produces a spatial mask, "
         "see this script's own module docstring for why that isn't faked here")

    if a.thresholds is None:
        per_class, class_avg = score_at_threshold(Y, Pprob, a.threshold, spec.locations)
        print(f"\n=== RSNA ROI classifier -- official Task-2 classification metrics "
             f"(n={len(cases)} held-out cases, threshold={a.threshold}) ===")
        for k in ("precision", "recall", "f1", "mcc"):
            print(f"  {k:<10}{class_avg[k]:.4f}")
        print("\nCompare against the rule's own reference point: "
             "dice 0.0052, precision 0.3596, recall 0.3253, mcc 0.3530")
        write_per_class_csv(per_class, a.out_csv or C.LOG_ROOT / "task1_classifier_official_metrics.csv")
        return

    print(f"\n=== RSNA ROI classifier -- threshold sweep (n={len(cases)} held-out cases) ===")
    print(f"  {'threshold':<10}{'precision':<11}{'recall':<9}{'f1':<9}{'mcc':<9}")
    rows = []
    for t in sorted(a.thresholds):
        _, class_avg = score_at_threshold(Y, Pprob, t, spec.locations)
        rows.append((t, class_avg))
        print(f"  {t:<10.3f}{class_avg['precision']:<11.4f}{class_avg['recall']:<9.4f}"
             f"{class_avg['f1']:<9.4f}{class_avg['mcc']:<9.4f}")

    best_f1_t, best_f1_avg = max(rows, key=lambda r: (r[1]["f1"] if r[1]["f1"] == r[1]["f1"] else -1))
    best_mcc_t, best_mcc_avg = max(rows, key=lambda r: (r[1]["mcc"] if r[1]["mcc"] == r[1]["mcc"] else -1))
    print(f"\nbest F1  @ threshold={best_f1_t:.3f}: precision={best_f1_avg['precision']:.4f} "
         f"recall={best_f1_avg['recall']:.4f} f1={best_f1_avg['f1']:.4f} mcc={best_f1_avg['mcc']:.4f}")
    print(f"best MCC @ threshold={best_mcc_t:.3f}: precision={best_mcc_avg['precision']:.4f} "
         f"recall={best_mcc_avg['recall']:.4f} f1={best_mcc_avg['f1']:.4f} mcc={best_mcc_avg['mcc']:.4f}")
    print("\nCompare against the rule's own reference point: "
         "dice 0.0052, precision 0.3596, recall 0.3253, mcc 0.3530")

    out_csv = a.out_csv or C.LOG_ROOT / "task1_classifier_threshold_sweep.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["threshold", "precision", "recall", "f1", "mcc"])
        for t, avg in rows:
            w.writerow([t, avg["precision"], avg["recall"], avg["f1"], avg["mcc"]])
    print(f"\nsweep table written to {out_csv}")


if __name__ == "__main__":
    main()
