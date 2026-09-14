"""
Replicates the official TopAneu-26 Task 2 grand-challenge evaluator, verified
directly against its published source rather than reconstructed from the
README alone:
  https://github.com/Bangulli/TopAneu-26/blob/main/eval/task2/evaluate.py
  https://github.com/CoWBenchmark/TopBrain_Eval_Metrics (Dice/HD95, cited
  by the TopAneu-26 README as the exact implementation it reuses)

This is a DIFFERENT, harsher-in-two-ways/looser-in-one-way scoring
convention than `assign_location_rule.py: score()`, which everything in
Experiments 1-3 was measured with up to now:

  Dice / VOLSIM   Computed per (case, class); if the class is absent from
                  EITHER the prediction or the ground truth in that case
                  (including both absent), the score is a hard 0.0 -- not
                  skipped. Averaged as a plain per-case mean for each class
                  (NOT pooled by summing intersections/volumes across the
                  whole cohort first, which is what `score()` does and
                  which lets a few missed cases get diluted into a large
                  denominator instead of each contributing a hard 0).

  HD95            Per (case, class); if the class is absent from either
                  side, the penalty is a fixed HD95_UPPER_BOUND = 290mm
                  (the official constant -- "roughly the maximum distance
                  across a human head") rather than being excluded. This
                  is why official HD95 numbers can look dramatically worse
                  (tens-hundreds of mm) than a version that only averages
                  over the cases where a class was actually detected.

  Precision/Recall/F1/MCC   TP/FP/FN/TN are PRESENCE-based, not overlap-
                  based: a class counts as a hit (TP) if it appears
                  anywhere in both the predicted and true masks for that
                  case, with no requirement that the predicted and true
                  voxels for that class spatially coincide. Counts are
                  summed across all cases before the ratios are computed
                  once at the end (same aggregation style `score()` uses,
                  just a looser per-case unit).

Final per-metric "class-average" score (what the challenge ranks
submissions by) is the mean of each metric's 52 per-class values, NaN
ignored (NaN can only arise here from Precision/Recall/F1/MCC's own
division-by-zero, e.g. a class with zero real occurrences anywhere in the
scored cohort).

Usage: call `official_score(cases, preds, gts, loc_value, n_loc)` with the
exact same arguments as `assign_location_rule.py: score()` -- it's a
drop-in second opinion on the same in-memory prediction/ground-truth dicts,
not a separate pipeline.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

HD95_UPPER_BOUND = 290.0  # mm; the official constant (roughly max distance across a human head)
HD95_PAD_VOX = 20


def _official_hd95(pred_mask: np.ndarray, gt_mask: np.ndarray, spacing) -> float:
    """290mm penalty whenever either mask lacks the class in this case;
    otherwise the real 95th-percentile bidirectional surface distance,
    cropped to the union's bounding box for speed (same trick as
    evaluate_location.py's hd95(), just a different empty-mask convention)."""
    if not pred_mask.any() or not gt_mask.any():
        return HD95_UPPER_BOUND
    union = pred_mask | gt_mask
    idx = np.argwhere(union)
    lo = np.maximum(idx.min(0) - HD95_PAD_VOX, 0)
    hi = np.minimum(idx.max(0) + HD95_PAD_VOX + 1, union.shape)
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    p, g = pred_mask[sl], gt_mask[sl]
    p_surf = p & ~binary_erosion(p)
    g_surf = g & ~binary_erosion(g)
    dt_g = distance_transform_edt(~g, sampling=spacing)
    dt_p = distance_transform_edt(~p, sampling=spacing)
    d = np.concatenate([dt_g[p_surf], dt_p[g_surf]])
    if d.size == 0:
        return HD95_UPPER_BOUND
    return float(np.percentile(d, 95))


def official_score(cases: list[str], preds: dict, gts: dict, loc_value: dict, n_loc: int):
    """Returns (per_class: dict[name -> metrics], class_avg: dict[metric -> float]).
    `preds`/`gts` use the exact same {case: (mask, spacing)} / {case: mask}
    shapes as assign_location_rule.py's score()."""
    id_to_name = {v: k for k, v in loc_value.items()}

    dice_sum = np.zeros(n_loc); vs_sum = np.zeros(n_loc); hd_sum = np.zeros(n_loc)
    n_cases_scored = 0
    tp = np.zeros(n_loc); fp = np.zeros(n_loc); fn = np.zeros(n_loc); tn = np.zeros(n_loc)

    for case in cases:
        pred, spacing = preds[case]
        gt = gts[case]
        pred_locs = set(int(v) for v in np.unique(pred)) - {0}
        gt_locs = set(int(v) for v in np.unique(gt)) - {0}
        n_cases_scored += 1

        for c in range(1, n_loc + 1):
            pm, gm = pred == c, gt == c
            p_present, g_present = c in pred_locs, c in gt_locs

            if p_present and g_present:
                pn, gn = int(pm.sum()), int(gm.sum())
                inter = int((pm & gm).sum())
                dice_sum[c - 1] += (2 * inter / (pn + gn)) if (pn + gn) else 0.0
                vs_sum[c - 1] += 1.0 - (abs(pn - gn) / (pn + gn)) if (pn + gn) else 0.0
            # else: contributes 0.0 to both sums -- already the initial value

            hd_sum[c - 1] += _official_hd95(pm, gm, spacing)

            if p_present and g_present:
                tp[c - 1] += 1
            elif p_present and not g_present:
                fp[c - 1] += 1
            elif g_present and not p_present:
                fn[c - 1] += 1
            else:
                tn[c - 1] += 1

    with np.errstate(invalid="ignore", divide="ignore"):
        dice = dice_sum / n_cases_scored
        vs = vs_sum / n_cases_scored
        hd95 = hd_sum / n_cases_scored
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * tp / (2 * tp + fp + fn)
        mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    per_class = {}
    for c in range(n_loc):
        name = id_to_name[c + 1]
        per_class[name] = dict(
            dice=float(dice[c]), volsim=float(vs[c]), hd95=float(hd95[c]),
            precision=float(precision[c]), recall=float(recall[c]),
            f1=float(f1[c]), mcc=float(mcc[c]),
            tp=int(tp[c]), fp=int(fp[c]), fn=int(fn[c]), tn=int(tn[c]))

    class_avg = {k: float(np.nanmean([per_class[name][k] for name in per_class]))
                for k in ("dice", "volsim", "hd95", "precision", "recall", "f1", "mcc")}
    return per_class, class_avg


def print_official(per_class: dict, class_avg: dict, n_cases: int, out_csv=None):
    print(f"\n=== official TopAneu-26 Task 2 metrics (n={n_cases} cases) ===")
    for k in ("dice", "volsim", "hd95", "precision", "recall", "f1", "mcc"):
        print(f"  {k:<10}{class_avg[k]:.4f}")

    if out_csv is not None:
        import csv
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["class", "dice", "volsim", "hd95", "precision", "recall",
                       "f1", "mcc", "tp", "fp", "fn", "tn"])
            for name, m in per_class.items():
                w.writerow([name, m["dice"], m["volsim"], m["hd95"], m["precision"],
                           m["recall"], m["f1"], m["mcc"], m["tp"], m["fp"], m["fn"], m["tn"]])
        print(f"  per-class detail written to {out_csv}")
