"""
Experiment Alpha, step 3: 3D connected-component volume-threshold
postprocessing on top of a chosen prediction set (default: the existing
baseline -- single fold, own validation cases, TTA on -- since that is the
pipeline's actual current best/least-leaked number; ensemble5 is
leaked/optimistic and notta is expected weaker, see the other two scripts).

Applied on top of the ONE chosen base variant only (not stacked with
notta/ensemble5), per the task's own "measure each step individually" ask.

  1. Label every case's predicted binary mask into 3D connected components
     (26-connectivity), compute each component's physical volume in mm^3
     (voxel count * spacing product).
  2. Report the pooled volume distribution across all cases/components (so
     the threshold grid below is chosen with real numbers in view, not
     blind).
  3. Sweep a small, fixed threshold grid (mm^3) -- deliberately coarse (a
     handful of round numbers, not a fine per-case-tuned search) to limit
     how much picking "best on this eval set" can overfit that same set.
     For each threshold, strip components below it and recompute the same
     pooled Dice/IoU/precision/recall used by expAlpha_evaluate_variants.py.
  4. Write the best-by-pooled-Dice threshold's masks to
     EXPALPHA_PRED_DIR/postproc_best/ and append a "postproc_best" row (plus
     the full sweep table) so it's directly comparable to the baseline row.

    python -m topaneu_rsna.seg.expAlpha_postprocess_cc --base_variant baseline
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy import ndimage
from tqdm import tqdm

from topaneu_rsna import config as C
from topaneu_rsna.seg.expAlpha_evaluate_variants import (
    append_result_row, evaluate, pred_paths_for,
)
from topaneu_rsna.utils import io as uio

THRESHOLDS_MM3 = [0, 2, 5, 10, 20, 50, 100, 200]


def label_components(pred_by_case: dict):
    """One pass: read + label every case once, cache (labels, sizes_vox,
    spacing, meta) so the threshold sweep below never re-reads/re-labels."""
    cache = {}
    all_component_vols = []
    for case, pp in tqdm(sorted(pred_by_case.items()), desc="labeling"):
        pred, meta = uio.read(pp)
        lab, n = ndimage.label(pred > 0, structure=np.ones((3, 3, 3)))
        if n == 0:
            cache[case] = (lab, np.array([]), meta, pred.shape)
            continue
        voxel_mm3 = float(np.prod(meta["spacing"]))
        sizes_vox = ndimage.sum(np.ones_like(lab), lab, index=range(1, n + 1))
        sizes_mm3 = sizes_vox * voxel_mm3
        cache[case] = (lab, sizes_mm3, meta, pred.shape)
        all_component_vols.extend(sizes_mm3.tolist())
    return cache, np.asarray(all_component_vols)


def apply_threshold(cache: dict, threshold_mm3: float, out_dir: Path | None):
    """Return {case: binary mask array} with components < threshold zeroed.
    Optionally writes each case's mask to out_dir (only used for the final
    chosen threshold, not every sweep point)."""
    masks = {}
    for case, (lab, sizes_mm3, meta, shape) in cache.items():
        if lab.max() == 0:
            out = np.zeros(shape, dtype=np.uint8)
        else:
            keep = np.where(sizes_mm3 >= threshold_mm3)[0] + 1
            out = np.isin(lab, keep).astype(np.uint8)
        masks[case] = out
        if out_dir is not None:
            uio.write(out, meta, out_dir / f"{case}{C.LABEL_SUFFIX}")
    return masks


def evaluate_masks_in_memory(masks: dict, gt_dir: Path):
    """Same metric definitions as expAlpha_evaluate_variants.evaluate(), but
    against in-memory arrays instead of re-reading prediction files from
    disk for every threshold in the sweep."""
    inter = pred_sum = gt_sum = 0
    tp = fp = fn = tn = 0
    per_case_dice = []
    for case, pm in masks.items():
        gp = gt_dir / f"{case}{C.LABEL_SUFFIX}"
        if not gp.exists():
            continue
        gt, _ = uio.read(gp)
        if pm.shape != gt.shape:
            # Same known issue as expAlpha_evaluate_variants.py's evaluate() --
            # a handful of Dataset304 raw files had a stale ROI-cropped-shape
            # file where a native whole-head shape should be. Skip rather than
            # crash the whole threshold sweep over one bad case.
            print(f"[WARNING] shape mismatch for {case}: pred={pm.shape} "
                 f"gt={gt.shape} -- skipping.")
            continue
        gm = gt > 0
        pm_b = pm > 0
        pn, gn = int(pm_b.sum()), int(gm.sum())
        it = int((pm_b & gm).sum())
        inter += it; pred_sum += pn; gt_sum += gn
        if gn > 0 and it > 0:
            tp += 1
        elif gn > 0 and it == 0:
            fn += 1
        elif gn == 0 and pn > 0:
            fp += 1
        else:
            tn += 1
        per_case_dice.append(1.0 if pn + gn == 0 else 2 * it / (pn + gn))

    per_case_dice = np.asarray(per_case_dice)
    with np.errstate(invalid="ignore", divide="ignore"):
        pooled_dice = 2 * inter / (pred_sum + gt_sum) if (pred_sum + gt_sum) else float("nan")
        pooled_iou = inter / (pred_sum + gt_sum - inter) if (pred_sum + gt_sum - inter) else float("nan")
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
    return dict(
        n_cases=len(per_case_dice), pooled_dice=pooled_dice, pooled_iou=pooled_iou,
        precision=precision, recall=recall, tp=tp, fp=fp, fn=fn, tn=tn,
        case_dice_mean=float(per_case_dice.mean()),
        case_dice_median=float(np.median(per_case_dice)),
        case_dice_std=float(per_case_dice.std()),
        case_dice_p25=float(np.percentile(per_case_dice, 25)),
        case_dice_p75=float(np.percentile(per_case_dice, 75)),
        case_dice_min=float(per_case_dice.min()),
        case_dice_max=float(per_case_dice.max()),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_variant", choices=["baseline", "notta", "ensemble5"],
                    default="baseline")
    ap.add_argument("--dataset", type=int, default=C.DS_ANEURYSM)
    ap.add_argument("--trainer", default=C.TRAINER_LOC)
    ap.add_argument("--plans", default=C.PLANS_RESENC)
    ap.add_argument("--out", type=Path, default=C.EXPALPHA_RESULTS_CSV)
    ap.add_argument("--sweep_csv", type=Path,
                    default=C.LOG_ROOT / "task2_expAlpha_postproc_sweep.csv")
    a = ap.parse_args()

    gt_dir = C.nnUNet_raw / f"Dataset{a.dataset:03d}_{C.DS_NAMES[a.dataset]}" / "labelsTr"
    pred_by_case = pred_paths_for(a.base_variant, a.dataset, a.trainer, a.plans)
    print(f"[expAlpha postproc] base_variant={a.base_variant}: {len(pred_by_case)} cases")

    before = evaluate(pred_by_case, gt_dir)
    print(f"[expAlpha postproc] before (threshold=0, i.e. no removal): "
          f"pooled_dice={before['pooled_dice']:.4f}")

    cache, all_vols = label_components(pred_by_case)
    print(f"\n[expAlpha postproc] component volume distribution "
          f"(n={len(all_vols)} components across all cases):")
    if len(all_vols):
        for p in (0, 5, 25, 50, 75, 95, 100):
            print(f"  p{p:<3d}: {np.percentile(all_vols, p):8.2f} mm^3")

    sweep_rows = []
    best = None
    for thr in THRESHOLDS_MM3:
        masks = apply_threshold(cache, thr, out_dir=None)
        m = evaluate_masks_in_memory(masks, gt_dir)
        sweep_rows.append((thr, m))
        print(f"  threshold={thr:>5} mm^3  pooled_dice={m['pooled_dice']:.4f}  "
              f"precision={m['precision']:.4f}  recall={m['recall']:.4f}  "
              f"fp={m['fp']}  fn={m['fn']}")
        if best is None or m["pooled_dice"] > best[1]["pooled_dice"]:
            best = (thr, m)

    a.sweep_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(a.sweep_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["threshold_mm3", "pooled_dice", "pooled_iou", "precision",
                    "recall", "tp", "fp", "fn", "tn", "case_dice_mean",
                    "case_dice_median", "case_dice_std"])
        for thr, m in sweep_rows:
            w.writerow([thr, m["pooled_dice"], m["pooled_iou"], m["precision"],
                        m["recall"], m["tp"], m["fp"], m["fn"], m["tn"],
                        m["case_dice_mean"], m["case_dice_median"], m["case_dice_std"]])

    best_thr, best_m = best
    print(f"\n[expAlpha postproc] BEST threshold = {best_thr} mm^3 "
          f"(pooled_dice {before['pooled_dice']:.4f} -> {best_m['pooled_dice']:.4f})")

    out_dir = C.EXPALPHA_PRED_DIR / "postproc_best"
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_threshold(cache, best_thr, out_dir=out_dir)

    note = (f"CC volume threshold={best_thr}mm^3 chosen by sweeping "
            f"{THRESHOLDS_MM3} against this same eval cohort (mild "
            f"threshold-selection optimism, disclosed here) on top of "
            f"base_variant={a.base_variant}.")
    append_result_row("postproc_best", best_m, a.out, note)
    print(f"\nbest-threshold masks written to {out_dir}")
    print(f"full sweep table written to {a.sweep_csv}")
    print(f"row appended to {a.out}")


if __name__ == "__main__":
    main()
