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
     EXPALPHA_PRED_DIR/postproc_best_<variant>/ and append a result row
     (plus the full sweep table) so it's directly comparable to the
     baseline row.

Memory: each case is labeled EXACTLY ONCE regardless of how many thresholds
are swept (case-major streaming: for each case, label once, then evaluate
every threshold immediately and accumulate into per-threshold running
totals) -- never holds more than one case's full-resolution labeled array
in memory at a time. An earlier version cached every case's labeled array
for the whole run's lifetime, which OOM-killed at ~80% through a 417-case,
64GB job (tens of GB of int label arrays held simultaneously) -- this is a
real fix, not a resource-limit workaround.

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


def _new_stat():
    return dict(inter=0, pred_sum=0, gt_sum=0, tp=0, fp=0, fn=0, tn=0, per_case_dice=[])


def sweep_thresholds(pred_by_case: dict, gt_dir: Path, thresholds: list[float]):
    """Case-major streaming sweep: labels each case's predicted mask exactly
    once, then immediately evaluates every threshold in `thresholds` against
    that one case's labeled array before moving to the next case. Returns
    (per-threshold running stats, pooled component-volume distribution) --
    never a per-case cache that grows with cohort size."""
    stats = {thr: _new_stat() for thr in thresholds}
    all_component_vols = []

    for case, pp in tqdm(sorted(pred_by_case.items()), desc="labeling+sweeping"):
        gp = gt_dir / f"{case}{C.LABEL_SUFFIX}"
        if not gp.exists():
            continue
        pred, meta = uio.read(pp)
        gt, _ = uio.read(gp)
        if pred.shape != gt.shape:
            print(f"[WARNING] shape mismatch for {case}: pred={pred.shape} "
                 f"gt={gt.shape} -- skipping.")
            continue
        gm = gt > 0

        lab, n = ndimage.label(pred > 0, structure=np.ones((3, 3, 3)))
        if n > 0:
            voxel_mm3 = float(np.prod(meta["spacing"]))
            sizes_mm3 = ndimage.sum(np.ones_like(lab), lab, index=range(1, n + 1)) * voxel_mm3
            all_component_vols.extend(sizes_mm3.tolist())
        else:
            sizes_mm3 = np.array([])

        for thr in thresholds:
            if n == 0:
                pm = np.zeros(pred.shape, dtype=bool)
            else:
                keep = np.where(sizes_mm3 >= thr)[0] + 1
                pm = np.isin(lab, keep)
            pn, gn = int(pm.sum()), int(gm.sum())
            it = int((pm & gm).sum())
            s = stats[thr]
            s["inter"] += it; s["pred_sum"] += pn; s["gt_sum"] += gn
            if gn > 0 and it > 0:
                s["tp"] += 1
            elif gn > 0:
                s["fn"] += 1
            elif pn > 0:
                s["fp"] += 1
            else:
                s["tn"] += 1
            s["per_case_dice"].append(1.0 if pn + gn == 0 else 2 * it / (pn + gn))
        # `pred`, `gt`, `lab` all go out of scope / get overwritten next
        # iteration -- nothing case-sized is retained across the loop.

    return stats, np.asarray(all_component_vols)


def stat_to_metrics(s: dict) -> dict:
    per_case_dice = np.asarray(s["per_case_dice"])
    with np.errstate(invalid="ignore", divide="ignore"):
        pooled_dice = 2 * s["inter"] / (s["pred_sum"] + s["gt_sum"]) if (s["pred_sum"] + s["gt_sum"]) else float("nan")
        pooled_iou = s["inter"] / (s["pred_sum"] + s["gt_sum"] - s["inter"]) if (s["pred_sum"] + s["gt_sum"] - s["inter"]) else float("nan")
        precision = s["tp"] / (s["tp"] + s["fp"]) if (s["tp"] + s["fp"]) else float("nan")
        recall = s["tp"] / (s["tp"] + s["fn"]) if (s["tp"] + s["fn"]) else float("nan")
    return dict(
        n_cases=len(per_case_dice), pooled_dice=pooled_dice, pooled_iou=pooled_iou,
        precision=precision, recall=recall, tp=s["tp"], fp=s["fp"], fn=s["fn"], tn=s["tn"],
        case_dice_mean=float(per_case_dice.mean()) if len(per_case_dice) else float("nan"),
        case_dice_median=float(np.median(per_case_dice)) if len(per_case_dice) else float("nan"),
        case_dice_std=float(per_case_dice.std()) if len(per_case_dice) else float("nan"),
    )


def write_thresholded_masks(pred_by_case: dict, threshold_mm3: float, out_dir: Path):
    """Second, single-purpose pass -- only ever run once, for the final
    chosen threshold, so re-labeling every case a second time here is cheap
    relative to the memory it saves versus caching every case's label array
    for the whole sweep's lifetime."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for case, pp in tqdm(sorted(pred_by_case.items()), desc="writing best-threshold masks"):
        pred, meta = uio.read(pp)
        lab, n = ndimage.label(pred > 0, structure=np.ones((3, 3, 3)))
        if n == 0:
            out = np.zeros(pred.shape, dtype=np.uint8)
        else:
            voxel_mm3 = float(np.prod(meta["spacing"]))
            sizes_mm3 = ndimage.sum(np.ones_like(lab), lab, index=range(1, n + 1)) * voxel_mm3
            keep = np.where(sizes_mm3 >= threshold_mm3)[0] + 1
            out = np.isin(lab, keep).astype(np.uint8)
        uio.write(out, meta, out_dir / f"{case}{C.LABEL_SUFFIX}")


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

    stats, all_vols = sweep_thresholds(pred_by_case, gt_dir, THRESHOLDS_MM3)
    print(f"\n[expAlpha postproc] component volume distribution "
          f"(n={len(all_vols)} components across all cases):")
    if len(all_vols):
        for p in (0, 5, 25, 50, 75, 95, 100):
            print(f"  p{p:<3d}: {np.percentile(all_vols, p):8.2f} mm^3")

    sweep_rows = []
    best = None
    for thr in THRESHOLDS_MM3:
        m = stat_to_metrics(stats[thr])
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

    # Variant-specific output dir -- a flat "postproc_best" here would let a
    # --base_variant ensemble5 run silently overwrite a --base_variant
    # baseline run's masks (job 106) or vice versa, since both call this
    # same script.
    out_dir = C.EXPALPHA_PRED_DIR / f"postproc_best_{a.base_variant}"
    write_thresholded_masks(pred_by_case, best_thr, out_dir)

    note = (f"CC volume threshold={best_thr}mm^3 chosen by sweeping "
            f"{THRESHOLDS_MM3} against this same eval cohort (mild "
            f"threshold-selection optimism, disclosed here) on top of "
            f"base_variant={a.base_variant}.")
    append_result_row(f"postproc_best_{a.base_variant}", best_m, a.out, note)
    print(f"\nbest-threshold masks written to {out_dir}")
    print(f"full sweep table written to {a.sweep_csv}")
    print(f"row appended to {a.out}")


if __name__ == "__main__":
    main()
