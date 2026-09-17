"""
Experiment Delta, step 3: score the two modality-specific models (CTA-only,
MRA-only) and pool them back into ONE combined Dice, directly comparable to
Dataset304's existing single-merged-model baseline (~0.60) on the same
overall cohort (every case scored exactly once, by whichever fold of its
own modality's model held it out -- same pooling methodology
evaluate_location.py already uses for Dataset304 itself).

Reports each modality's own number separately too (diagnostic: the split
might help one modality much more than the other, or even hurt one of them
if its own cohort is too small for a stable 5-fold CV).

    python -m topaneu_rsna.seg.evaluate_modality_split
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from tqdm import tqdm

from topaneu_rsna import config as C
from topaneu_rsna.seg.assign_location_rule import load_binary_pred_paths
from topaneu_rsna.utils import io as uio


def voxel_sums(pred_by_case: dict, gt_dir: Path):
    """Same binary metric definitions as evaluate_location.py, but also
    returns the raw pooled sums (inter/pred_sum/gt_sum/tp/fp/fn/tn) so two
    disjoint cohorts (CTA cases, MRA cases) can be combined into one true
    pooled Dice afterward, not just averaged after the fact."""
    inter = pred_sum = gt_sum = 0
    tp = fp = fn = tn = 0
    per_case_dice = []
    for case, pp in tqdm(sorted(pred_by_case.items()), desc=f"scoring {gt_dir.parent.name}"):
        gp = gt_dir / f"{case}{C.LABEL_SUFFIX}"
        if not gp.exists():
            continue
        pred, _ = uio.read(pp)
        gt, _ = uio.read(gp)
        pm, gm = pred > 0, gt > 0
        pn, gn = int(pm.sum()), int(gm.sum())
        it = int((pm & gm).sum())
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
    return dict(inter=inter, pred_sum=pred_sum, gt_sum=gt_sum,
               tp=tp, fp=fp, fn=fn, tn=tn, per_case_dice=per_case_dice)


def rates(s: dict) -> dict:
    with np.errstate(invalid="ignore", divide="ignore"):
        dice = 2 * s["inter"] / (s["pred_sum"] + s["gt_sum"]) if (s["pred_sum"] + s["gt_sum"]) else float("nan")
        iou = s["inter"] / (s["pred_sum"] + s["gt_sum"] - s["inter"]) if (s["pred_sum"] + s["gt_sum"] - s["inter"]) else float("nan")
        precision = s["tp"] / (s["tp"] + s["fp"]) if (s["tp"] + s["fp"]) else float("nan")
        recall = s["tp"] / (s["tp"] + s["fn"]) if (s["tp"] + s["fn"]) else float("nan")
    dice_arr = np.asarray(s["per_case_dice"])
    return dict(
        n_cases=len(dice_arr), pooled_dice=dice, pooled_iou=iou,
        precision=precision, recall=recall,
        tp=s["tp"], fp=s["fp"], fn=s["fn"], tn=s["tn"],
        case_dice_mean=float(dice_arr.mean()) if len(dice_arr) else float("nan"),
        case_dice_median=float(np.median(dice_arr)) if len(dice_arr) else float("nan"),
        case_dice_std=float(dice_arr.std()) if len(dice_arr) else float("nan"),
    )


def combine(a: dict, b: dict) -> dict:
    return dict(
        inter=a["inter"] + b["inter"],
        pred_sum=a["pred_sum"] + b["pred_sum"],
        gt_sum=a["gt_sum"] + b["gt_sum"],
        tp=a["tp"] + b["tp"], fp=a["fp"] + b["fp"],
        fn=a["fn"] + b["fn"], tn=a["tn"] + b["tn"],
        per_case_dice=a["per_case_dice"] + b["per_case_dice"],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trainer", default=C.TRAINER_LOC)
    ap.add_argument("--plans", default=C.PLANS_RESENC)
    ap.add_argument("--out", type=Path,
                    default=C.LOG_ROOT / "task2_expDelta_modality_split_results.csv")
    a = ap.parse_args()

    cta_gt = C.nnUNet_raw / f"Dataset{C.DS_ANEURYSM_CTA:03d}_{C.DS_NAMES[C.DS_ANEURYSM_CTA]}" / "labelsTr"
    mra_gt = C.nnUNet_raw / f"Dataset{C.DS_ANEURYSM_MRA:03d}_{C.DS_NAMES[C.DS_ANEURYSM_MRA]}" / "labelsTr"

    cta_preds = load_binary_pred_paths(C.DS_ANEURYSM_CTA, a.trainer, a.plans, folds=range(5))
    mra_preds = load_binary_pred_paths(C.DS_ANEURYSM_MRA, a.trainer, a.plans, folds=range(5))
    print(f"[expDelta] CTA: {len(cta_preds)} predicted cases  |  "
          f"MRA: {len(mra_preds)} predicted cases")

    cta_sums = voxel_sums(cta_preds, cta_gt)
    mra_sums = voxel_sums(mra_preds, mra_gt)
    combined_sums = combine(cta_sums, mra_sums)

    results = {
        "cta_only": rates(cta_sums),
        "mra_only": rates(mra_sums),
        "combined_modality_split": rates(combined_sums),
    }

    a.out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["variant", "n_cases", "pooled_dice", "pooled_iou", "precision", "recall",
              "tp", "fp", "fn", "tn", "case_dice_mean", "case_dice_median", "case_dice_std"]
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for name, m in results.items():
            row = dict(m); row["variant"] = name
            w.writerow({k: row.get(k, "") for k in fields})

    print(f"\n{'variant':<28}{'pooled_dice':>12}{'precision':>12}{'recall':>12}{'n_cases':>10}")
    for name, m in results.items():
        print(f"{name:<28}{m['pooled_dice']:>12.4f}{m['precision']:>12.4f}"
              f"{m['recall']:>12.4f}{m['n_cases']:>10d}")

    print(f"\nCompare 'combined_modality_split' pooled_dice above directly against "
          f"Dataset304's existing baseline (~0.60, see task2_seg_eval_aneurysm.csv / "
          f"the pooled number already documented in this pipeline) -- same cohort, "
          f"same pooling methodology, only difference is one merged model vs two "
          f"modality-specific models.")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
