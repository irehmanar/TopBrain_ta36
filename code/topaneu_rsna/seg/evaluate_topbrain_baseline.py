"""
TopBrain 2026 TA36 fine-tune, "before" number: run Model 2's ORIGINAL,
never-fine-tuned checkpoint (Dataset302, TRAINER_M2, fold_all) on the exact
same held-out TopBrain cases that build_topbrain_finetune_dataset.py carved
out, and score it against their v2 (36-class) ground truth.

Model 2 was never trained or validated on these cases (they're a genuinely
independent cohort from the organizers), so this is the real baseline the
fine-tuned run (jobs/41_topbrain_finetune's training job, whose own
post-training validation/summary.json gives the "after" number on the SAME
5 cases via splits_final.json) needs to beat -- without this, "fine-tuning
helped" would be an assumption, not a measurement.

    python -m topaneu_rsna.seg.evaluate_topbrain_baseline
"""
from __future__ import annotations

import argparse
import csv
import json

import numpy as np
import SimpleITK as sitk
import torch

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

from topaneu_rsna import config as C

# Same OOM-avoidance ladder proven in docker/task2_rule_based/inference.py and
# docker/TopBrain_Algo_Submission/inference.py for this exact architecture.
_PATCH_LADDER = ([96, 192, 192], [64, 160, 160], [64, 128, 128], [48, 96, 96])


def _sitk_to_nnunet_input(img: sitk.Image):
    arr = sitk.GetArrayFromImage(img).astype(np.float32)[None]
    spacing_zyx = tuple(img.GetSpacing()[::-1])
    return arr, {"spacing": spacing_zyx}


def _predict_with_oom_fallback(predictor, arr, props):
    last_err = None
    for patch in _PATCH_LADDER:
        predictor.configuration_manager.configuration["patch_size"] = list(patch)
        try:
            return predictor.predict_single_npy_array(arr, props)
        except torch.OutOfMemoryError as e:
            last_err = e
            print(f"[patch] OOM at patch_size={patch}, shrinking", flush=True)
            torch.cuda.empty_cache()
    raise last_err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_id", type=int, default=C.DS_TOPBRAIN_FINETUNE)
    ap.add_argument("--fold_idx", type=int, default=0,
                    help="which entry of splits_final.json to read the held-out ('val') "
                         "case list from")
    ap.add_argument("--out", type=type(C.LOG_ROOT), default=C.LOG_ROOT / "topbrain_baseline_dice.csv")
    a = ap.parse_args()

    dataset_name = f"Dataset{a.dataset_id:03d}_{C.DS_NAMES[a.dataset_id]}"
    raw_dir = C.nnUNet_raw / dataset_name
    splits_path = C.nnUNet_preprocessed / dataset_name / "splits_final.json"
    splits = json.loads(splits_path.read_text())
    held_out = splits[a.fold_idx]["val"]
    print(f"[baseline] {len(held_out)} held-out cases from {splits_path}: {held_out}")

    m2_dir = C.seg_model_dir(C.DS_VESSEL, C.TRAINER_M2)
    print(f"[baseline] loading Model 2's ORIGINAL checkpoint from {m2_dir}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    predictor = nnUNetPredictor(
        tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
        perform_everything_on_device=False,
        device=device, verbose=False, verbose_preprocessing=False, allow_tqdm=True)
    predictor.initialize_from_trained_model_folder(
        str(m2_dir), use_folds=("all",), checkpoint_name="checkpoint_final.pth")

    n_cls = 36
    inter = np.zeros(n_cls); pred_sum = np.zeros(n_cls); gt_sum = np.zeros(n_cls)

    rows = []
    for case in held_out:
        img = sitk.ReadImage(str(raw_dir / "imagesTr" / f"{case}_0000.nii.gz"))
        gt_img = sitk.ReadImage(str(raw_dir / "labelsTr" / f"{case}.nii.gz"))
        gt = sitk.GetArrayFromImage(gt_img).astype(np.uint8)

        arr, props = _sitk_to_nnunet_input(img)
        pred = _predict_with_oom_fallback(predictor, arr, props)

        case_dice = []
        for c in range(1, n_cls + 1):
            pm, gm = pred == c, gt == c
            pn, gn = int(pm.sum()), int(gm.sum())
            it = int((pm & gm).sum())
            inter[c - 1] += it; pred_sum[c - 1] += pn; gt_sum[c - 1] += gn
            d = (2 * it / (pn + gn)) if (pn + gn) > 0 else float("nan")
            case_dice.append(d)
        rows.append([case] + case_dice)
        print(f"[baseline] {case}: mean dice over present classes = "
             f"{np.nanmean(case_dice):.4f}")

    with np.errstate(invalid="ignore", divide="ignore"):
        pooled_dice = 2 * inter / (pred_sum + gt_sum)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case"] + [f"class_{c}" for c in range(1, n_cls + 1)])
        w.writerows(rows)
        w.writerow(["POOLED"] + list(pooled_dice))

    print(f"[baseline] pooled per-class dice: {np.round(pooled_dice, 4).tolist()}")
    print(f"[baseline] pooled mean dice (36 classes): {np.nanmean(pooled_dice):.4f}")
    print(f"[baseline] written to {a.out}")


if __name__ == "__main__":
    main()
