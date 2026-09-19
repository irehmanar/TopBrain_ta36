"""Quantify the real accuracy cost of what the Docker submission had to change
to fit the T4's 16GB VRAM: patch_size [128,256,256] -> [96,192,192], plus TTA
(mirroring) off. The existing ~0.60 pooled-Dice baseline (job 14 / expAlpha's
notta comparison, see 33_expAlpha_binary_postproc/README.md) already
establishes the honest, non-leaked methodology: each fold predicts ONLY its
own held-out validation cases with ONLY that fold's model, so no case is ever
seen by the model that predicts it. This script follows the exact same
protocol, just calling nnUNetPredictor directly (not the nnUNetv2_predict
CLI, which has no way to override patch_size) so the deployed inference.py's
own patch_size/use_mirroring settings can be tested. Metrics reproduce
evaluate_location.py's formulas exactly (case-level TP/FP/FN/TN -> precision/
recall/MCC, pooled-voxel Dice/VS, mean HD95) so the printed numbers are
directly comparable to the documented ~0.60 baseline and to the grand-
challenge leaderboard's own metric definitions.

Runs TWO settings back to back on the identical case set for a fair,
paired comparison:
  --patch 128 256 256 --mirror   (matches the existing ~0.60 TTA-on baseline)
  --patch  96 192 192 --no-mirror (matches what's actually deployed in
                                    docker/task2_rule_based/inference.py)

    python -m topaneu_rsna.seg.patch_size_ablation
"""
from __future__ import annotations

import argparse
import csv

import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt
from tqdm import tqdm

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

from topaneu_rsna import config as C
from topaneu_rsna.utils import io as uio

HD95_PAD_VOX = 20


def _bbox(mask, pad, shape):
    idx = np.argwhere(mask)
    lo = np.maximum(idx.min(0) - pad, 0)
    hi = np.minimum(idx.max(0) + pad + 1, shape)
    return tuple(slice(a, b) for a, b in zip(lo, hi))


def hd95(pred, gt, spacing):
    union = pred | gt
    if not union.any():
        return None
    sl = _bbox(union, HD95_PAD_VOX, pred.shape)
    p, g = pred[sl], gt[sl]
    if not p.any() or not g.any():
        return None
    p_surf = p & ~binary_erosion(p)
    g_surf = g & ~binary_erosion(g)
    dt_g = distance_transform_edt(~g, sampling=spacing)
    dt_p = distance_transform_edt(~p, sampling=spacing)
    d = np.concatenate([dt_g[p_surf], dt_p[g_surf]])
    return float(np.percentile(d, 95))


def run_one_setting(dataset_id: int, trainer: str, plans: str, folds: list[int],
                    patch_size: list[int], use_mirroring: bool, label: str,
                    allowed_axes=None, tile_step: float = 0.5):
    model_dir = (C.nnUNet_results / f"Dataset{dataset_id:03d}_{C.DS_NAMES[dataset_id]}"
                / f"{trainer}__{plans}__3d_fullres")
    images_dir = C.nnUNet_raw / f"Dataset{dataset_id:03d}_{C.DS_NAMES[dataset_id]}" / "imagesTr"
    gt_dir = C.nnUNet_raw / f"Dataset{dataset_id:03d}_{C.DS_NAMES[dataset_id]}" / "labelsTr"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tp = fp = fn = tn = inter = pred_sum = gt_sum = 0
    hd_sum, hd_n, n_cases = 0.0, 0, 0

    for fold in folds:
        val_dir = model_dir / f"fold_{fold}" / "validation"
        cases = uio.list_cases(val_dir, C.LABEL_SUFFIX)
        if not cases:
            print(f"[{label}] fold {fold}: no validation cases found, skipping")
            continue

        predictor = nnUNetPredictor(
            tile_step_size=tile_step, use_gaussian=True, use_mirroring=use_mirroring,
            perform_everything_on_device=False,
            device=device, verbose=False, verbose_preprocessing=False, allow_tqdm=False)
        predictor.initialize_from_trained_model_folder(
            str(model_dir), use_folds=(fold,), checkpoint_name="checkpoint_final.pth")
        predictor.configuration_manager.configuration["patch_size"] = list(patch_size)
        if allowed_axes is not None:
            predictor.allowed_mirroring_axes = tuple(allowed_axes)

        for case in tqdm(cases, desc=f"[{label}] fold {fold}"):
            gp = gt_dir / f"{case}{C.LABEL_SUFFIX}"
            if not gp.exists():
                continue
            img_arr, meta = uio.read(images_dir / f"{case}{C.IMAGE_SUFFIX}")
            gt, _ = uio.read(gp)
            gt = gt.astype(bool)

            props = {"spacing": meta["spacing"]}
            pred = predictor.predict_single_npy_array(
                img_arr[None].astype(np.float32), props) > 0

            n_cases += 1
            pn, gn = int(pred.sum()), int(gt.sum())
            it = int((pred & gt).sum())
            inter += it; pred_sum += pn; gt_sum += gn
            if gn > 0 and it > 0:
                tp += 1
            elif gn > 0 and it == 0:
                fn += 1
            elif gn == 0 and pn > 0:
                fp += 1
            else:
                tn += 1
            if pn > 0 and gn > 0:
                h = hd95(pred, gt, meta["spacing"])
                if h is not None:
                    hd_sum += h; hd_n += 1

        del predictor
        torch.cuda.empty_cache()

        run_dice = 2 * inter / (pred_sum + gt_sum) if (pred_sum + gt_sum) else float("nan")
        print(f"[{label}] after fold {fold}: cases={n_cases} dice={run_dice:.4f} "
              f"tp={tp} fp={fp} fn={fn} tn={tn} inter={inter} pred_sum={pred_sum} "
              f"gt_sum={gt_sum}", flush=True)

    with np.errstate(invalid="ignore", divide="ignore"):
        dice = 2 * inter / (pred_sum + gt_sum) if (pred_sum + gt_sum) else float("nan")
        vs = 1 - abs(pred_sum - gt_sum) / (pred_sum + gt_sum) if (pred_sum + gt_sum) else float("nan")
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = (tp * tn - fp * fn) / mcc_den if mcc_den else float("nan")
        hd = hd_sum / hd_n if hd_n else float("nan")

    return dict(label=label, n_cases=n_cases, patch_size=patch_size, mirror=use_mirroring,
               dice=dice, vs=vs, hd95=hd, precision=precision, recall=recall, mcc=mcc,
               tp=tp, fp=fp, fn=fn, tn=tn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=int, default=C.DS_ANEURYSM)
    ap.add_argument("--trainer", default=C.TRAINER_LOC)
    ap.add_argument("--plans", default=C.PLANS_RESENC)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--settings", choices=["both", "baseline", "deployed", "mirror_x", "p112", "p112_mirror_x",
                             "p112_step033", "p112_mirror_xy"],
                    default="both")
    a = ap.parse_args()

    out = a.out or str(C.LOG_ROOT / f"patch_size_ablation_{a.settings}.csv")

    settings = [
        dict(patch_size=[128, 256, 256], use_mirroring=True, label="baseline_128_256_256_mirror"),
        dict(patch_size=[96, 192, 192], use_mirroring=False, label="deployed_96_192_192_nomirror"),
    ]
    if a.settings == "baseline":
        settings = settings[:1]
    elif a.settings == "deployed":
        settings = settings[1:]
    elif a.settings == "mirror_x":
        # axis index 2 = last spatial axis = left-right; nnU-Net's own mirror
        # list is [z, y, x], so (2,) gives exactly one flip (2x compute).
        settings = [dict(patch_size=[96, 192, 192], use_mirroring=True,
                         allowed_axes=(2,), label="p96_mirror_x_only")]
    elif a.settings == "p112_step033":
        settings = [dict(patch_size=[112, 224, 224], use_mirroring=True, allowed_axes=(2,),
                         tile_step=0.33, label="p112_mirror_x_step033")]
    elif a.settings == "p112_mirror_xy":
        # 4x compute (flips over y, x and both): a ceiling for what more test-
        # time mirroring could give, not deployable on the T4 within 12 minutes
        settings = [dict(patch_size=[112, 224, 224], use_mirroring=True, allowed_axes=(1, 2),
                         label="p112_mirror_xy")]
    elif a.settings == "p112_mirror_x":
        settings = [dict(patch_size=[112, 224, 224], use_mirroring=True,
                         allowed_axes=(2,), label="p112_mirror_x")]
    elif a.settings == "p112":
        settings = [dict(patch_size=[112, 224, 224], use_mirroring=False,
                         label="p112_224_224_nomirror")]

    rows = []
    for s in settings:
        r = run_one_setting(a.dataset, a.trainer, a.plans, a.folds, **s)
        rows.append(r)
        print(f"\n=== {r['label']}: {r['n_cases']} cases, patch={r['patch_size']}, "
              f"mirror={r['mirror']} ===")
        for k in ("dice", "vs", "hd95", "precision", "recall", "mcc"):
            print(f"  {k:<10} {r[k]:.4f}")
        print(f"  tp={r['tp']} fp={r['fp']} fn={r['fn']} tn={r['tn']}")

    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "n_cases", "patch_size", "mirror", "dice", "vs", "hd95",
                    "precision", "recall", "mcc", "tp", "fp", "fn", "tn"])
        for r in rows:
            w.writerow([r["label"], r["n_cases"], r["patch_size"], r["mirror"], r["dice"],
                       r["vs"], r["hd95"], r["precision"], r["recall"], r["mcc"],
                       r["tp"], r["fp"], r["fn"], r["tn"]])

    if len(rows) == 2:
        print(f"\ndelta dice (deployed - baseline): {rows[1]['dice'] - rows[0]['dice']:+.4f}")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
