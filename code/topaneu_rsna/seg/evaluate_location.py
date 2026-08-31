"""
Task 2 scoring: aneurysm segmentation.

Works against Dataset304_TopAneuAneurysm (binary, whole-head, --dataset 304,
the default), Dataset305_TopAneuAneurysmROI (binary, ROI-cropped + vessel
channel, --dataset 305), Dataset306_TopAneuAneurysmVesselness (binary,
whole-head + Frangi vesselness channel, --dataset 306), or, for reference,
the paused 52-class Dataset303_TopAneuLocation via --dataset 303.

Stitches together the 5 folds' nnU-Net validation predictions -- together they
cover every training case exactly once as held-out data, so this is an honest
whole-cohort estimate of what the grand-challenge leaderboard will show.

Reproduces the metrics documented at https://topaneu-26.grand-challenge.org/evaluation/
per class:
  DSC   = 2*|A&B| / (|A|+|B|)                              voxels pooled over all cases
  VS    = 1 - |A|-|B|| / (|A|+|B|)                          same pooling
  HD95  = 95th percentile bidirectional surface distance    averaged over cases
          where both prediction and ground truth are non-empty
  Precision / Recall / MCC computed from case-level TP/FP/FN, where "TP" is a
  case with non-zero overlap between predicted and ground-truth masks for
  that class, FN a case where ground truth is present but missed entirely,
  and FP a case where the class was predicted but no ground truth exists.

    python -m topaneu_rsna.seg.evaluate_location
    python -m topaneu_rsna.seg.evaluate_location --dataset 303   # reference only
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt
from tqdm import tqdm

from topaneu_rsna import config as C
from topaneu_rsna.utils import io as uio

HD95_PAD_VOX = 20  # crop margin around the pred/gt union before running the EDT


def _bbox(mask, pad, shape):
    idx = np.argwhere(mask)
    lo = np.maximum(idx.min(0) - pad, 0)
    hi = np.minimum(idx.max(0) + pad + 1, shape)
    return tuple(slice(a, b) for a, b in zip(lo, hi))


def hd95(pred, gt, spacing):
    """95th-percentile symmetric surface distance in mm, or None if undefined."""
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


def class_names_for(dataset_id: int) -> list[str]:
    if dataset_id in (C.DS_ANEURYSM, C.DS_ANEURYSM_ROI, C.DS_ANEURYSM_VESSELNESS):
        return ["aneurysm"]
    if dataset_id == C.DS_LOCATION:
        return C.load_labels().locations
    raise ValueError(f"no known class list for dataset {dataset_id}")


def evaluate(dataset_id: int, trainer: str, plans: str, folds: list[int],
            class_names: list[str]):
    gt_dir = C.nnUNet_raw / f"Dataset{dataset_id:03d}_{C.DS_NAMES[dataset_id]}" / "labelsTr"
    model_dir = (C.nnUNet_results / f"Dataset{dataset_id:03d}_{C.DS_NAMES[dataset_id]}"
                 / f"{trainer}__{plans}__3d_fullres")

    n_cls = len(class_names)
    inter = np.zeros(n_cls); pred_sum = np.zeros(n_cls); gt_sum = np.zeros(n_cls)
    tp = np.zeros(n_cls); fp = np.zeros(n_cls); fn = np.zeros(n_cls); tn = np.zeros(n_cls)
    hd_sum = np.zeros(n_cls); hd_n = np.zeros(n_cls)

    n_cases = 0
    for fold in folds:
        val_dir = model_dir / f"fold_{fold}" / "validation"
        cases = uio.list_cases(val_dir, C.LABEL_SUFFIX)
        for case in tqdm(cases, desc=f"fold {fold}"):
            gp = gt_dir / f"{case}{C.LABEL_SUFFIX}"
            if not gp.exists():
                continue
            pred, pmeta = uio.read(val_dir / f"{case}.nii.gz")
            gt, _ = uio.read(gp)
            n_cases += 1
            for c in range(1, n_cls + 1):
                pm, gm = pred == c, gt == c
                pn, gn = int(pm.sum()), int(gm.sum())
                it = int((pm & gm).sum())
                inter[c - 1] += it; pred_sum[c - 1] += pn; gt_sum[c - 1] += gn

                if gn > 0 and it > 0:
                    tp[c - 1] += 1
                elif gn > 0 and it == 0:
                    fn[c - 1] += 1
                elif gn == 0 and pn > 0:
                    fp[c - 1] += 1
                else:
                    tn[c - 1] += 1

                if pn > 0 and gn > 0:
                    h = hd95(pm, gm, pmeta["spacing"])
                    if h is not None:
                        hd_sum[c - 1] += h; hd_n[c - 1] += 1

    with np.errstate(invalid="ignore", divide="ignore"):
        dice = 2 * inter / (pred_sum + gt_sum)
        vs = 1 - np.abs(pred_sum - gt_sum) / (pred_sum + gt_sum)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        mcc_num = tp * tn - fp * fn
        mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = mcc_num / mcc_den
        hd = hd_sum / hd_n

    return n_cases, dict(dice=dice, vs=vs, hd95=hd, precision=precision,
                         recall=recall, mcc=mcc, tp=tp, fp=fp, fn=fn, tn=tn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=int, default=C.DS_ANEURYSM)
    ap.add_argument("--trainer", default=C.TRAINER_LOC)
    ap.add_argument("--plans", default=C.PLANS_RESENC)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    class_names = class_names_for(a.dataset)
    out = a.out or C.LOG_ROOT / f"seg_eval_{C.DS_NAMES[a.dataset]}.csv"

    n_cases, m = evaluate(a.dataset, a.trainer, a.plans, a.folds, class_names)
    print(f"pooled {n_cases} held-out cases across folds {a.folds}\n")

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "dice", "vs", "hd95_mm", "precision", "recall", "mcc",
                    "tp", "fp", "fn", "tn"])
        for i, name in enumerate(class_names):
            w.writerow([name, m["dice"][i], m["vs"][i], m["hd95"][i], m["precision"][i],
                        m["recall"][i], m["mcc"][i], int(m["tp"][i]), int(m["fp"][i]),
                        int(m["fn"][i]), int(m["tn"][i])])

    def nanmean(x):
        return float(np.nanmean(x))

    print(f"{'metric':<12}{'mean over classes':>20}")
    for k in ("dice", "vs", "hd95", "precision", "recall", "mcc"):
        print(f"{k:<12}{nanmean(m[k]):>20.4f}")
    print(f"\nper-class detail written to {out}")


if __name__ == "__main__":
    main()
