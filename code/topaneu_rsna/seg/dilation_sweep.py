"""Does growing the predicted aneurysm mask by a few voxels improve Dice,
VolSim and HD95? Jobs 134-143 showed the binary model's pooled predicted volume
is only ~58% of the ground-truth volume (job 143: pred 1.28M vs GT 2.22M voxels,
voxel recall ~0.48), i.e. it looks systematically under-segmented, which hurts
three of the seven leaderboard metrics (Dice, VolSim, HD95).

CPU only: reads the honest held-out predictions nnU-Net already wrote at
training time (fold_*/validation/, each case predicted only by the fold that
did not train on it) plus the ground-truth labels, so nothing here re-runs a
model. Note these are the single-fold, [128,256,256], full-TTA predictions, not
the v3 container's exact settings; a mask-growing gain is only trustworthy if it
shows up in the per-case mean as well as the pooled number (a few giant cases
dominate the pooled voxel sums).

Configs: "all_k" grows every component by k voxels (6-connected, k iterations);
"big_k" grows only components of at least BIG_VOX voxels.

    python -m topaneu_rsna.seg.dilation_sweep
"""
from __future__ import annotations

import argparse
import csv

import numpy as np
from scipy import ndimage
from tqdm import tqdm

from topaneu_rsna import config as C
from topaneu_rsna.utils import io as uio
from topaneu_rsna.seg.evaluate_location import hd95

BIG_VOX = 3000
PAD = 6
ALL_K = [0, 1, 2, 3, 4]
BIG_K = [1, 2, 3]
HD_CONFIGS = {"all_0", "all_1", "all_2", "all_3", "big_1", "big_2"}
CONFIGS = [f"all_{k}" for k in ALL_K] + [f"big_{k}" for k in BIG_K]


def grow(mask, k):
    for _ in range(k):
        mask = ndimage.binary_dilation(mask)
    return mask


def variants(pc):
    """All config masks inside the cropped prediction box `pc`."""
    out = {}
    cur = pc
    out["all_0"] = pc
    for k in range(1, max(ALL_K) + 1):
        cur = ndimage.binary_dilation(cur)
        out[f"all_{k}"] = cur
    lab, n = ndimage.label(pc)
    if n:
        sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
        big_ids = np.where(sizes >= BIG_VOX)[0] + 1
        big = np.isin(lab, big_ids)
    else:
        big = np.zeros_like(pc)
    rest = pc & ~big
    for k in BIG_K:
        out[f"big_{k}"] = rest | grow(big, k)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=int, default=C.DS_ANEURYSM)
    ap.add_argument("--trainer", default=C.TRAINER_LOC)
    ap.add_argument("--plans", default=C.PLANS_RESENC)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default=str(C.LOG_ROOT / "dilation_sweep.csv"))
    a = ap.parse_args()

    name = f"Dataset{a.dataset:03d}_{C.DS_NAMES[a.dataset]}"
    model_dir = C.nnUNet_results / name / f"{a.trainer}__{a.plans}__3d_fullres"
    gt_dir = C.nnUNet_raw / name / "labelsTr"

    st = {c: dict(inter=0, ps=0, gs=0, case_dice=[], hd_sum=0.0, hd_n=0) for c in CONFIGS}
    ratios = []
    for fold in a.folds:
        val_dir = model_dir / f"fold_{fold}" / "validation"
        for case in tqdm(uio.list_cases(val_dir, C.LABEL_SUFFIX), desc=f"fold {fold}"):
            gp = gt_dir / f"{case}{C.LABEL_SUFFIX}"
            if not gp.exists():
                continue
            pred, meta = uio.read(val_dir / f"{case}{C.LABEL_SUFFIX}")
            pred = pred > 0
            gt = uio.read(gp)[0] > 0
            gs = int(gt.sum())
            box = ndimage.find_objects(pred.astype(np.uint8))
            if box:
                sl = tuple(slice(max(s.start - PAD, 0), min(s.stop + PAD, d))
                           for s, d in zip(box[0], pred.shape))
                var = variants(pred[sl])
            else:
                sl, var = None, None
            if var is not None and gs > 0 and pred.sum() > 0:
                ratios.append(float(pred.sum()) / gs)
            for c in CONFIGS:
                s = st[c]
                if var is None:
                    ps = it = 0; full = None
                else:
                    m = var[c]; ps = int(m.sum()); it = int((m & gt[sl]).sum())
                    full = None
                s["inter"] += it; s["ps"] += ps; s["gs"] += gs
                if gs > 0:
                    s["case_dice"].append(2 * it / (ps + gs) if ps + gs else 0.0)
                if c in HD_CONFIGS and var is not None and ps > 0 and gs > 0:
                    full = np.zeros_like(pred)
                    full[sl] = var[c]
                    h = hd95(full, gt, meta["spacing"])
                    if h is not None:
                        s["hd_sum"] += h; s["hd_n"] += 1

    print(f"\nmedian pred/GT volume ratio (cases with both): {np.median(ratios):.2f}; "
          f"mean {np.mean(ratios):.2f}")
    print(f"{'config':<8}{'pool_dice':>10}{'volsim':>8}{'case_dice':>10}{'hd95':>8}"
          f"{'pred/gt':>9}")
    rows = []
    for c in CONFIGS:
        s = st[c]
        dice = 2 * s["inter"] / (s["ps"] + s["gs"])
        vs = 1 - abs(s["ps"] - s["gs"]) / (s["ps"] + s["gs"])
        cd = float(np.mean(s["case_dice"]))
        hd = s["hd_sum"] / s["hd_n"] if s["hd_n"] else float("nan")
        print(f"{c:<8}{dice:10.4f}{vs:8.4f}{cd:10.4f}{hd:8.2f}{s['ps'] / s['gs']:9.2f}",
              flush=True)
        rows.append([c, dice, vs, cd, hd, s["ps"] / s["gs"]])
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "pooled_dice", "volsim", "mean_case_dice", "hd95", "pred_over_gt"])
        w.writerows(rows)
    print(f"written to {a.out}")


if __name__ == "__main__":
    main()
