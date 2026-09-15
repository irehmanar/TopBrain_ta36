"""
Experiment A, step 2: build Dataset313_TopAneuVesselCondSeg -- whole-head,
native per-case grid, 2-channel input, 53-class label (background + 52
locations). Mirrors build_location_conditioned_dataset.py's build_model1()/
build_gt_vessel() pattern exactly (same channel-write convention, same
geo.crop_pad shape-reconciliation call), swapping the second channel for job
78's whole-head Model 2 vessel prediction (VESSEL_PRED_M2_FULLHEAD) --
whole-head Model 2 inference has never been run before in this pipeline
(every prior use was a coarse-ROI crop or the oracle ground truth), so this
is the first dataset in this pipeline to condition on a REAL (non-oracle,
non-cropped) vessel channel.

Cases listed in EXPA_HOLDOUT_JSON's "holdout" set are excluded from
imagesTr/labelsTr entirely (so Dataset313's fold_all training never sees
them) and instead written to a separate staging folder
(expA_holdout_staging/) with the same 2-channel convention, for job 83's
prediction pass.

Run seg/build_expA_holdout.py first if EXPA_HOLDOUT_JSON doesn't exist yet.

    python -m topaneu_rsna.seg.build_expA_vessel_cond_dataset
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from tqdm import tqdm

from topaneu_rsna import config as C
from topaneu_rsna.utils import geometry as geo
from topaneu_rsna.utils import io as uio


def _root() -> "Path":
    return C.nnUNet_raw / f"Dataset{C.DS_VESSELCOND_SEG:03d}_{C.DS_NAMES[C.DS_VESSELCOND_SEG]}"


def _write_case(img_dir, case, img, meta, cond, lab=None):
    cond = geo.crop_pad(cond.astype(np.float32), (0, 0, 0), img.shape)
    uio.write(img.astype(np.float32), meta, img_dir / f"{case}_0000.nii.gz")
    uio.write(cond, meta, img_dir / f"{case}_0001.nii.gz")
    if lab is not None:
        (img_dir.parent / "labelsTr").mkdir(parents=True, exist_ok=True)
        uio.write(lab.astype(np.uint8), meta, img_dir.parent / "labelsTr" / f"{case}.nii.gz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    if not C.EXPA_HOLDOUT_JSON.exists():
        raise FileNotFoundError(
            f"{C.EXPA_HOLDOUT_JSON} missing. Run: "
            f"python -m topaneu_rsna.seg.build_expA_holdout")
    split = json.loads(C.EXPA_HOLDOUT_JSON.read_text())
    train_cases, holdout_cases = set(split["train"]), set(split["holdout"])

    spec = C.load_labels()
    root = _root()
    (root / "imagesTr").mkdir(parents=True, exist_ok=True)
    (root / "labelsTr").mkdir(parents=True, exist_ok=True)
    holdout_dir = C.WORK / "expA_holdout_staging"
    holdout_dir.mkdir(parents=True, exist_ok=True)

    cases = uio.list_cases(C.IMAGES_DIR, C.IMAGE_SUFFIX)
    if a.limit:
        cases = cases[:a.limit]

    n_train, n_holdout, n_skipped = 0, 0, 0
    for case in tqdm(cases, desc="expA-vessel-cond"):
        img_p = C.IMAGES_DIR / f"{case}{C.IMAGE_SUFFIX}"
        cond_p = C.VESSEL_PRED_M2_FULLHEAD / f"{case}.nii.gz"
        lab_p = C.LOCATION_MASKS / f"{case}{C.LABEL_SUFFIX}"
        if not (img_p.exists() and cond_p.exists() and lab_p.exists()):
            n_skipped += 1
            continue

        img, meta = uio.read(img_p)
        cond, _ = uio.read(cond_p)

        if case in train_cases:
            lab, _ = uio.read(lab_p)
            _write_case(root / "imagesTr", case, img, meta, cond, lab)
            n_train += 1
        elif case in holdout_cases:
            _write_case(holdout_dir, case, img, meta, cond, lab=None)
            n_holdout += 1
        else:
            n_skipped += 1   # case exists but isn't in either half of the split

    labels = {"background": 0}
    labels.update({loc: i + 1 for i, loc in enumerate(spec.locations)})
    (root / "dataset.json").write_text(json.dumps({
        "channel_names": {"0": "CTA_MRA", "1": "model2_vessel_prediction_fullhead"},
        "labels": labels,
        "numTraining": int(n_train),
        "file_ending": ".nii.gz",
    }, indent=2))

    print(f"{root}  ({n_train} training cases, {len(labels) - 1} classes)")
    print(f"{holdout_dir}  ({n_holdout} holdout cases staged for job 83's prediction)")
    if n_skipped:
        print(f"[warn] {n_skipped} cases skipped (missing image/vessel-pred/label, "
             f"or absent from expA_holdout_cases.json's split)")


if __name__ == "__main__":
    main()
