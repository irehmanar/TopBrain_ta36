"""
Task 2, 52-class location segmentation, conditioned on an existing vessel
model's own prediction as a second input channel -- revisiting the direct
52-class attempt that collapsed as Dataset303, this time with two changes:

  1. a second input channel carrying a vessel/vessel-group model's own
     prediction, since aneurysms only occur on vessels;
  2. training initialized from that same model's trained weights
     (nnUNetv2_train -pretrained_weights) instead of random init -- the same
     transfer-learning lever that took the ROI classifier from 0.794 to 0.902
     AUC when pretrained from Model 2.

Two variants, both built entirely from job 05/06's *existing* outputs -- no
new segmentation inference needed either way:

  --source model1  Dataset307, whole-head, native per-case grid.
                    ch0 = raw image (IMAGES_DIR)
                    ch1 = Model 1's 3-class vessel-group prediction
                          (COARSE_PRED_DIR, job 05 -- already on this grid,
                          since Dataset301's imagesTr are symlinks to
                          IMAGES_DIR and nnU-Net predicts back onto the
                          input's own grid)
                    label = LOCATION_MASKS (whole-head, 52 classes)

  --source model2  Dataset308, coarse-ROI-cropped grid (140mm cube,
                    FINE_SPACING).
                    ch0 = COARSE_ROI_DIR/<case>_0000.nii.gz (job 05's crop)
                    ch1 = VESSEL_PRED_M2/<case>.nii.gz (job 06's Model 2
                          prediction, computed directly on that same crop)
                    label = COARSE_ROI_DIR/<case>_location.nii.gz (job 05's
                          matching cropped location mask)

Note this doesn't by itself fix Dataset303's diagnosed root cause (most
training patches contain at most one of 52 classes -- too sparse for the
loss to escape predicting all-background); channel-conditioning and warm-
starting give the network anatomical priors and a head start, but if the
collapse persists, pair this with class-balanced case sampling next.

    python -m topaneu_rsna.seg.build_location_conditioned_dataset --source model1
    python -m topaneu_rsna.seg.build_location_conditioned_dataset --source model2

Then train with, e.g.:
    nnUNetv2_train 307 3d_fullres all -p nnUNetResEncUNetMPlans \\
        -tr RSNA2025Trainer_moreDAv7 -pretrained_weights <Model1 checkpoint_final.pth>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from topaneu_rsna import config as C
from topaneu_rsna.utils import geometry as geo
from topaneu_rsna.utils import io as uio

DS_ID = {"model1": C.DS_LOCATION_M1COND, "model2": C.DS_LOCATION_M2COND}


def pretrained_checkpoint(source: str) -> Path:
    if source == "model1":
        return C.seg_model_dir(C.DS_COARSE, C.TRAINER_M1) / "fold_all" / "checkpoint_final.pth"
    return C.seg_model_dir(C.DS_VESSEL, C.TRAINER_M2) / "fold_all" / "checkpoint_final.pth"


def _root(ds_id: int) -> Path:
    return C.nnUNet_raw / f"Dataset{ds_id:03d}_{C.DS_NAMES[ds_id]}"


def build_model1(root: Path, limit: int | None):
    cases = uio.list_cases(C.IMAGES_DIR, C.IMAGE_SUFFIX)
    if limit:
        cases = cases[:limit]
    n = 0
    for case in tqdm(cases, desc="model1-conditioned"):
        img_p = C.IMAGES_DIR / f"{case}{C.IMAGE_SUFFIX}"
        cond_p = C.COARSE_PRED_DIR / f"{case}.nii.gz"
        lab_p = C.LOCATION_MASKS / f"{case}{C.LABEL_SUFFIX}"
        if not (img_p.exists() and cond_p.exists() and lab_p.exists()):
            continue

        img, meta = uio.read(img_p)
        cond, _ = uio.read(cond_p)
        cond = geo.crop_pad(cond.astype(np.uint8), (0, 0, 0), img.shape)
        lab, _ = uio.read(lab_p)

        uio.write(img.astype(np.float32), meta, root / "imagesTr" / f"{case}_0000.nii.gz")
        uio.write(cond.astype(np.float32), meta, root / "imagesTr" / f"{case}_0001.nii.gz")
        uio.write(lab.astype(np.uint8), meta, root / "labelsTr" / f"{case}.nii.gz")
        n += 1
    return n


def build_model2(root: Path, limit: int | None):
    cases = uio.list_cases(C.COARSE_ROI_DIR, C.IMAGE_SUFFIX)
    if limit:
        cases = cases[:limit]
    n = 0
    for case in tqdm(cases, desc="model2-conditioned"):
        img_p = C.COARSE_ROI_DIR / f"{case}{C.IMAGE_SUFFIX}"
        cond_p = C.VESSEL_PRED_M2 / f"{case}.nii.gz"
        lab_p = C.COARSE_ROI_DIR / f"{case}_location{C.LABEL_SUFFIX}"
        if not (img_p.exists() and cond_p.exists() and lab_p.exists()):
            continue

        img, meta = uio.read(img_p)
        cond, _ = uio.read(cond_p)
        cond = geo.crop_pad(cond.astype(np.uint8), (0, 0, 0), img.shape)
        lab, _ = uio.read(lab_p)
        lab = geo.crop_pad(lab.astype(np.uint8), (0, 0, 0), img.shape)

        uio.write(img.astype(np.float32), meta, root / "imagesTr" / f"{case}_0000.nii.gz")
        uio.write(cond.astype(np.float32), meta, root / "imagesTr" / f"{case}_0001.nii.gz")
        uio.write(lab.astype(np.uint8), meta, root / "labelsTr" / f"{case}.nii.gz")
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["model1", "model2"], required=True)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    spec = C.load_labels()
    ds_id = DS_ID[a.source]
    root = _root(ds_id)
    (root / "imagesTr").mkdir(parents=True, exist_ok=True)
    (root / "labelsTr").mkdir(parents=True, exist_ok=True)

    n = build_model1(root, a.limit) if a.source == "model1" else build_model2(root, a.limit)

    labels = {"background": 0}
    labels.update({loc: i + 1 for i, loc in enumerate(spec.locations)})
    (root / "dataset.json").write_text(json.dumps({
        "channel_names": {"0": "CTA_MRA", "1": f"{a.source}_prediction"},
        "labels": labels,
        "numTraining": int(n),
        "file_ending": ".nii.gz",
    }, indent=2))

    ck = pretrained_checkpoint(a.source)
    print(f"{root}  ({n} cases, {len(labels) - 1} classes)")
    print(f"pretrained checkpoint for -pretrained_weights: {ck}"
         + ("" if ck.exists() else "  !! NOT FOUND YET"))


if __name__ == "__main__":
    main()
