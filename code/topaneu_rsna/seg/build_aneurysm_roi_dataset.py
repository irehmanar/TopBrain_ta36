"""
Build Dataset305_TopAneuAneurysmROI: binary aneurysm segmentation, ROI-cropped and
vessel-conditioned. A separate dataset id from Dataset304/DS_ANEURYSM on purpose --
304's whole-head build+train (jobs 12-14) was already running when this was added,
so this targets 305 instead of overwriting it; see jobs 15-17.

Reuses exactly the geometry jobs 5/6 already computed for the ROI classifier --
Model 1's coarse crop (coarse_roi/) and Model 2's predicted vessel labels
(vessel_pred_m2/) -- so no new GPU inference is needed here, just re-packaging
into nnU-Net's raw dataset layout:

  imagesTr/<case>_0000.nii.gz   the same tight 128x256x256 crop final_roi.py
                                 uses (Model 2's predicted vessel mask + margin,
                                 centered to FINAL_ROI_SIZE), raw intensities
                                 (nnU-Net does its own normalisation)
  imagesTr/<case>_0001.nii.gz   Model 2's predicted vessel mask, binarised to
                                 "on some vessel" 0/1, cropped to the same box
                                 -- aneurysms only occur on vessels, so this
                                 channel tells the network where to look instead
                                 of it having to relocate vessels from scratch
  labelsTr/<case>.nii.gz        location_masks collapsed to binary aneurysm/
                                 background, cropped to the same box

Both the geometry and the second input channel are prediction-time-safe: at
inference the same Model-1 -> Model-2 -> crop chain (jobs 5/6) is what produces
them, so training never sees information unavailable at test time.

    python -m topaneu_rsna.seg.build_aneurysm_roi_dataset
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


def _root():
    return C.nnUNet_raw / f"Dataset{C.DS_ANEURYSM_ROI:03d}_{C.DS_NAMES[C.DS_ANEURYSM_ROI]}"


def _write_json(root, n):
    (root / "dataset.json").write_text(json.dumps({
        "channel_names": {"0": "CTA_MRA", "1": "vessel_mask"},
        "labels": {"background": 0, "aneurysm": 1},
        "numTraining": int(n),
        "file_ending": ".nii.gz",
    }, indent=2))


def process(case, roi_dir, m2_dir, root, with_labels) -> bool:
    ip = roi_dir / f"{case}{C.IMAGE_SUFFIX}"
    p2 = m2_dir / f"{case}.nii.gz"
    if not ip.exists() or not p2.exists():
        return False

    img, meta = uio.read(ip)
    v2, _ = uio.read(p2)
    v2 = geo.crop_pad(v2.astype(np.uint8), (0, 0, 0), img.shape)

    lo, hi = geo.tight_bounds(v2, C.ROI_REFINE_MARGIN_MM, C.FINE_SPACING)
    if lo is None:
        lo, hi = np.zeros(3, int), np.asarray(img.shape, int)
    lo, hi = geo.center_to_size(lo, hi, C.FINAL_ROI_SIZE)

    uio.write(geo.crop_pad(img, lo, hi).astype(np.float32), meta,
              root / "imagesTr" / f"{case}_0000.nii.gz")
    uio.write((geo.crop_pad(v2, lo, hi) > 0).astype(np.float32), meta,
              root / "imagesTr" / f"{case}_0001.nii.gz")

    if with_labels:
        lp = roi_dir / f"{case}_location{C.LABEL_SUFFIX}"
        if lp.exists():
            loc, _ = uio.read(lp)
            aneu = (geo.crop_pad(loc.astype(np.uint8), lo, hi) > 0).astype(np.uint8)
            uio.write(aneu, meta, root / "labelsTr" / f"{case}.nii.gz")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roi_dir", type=Path, default=C.COARSE_ROI_DIR)
    ap.add_argument("--m2_dir", type=Path, default=C.VESSEL_PRED_M2)
    ap.add_argument("--no_labels", action="store_true")
    a = ap.parse_args()

    root = _root()
    (root / "imagesTr").mkdir(parents=True, exist_ok=True)
    (root / "labelsTr").mkdir(parents=True, exist_ok=True)

    cases = sorted(p.name[: -len(C.IMAGE_SUFFIX)]
                   for p in a.roi_dir.glob(f"*{C.IMAGE_SUFFIX}"))
    if not cases:
        raise SystemExit(f"no coarse ROI images in {a.roi_dir} -- run job 05 first")

    n = sum(process(c, a.roi_dir, a.m2_dir, root, not a.no_labels)
           for c in tqdm(cases, desc="aneurysm-roi"))
    if n < len(cases):
        print(f"WARNING: {len(cases) - n} cases skipped (missing Model 2 prediction "
              f"in {a.m2_dir} -- run job 06 first)")

    _write_json(root, n)
    print(f"{root}  ({n} cases, 2 input channels, 1 foreground class)")


if __name__ == "__main__":
    main()
