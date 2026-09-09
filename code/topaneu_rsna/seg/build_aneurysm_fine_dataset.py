"""
Fine stage of the dedicated aneurysm-segmentation cascade (see config.py
DS_ANEURYSM_COARSE / DS_ANEURYSM_FINE_RAW / DS_ANEURYSM_FINE_VESSEL).

Consumes seg/aneurysm_coarse_roi.py's crop -- already at FINE_SPACING,
already cropped, no geometry recomputed here. Two variants:

  --with_vessel_channel absent  Dataset310 (Experiment A): ch0 = cropped
                                 image only.
  --with_vessel_channel present Dataset311 (Experiment B): ch0 = cropped
                                 image, ch1 = Model 2's vessel prediction on
                                 this same crop (ANEURYSM_VESSEL_PRED_M2 --
                                 a fresh inference pass, since Model 2 was
                                 never run on this new crop before).

Both: label = the crop's 52-class location label collapsed to binary
aneurysm/background (per-location assignment is a separate, later concern,
same as Dataset304/305).

    python -m topaneu_rsna.seg.build_aneurysm_fine_dataset
    python -m topaneu_rsna.seg.build_aneurysm_fine_dataset --with_vessel_channel
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


def _root(ds_id: int) -> Path:
    return C.nnUNet_raw / f"Dataset{ds_id:03d}_{C.DS_NAMES[ds_id]}"


def process(case: str, roi_dir: Path, m2_dir: Path | None, root: Path,
           with_labels: bool) -> bool:
    ip = roi_dir / f"{case}{C.IMAGE_SUFFIX}"
    if not ip.exists():
        return False

    img, meta = uio.read(ip)
    uio.write(img.astype(np.float32), meta, root / "imagesTr" / f"{case}_0000.nii.gz")

    if m2_dir is not None:
        p2 = m2_dir / f"{case}.nii.gz"
        if not p2.exists():
            return False
        v2, _ = uio.read(p2)
        v2 = geo.crop_pad(v2.astype(np.uint8), (0, 0, 0), img.shape)
        uio.write((v2 > 0).astype(np.float32), meta, root / "imagesTr" / f"{case}_0001.nii.gz")

    if with_labels:
        lp = roi_dir / f"{case}_location{C.LABEL_SUFFIX}"
        if lp.exists():
            loc, _ = uio.read(lp)
            aneu = (geo.crop_pad(loc.astype(np.uint8), (0, 0, 0), img.shape) > 0).astype(np.uint8)
            uio.write(aneu, meta, root / "labelsTr" / f"{case}.nii.gz")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roi_dir", type=Path, default=C.ANEURYSM_COARSE_ROI_DIR)
    ap.add_argument("--m2_dir", type=Path, default=C.ANEURYSM_VESSEL_PRED_M2)
    ap.add_argument("--with_vessel_channel", action="store_true")
    ap.add_argument("--no_labels", action="store_true")
    a = ap.parse_args()

    ds_id = C.DS_ANEURYSM_FINE_VESSEL if a.with_vessel_channel else C.DS_ANEURYSM_FINE_RAW
    root = _root(ds_id)
    (root / "imagesTr").mkdir(parents=True, exist_ok=True)
    (root / "labelsTr").mkdir(parents=True, exist_ok=True)

    m2_dir = a.m2_dir if a.with_vessel_channel else None

    cases = sorted(p.name[: -len(C.IMAGE_SUFFIX)]
                   for p in a.roi_dir.glob(f"*{C.IMAGE_SUFFIX}"))
    if not cases:
        raise SystemExit(f"no crops in {a.roi_dir} -- run aneurysm_coarse_roi.py first")

    n = sum(process(c, a.roi_dir, m2_dir, root, not a.no_labels)
           for c in tqdm(cases, desc="aneurysm-fine"))
    if n < len(cases):
        msg = f"WARNING: {len(cases) - n} cases skipped"
        if a.with_vessel_channel:
            msg += f" (missing Model 2 prediction in {m2_dir} -- run that inference job first)"
        print(msg)

    channel_names = {"0": "CTA_MRA"}
    if a.with_vessel_channel:
        channel_names["1"] = "vessel_mask"
    (root / "dataset.json").write_text(json.dumps({
        "channel_names": channel_names,
        "labels": {"background": 0, "aneurysm": 1},
        "numTraining": int(n),
        "file_ending": ".nii.gz",
    }, indent=2))
    print(f"{root}  ({n} cases, {len(channel_names)} input channel(s), 1 foreground class)")


if __name__ == "__main__":
    main()
