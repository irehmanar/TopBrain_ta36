"""
Stage 2 -- ROI refine + classifier cache.

Model 2's vessel segmentation gives a tight bbox; it is expanded by
ROI_REFINE_MARGIN_MM (15 mm z / 30 mm in-plane, matching the author's
"margin15_30" prediction directory) and re-cropped to 128x256x256.

Writes one .npz per case:
    image       float16 (1, 128, 256, 256)   z-scored
    vessel_m2   uint8   (128, 256, 256)      Model 2 label map
    vessel_m3   uint8   (128, 256, 256)      Model 3 label map (second mask branch)
    loc         uint8   (n_loc,)             per-location presence          [train]
    typ         uint8   (n_type,)            per-type presence              [train]
    points      float32 (M, 3)               aneurysm centres, z/y/x voxels [train]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from topaneu_rsna import config as C
from topaneu_rsna.utils import geometry as geo
from topaneu_rsna.utils import io as uio


def process(case, spec, m2_dir, m3_dir, roi_dir, out_dir, with_labels) -> bool:
    img, _ = uio.read(roi_dir / f"{case}{C.IMAGE_SUFFIX}")
    p2 = m2_dir / f"{case}.nii.gz"
    if not p2.exists():
        return False
    v2, _ = uio.read(p2)
    v2 = geo.crop_pad(v2.astype(np.uint8), (0, 0, 0), img.shape)

    p3 = (m3_dir / f"{case}.nii.gz") if m3_dir else None
    if p3 is not None and p3.exists():
        v3, _ = uio.read(p3)
        v3 = geo.crop_pad(v3.astype(np.uint8), (0, 0, 0), img.shape)
    else:
        v3 = v2

    lo, hi = geo.tight_bounds(v2, C.ROI_REFINE_MARGIN_MM, C.FINE_SPACING)
    if lo is None:
        lo, hi = np.zeros(3, int), np.asarray(img.shape, int)
    lo, hi = geo.center_to_size(lo, hi, C.FINAL_ROI_SIZE)

    out = {
        "image": uio.zscore(geo.crop_pad(img, lo, hi))[None].astype(np.float16),
        "vessel_m2": geo.crop_pad(v2, lo, hi).astype(np.uint8),
        "vessel_m3": geo.crop_pad(v3, lo, hi).astype(np.uint8),
    }

    if with_labels:
        lp = roi_dir / f"{case}_location{C.LABEL_SUFFIX}"
        if lp.exists():
            loc_map = geo.crop_pad(uio.read(lp)[0].astype(np.uint8), lo, hi)
            present = set(np.unique(loc_map).tolist())
            loc = np.array([1 if (i + 1) in present else 0
                            for i in range(spec.n_loc)], np.uint8)
            pts = []
            for i in range(spec.n_loc):
                if loc[i]:
                    pts.extend(geo.component_centroids(loc_map, i + 1))
            out["loc"] = loc
            out["points"] = (np.asarray(pts, np.float32) if pts
                             else np.zeros((0, 3), np.float32))
        tp = roi_dir / f"{case}_type{C.LABEL_SUFFIX}"
        if tp.exists() and spec.n_type:
            tmap = geo.crop_pad(uio.read(tp)[0].astype(np.uint8), lo, hi)
            pres = set(np.unique(tmap).tolist())
            out["typ"] = np.array([1 if (i + 1) in pres else 0
                                   for i in range(spec.n_type)], np.uint8)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"{case}.npz", **out)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m2_dir", type=Path, default=C.VESSEL_PRED_M2)
    ap.add_argument("--m3_dir", type=Path, default=C.VESSEL_PRED_M3)
    ap.add_argument("--roi_dir", type=Path, default=C.COARSE_ROI_DIR)
    ap.add_argument("--out_dir", type=Path, default=C.CLS_CACHE_DIR)
    ap.add_argument("--no_labels", action="store_true")
    a = ap.parse_args()

    spec = C.load_labels()
    cases = sorted(p.name[: -len(C.IMAGE_SUFFIX)]
                   for p in a.roi_dir.glob(f"*{C.IMAGE_SUFFIX}"))
    ok = sum(process(c, spec, a.m2_dir, a.m3_dir, a.roi_dir, a.out_dir,
                     not a.no_labels) for c in tqdm(cases))
    print(f"{ok}/{len(cases)} cached -> {a.out_dir}")


if __name__ == "__main__":
    main()
