"""
Stage 1 -- Coarse Scan.

Consumes Model 1's 3-class vessel-group prediction and writes, per case, a
140x140x140 mm cube resampled to FINE_SPACING, plus the aligned vessel /
location / type labels so stage 2 and 3 never touch the master data again.
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


def process(case, pred_dir: Path, out_dir: Path, with_labels: bool) -> bool:
    pred, pmeta = uio.read(pred_dir / f"{case}.nii.gz")
    fg = (pred > 0).astype(np.uint8)

    c = geo.dbscan_centroid(fg, pmeta["spacing"], C.DBSCAN_EPS_MM,
                            C.DBSCAN_MIN_SAMPLES, C.DBSCAN_MAX_POINTS)
    fallback = c is None
    if fallback:
        c = np.asarray(fg.shape, np.float64) / 2.0
    center_mm = c * np.asarray(pmeta["spacing"], np.float64)

    img, meta = uio.read(C.IMAGES_DIR / f"{case}{C.IMAGE_SUFFIX}")
    img, meta = uio.resample(img, meta, C.FINE_SPACING, is_label=False)
    center_vox = center_mm / np.asarray(C.FINE_SPACING, np.float64)
    lo, hi = geo.cube_bounds(center_vox, C.COARSE_ROI_MM, C.FINE_SPACING)

    out_dir.mkdir(parents=True, exist_ok=True)
    uio.write(geo.crop_pad(img, lo, hi).astype(np.float32), meta,
              out_dir / f"{case}{C.IMAGE_SUFFIX}")

    if with_labels:
        for tag, src in (("vessel", C.VESSEL_MASKS),
                         ("location", C.LOCATION_MASKS),
                         ("type", C.TYPE_MASKS)):
            p = src / f"{case}{C.LABEL_SUFFIX}"
            if not p.exists():
                continue
            lab, lmeta = uio.read(p)
            lab, _ = uio.resample(lab, lmeta, C.FINE_SPACING, is_label=True)
            uio.write(geo.crop_pad(lab, lo, hi).astype(np.uint8), meta,
                      out_dir / f"{case}_{tag}{C.LABEL_SUFFIX}")

    (out_dir / f"{case}_coarse.json").write_text(json.dumps({
        "case": case, "lo": [int(v) for v in lo], "hi": [int(v) for v in hi],
        "dbscan_fallback": bool(fallback)}, indent=2))
    return fallback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", type=Path, default=C.COARSE_PRED_DIR)
    ap.add_argument("--out_dir", type=Path, default=C.COARSE_ROI_DIR)
    ap.add_argument("--no_labels", action="store_true")
    a = ap.parse_args()

    cases = sorted(p.name[: -len(".nii.gz")] for p in a.pred_dir.glob("*.nii.gz"))
    if not cases:
        raise SystemExit(f"no Model 1 predictions in {a.pred_dir}")
    nf = sum(process(c, a.pred_dir, a.out_dir, not a.no_labels) for c in tqdm(cases))
    print(f"{len(cases)} ROIs -> {a.out_dir}")
    if nf:
        print(f"WARNING: {nf} cases had an empty Model-1 prediction (centre fallback).")


if __name__ == "__main__":
    main()
