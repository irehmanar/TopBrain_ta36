"""
Precompute a multi-scale Frangi vesselness map for any TopAneu case not already
covered by Dataset104_TopAneuLocationVesselness (see config.LEGACY_VESSELNESS_DIR
and prep/build_aneurysm_vesselness_dataset.py, which prefers that already-computed
data and only needs this script's output for the gap). A single full Frangi run
over the whole cohort took 3 days last time, so build_aneurysm_vesselness_dataset.py
is written to avoid re-running it wherever Dataset104 already has the case.

Ported from a standalone script the author ran locally to build that 52-class
location+vesselness dataset -- same Frangi algorithm, wired into this pipeline's
config/io conventions instead of hardcoded input/output folders.

For each case:
  - build a rough head/brain mask (CTA: HU soft-tissue window; MRA: intensity
    percentile) from the case name's "_ct_"/"_mr_" tag (TopAneu naming:
    topaneu_center<N>_<ct|mr>_<id>)
  - crop to that mask's bounding box so the Hessian is never computed over
    background/skull
  - run a 3-scale Frangi filter inside the crop, paste the result back into a
    full-size zero volume, and write it with the source image's spacing/origin/
    direction (via utils.io.write) so it lines up as an extra nnU-Net channel

Resumable: skips any case that already has an output file, so a killed/re-queued
job just continues where it left off.

    python -m topaneu_rsna.prep.compute_vesselness
"""
from __future__ import annotations

import argparse
import gc
import time
import traceback
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from skimage.filters import frangi
from skimage.morphology import binary_closing, ball
from skimage.measure import label
from tqdm import tqdm

from topaneu_rsna import config as C
from topaneu_rsna.utils import io as uio

CTA_WINDOW = (0, 400)


def detect_modality(case: str) -> str:
    name = case.lower()
    if "_ct_" in name:
        return "CTA"
    if "_mr_" in name:
        return "MRA"
    return "MRA"  # TopAneu ships only CT/MR; default to the percentile-based path


def brain_mask_cta(arr, hu_low=-20, hu_high=180):
    """Rough head mask for CTA to strip skull + extracranial soft tissue."""
    soft_tissue = (arr > hu_low) & (arr < hu_high)
    soft_tissue = ndimage.binary_fill_holes(soft_tissue)
    lbl = label(soft_tissue)
    if lbl.max() > 0:
        sizes = ndimage.sum(soft_tissue, lbl, range(1, lbl.max() + 1))
        biggest = np.argmax(sizes) + 1
        mask = lbl == biggest
    else:
        mask = soft_tissue
    mask = binary_closing(mask, ball(2))
    mask = ndimage.binary_erosion(mask, iterations=2)
    return mask


def brain_mask_mra(arr, pct_thresh=5):
    """Simple foreground mask for MRA -- vessels/tissue are bright, background near-zero."""
    thresh = np.percentile(arr, pct_thresh)
    fg = arr > max(thresh, 1e-3)
    fg = ndimage.binary_fill_holes(fg)
    lbl = label(fg)
    if lbl.max() > 0:
        sizes = ndimage.sum(fg, lbl, range(1, lbl.max() + 1))
        biggest = np.argmax(sizes) + 1
        mask = lbl == biggest
    else:
        mask = fg
    mask = binary_closing(mask, ball(2))
    return mask


def preprocess_cta(arr, clip_low=0, clip_high=400):
    mask = brain_mask_cta(arr)
    clipped = np.clip(arr, clip_low, clip_high)
    clipped = np.where(mask, clipped, clip_low)
    norm = (clipped - clip_low) / (clip_high - clip_low + 1e-6)
    return norm.astype(np.float32), mask


def preprocess_mra(arr, p_low=1, p_high=99):
    mask = brain_mask_mra(arr)
    lo, hi = np.percentile(arr, [p_low, p_high])
    clipped = np.clip(arr, lo, hi)
    norm = (clipped - lo) / (hi - lo + 1e-6)
    norm = np.where(mask, norm, 0.0)
    return norm.astype(np.float32), mask


def _crop_to_mask(norm, mask, pad=3):
    coords = np.argwhere(mask)
    z0, y0, x0 = coords.min(axis=0)
    z1, y1, x1 = coords.max(axis=0) + 1
    z0, y0, x0 = max(z0 - pad, 0), max(y0 - pad, 0), max(x0 - pad, 0)
    z1 = min(z1 + pad, norm.shape[0])
    y1 = min(y1 + pad, norm.shape[1])
    x1 = min(x1 + pad, norm.shape[2])
    bbox = (z0, z1, y0, y1, x0, x1)
    return norm[z0:z1, y0:y1, x0:x1], mask[z0:z1, y0:y1, x0:x1], bbox


def compute_vesselness(arr, spacing, modality, sigmas=None, cta_window=CTA_WINDOW,
                        presmooth_sigma=0.5, intensity_gate_pct=60):
    """Crop to the brain-mask bbox before running Frangi, float32 throughout.
    Returns a full-size (same shape as `arr`) vesselness map and the head mask."""
    if modality == "CTA":
        norm, mask = preprocess_cta(arr, *cta_window)
        gate_active = True
    else:
        norm, mask = preprocess_mra(arr)
        gate_active = False

    norm_crop, mask_crop, bbox = _crop_to_mask(norm, mask)
    del norm, mask
    gc.collect()

    norm_crop = gaussian_filter(norm_crop, sigma=presmooth_sigma).astype(np.float32)

    if sigmas is None:
        base = min(spacing[1], spacing[2])
        sigmas = np.arange(1.0, 2.5, 0.5) / base  # 3 scales
        sigmas = sigmas[sigmas > 0.3]

    in_mask_vals = norm_crop[mask_crop]
    hessian_scale_hint = np.std(in_mask_vals) if in_mask_vals.size else None

    vess_crop = frangi(norm_crop, sigmas=sigmas, black_ridges=False,
                        gamma=hessian_scale_hint).astype(np.float32)

    if gate_active:
        gate_thresh = np.percentile(in_mask_vals, intensity_gate_pct) if in_mask_vals.size else 0
        vess_crop = np.where(norm_crop >= gate_thresh, vess_crop, 0.0)

    del norm_crop, in_mask_vals
    gc.collect()

    vess_crop = np.where(mask_crop, vess_crop, 0.0).astype(np.float32)

    z0, z1, y0, y1, x0, x1 = bbox
    vess = np.zeros(arr.shape, dtype=np.float32)
    vess[z0:z1, y0:y1, x0:x1] = vess_crop

    full_mask = np.zeros(arr.shape, dtype=bool)
    full_mask[z0:z1, y0:y1, x0:x1] = mask_crop

    del vess_crop, mask_crop
    gc.collect()

    return vess, full_mask


def process_one(case: str, out_dir: Path) -> str:
    out_path = out_dir / f"{case}.nii.gz"
    if out_path.exists():
        return "already_done"

    ip = C.IMAGES_DIR / f"{case}{C.IMAGE_SUFFIX}"
    if not ip.exists():
        return "skipped"

    t0 = time.time()
    arr, meta = uio.read(ip)
    orig_shape = arr.shape
    modality = detect_modality(case)

    vess, mask = compute_vesselness(arr.astype(np.float32), meta["spacing"], modality)

    assert vess.shape == orig_shape, (
        f"Shape mismatch for {case}: input {orig_shape} vs vesselness {vess.shape}")

    uio.write(vess, meta, out_path)

    print(f"[OK] {case} ({modality}) shape={orig_shape} "
          f"mask={mask.mean() * 100:.1f}% vess_max={vess.max():.3f} "
          f"({time.time() - t0:.1f}s)")

    del arr, vess, mask
    gc.collect()
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", type=Path, default=C.VESSELNESS_DIR)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    a.output_dir.mkdir(parents=True, exist_ok=True)

    cases = uio.list_cases(C.IMAGES_DIR, C.IMAGE_SUFFIX)
    if a.limit:
        cases = cases[:a.limit]
    print(f"Found {len(cases)} case(s) in {C.IMAGES_DIR}")
    print(f"Writing vesselness maps to {a.output_dir}\n")

    failed = []
    counts = {"ok": 0, "skipped": 0, "already_done": 0, "failed": 0}
    for case in tqdm(cases, desc="vesselness"):
        try:
            counts[process_one(case, a.output_dir)] += 1
        except MemoryError:
            print(f"[FAIL] {case}: MemoryError -- volume too large for available RAM")
            failed.append(case)
            counts["failed"] += 1
            gc.collect()
        except Exception as e:
            print(f"[FAIL] {case}: {e}")
            traceback.print_exc()
            failed.append(case)
            counts["failed"] += 1
            gc.collect()

    print("\n--- Summary ---")
    for k, v in counts.items():
        print(f"  {k}: {v}")
    if failed:
        fail_log = a.output_dir / "failed_cases.txt"
        fail_log.write_text("\n".join(failed))
        print(f"\n{len(failed)} case(s) failed -- logged to {fail_log}")


if __name__ == "__main__":
    main()
