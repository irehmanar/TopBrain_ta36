"""SimpleITK helpers. Array axis order is always (z, y, x)."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import SimpleITK as sitk


def read(path):
    img = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(img), {
        "spacing": tuple(img.GetSpacing()[::-1]),
        "origin": img.GetOrigin(),
        "direction": img.GetDirection(),
    }


def write(arr, meta, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img = sitk.GetImageFromArray(np.ascontiguousarray(arr))
    img.SetSpacing(tuple(float(s) for s in meta["spacing"][::-1]))
    img.SetOrigin(tuple(meta["origin"]))
    img.SetDirection(tuple(meta["direction"]))
    sitk.WriteImage(img, str(path), useCompression=True)


def resample(arr, meta, new_spacing, is_label=False):
    old = np.asarray(meta["spacing"], np.float64)
    new = np.asarray(new_spacing, np.float64)
    if np.allclose(old, new):
        return arr, dict(meta)
    img = sitk.GetImageFromArray(np.ascontiguousarray(arr))
    img.SetSpacing(tuple(old[::-1])); img.SetOrigin(tuple(meta["origin"]))
    img.SetDirection(tuple(meta["direction"]))
    size = np.asarray(img.GetSize(), np.float64)
    new_size = np.maximum(1, np.round(size * (old[::-1] / new[::-1]))).astype(int)
    rs = sitk.ResampleImageFilter()
    rs.SetOutputSpacing(tuple(float(v) for v in new[::-1]))
    rs.SetSize([int(v) for v in new_size])
    rs.SetOutputOrigin(img.GetOrigin()); rs.SetOutputDirection(img.GetDirection())
    rs.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkBSpline)
    rs.SetDefaultPixelValue(0)
    out = rs.Execute(img)
    m = dict(meta); m["spacing"] = tuple(float(v) for v in new); m["origin"] = out.GetOrigin()
    return sitk.GetArrayFromImage(out), m


def resample_to_reference(moving_path, reference_path, is_label=False):
    """Resample the volume at `moving_path` onto `reference_path`'s exact grid
    (same size/spacing/origin/direction). Uses an identity transform -- this
    assumes both volumes already sit in the same physical (world) coordinate
    frame (e.g. two different crops/resamplings of the same source scan), not
    that they need registering. Returns (array, meta) with `reference_path`'s
    geometry, ready to hand to `write`."""
    ref = sitk.ReadImage(str(reference_path))
    mov = sitk.ReadImage(str(moving_path))
    interp = sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear
    out = sitk.Resample(mov, ref, sitk.Transform(), interp, 0.0, sitk.sitkFloat32)
    return sitk.GetArrayFromImage(out), {
        "spacing": tuple(ref.GetSpacing()[::-1]),
        "origin": ref.GetOrigin(),
        "direction": ref.GetDirection(),
    }


def physical_overlap_frac(path_a, path_b) -> float:
    """Fraction of image A's world-space bounding box that B's bounding box
    covers -- a cheap (header-only) sanity check that two differently-gridded
    volumes plausibly represent the same physical region before resampling
    one onto the other's grid."""
    def bbox(path):
        r = sitk.ImageFileReader()
        r.SetFileName(str(path))
        r.ReadImageInformation()  # header only, no pixel data
        size = np.asarray(r.GetSize(), np.float64)
        origin = np.asarray(r.GetOrigin(), np.float64)
        spacing = np.asarray(r.GetSpacing(), np.float64)
        direction = np.asarray(r.GetDirection(), np.float64).reshape(3, 3)
        corners = [origin + direction @ (spacing * np.asarray(idx))
                  for idx in ((0, 0, 0), (size[0] - 1, 0, 0), (0, size[1] - 1, 0),
                             (0, 0, size[2] - 1), (size[0] - 1, size[1] - 1, 0),
                             (size[0] - 1, 0, size[2] - 1), (0, size[1] - 1, size[2] - 1),
                             size - 1)]
        arr = np.asarray(corners)
        return arr.min(0), arr.max(0)

    lo_a, hi_a = bbox(path_a)
    lo_b, hi_b = bbox(path_b)
    lo, hi = np.maximum(lo_a, lo_b), np.minimum(hi_a, hi_b)
    inter = float(np.prod(np.maximum(hi - lo, 0)))
    vol_a = float(np.prod(hi_a - lo_a))
    return inter / vol_a if vol_a > 0 else 0.0


def zscore(img: np.ndarray) -> np.ndarray:
    """nnU-Net style per-volume z-score (the author used nnU-Net normalisation)."""
    x = img.astype(np.float32)
    m, s = float(x.mean()), float(x.std())
    return (x - m) / (s if s > 1e-8 else 1.0)


def list_cases(d, suffix):
    return sorted(p.name[: -len(suffix)] for p in Path(d).glob(f"*{suffix}"))
