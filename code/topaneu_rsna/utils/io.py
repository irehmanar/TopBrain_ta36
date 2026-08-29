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


def zscore(img: np.ndarray) -> np.ndarray:
    """nnU-Net style per-volume z-score (the author used nnU-Net normalisation)."""
    x = img.astype(np.float32)
    m, s = float(x.mean()), float(x.std())
    return (x - m) / (s if s > 1e-8 else 1.0)


def list_cases(d, suffix):
    return sorted(p.name[: -len(suffix)] for p in Path(d).glob(f"*{suffix}"))
