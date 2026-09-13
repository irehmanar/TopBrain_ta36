from __future__ import annotations
import numpy as np


def dbscan_centroid(mask, spacing, eps_mm, min_samples, max_points=30000, seed=0):
    from sklearn.cluster import DBSCAN
    pts = np.argwhere(mask > 0)
    if pts.size == 0:
        return None
    rng = np.random.default_rng(seed)
    sub = pts[rng.choice(len(pts), max_points, replace=False)] if len(pts) > max_points else pts
    mm = sub * np.asarray(spacing, np.float64)[None]
    lab = DBSCAN(eps=eps_mm, min_samples=min_samples, n_jobs=-1).fit_predict(mm)
    ok = lab >= 0
    if not ok.any():
        return pts.mean(0)
    ids, cnt = np.unique(lab[ok], return_counts=True)
    return sub[lab == ids[cnt.argmax()]].mean(0)


def cube_bounds(center_vox, size_mm, spacing):
    sp = np.asarray(spacing, np.float64)
    n = np.round(np.asarray(size_mm, np.float64) / sp).astype(int)
    lo = np.round(np.asarray(center_vox, np.float64) - n / 2.0).astype(int)
    return lo, lo + n


def tight_bounds(mask, margin_mm, spacing):
    pts = np.argwhere(mask > 0)
    if pts.size == 0:
        return None, None
    pad = np.ceil(np.asarray(margin_mm, np.float64) / np.asarray(spacing, np.float64)).astype(int)
    return pts.min(0) - pad, pts.max(0) + 1 + pad


def center_to_size(lo, hi, target):
    c = (np.asarray(lo, np.float64) + np.asarray(hi, np.float64)) / 2.0
    t = np.asarray(target, np.float64)
    new_lo = np.round(c - t / 2.0).astype(int)
    return new_lo, new_lo + np.asarray(target, int)


def crop_pad(arr, lo, hi, cval=0):
    lo = np.asarray(lo, int); hi = np.asarray(hi, int)
    out = np.full(tuple(hi - lo), cval, dtype=arr.dtype)
    s_lo = np.clip(lo, 0, arr.shape); s_hi = np.clip(hi, 0, arr.shape)
    if np.any(s_hi <= s_lo):
        return out
    d_lo = s_lo - lo; d_hi = d_lo + (s_hi - s_lo)
    out[d_lo[0]:d_hi[0], d_lo[1]:d_hi[1], d_lo[2]:d_hi[2]] = \
        arr[s_lo[0]:s_hi[0], s_lo[1]:s_hi[1], s_lo[2]:s_hi[2]]
    return out


def component_centroids(labelmap, class_value, min_voxels=3):
    """Centroids (z,y,x) of each connected component of one class."""
    from scipy import ndimage
    m = labelmap == class_value
    if not m.any():
        return []
    lab, n = ndimage.label(m)
    out = []
    for i in range(1, n + 1):
        pts = np.argwhere(lab == i)
        if len(pts) >= min_voxels:
            out.append(pts.mean(0))
    return out


def nearest_vessel_label(instance_mask, vessel_map, spacing, tau_mm=4.0, pad_mm=20.0):
    """Host-vessel lookup for one lesion instance against a multi-class vessel
    label map, following the "distance to each other label" reasoning in the
    TopAneu rule-based localisation paper: the vessel segmenter has no
    aneurysm class, so the lesion itself is usually painted as vessel by
    whichever label the sac sits in, making raw overlap unreliable -- what
    stays reliable is which *other* labels are within a small margin.

    Returns (vessel_label_id, distance_mm, touching) where vessel_label_id is
    the nearest nonzero label in `vessel_map` (an int, or None if nothing is
    within `tau_mm`), distance_mm is its distance from the instance (0.0 if
    touching), and `touching` lists every label touching the instance at
    zero distance (for later, more elaborate junction handling -- unused by
    the simple first-pass caller, which only takes the single nearest one).
    """
    from scipy import ndimage

    sp = np.asarray(spacing, np.float64)
    idx = np.argwhere(instance_mask)
    pad = np.ceil(pad_mm / sp).astype(int)
    lo = np.maximum(idx.min(0) - pad, 0)
    hi = np.minimum(idx.max(0) + pad + 1, instance_mask.shape)
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    inst = instance_mask[sl]
    ves = vessel_map[sl]

    touch_zone = ndimage.binary_dilation(inst, iterations=2)
    touching = sorted(int(v) for v in np.unique(ves[touch_zone & (ves > 0)]))
    if touching:
        # tie-break by contact-patch size (largest first)
        counts = {v: int((ves[touch_zone] == v).sum()) for v in touching}
        best = max(touching, key=lambda v: counts[v])
        return best, 0.0, touching

    labels_present = np.unique(ves[ves > 0])
    if labels_present.size == 0:
        return None, None, []

    dt_inst = ndimage.distance_transform_edt(~inst, sampling=sp)
    best_label, best_dist = None, np.inf
    for v in labels_present:
        d = float(dt_inst[ves == v].min())
        if d < best_dist:
            best_label, best_dist = int(v), d
    if best_dist > tau_mm:
        return None, best_dist, []
    return best_label, best_dist, []
