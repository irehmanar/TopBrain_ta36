"""
Arc-length sub-parcellation for the 13 vessel labels that host more than one
of the 52 aneurysm-location classes (see seg/build_vessel_location_prior.py
and seg/assign_location_rule.py). This replaces the flat cohort-majority
fallback -- "always guess this vessel's single most common location" -- with
"where along this vessel does the lesion actually sit," following Paper 1's
sub-parcellation step (geodesic arc position, cohort-median fallback).

Two declared (hand-authored, not learned) anatomical tables live here,
analogous in spirit to labels.json's own location_to_vessel map:

  PROXIMAL_ANCHOR  per shared vessel, which neighbouring vessel(s) mark its
                    proximal (upstream/inflow) end -- e.g. BA's proximal end
                    is where the two vertebral arteries converge, so its
                    anchor is R-VA/L-VA. Used only to fix which of a
                    per-case skeleton's two endpoints is arc-fraction 0.0 vs
                    1.0 -- consistently across cases, without needing a
                    registered/canonical brain template.
  JUNCTION_BRANCH   per junction-type location (name contains "junction",
                    "bifurcation", or "terminus"), the neighbouring branch
                    vessel(s) whose take-off defines it. Paper 1 handles
                    junctions as the voxel contact patch between host and
                    branch, not as a position along the host -- so these
                    locations are checked by contact-patch proximity first,
                    and only fall through to the arc-fraction bucket lookup
                    if no branch vessel is present in this case's map (e.g.
                    a missed segmentation) to test contact against.

Both tables were authored from the vessel/location names in labels.json
(e.g. "R-3.2 ICA C6-OA-junction" implies the branch is "R-OA") rather than
from an independent anatomical reference, so treat them as a first-pass
declaration to sanity-check, not ground truth -- the same caveat labels.json
itself carries.
"""
from __future__ import annotations

import collections
from typing import NamedTuple

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

PROXIMAL_ANCHOR: dict[str, list[str]] = {
    "BA": ["R-VA", "L-VA"],
    "R-ICA-C6-C7": ["R-ICA-C1-C5"],
    "L-ICA-C6-C7": ["L-ICA-C1-C5"],
    "R-M1": ["R-ICA-C6-C7"],
    "L-M1": ["L-ICA-C6-C7"],
    "R-M2": ["R-M1"],
    "L-M2": ["L-M1"],
    "R-A1A2": ["Acom", "R-ICA-C6-C7"],
    "L-A1A2": ["Acom", "L-ICA-C6-C7"],
    "R-A3": ["R-A1A2"],
    "L-A3": ["L-A1A2"],
    "R-PICA": ["R-VA"],
    "L-PICA": ["L-VA"],
}

# location name -> branch vessel(s) whose contact patch with the host vessel
# defines it. Only locations whose name marks them as junction/bifurcation/
# terminus points appear here; plain trunk/segment locations are resolved by
# arc-fraction alone.
JUNCTION_BRANCH: dict[str, list[str]] = {
    "1.5 VA-BA junction": ["R-VA", "L-VA"],
    "R-1.7 BA-AICA junction": ["R-AICA"],
    "L-1.7 BA-AICA junction": ["L-AICA"],
    "R-1.9 BA-SCA junction": ["R-SCA"],
    "L-1.9 BA-SCA junction": ["L-SCA"],
    "R-1.3 VA-PICA junction": ["R-VA"],
    "L-1.3 VA-PICA junction": ["L-VA"],
    "R-3.2 ICA C6-OA-junction": ["R-OA"],
    "L-3.2 ICA C6-OA-junction": ["L-OA"],
    "R-3.7 ICA C7-terminus": ["R-M1", "R-A1A2"],
    "L-3.7 ICA C7-terminus": ["L-M1", "L-A1A2"],
    "R-5.2 M1 early bifurcation": ["R-M2"],
    "L-5.2 M1 early bifurcation": ["L-M2"],
    "R-5.3 M1-M2 junction": ["R-M1"],
    "L-5.3 M1-M2 junction": ["L-M1"],
}

CONTACT_TAU_MM = 4.0


def _largest_component(mask: np.ndarray) -> np.ndarray:
    lab, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, lab, index=np.arange(1, n + 1))
    return lab == (int(np.argmax(sizes)) + 1)


class Skeleton(NamedTuple):
    path_mm: np.ndarray       # (N, 3) ordered centerline points, arbitrary end first
    cumlen_mm: np.ndarray     # (N,) cumulative arc length from path_mm[0]
    total_mm: float
    proximal_first: bool | None   # True: path_mm[0] is frac 0.0; None: unknown


def extract_skeleton(vessel_mask: np.ndarray, spacing) -> Skeleton | None:
    """Skeletonize one vessel label's mask (already boolean, one case) and
    order its voxels into a single centerline path via a tree-diameter (two-
    BFS-sweep) heuristic: the true endpoints of a possibly-noisy skeleton
    are approximated as the two mutually-farthest points, and the path
    between them is their shortest path through the skeleton's own voxel
    adjacency graph. This naturally ignores short spurs from skeletonization
    noise, since a spur is never on the shortest path between the two
    farthest points.

    Cropped internally to the mask's own bounding box before skeletonizing
    (skeletonize cost scales with array size, not just mask size, and this
    runs once per shared vessel per case) -- the crop offset is added back
    so the returned `path_mm` stays in the *original*, uncropped volume's
    voxel/mm frame, matching whatever frame a caller computes a lesion
    centroid in from that same original vessel_map."""
    idx0 = np.argwhere(vessel_mask)
    if len(idx0) < 5:
        return None
    lo = idx0.min(0)
    hi = idx0.max(0) + 1
    crop = vessel_mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]

    m = _largest_component(crop)
    if int(m.sum()) < 5:
        return None

    from skimage.morphology import skeletonize
    skel = skeletonize(m)
    pts = np.argwhere(skel) + lo[None]   # back to the original volume's voxel frame
    if len(pts) < 2:
        return None

    sp = np.asarray(spacing, np.float64)
    pts_mm = pts * sp[None]

    idx_of = {tuple(p): i for i, p in enumerate(pts.tolist())}
    offsets = [(dz, dy, dx) for dz in (-1, 0, 1) for dy in (-1, 0, 1)
              for dx in (-1, 0, 1) if (dz, dy, dx) != (0, 0, 0)]
    neighbors = [[] for _ in pts]
    for i, p in enumerate(pts.tolist()):
        for dz, dy, dx in offsets:
            j = idx_of.get((p[0] + dz, p[1] + dy, p[2] + dx))
            if j is not None:
                neighbors[i].append(j)

    def bfs(start):
        dist = {start: 0.0}
        prev = {start: None}
        q = collections.deque([start])
        while q:
            u = q.popleft()
            for v in neighbors[u]:
                if v not in dist:
                    dist[v] = dist[u] + float(np.linalg.norm(pts_mm[v] - pts_mm[u]))
                    prev[v] = u
                    q.append(v)
        far = max(dist, key=dist.get)
        return far, dist, prev

    a, _, _ = bfs(0)
    b, _, prev = bfs(a)

    path_idx = [b]
    while prev[path_idx[-1]] is not None:
        path_idx.append(prev[path_idx[-1]])
    path_idx.reverse()

    path_mm = pts_mm[path_idx]
    seg = np.linalg.norm(np.diff(path_mm, axis=0), axis=1)
    cumlen = np.concatenate([[0.0], np.cumsum(seg)])
    return Skeleton(path_mm, cumlen, float(cumlen[-1]), None)


def orient_skeleton(skel: Skeleton, vessel_map: np.ndarray, spacing,
                    anchor_names: list[str], name_to_id: dict) -> Skeleton:
    """Decide which end of `skel.path_mm` is arc-fraction 0.0, using whichever
    endpoint sits nearer any voxel of `anchor_names` in this case's own
    vessel_map. Returns `skel` unchanged (proximal_first=None) if none of the
    anchor vessels are present in this case -- callers should treat that as
    "orientation unknown" and fall back to the flat majority table."""
    ids = [name_to_id[a] for a in anchor_names if a in name_to_id]
    anchor_mask = np.isin(vessel_map, ids) if ids else np.zeros_like(vessel_map, dtype=bool)
    if not anchor_mask.any():
        return skel
    sp = np.asarray(spacing, np.float64)
    anchor_pts_mm = np.argwhere(anchor_mask) * sp[None]
    tree = cKDTree(anchor_pts_mm)
    d0, _ = tree.query(skel.path_mm[0])
    d1, _ = tree.query(skel.path_mm[-1])
    return skel._replace(proximal_first=bool(d0 <= d1))


def arc_fraction(skel: Skeleton, point_mm) -> tuple[float | None, float]:
    """(fraction in [0,1] from the proximal end, or None if orientation is
    unknown; distance in mm from `point_mm` to the nearest skeleton point)."""
    d = np.linalg.norm(skel.path_mm - np.asarray(point_mm)[None], axis=1)
    i = int(np.argmin(d))
    dist = float(d[i])
    if skel.total_mm <= 0 or skel.proximal_first is None:
        return None, dist
    f = skel.cumlen_mm[i] / skel.total_mm
    return (float(f) if skel.proximal_first else float(1.0 - f)), dist


def mirror_location_name(loc: str) -> str | None:
    """"R-..." <-> "L-...", or None for an unsided location (e.g. "1.4 BA
    trunk", "4.1 Acom complex") -- used to find a junction candidate's
    mirrored counterpart for the side-reconciliation check below."""
    if loc.startswith("R-"):
        return "L-" + loc[2:]
    if loc.startswith("L-"):
        return "R-" + loc[2:]
    return None


def estimate_midline(vessel_map: np.ndarray, spacing,
                     name_to_id: dict) -> tuple[float, bool] | None:
    """Sagittal midline position (mm, along the last/x array axis) and which
    side sits at larger x, calibrated from THIS case's own paired R-/L-
    vessel labels rather than assumed from image geometry -- Paper 1's own
    side-reconciliation step fits its midline the same way (from the vessel
    map's own left/right label pairs), since a fixed image-center guess
    doesn't hold under arbitrary patient positioning/cropping.

    Returns (midline_x_mm, positive_side_is_r), or None if this case has no
    usable paired vessel (both sides present and non-empty) to calibrate
    from -- callers should skip the side check entirely in that case rather
    than guess."""
    sp = np.asarray(spacing, np.float64)
    mids, diffs = [], []
    seen = set()
    for v, vid in name_to_id.items():
        if not v.startswith("R-") or v in seen:
            continue
        lv = "L-" + v[2:]
        lid = name_to_id.get(lv)
        if lid is None:
            continue
        seen.add(v); seen.add(lv)
        rmask, lmask = vessel_map == vid, vessel_map == lid
        if not rmask.any() or not lmask.any():
            continue
        rx = np.argwhere(rmask).mean(0)[-1] * sp[-1]
        lx = np.argwhere(lmask).mean(0)[-1] * sp[-1]
        mids.append((rx + lx) / 2.0)
        diffs.append(rx - lx)
    if not mids:
        return None
    return float(np.median(mids)), bool(np.median(diffs) > 0)


def reconcile_side(loc: str, inst_mask: np.ndarray, vessel_map: np.ndarray,
                   spacing, name_to_id: dict, candidate_locs: set) -> str:
    """Paper 1's side-reconciliation step: when a lesion's own spatial
    position contradicts the side implied by its assigned (possibly
    mislabeled-by-segmentation-noise) vessel/junction match, trust the
    geometry and re-side the label -- but only if the mirrored location is
    itself actually a legal candidate on this vessel (`candidate_locs`) and
    a midline could be calibrated for this case; otherwise returns `loc`
    unchanged. This matters specifically for junction-type locations shared
    by an unpaired midline vessel (e.g. BA hosts both "R-1.9 BA-SCA
    junction" and "L-1.9 BA-SCA junction"), where nothing upstream already
    forces the correct side the way a paired host vessel (e.g. "L-PICA")
    does automatically."""
    mirrored = mirror_location_name(loc)
    if mirrored is None or mirrored not in candidate_locs:
        return loc
    mid = estimate_midline(vessel_map, spacing, name_to_id)
    if mid is None:
        return loc
    midline_x, positive_is_r = mid
    sp = np.asarray(spacing, np.float64)
    lesion_x = np.argwhere(inst_mask).mean(0)[-1] * sp[-1]
    implied_r = (lesion_x > midline_x) == positive_is_r
    assigned_is_r = loc.startswith("R-")
    return loc if implied_r == assigned_is_r else mirrored


def contact_patch_distance(instance_mask: np.ndarray, vessel_map: np.ndarray,
                           spacing, host_id: int, branch_names: list[str],
                           name_to_id: dict, pad_mm: float = 20.0) -> float | None:
    """Distance in mm from `instance_mask` to the voxel contact patch between
    the host vessel (`host_id`) and any of `branch_names` -- Paper 1's
    junction-object treatment (a junction is not a label of its own, it's
    where two labels touch). Returns None if the branch vessel isn't present
    in this case (segmentation miss) or the two labels never touch."""
    ids = [name_to_id[b] for b in branch_names if b in name_to_id]
    if not ids:
        return None
    sp = np.asarray(spacing, np.float64)

    idx = np.argwhere(instance_mask)
    pad = np.ceil(pad_mm / sp).astype(int)
    lo = np.maximum(idx.min(0) - pad, 0)
    hi = np.minimum(idx.max(0) + pad + 1, instance_mask.shape)
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    ves = vessel_map[sl]
    inst = instance_mask[sl]

    host = ves == host_id
    branch = np.isin(ves, ids)
    if not host.any() or not branch.any():
        return None
    contact = ndimage.binary_dilation(host, iterations=1) & branch
    contact |= ndimage.binary_dilation(branch, iterations=1) & host
    if not contact.any():
        return None

    contact_pts_mm = np.argwhere(contact) * sp[None]
    inst_pts_mm = np.argwhere(inst) * sp[None]
    tree = cKDTree(contact_pts_mm)
    d, _ = tree.query(inst_pts_mm)
    return float(d.min())
