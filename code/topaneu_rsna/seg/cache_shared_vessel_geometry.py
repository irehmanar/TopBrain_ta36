"""
Experiment Gamma ensemble, step 1: cache every instance's PRIOR-INDEPENDENT
geometry exactly once, so a bootstrap-prior ensemble (seg/ensemble_rule_bootstrap.py)
never has to re-skeletonize/re-search for junction contact patches per
bootstrap replicate -- only the final bucket/majority-table lookup, which is
cheap pure-Python, needs to change per bootstrap.

Every quantity `_junction_candidate` and `_arc_candidate` (assign_location_rule.py)
compute is either purely geometric (host vessel, junction contact-patch
distance to each candidate branch, raw continuous arc-length fraction) or
purely a function of the prior (which bucket a fraction falls in, whether
that bucket is low_sample, the flat majority table). This script computes
and stores only the geometric half:

  case, instance_idx, vessel, size, host_dist_mm, true_class,
  n_locs (1 = single-location vessel, resolved once here, no prior needed),
  single_location (the vessel's one location, if n_locs == 1),
  junction_best_loc, junction_best_dist_mm (best candidate + distance,
    independent of any prior -- _junction_candidate never reads one),
  arc_fraction (raw, continuous -- independent of any prior's buckets)

Reuses assign_location_rule.py's own `_junction_candidate` directly (so this
can never silently drift from what the deployed rule actually checks) and
build_feature_table.py's `raw_arc_fraction` helper for the arc-length value.

    python -m topaneu_rsna.seg.cache_shared_vessel_geometry
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy import ndimage
from tqdm import tqdm

from topaneu_rsna import config as C
from topaneu_rsna.utils import geometry as geo
from topaneu_rsna.utils import io as uio
from topaneu_rsna.utils import vessel_skeleton as vsk
from topaneu_rsna.seg.assign_location_rule import (
    build_vessel_to_locations, load_binary_pred_paths, oracle_binary_pred_paths,
    _junction_candidate)
from topaneu_rsna.seg.build_feature_table import raw_arc_fraction

FIELDNAMES = ["case", "instance_idx", "vessel", "size", "host_dist_mm",
             "true_class", "n_locs", "single_location",
             "junction_best_loc", "junction_best_dist_mm", "arc_fraction"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary_dataset", type=int, default=C.DS_ANEURYSM)
    ap.add_argument("--trainer", default=C.TRAINER_LOC)
    ap.add_argument("--plans", default=C.PLANS_RESENC)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--oracle_binary", action="store_true")
    ap.add_argument("--binary_pred_dir", type=Path, default=None,
                    help="use a flat directory of already-computed <case>.nii.gz "
                         "binary predictions instead of --binary_dataset's own "
                         "fold_*/validation/ split -- see assign_location_rule.py's "
                         "own --binary_pred_dir for the identical convention. "
                         "Overrides --oracle_binary/--binary_dataset/--folds.")
    ap.add_argument("--vessel_source", choices=["gt", "pred"], default="gt")
    ap.add_argument("--vessel_pred_dir", type=Path, default=None,
                    help="required when --vessel_source pred -- a directory of "
                         "already-computed <case>.nii.gz vessel predictions, e.g. "
                         "job 78's whole-head Model 2 output")
    ap.add_argument("--tau_mm", type=float, default=4.0)
    ap.add_argument("--junction_tau_mm", type=float, default=2.0,
                    help="candidacy radius for the junction contact-patch check "
                         "-- prior-independent, but still a real tunable, so it "
                         "must match whatever ensemble_rule_bootstrap.py is told "
                         "to assume when it reuses this cache")
    ap.add_argument("--min_voxels", type=int, default=3)
    ap.add_argument("--out", type=Path,
                    default=C.LOG_ROOT / "task2_shared_vessel_geometry_cache.csv")
    a = ap.parse_args()

    if a.vessel_source == "pred" and a.vessel_pred_dir is None:
        raise SystemExit("--vessel_source pred requires --vessel_pred_dir")

    spec = C.load_labels()
    vessel_to_locations = build_vessel_to_locations(spec)
    loc_value = {loc: i + 1 for i, loc in enumerate(spec.locations)}
    name_to_id = {v: i + 1 for i, v in enumerate(spec.vessels)}

    if a.binary_pred_dir is not None:
        bin_paths = {case: a.binary_pred_dir / f"{case}{C.LABEL_SUFFIX}"
                    for case in uio.list_cases(a.binary_pred_dir, C.LABEL_SUFFIX)}
        print(f"{len(bin_paths)} binary predictions read directly from {a.binary_pred_dir}")
    elif a.oracle_binary:
        bin_paths = oracle_binary_pred_paths()
    else:
        bin_paths = load_binary_pred_paths(a.binary_dataset, a.trainer, a.plans, a.folds)
    print(f"{len(bin_paths)} held-out binary predictions")

    rows = []
    for case, bp in tqdm(bin_paths.items(), desc="caching geometry"):
        binmask, meta = uio.read(bp)
        binmask = binmask > 0
        spacing = meta["spacing"]

        vp = (C.VESSEL_MASKS / f"{case}{C.LABEL_SUFFIX}" if a.vessel_source == "gt"
             else a.vessel_pred_dir / f"{case}{C.LABEL_SUFFIX}")
        gt_p = C.LOCATION_MASKS / f"{case}{C.LABEL_SUFFIX}"
        if not vp.exists() or not gt_p.exists():
            continue
        vessel_map, _ = uio.read(vp)
        gt, _ = uio.read(gt_p)

        lab, n = ndimage.label(binmask)
        for i in range(1, n + 1):
            inst = lab == i
            size = int(inst.sum())
            if size < a.min_voxels:
                continue
            vessel_id, host_dist, _ = geo.nearest_vessel_label(
                inst, vessel_map, spacing, tau_mm=a.tau_mm)

            vals, counts = np.unique(gt[inst], return_counts=True)
            nz = vals != 0
            true_id = int(vals[nz][np.argmax(counts[nz])]) if nz.any() else 0
            true_name = spec.locations[true_id - 1] if true_id else "background"

            row = dict(case=case, instance_idx=i, size=size, true_class=true_name,
                      vessel="", host_dist_mm="", n_locs=0, single_location="",
                      junction_best_loc="", junction_best_dist_mm="", arc_fraction="")

            if vessel_id is not None:
                vessel_name = spec.vessels[vessel_id - 1]
                locs = vessel_to_locations.get(vessel_name, [])
                row.update(vessel=vessel_name, host_dist_mm=host_dist, n_locs=len(locs))

                if len(locs) == 1:
                    row["single_location"] = locs[0]
                elif len(locs) > 1:
                    j_loc, j_dist = _junction_candidate(
                        inst, vessel_map, spacing, vessel_id, locs, name_to_id,
                        a.junction_tau_mm)
                    if j_loc is not None:
                        # Side reconciliation is purely geometric (lesion
                        # position + per-case midline calibration), never
                        # prior-dependent -- safe and correct to resolve once
                        # here rather than per bootstrap replicate. Applied
                        # unconditionally (not just when junction ultimately
                        # wins the competition) since it's a no-op whenever
                        # arc wins instead and this value is simply unused.
                        j_loc = vsk.reconcile_side(j_loc, inst, vessel_map, spacing,
                                                   name_to_id, set(locs))
                    frac = raw_arc_fraction(inst, vessel_map, spacing, vessel_name,
                                            vessel_id, name_to_id)
                    row.update(
                        junction_best_loc=(j_loc or ""),
                        junction_best_dist_mm=("" if j_dist == np.inf else j_dist),
                        arc_fraction=("" if frac is None else frac))
            rows.append(row)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} instances cached to {a.out}")


if __name__ == "__main__":
    main()
