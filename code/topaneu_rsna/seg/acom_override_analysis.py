"""Test whether the biggest single location error in the honest single-fold
run (job 124: 23 real Acom-complex aneurysms assigned to L-4.2 A1 / R-4.3 A2)
comes from nearest_vessel_label() picking the A1A2 label because it has the
largest contact patch, even though the Acom label is also right there.

For every detected instance this records the distance (mm) to the Acom vessel
label and the full list of touching vessel labels, then replays the assignment
under an override rule ("if the host vessel is L-A1A2/R-A1A2 and the Acom label
is within D mm, call it 4.1 Acom complex") for several D, joining on
(case, instance_idx) to job 124's per-instance CSV so the CHANGE in the number
of correct instances is measured directly against the current behaviour.
No model inference: reads existing binary + vessel predictions only.

    python -m topaneu_rsna.seg.acom_override_analysis
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from tqdm import tqdm

from topaneu_rsna import config as C
from topaneu_rsna.utils import geometry as geo
from topaneu_rsna.utils import io as uio
from topaneu_rsna.seg.assign_location_rule import load_binary_pred_paths

A1A2_VESSELS = {"L-A1A2", "R-A1A2"}
ACOM_VESSEL = "Acom"
ACOM_LOCATION = "4.1 Acom complex"
PAD_MM = 20.0


def acom_distance(inst, vessel_map, spacing, acom_id):
    sp = np.asarray(spacing, np.float64)
    idx = np.argwhere(inst)
    pad = np.ceil(PAD_MM / sp).astype(int)
    lo = np.maximum(idx.min(0) - pad, 0)
    hi = np.minimum(idx.max(0) + pad + 1, inst.shape)
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    ves = vessel_map[sl]
    target = ves == acom_id
    if not target.any():
        return float("inf")
    dt = ndimage.distance_transform_edt(~inst[sl], sampling=sp)
    return float(dt[target].min())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vessel_pred_dir", type=Path,
                    default=C.SCRATCH_ROOT / "work" / "vessel_pred_m2_fullhead")
    ap.add_argument("--instances_csv", type=Path,
                    default=C.LOG_ROOT / "task2_rule_instances_realvessel_singlefold.csv")
    ap.add_argument("--out", type=Path,
                    default=C.LOG_ROOT / "task2_acom_override_analysis.csv")
    ap.add_argument("--min_voxels", type=int, default=3)
    a = ap.parse_args()

    spec = C.load_labels()
    name_to_id = {v: i + 1 for i, v in enumerate(spec.vessels)}
    acom_id = name_to_id[ACOM_VESSEL]

    bin_paths = load_binary_pred_paths(C.DS_ANEURYSM, C.TRAINER_LOC, C.PLANS_RESENC,
                                       [0, 1, 2, 3, 4])
    rows = []
    for case, bp in tqdm(bin_paths.items(), desc="acom distance"):
        vp = a.vessel_pred_dir / f"{case}{C.LABEL_SUFFIX}"
        if not vp.exists():
            continue
        binmask, meta = uio.read(bp)
        vessel_map, _ = uio.read(vp)
        lab, n = ndimage.label(binmask > 0)
        for i in range(1, n + 1):
            inst = lab == i
            size = int(inst.sum())
            if size < a.min_voxels:
                continue
            vessel_id, host_dist, touching = geo.nearest_vessel_label(
                inst, vessel_map, meta["spacing"], tau_mm=4.0)
            rows.append(dict(
                case=case, instance_idx=i, size=size,
                host_vessel=(spec.vessels[vessel_id - 1] if vessel_id else ""),
                touches_acom=(acom_id in touching),
                acom_dist_mm=acom_distance(inst, vessel_map, meta["spacing"], acom_id)))

    new = pd.DataFrame(rows)
    old = pd.read_csv(a.instances_csv)
    m = old.merge(new[["case", "instance_idx", "size", "host_vessel", "touches_acom",
                       "acom_dist_mm"]],
                  on=["case", "instance_idx"], suffixes=("", "_new"))
    m.to_csv(a.out, index=False)
    print(f"joined {len(m)} of {len(old)} job-124 instances "
          f"(sanity: vessel matches for {(m.vessel == m.host_vessel).mean():.3f})")

    base_correct = int(m.correct.sum())
    print(f"\nbaseline correct: {base_correct} of {len(m)}")
    a1 = m[m.host_vessel.isin(A1A2_VESSELS)]
    print(f"instances hosted by A1A2: {len(a1)}; true Acom among them: "
          f"{int((a1.true_class == ACOM_LOCATION).sum())}; "
          f"currently correct: {int(a1.correct.sum())}")
    print("\nacom_dist_mm quantiles, by true class group, for A1A2-hosted instances:")
    grp = a1.assign(g=np.where(a1.true_class == ACOM_LOCATION, "true=Acom",
                        np.where(a1.true_class == "background", "background", "true=A1/A2/other")))
    print(grp.groupby("g").acom_dist_mm.describe()[["count", "25%", "50%", "75%", "max"]])

    print("\noverride rule: A1A2-hosted and acom_dist<=D -> 4.1 Acom complex")
    print("   D  flipped  fixed  broken  net  total_correct")
    for D in [0, 1, 2, 3, 4, 6, 8, 10, 15]:
        sel = m.host_vessel.isin(A1A2_VESSELS) & (m.acom_dist_mm <= D)
        new_ok = np.where(sel, m.true_class == ACOM_LOCATION, m.correct)
        fixed = int((sel & ~m.correct & (m.true_class == ACOM_LOCATION)).sum())
        broken = int((sel & m.correct & (m.true_class != ACOM_LOCATION)).sum())
        print(f"{D:4d} {int(sel.sum()):8d} {fixed:6d} {broken:7d} {fixed - broken:4d} "
              f"{int(new_ok.sum()):13d}")


if __name__ == "__main__":
    main()
