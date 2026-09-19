"""Why are large detected instances so often assigned to the wrong location?

In the honest single-fold run (job 124) nearly every instance of >= ~4000 voxels
hosted by ICA C1-C5 or VA was wrong (true class: C7 Pcom junction, C7 non-branch,
BA trunk, BA tip ...), while small instances were mostly right. The host vessel
is the touching label with the LARGEST contact patch, which for a large sac
tends to be the long parent artery rather than the vessel at the neck.

This records, for every instance of at least MIN_VOX voxels, ALL vessel labels
it touches and how many voxels each contacts, next to the current host, the
current assignment and the ground-truth class, so a better host-selection rule
can be read off the table (or scored) instead of guessed. CPU only, existing
predictions only.

    python -m topaneu_rsna.seg.host_contacts
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

PAD_MM = 20.0


def contacts(inst, vessel_map, spacing, names):
    sp = np.asarray(spacing, np.float64)
    idx = np.argwhere(inst)
    pad = np.ceil(PAD_MM / sp).astype(int)
    lo = np.maximum(idx.min(0) - pad, 0)
    hi = np.minimum(idx.max(0) + pad + 1, inst.shape)
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    zone = ndimage.binary_dilation(inst[sl], iterations=2)
    ves = vessel_map[sl]
    vals, cnt = np.unique(ves[zone & (ves > 0)], return_counts=True)
    order = np.argsort(-cnt)
    return ";".join(f"{names[int(vals[i]) - 1]}:{int(cnt[i])}" for i in order)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vessel_pred_dir", type=Path,
                    default=C.SCRATCH_ROOT / "work" / "vessel_pred_m2_fullhead")
    ap.add_argument("--instances_csv", type=Path,
                    default=C.LOG_ROOT / "task2_rule_instances_realvessel_singlefold.csv")
    ap.add_argument("--out", type=Path, default=C.LOG_ROOT / "task2_host_contacts.csv")
    ap.add_argument("--min_vox", type=int, default=3000)
    a = ap.parse_args()

    spec = C.load_labels()
    bin_paths = load_binary_pred_paths(C.DS_ANEURYSM, C.TRAINER_LOC, C.PLANS_RESENC,
                                       [0, 1, 2, 3, 4])
    rows = []
    for case, bp in tqdm(bin_paths.items(), desc="contacts"):
        vp = a.vessel_pred_dir / f"{case}{C.LABEL_SUFFIX}"
        if not vp.exists():
            continue
        binmask, meta = uio.read(bp)
        vessel_map, _ = uio.read(vp)
        lab, n = ndimage.label(binmask > 0)
        for i in range(1, n + 1):
            inst = lab == i
            size = int(inst.sum())
            if size < a.min_vox:
                continue
            vid, _, _ = geo.nearest_vessel_label(inst, vessel_map, meta["spacing"], tau_mm=4.0)
            rows.append(dict(case=case, instance_idx=i, size=size,
                             host=(spec.vessels[vid - 1] if vid else ""),
                             touching=contacts(inst, vessel_map, meta["spacing"], spec.vessels)))

    new = pd.DataFrame(rows)
    old = pd.read_csv(a.instances_csv)[["case", "instance_idx", "assigned", "true_class",
                                        "correct"]]
    m = new.merge(old, on=["case", "instance_idx"], how="left")
    m.to_csv(a.out, index=False)
    print(f"{len(m)} instances of >= {a.min_vox} voxels; currently correct: "
          f"{int(m.correct.fillna(False).sum())} of {len(m)}")
    pd.set_option("display.width", 250, "display.max_colwidth", 70, "display.max_rows", 200)
    show = m.sort_values("size", ascending=False)[
        ["size", "host", "true_class", "correct", "touching"]]
    print("\nlargest first (touching = vessel:contact voxels, biggest first):")
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
