"""
Create the two nnU-Net datasets on scratch.

  --stage coarse : Dataset301_TopAneuVesselGroup, 3 vessel groups, native spacing
                   (nnU-Net is told to resample to 1 mm via the ForcedLowres planner)
  --stage vessel : Dataset302_TopAneuVessel, all V vessel classes, native spacing
                   (spacing is forced to (0.80,0.45,0.44) in the plans step)

Both read from $TOPANEU_DATA, which is never modified.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from topaneu_rsna import config as C
from topaneu_rsna.utils import io as uio


def _root(ds_id):
    return C.nnUNet_raw / f"Dataset{ds_id:03d}_{C.DS_NAMES[ds_id]}"


def _write_json(root, labels, n, ending=".nii.gz"):
    (root / "dataset.json").write_text(json.dumps({
        "channel_names": {"0": "CTA_MRA"},
        "labels": labels,
        "numTraining": int(n),
        "file_ending": ending,
    }, indent=2))


def _link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src.resolve(), dst)


def build(stage: str, limit: int | None):
    spec = C.load_labels()
    ds_id = C.DS_COARSE if stage == "coarse" else C.DS_VESSEL
    root = _root(ds_id)
    (root / "imagesTr").mkdir(parents=True, exist_ok=True)
    (root / "labelsTr").mkdir(parents=True, exist_ok=True)

    cases = uio.list_cases(C.IMAGES_DIR, C.IMAGE_SUFFIX)
    if limit:
        cases = cases[:limit]

    # vessel label value -> coarse group (1..3)
    v_to_group = np.zeros(spec.n_vessel + 1, dtype=np.uint8)
    for i, v in enumerate(spec.vessels):
        v_to_group[i + 1] = spec.coarse_groups[v]

    n = 0
    for case in tqdm(cases, desc=stage):
        vp = C.VESSEL_MASKS / f"{case}{C.LABEL_SUFFIX}"
        ip = C.IMAGES_DIR / f"{case}{C.IMAGE_SUFFIX}"
        if not vp.exists() or not ip.exists():
            continue

        # images are identical for both datasets -> symlink, never copy 400+ volumes
        _link_or_copy(ip, root / "imagesTr" / f"{case}_0000.nii.gz")

        if stage == "vessel":
            _link_or_copy(vp, root / "labelsTr" / f"{case}.nii.gz")
        else:
            lab, meta = uio.read(vp)
            grp = v_to_group[np.clip(lab.astype(np.int64), 0, spec.n_vessel)]
            uio.write(grp.astype(np.uint8), meta, root / "labelsTr" / f"{case}.nii.gz")
        n += 1

    if stage == "vessel":
        labels = {"background": 0}
        labels.update({v: i + 1 for i, v in enumerate(spec.vessels)})
    else:
        labels = {"background": 0, "posterior_basilar": 1, "mca": 2, "other": 3}
    _write_json(root, labels, n)
    print(f"{root}  ({n} cases, {len(labels) - 1} classes)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["coarse", "vessel"], required=True)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    build(a.stage, a.limit)
