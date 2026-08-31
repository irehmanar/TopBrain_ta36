"""
Build Dataset306_TopAneuAneurysmVesselness: binary aneurysm segmentation, whole-head,
with a Frangi vesselness map as a second input channel.

Same images/spacing/labels as Dataset304/DS_ANEURYSM (prep/build_nnunet_datasets.py
--stage aneurysm): location_masks collapsed to a single foreground class. The only
difference is channel 1, a multi-scale Frangi vesselness response meant to give the
network an explicit "here's where the tubular vessel structures are" prior, since
aneurysms only occur on vessels.

Vesselness source, checked per case in this order (no recomputation if avoidable --
the Frangi run took 3 days last time):
  1. LEGACY_VESSELNESS_DIR: imagesTr/<case>_0001.nii.gz inside the already-built
     01_rsna_pipeline/Dataset104_TopAneuLocationVesselness.
  2. VESSELNESS_DIR: <case>.nii.gz written by prep/compute_vesselness.py (job 18),
     for any case Dataset104 doesn't cover.
A case missing from both is skipped with a warning; run job 18 to fill the gap.

Dataset104 was built on a different, native/un-resampled voxel grid than this
pipeline's own images (its volumes vary in shape case to case, e.g. 264x426x352,
while this pipeline's imagesTr is a fixed 128x256x256 crop) -- nnU-Net's dataset
integrity check rejects mismatched channel shapes outright. So a legacy vesselness
map is NOT symlinked as-is: it's resampled onto this case's own _0000 image's grid
(same size/spacing/origin/direction) first, via utils.io.resample_to_reference,
under the assumption both volumes already share the same physical (world)
coordinate frame -- i.e. they're different crops/resamplings of the same scan, not
volumes needing registration. A cheap header-only bounding-box overlap check
(utils.io.physical_overlap_frac) guards against a case where that assumption is
wrong: too little physical overlap between the two grids means the resampled
vesselness would land somewhere physically wrong in the reference frame, so that
case is skipped rather than silently fed to training. Freshly computed vesselness
(source 2 above) needs no resampling -- compute_vesselness.py already writes it on
the exact grid of this pipeline's own image, by construction.

    python -m topaneu_rsna.prep.build_aneurysm_vesselness_dataset
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from topaneu_rsna import config as C
from topaneu_rsna.utils import io as uio
from topaneu_rsna.prep.build_nnunet_datasets import _link_or_copy

MIN_OVERLAP_FRAC = 0.3  # below this, treat the legacy vesselness grid as misaligned


def _root():
    return (C.nnUNet_raw
            / f"Dataset{C.DS_ANEURYSM_VESSELNESS:03d}_{C.DS_NAMES[C.DS_ANEURYSM_VESSELNESS]}")


def _write_json(root, n):
    (root / "dataset.json").write_text(json.dumps({
        "channel_names": {"0": "CTA_MRA", "1": "vesselness"},
        "labels": {"background": 0, "aneurysm": 1},
        "numTraining": int(n),
        "file_ending": ".nii.gz",
    }, indent=2))


def _place_vesselness(case: str, ip: Path, legacy_dir: Path, computed_dir: Path,
                      out_path: Path) -> str:
    """Returns 'legacy', 'computed', 'misaligned', or 'missing'."""
    legacy = legacy_dir / f"{case}_0001.nii.gz"
    if legacy.exists():
        overlap = uio.physical_overlap_frac(ip, legacy)
        if overlap < MIN_OVERLAP_FRAC:
            print(f"[SKIP] {case}: legacy vesselness only overlaps {overlap:.0%} of the "
                  f"reference image's world-space bbox (< {MIN_OVERLAP_FRAC:.0%}) -- "
                  f"treating as a grid mismatch, not resampling it in")
            return "misaligned"
        vess, meta = uio.resample_to_reference(legacy, ip, is_label=False)
        uio.write(vess.astype(np.float32), meta, out_path)
        return "legacy"

    computed = computed_dir / f"{case}.nii.gz"
    if computed.exists():
        _link_or_copy(computed, out_path)
        return "computed"

    return "missing"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--legacy_vesselness_dir", type=Path, default=C.LEGACY_VESSELNESS_DIR)
    ap.add_argument("--vesselness_dir", type=Path, default=C.VESSELNESS_DIR)
    ap.add_argument("--min_overlap", type=float, default=MIN_OVERLAP_FRAC)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    global MIN_OVERLAP_FRAC
    MIN_OVERLAP_FRAC = a.min_overlap

    root = _root()
    (root / "imagesTr").mkdir(parents=True, exist_ok=True)
    (root / "labelsTr").mkdir(parents=True, exist_ok=True)

    cases = uio.list_cases(C.IMAGES_DIR, C.IMAGE_SUFFIX)
    if a.limit:
        cases = cases[:a.limit]

    n = 0
    counts = {"legacy": 0, "computed": 0, "misaligned": 0, "missing": 0, "no_label": 0}
    for case in tqdm(cases, desc="aneurysm-vesselness"):
        ip = C.IMAGES_DIR / f"{case}{C.IMAGE_SUFFIX}"
        lp = C.LOCATION_MASKS / f"{case}{C.LABEL_SUFFIX}"
        if not (ip.exists() and lp.exists()):
            counts["no_label" if ip.exists() else "missing"] += 1
            continue

        status = _place_vesselness(case, ip, a.legacy_vesselness_dir, a.vesselness_dir,
                                   root / "imagesTr" / f"{case}_0001.nii.gz")
        counts[status] += 1
        if status in ("misaligned", "missing"):
            continue

        _link_or_copy(ip, root / "imagesTr" / f"{case}_0000.nii.gz")
        lab, meta = uio.read(lp)
        uio.write((lab > 0).astype(np.uint8), meta, root / "labelsTr" / f"{case}.nii.gz")
        n += 1

    skipped = counts["misaligned"] + counts["missing"] + counts["no_label"]
    if skipped:
        print(f"WARNING: {skipped} case(s) skipped -- {counts['missing']} missing "
              f"image/label, {counts['no_label']} missing label, "
              f"{counts['misaligned']} legacy vesselness grid didn't overlap enough "
              f"(run jobs/18_compute_vesselness.sbatch to fill any of these gaps)")

    _write_json(root, n)
    print(f"{root}  ({n} cases: {counts['legacy']} from Dataset104 (resampled), "
          f"{counts['computed']} freshly computed, 2 input channels, 1 foreground class)")


if __name__ == "__main__":
    main()
