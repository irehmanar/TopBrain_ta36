"""
Build Dataset306_TopAneuAneurysmVesselness: binary aneurysm segmentation, whole-head,
with a Frangi vesselness map as a second input channel.

Same images/spacing/labels as Dataset304/DS_ANEURYSM (prep/build_nnunet_datasets.py
--stage aneurysm): location_masks collapsed to a single foreground class. The only
difference is channel 1, a multi-scale Frangi vesselness response meant to give the
network an explicit "here's where the tubular vessel structures are" prior, since
aneurysms only occur on vessels.

Vesselness source is VESSELNESS_DIR, written by prep/compute_vesselness.py (job 18)
directly on this pipeline's own image grid -- run job 18 to completion before this.
[TOPANEU] An earlier version of this script defaulted to reusing vesselness already
computed for the unrelated 01_rsna_pipeline/Dataset104_TopAneuLocationVesselness
experiment (to skip the ~3-day Frangi run), resampled onto this pipeline's grid.
That resampling relied on an unverified assumption -- that Dataset104's native/
un-resampled grid shares the same physical (world) coordinate frame as this
pipeline's images -- which is exactly the kind of silent-misalignment risk not
worth taking for training data; recomputing vesselness directly on this pipeline's
own images sidesteps the question entirely. Pass --legacy_vesselness_dir to opt
back into that reuse path if ever needed; it's kept below but unused by default.

Separately, a handful of cases (seen so far: some center4_ct_* cases) have a
location_masks label at native resolution instead of this pipeline's image grid,
e.g. shape (274,369,355) vs the image's (128,256,256) -- unrelated to vesselness.
Checked with a cheap header-only shape/spacing comparison (mirroring nnU-Net's own
integrity check) so the common case (already matching) is untouched; a mismatch is
resampled onto the image's grid (nearest-neighbor, since it's categorical), guarded
by the same physical-overlap sanity check used for the legacy vesselness path
below, or skipped if the overlap looks too small to trust.

nnU-Net's own dataset-integrity check requires exact shape/spacing agreement
between every channel and the label for a case, so any mismatch left as-is fails
the build outright.

    python -m topaneu_rsna.prep.build_aneurysm_vesselness_dataset
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm

from topaneu_rsna import config as C
from topaneu_rsna.utils import io as uio
from topaneu_rsna.prep.build_nnunet_datasets import _link_or_copy

MIN_OVERLAP_FRAC = 0.3  # below this, treat a mismatched grid as misaligned, not resamplable


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


def _place_vesselness(case: str, ip: Path, legacy_dir: Path | None, computed_dir: Path,
                      out_path: Path) -> str:
    """Returns 'legacy', 'computed', 'misaligned', or 'missing'."""
    if legacy_dir is not None:
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


def _geometry_matches(ip: Path, lp: Path) -> bool:
    """Mirrors nnU-Net's own integrity check (np.allclose, i.e. essentially exact --
    default rtol=1e-5/atol=1e-8) so a case that passes here also passes that check."""
    hi, hl = uio.read_header(ip), uio.read_header(lp)
    if hi["shape"] != hl["shape"]:
        return False
    return np.allclose(hi["spacing"], hl["spacing"])


def _place_label(case: str, ip: Path, lp: Path, out_path: Path) -> str:
    """Returns 'ok', 'resampled', or 'misaligned'."""
    if _geometry_matches(ip, lp):
        lab, meta = uio.read(lp)
        uio.write((lab > 0).astype(np.uint8), meta, out_path)
        return "ok"

    overlap = uio.physical_overlap_frac(ip, lp)
    if overlap < MIN_OVERLAP_FRAC:
        print(f"[SKIP] {case}: location_masks is on a different grid than the image and "
              f"only overlaps {overlap:.0%} of its world-space bbox (< {MIN_OVERLAP_FRAC:.0%}) "
              f"-- treating as a grid mismatch, not resampling it in")
        return "misaligned"

    lab, meta = uio.resample_to_reference(lp, ip, is_label=True)
    uio.write((lab > 0).astype(np.uint8), meta, out_path)
    return "resampled"


def main():
    global MIN_OVERLAP_FRAC
    ap = argparse.ArgumentParser()
    ap.add_argument("--legacy_vesselness_dir", type=Path, default=None,
                    help="opt back into reusing/resampling Dataset104's vesselness "
                         "(unused by default -- pass e.g. %s to enable" % C.LEGACY_VESSELNESS_DIR)
    ap.add_argument("--vesselness_dir", type=Path, default=C.VESSELNESS_DIR)
    ap.add_argument("--min_overlap", type=float, default=MIN_OVERLAP_FRAC)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    MIN_OVERLAP_FRAC = a.min_overlap

    root = _root()
    if root.exists():  # avoid stale files (e.g. bad labels) from a previous failed build
        shutil.rmtree(root)
    (root / "imagesTr").mkdir(parents=True, exist_ok=True)
    (root / "labelsTr").mkdir(parents=True, exist_ok=True)

    cases = uio.list_cases(C.IMAGES_DIR, C.IMAGE_SUFFIX)
    if a.limit:
        cases = cases[:a.limit]

    n = 0
    counts = {"legacy": 0, "computed": 0, "misaligned": 0, "missing": 0, "no_label": 0,
             "label_ok": 0, "label_resampled": 0, "label_misaligned": 0}
    for case in tqdm(cases, desc="aneurysm-vesselness"):
        ip = C.IMAGES_DIR / f"{case}{C.IMAGE_SUFFIX}"
        lp = C.LOCATION_MASKS / f"{case}{C.LABEL_SUFFIX}"
        if not (ip.exists() and lp.exists()):
            counts["no_label" if ip.exists() else "missing"] += 1
            continue

        vstatus = _place_vesselness(case, ip, a.legacy_vesselness_dir, a.vesselness_dir,
                                    root / "imagesTr" / f"{case}_0001.nii.gz")
        counts[vstatus] += 1
        if vstatus in ("misaligned", "missing"):
            continue

        lstatus = _place_label(case, ip, lp, root / "labelsTr" / f"{case}.nii.gz")
        counts[f"label_{lstatus}"] += 1
        if lstatus == "misaligned":
            (root / "imagesTr" / f"{case}_0001.nii.gz").unlink(missing_ok=True)
            continue

        _link_or_copy(ip, root / "imagesTr" / f"{case}_0000.nii.gz")
        n += 1

    skipped = counts["misaligned"] + counts["missing"] + counts["no_label"] + counts["label_misaligned"]
    if skipped:
        print(f"WARNING: {skipped} case(s) skipped -- {counts['missing']} missing "
              f"image/label, {counts['no_label']} missing label, "
              f"{counts['misaligned']} legacy vesselness grid didn't overlap enough, "
              f"{counts['label_misaligned']} location_masks grid didn't overlap enough "
              f"(run jobs/18_compute_vesselness.sbatch to fill any vesselness gaps)")

    _write_json(root, n)
    print(f"{root}  ({n} cases: vesselness {counts['legacy']} from Dataset104 (resampled) + "
          f"{counts['computed']} freshly computed; label {counts['label_resampled']} "
          f"resampled onto the image grid; 2 input channels, 1 foreground class)")


if __name__ == "__main__":
    main()
