"""
The one clean, controlled test of vessel-conditioning for binary aneurysm
segmentation that was never actually run. Builds Dataset317, identical to
Dataset304 (jobs/04_segmentation_simple_2class) in every respect that
matters for a fair comparison:

  - the SAME 417 cases (Dataset304's own imagesTr/labelsTr inclusion rule:
    every case with both an image and a location_masks label)
  - the SAME whole-head, native-per-case grid (no ROI cropping -- unlike
    Dataset305, which changed the crop AND added a vessel channel at the
    same time, confounding the two)
  - the SAME real 5-fold split -- copied byte-for-byte from Dataset304's own
    splits_final.json (not regenerated), so "fold 2" here is the exact same
    train/val case split as "fold 2" in Dataset304
  - the SAME nnUNetResEncUNetMPlans plan, reused via
    nnUNetv2_move_plans_between_datasets rather than re-derived, so patch
    size, spacing and normalization are identical too

The ONLY thing that differs is a second input channel: a binarized
("on some vessel" 0/1) real Model 2 whole-head vessel prediction, from
VESSEL_PRED_M2_FULLHEAD (job 78's cache -- already computed for every case
in this pipeline, TTA on, fold_all, so no new GPU inference is needed here).
This is the same 0/1 convention Dataset305 used for its own vessel channel,
not Dataset313's raw 36-class float channel (a binary presence signal is
the well-posed way to test "aneurysms form near vessels"; a raw class-id
integer has no meaningful ordinal relationship for the network to exploit).

Every earlier attempt at this hypothesis changed something else alongside
the channel:
  - Dataset305: ROI-cropped AND vessel-conditioned at once
  - Dataset306: a Frangi vesselness filter, not a real trained model
  - Dataset311: crashed before epoch 1, no result either way
This dataset isolates the one variable: image-only vs. image+vessel, with
everything else held fixed.

    python -m topaneu_rsna.seg.build_aneurysm_vesselcond_dataset
"""
from __future__ import annotations

import argparse
import json
import shutil

from topaneu_rsna import config as C
from topaneu_rsna.utils import io as uio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_id", type=int, default=C.DS_ANEURYSM_VESSELCOND,
                    help="317 = base Model 2 vessel channel, 318 = class-balanced "
                         "retrain's vessel channel")
    ap.add_argument("--vessel_pred_dir", type=type(C.VESSEL_PRED_M2_FULLHEAD),
                    default=C.VESSEL_PRED_M2_FULLHEAD,
                    help="whole-head vessel-prediction cache to read channel 1 from")
    ap.add_argument("--overwrite", action="store_true",
                    help="regenerate dataset.json/splits_final.json even if they already exist")
    a = ap.parse_args()

    # Dataset304's own inclusion rule (build_nnunet_datasets.py, stage="aneurysm"):
    # every case with both an image and a location_masks label.
    cases = uio.list_cases(C.IMAGES_DIR, C.IMAGE_SUFFIX)
    cases = [c for c in cases if (C.LOCATION_MASKS / f"{c}{C.LABEL_SUFFIX}").exists()]
    print(f"[vesselcond] {len(cases)} cases match Dataset304's own inclusion rule")

    missing_vessel = [c for c in cases if not (a.vessel_pred_dir / f"{c}.nii.gz").exists()]
    if missing_vessel:
        raise RuntimeError(
            f"{len(missing_vessel)} of Dataset304's {len(cases)} cases have no whole-head "
            f"vessel prediction in {a.vessel_pred_dir}: {missing_vessel[:10]}"
            f"{'...' if len(missing_vessel) > 10 else ''}. "
            f"This experiment is only a fair comparison to Dataset304 if it uses the exact "
            f"same case set -- generate the missing predictions before building this "
            f"dataset, do not silently drop cases.")
    print(f"[vesselcond] all {len(cases)} cases have a cached whole-head vessel prediction "
         f"in {a.vessel_pred_dir}")

    dataset_name = f"Dataset{a.dataset_id:03d}_{C.DS_NAMES[a.dataset_id]}"
    raw_dir = C.nnUNet_raw / dataset_name
    (raw_dir / "imagesTr").mkdir(parents=True, exist_ok=True)
    (raw_dir / "labelsTr").mkdir(parents=True, exist_ok=True)

    n_written = 0
    for case in cases:
        dst_img = raw_dir / "imagesTr" / f"{case}_0000.nii.gz"
        dst_vessel = raw_dir / "imagesTr" / f"{case}_0001.nii.gz"
        dst_lab = raw_dir / "labelsTr" / f"{case}{C.LABEL_SUFFIX}"
        if dst_img.exists() and dst_vessel.exists() and dst_lab.exists() and not a.overwrite:
            continue

        img, meta = uio.read(C.IMAGES_DIR / f"{case}{C.IMAGE_SUFFIX}")
        uio.write(img, meta, dst_img)

        vessel, _ = uio.read(a.vessel_pred_dir / f"{case}.nii.gz")
        vessel_binary = (vessel > 0).astype(vessel.dtype)
        uio.write(vessel_binary, meta, dst_vessel)

        lab, _ = uio.read(C.LOCATION_MASKS / f"{case}{C.LABEL_SUFFIX}")
        uio.write((lab > 0).astype(lab.dtype), meta, dst_lab)
        n_written += 1

    print(f"[vesselcond] wrote {n_written} new cases into {raw_dir} "
         f"({len(cases) - n_written} already present)")

    dataset_json_path = raw_dir / "dataset.json"
    dataset_json = {
        "channel_names": {"0": "CTA_MRA", "1": "vessel_mask"},
        "labels": {"background": 0, "aneurysm": 1},
        "numTraining": len(cases),
        "file_ending": ".nii.gz",
    }
    if dataset_json_path.exists() and not a.overwrite:
        print(f"[vesselcond] {dataset_json_path} already exists -- leaving it untouched")
    else:
        dataset_json_path.write_text(json.dumps(dataset_json, indent=4))
        print(f"[vesselcond] wrote {dataset_json_path}")

    # nnU-Net's DefaultPreprocessor reads dataset.json from nnUNet_preprocessed/,
    # not nnUNet_raw/ -- same gap hit (and fixed) for Dataset316's build.
    preprocessed_dir = C.nnUNet_preprocessed / dataset_name
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    (preprocessed_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=4))
    print(f"[vesselcond] synced {preprocessed_dir / 'dataset.json'}")

    # Copy Dataset304's own real 5-fold split byte-for-byte, so this experiment's
    # folds are guaranteed identical to Dataset304's, not just independently
    # re-derived from the same (fortunately identical) case list.
    src_splits = (C.nnUNet_preprocessed
                 / f"Dataset{C.DS_ANEURYSM:03d}_{C.DS_NAMES[C.DS_ANEURYSM]}"
                 / "splits_final.json")
    dst_splits = preprocessed_dir / "splits_final.json"
    if not src_splits.exists():
        raise FileNotFoundError(
            f"Dataset304's splits_final.json not found at {src_splits} -- it must exist "
            f"(Dataset304 already trained 5 real folds) so this experiment can reuse its "
            f"exact fold assignment.")
    if dst_splits.exists() and not a.overwrite:
        print(f"[vesselcond] {dst_splits} already exists -- leaving it untouched")
    else:
        shutil.copy2(src_splits, dst_splits)
        splits = json.loads(dst_splits.read_text())
        n_folds = len(splits)
        print(f"[vesselcond] copied Dataset304's {n_folds}-fold split to {dst_splits}")


if __name__ == "__main__":
    main()
