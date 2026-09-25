"""
TopBrain 2026 TA36 fine-tune, step 0: build Dataset316_TopBrainVesselFinetune
(nnU-Net raw layout) from the organizers' own TopBrain_Data release, and
carve out a small, fixed held-out split so the fine-tune can be judged
against a real, honest before/after Dice number rather than trusted blind.

Source layout expected under TOPBRAIN_DATA_ROOT (searched recursively so this
doesn't care whether the release's own batch-name subfolder is present):
    imagesTr_topbrain/<case>_0000.nii.gz
    labelsTr_topbrain_v2_topaneu36class/<case>.nii.gz   <- v2 ONLY. The v1_ct/
        v1_mr folders use different, per-modality, non-matching class counts
        (confirmed against Model 2's own dataset.json: v2 is an exact,
        label-for-label match, 1=BA...36=L-ICA-C1-C5).

Both image and label filenames already follow nnU-Net's own naming
convention (<case>_0000.nii.gz / <case>.nii.gz), so this is a straight copy
into nnUNet_raw, not a reformat.

Writes three things:
  1. nnUNet_raw/Dataset316_TopBrainVesselFinetune/{imagesTr,labelsTr}/  (ALL
     cases, train+held-out both live here -- nnU-Net's own do_split() reads
     val cases from the same preprocessed pool, it does not need them kept
     separate on disk)
  2. nnUNet_raw/Dataset316_TopBrainVesselFinetune/dataset.json -- channel
     names/labels copied verbatim from Dataset302's own dataset.json so the
     label ids match exactly (they already do; this just makes it explicit
     and machine-checked rather than assumed)
  3. nnUNet_preprocessed/Dataset316_TopBrainVesselFinetune/splits_final.json
     -- a single fold (index 0) with the held-out cases as 'val' and
     everything else as 'train'. Placed here (not in nnUNet_raw) because
     that's the exact path nnUNetTrainer.do_split() checks before generating
     its own 5-fold CV split.

Deterministic and idempotent like build_expA_holdout.py: won't touch an
existing dataset.json/splits_final.json unless --overwrite is passed, so
re-running this after fine-tuning has already started can't silently change
which cases were held out.

    python -m topaneu_rsna.seg.build_topbrain_finetune_dataset
"""
from __future__ import annotations

import argparse
import json
import shutil

import numpy as np

from topaneu_rsna import config as C

IMAGE_SUBDIR = "imagesTr_topbrain"
LABEL_SUBDIR = "labelsTr_topbrain_v2_topaneu36class"
IMAGE_SUFFIX = "_0000.nii.gz"
LABEL_SUFFIX = ".nii.gz"


def _find_one(root, name):
    if (root / name).is_dir():
        return root / name
    hits = sorted(root.glob(f"**/{name}"))
    if not hits:
        raise FileNotFoundError(f"could not find a '{name}' directory anywhere under {root}")
    if len(hits) > 1:
        raise RuntimeError(f"found multiple '{name}' directories under {root}: {hits} "
                           f"-- pass --data_root pointing directly at the right release folder.")
    return hits[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=type(C.TOPBRAIN_DATA_ROOT), default=C.TOPBRAIN_DATA_ROOT)
    ap.add_argument("--dataset_id", type=int, default=C.DS_TOPBRAIN_FINETUNE)
    ap.add_argument("--n_holdout", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true",
                    help="regenerate dataset.json/splits_final.json even if they already exist "
                         "(does NOT re-copy image/label files that are already present).")
    a = ap.parse_args()

    images_dir = _find_one(a.data_root, IMAGE_SUBDIR)
    labels_dir = _find_one(a.data_root, LABEL_SUBDIR)
    print(f"[topbrain finetune] images: {images_dir}")
    print(f"[topbrain finetune] labels: {labels_dir}")

    cases = sorted(p.name[: -len(IMAGE_SUFFIX)] for p in images_dir.glob(f"*{IMAGE_SUFFIX}"))
    missing_labels = [c for c in cases if not (labels_dir / f"{c}{LABEL_SUFFIX}").exists()]
    if missing_labels:
        raise RuntimeError(f"{len(missing_labels)} case(s) have an image but no v2 label: "
                           f"{missing_labels[:5]}{'...' if len(missing_labels) > 5 else ''}")
    print(f"[topbrain finetune] {len(cases)} cases with matching image + v2 label")

    dataset_name = f"Dataset{a.dataset_id:03d}_{C.DS_NAMES[a.dataset_id]}"
    raw_dir = C.nnUNet_raw / dataset_name
    (raw_dir / "imagesTr").mkdir(parents=True, exist_ok=True)
    (raw_dir / "labelsTr").mkdir(parents=True, exist_ok=True)

    n_copied = 0
    for c in cases:
        src_img = images_dir / f"{c}{IMAGE_SUFFIX}"
        dst_img = raw_dir / "imagesTr" / f"{c}{IMAGE_SUFFIX}"
        src_lbl = labels_dir / f"{c}{LABEL_SUFFIX}"
        dst_lbl = raw_dir / "labelsTr" / f"{c}{LABEL_SUFFIX}"
        if not dst_img.exists():
            shutil.copy2(src_img, dst_img)
            n_copied += 1
        if not dst_lbl.exists():
            shutil.copy2(src_lbl, dst_lbl)
    print(f"[topbrain finetune] copied {n_copied} new image files into {raw_dir} "
         f"({len(cases) - n_copied} already present)")

    dataset_json_path = raw_dir / "dataset.json"
    if dataset_json_path.exists() and not a.overwrite:
        print(f"[topbrain finetune] {dataset_json_path} already exists -- leaving it untouched "
             f"(pass --overwrite to regenerate).")
        dataset_json = json.loads(dataset_json_path.read_text())
    else:
        m2_path = C.nnUNet_raw / f"Dataset{C.DS_VESSEL:03d}_{C.DS_NAMES[C.DS_VESSEL]}" / "dataset.json"
        if not m2_path.exists():
            raise FileNotFoundError(
                f"Dataset302's own dataset.json not found at {m2_path} -- it must exist "
                f"(Dataset302 is already trained) so this script can copy its exact "
                f"channel_names/labels rather than re-deriving them.")
        m2_dataset_json = json.loads(m2_path.read_text())
        dataset_json = {
            "channel_names": m2_dataset_json["channel_names"],
            "labels": m2_dataset_json["labels"],
            "numTraining": len(cases),
            "file_ending": ".nii.gz",
        }
        dataset_json_path.write_text(json.dumps(dataset_json, indent=4))
        print(f"[topbrain finetune] wrote {dataset_json_path} (labels copied verbatim from "
             f"Dataset302's own dataset.json)")

    preprocessed_dir = C.nnUNet_preprocessed / dataset_name
    preprocessed_dir.mkdir(parents=True, exist_ok=True)

    # nnU-Net's own DefaultPreprocessor.run() reads dataset.json from
    # nnUNet_preprocessed/<dataset_name>/, NOT from nnUNet_raw (confirmed from
    # preprocessing/preprocessors/default_preprocessor.py) -- normally
    # nnUNetv2_extract_fingerprint copies it there as a side effect, but we
    # skip that step entirely (we borrow Dataset302's plan wholesale instead
    # of deriving a fresh one), so it has to be copied explicitly here.
    preprocessed_dataset_json_path = preprocessed_dir / "dataset.json"
    preprocessed_dataset_json_path.write_text(json.dumps(dataset_json, indent=4))
    print(f"[topbrain finetune] synced {preprocessed_dataset_json_path} "
         f"(required by nnUNetv2_preprocess, easy to miss since it's not in nnUNet_raw)")
    splits_path = preprocessed_dir / "splits_final.json"
    if splits_path.exists() and not a.overwrite:
        existing = json.loads(splits_path.read_text())
        print(f"[topbrain finetune] {splits_path} already exists "
             f"({len(existing[0]['val'])} held-out cases) -- leaving it untouched.")
    else:
        rng = np.random.default_rng(a.seed)
        holdout = sorted(rng.choice(cases, size=a.n_holdout, replace=False).tolist())
        train = sorted(set(cases) - set(holdout))
        splits = [{"train": train, "val": holdout}]
        splits_path.write_text(json.dumps(splits, indent=2))
        print(f"[topbrain finetune] {len(cases)} cases -> {len(train)} train / "
             f"{len(holdout)} held-out (seed={a.seed})")
        print(f"[topbrain finetune] held-out cases: {holdout}")
        print(f"[topbrain finetune] written to {splits_path}")


if __name__ == "__main__":
    main()
