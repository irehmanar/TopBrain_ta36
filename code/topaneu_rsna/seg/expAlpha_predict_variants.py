"""
Experiment Alpha, step 1: generate the two prediction sets NOT already on disk
from Dataset304's real 5-fold training (job 13) -- inference only, no
retraining, nothing under nnUNet_results/Dataset304.../fold_*/ is touched.

The existing baseline ("single fold, own validation cases only, TTA on" --
nnU-Net's default post-training validation pass) is NOT regenerated here; it
is read directly from fold_*/validation/ by evaluate_location.py's own
load_binary_pred_paths(), exactly as the ~0.60 pooled Dice number already
documented in this pipeline was computed.

Two variants, each written under EXPALPHA_PRED_DIR so nothing overwrites the
existing baseline outputs:

  --variant notta
      Same case set, same "own fold only" assignment as the baseline, but
      with test-time mirroring augmentation disabled (--disable_tta). Runs
      fold-by-fold: for fold f, predicts ONLY that fold's own held-out
      validation cases with ONLY that fold's model (-f f). Leak-free --
      isolates exactly the TTA contribution against the existing baseline.

  --variant ensemble5
      Every case in Dataset304's imagesTr, predicted with all 5 folds
      ensembled (-f 0 1 2 3 4, nnU-Net averages softmax probabilities across
      the 5 folds before argmax -- confirmed by reading
      nnunetv2/inference/predict_from_raw_data.py's own multi-fold loading
      path, not assumed). See config.py's EXPALPHA_PRED_DIR docstring for why
      this result is reported as a leaked/optimistic upper bound, not a valid
      generalization estimate: every case here was training data for 4 of
      the 5 ensembled models.

    python -m topaneu_rsna.seg.expAlpha_predict_variants --variant notta
    python -m topaneu_rsna.seg.expAlpha_predict_variants --variant ensemble5
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from topaneu_rsna import config as C
from topaneu_rsna.utils import io as uio


def _link_inputs(cases: list[str], src_images_dir: Path, dst_dir: Path):
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    for case in cases:
        src = src_images_dir / f"{case}{C.IMAGE_SUFFIX}"
        dst = dst_dir / f"{case}{C.IMAGE_SUFFIX}"
        dst.symlink_to(src.resolve())


def run_notta(dataset_id: int, trainer: str, plans: str):
    # built explicitly (not via config.py's seg_model_dir, which hardcodes
    # PLANS_RESENC and would silently ignore a non-default --plans here)
    model_dir = (C.nnUNet_results / f"Dataset{dataset_id:03d}_{C.DS_NAMES[dataset_id]}"
                / f"{trainer}__{plans}__3d_fullres")
    images_dir = C.nnUNet_raw / f"Dataset{dataset_id:03d}_{C.DS_NAMES[dataset_id]}" / "imagesTr"
    out_dir = C.EXPALPHA_PRED_DIR / "notta"
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_root = C.EXPALPHA_PRED_DIR / "_notta_inputs"

    for fold in range(5):
        val_dir = model_dir / f"fold_{fold}" / "validation"
        cases = uio.list_cases(val_dir, C.LABEL_SUFFIX)
        if not cases:
            print(f"[expAlpha] fold {fold}: no validation cases found, skipping")
            continue
        print(f"[expAlpha] fold {fold}: {len(cases)} own-validation cases, TTA disabled")
        in_dir = tmp_root / f"fold_{fold}"
        _link_inputs(cases, images_dir, in_dir)

        subprocess.run([
            "nnUNetv2_predict",
            "-i", str(in_dir),
            "-o", str(out_dir),
            "-d", str(dataset_id),
            "-c", "3d_fullres",
            "-p", plans,
            "-tr", trainer,
            "-f", str(fold),
            "--disable_tta",
            "-chk", "checkpoint_final.pth",
        ], check=True)

    shutil.rmtree(tmp_root, ignore_errors=True)
    print(f"[expAlpha] notta predictions written to {out_dir}")


def run_ensemble5(dataset_id: int, trainer: str, plans: str):
    images_dir = C.nnUNet_raw / f"Dataset{dataset_id:03d}_{C.DS_NAMES[dataset_id]}" / "imagesTr"
    out_dir = C.EXPALPHA_PRED_DIR / "ensemble5"
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = uio.list_cases(images_dir, C.IMAGE_SUFFIX)
    print(f"[expAlpha] ensembling folds 0-4 (TTA on, default) over all {len(cases)} "
          f"cases -- see config.py's EXPALPHA_PRED_DIR docstring: this is a leaked/"
          f"optimistic upper bound, NOT a fair generalization estimate, since every "
          f"case here was training data for 4 of the 5 ensembled fold models.")

    # Build a clean, symlink-only input dir with ONLY each case's _0000 file,
    # rather than pointing -i at the shared imagesTr folder directly -- that
    # folder turned out to contain stray _0001.nii.gz files (128x256x256
    # ROI-crop shape, from some other experiment) for at least a few cases,
    # which made nnUNetv2_predict think Dataset304 (declared single-channel)
    # had 2 input channels and crash on the shape mismatch. Same defensive
    # pattern run_notta() already uses, just applied here too now that
    # "read-only means safe" has been shown wrong.
    in_dir = C.EXPALPHA_PRED_DIR / "_ensemble5_inputs"
    _link_inputs(cases, images_dir, in_dir)

    subprocess.run([
        "nnUNetv2_predict",
        "-i", str(in_dir),
        "-o", str(out_dir),
        "-d", str(dataset_id),
        "-c", "3d_fullres",
        "-p", plans,
        "-tr", trainer,
        "-f", "0", "1", "2", "3", "4",
        "-chk", "checkpoint_final.pth",
    ], check=True)

    shutil.rmtree(in_dir, ignore_errors=True)
    print(f"[expAlpha] ensemble5 predictions written to {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["notta", "ensemble5"], required=True)
    ap.add_argument("--dataset", type=int, default=C.DS_ANEURYSM)
    ap.add_argument("--trainer", default=C.TRAINER_LOC)
    ap.add_argument("--plans", default=C.PLANS_RESENC)
    a = ap.parse_args()

    if a.variant == "notta":
        run_notta(a.dataset, a.trainer, a.plans)
    else:
        run_ensemble5(a.dataset, a.trainer, a.plans)


if __name__ == "__main__":
    main()
