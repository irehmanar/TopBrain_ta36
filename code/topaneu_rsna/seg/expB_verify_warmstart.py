"""
Experiment B pre-flight: CPU-only check that (1) the channel-expanded
Dataset304 fold_4 checkpoint actually loads into this classifier's
architecture, AND (2) a real forward pass at the ACTUAL crop size (read from
build_expB_crop_dataset.py's own crop_size.json) succeeds -- before spending
a GPU allocation finding out either doesn't work.

The forward-pass check exists because it's the one this script originally
lacked: job 91 only checked weight-loading (state_dict shapes) and PASSED,
but job 92 then crashed on GPU with a residual-block shape mismatch caused
by a crop size that wasn't an exact multiple of the encoder's real per-axis
downsampling factor -- a problem only a real forward pass through the actual
architecture would have caught. Fixed at the source (build_expB_crop_dataset.
py's pick_crop_size() now reads that factor from Dataset304's own plans.json
instead of guessing), but this script now also verifies it directly, so a
future crop-size change gets caught here again if it ever regresses.

There is no nnU-Net "dataset" for this standalone classifier (it's a plain
PyTorch pipeline, not an nnU-Net trainer/dataset), so
seg/verify_pretrained_weights.py's own approach (build the network from a
PREPROCESSED dataset's plans.json) doesn't apply here -- this instead calls
expB_train_classifier.build_model() directly (the exact function training/
eval will use) on CPU.

    python -m topaneu_rsna.seg.expB_verify_warmstart
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from topaneu_rsna import config as C
from topaneu_rsna.seg.expB_train_classifier import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expanded_ckpt", type=Path,
                    default=C.WORK / "checkpoints" / "expB_dataset304_fold4_3ch_checkpoint_final.pth")
    ap.add_argument("--dataset304_model_dir", type=Path, default=None)
    ap.add_argument("--crop_size_json", type=Path, default=C.EXPB_CROP_CACHE / "crop_size.json")
    a = ap.parse_args()

    spec = C.load_labels()
    n_classes = spec.n_loc + 1
    dataset304_model_dir = a.dataset304_model_dir or C.seg_model_dir(C.DS_ANEURYSM, C.TRAINER_LOC)

    if not a.expanded_ckpt.exists():
        raise SystemExit(f"[FAIL] expanded checkpoint not found at {a.expanded_ckpt} -- "
                         f"run expand_pretrained_channels.py first (--extra_channels 2)")

    model = build_model(a.expanded_ckpt, dataset304_model_dir, n_classes, torch.device("cpu"))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model built and warm-start loaded successfully "
         f"({n_params:,} parameters, {n_classes} output classes)")

    if not a.crop_size_json.exists():
        raise SystemExit(f"[FAIL] {a.crop_size_json} not found -- run "
                         f"build_expB_crop_dataset.py first, then rerun this check")
    crop_size = tuple(json.loads(a.crop_size_json.read_text())["crop_size"])
    print(f"running a dummy forward pass at the real crop size {crop_size} "
         f"(this is the check that would have caught job 92's shape-mismatch crash)")
    dummy = torch.randn(2, 3, *crop_size)
    with torch.no_grad():
        out = model(dummy)
    print(f"forward pass succeeded, output shape {tuple(out.shape)}")

    print(f"\nPASS -- warm start loads AND a real forward pass at crop size "
         f"{crop_size} succeeds")


if __name__ == "__main__":
    main()
