"""
Experiment B pre-flight: CPU-only check that the channel-expanded Dataset304
fold_4 checkpoint actually loads into this classifier's architecture, before
spending a GPU allocation finding out. There is no nnU-Net "dataset" for this
standalone classifier (it's a plain PyTorch pipeline, not an nnU-Net trainer/
dataset), so seg/verify_pretrained_weights.py's own approach (build the
network from a PREPROCESSED dataset's plans.json) doesn't apply here --
this instead calls expB_train_classifier.build_model() directly (the exact
function training/eval will use) on CPU, which already does the shape check
and raises if too many keys are missing.

    python -m topaneu_rsna.seg.expB_verify_warmstart
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from topaneu_rsna import config as C
from topaneu_rsna.seg.expB_train_classifier import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expanded_ckpt", type=Path,
                    default=C.WORK / "checkpoints" / "expB_dataset304_fold4_3ch_checkpoint_final.pth")
    ap.add_argument("--dataset304_model_dir", type=Path, default=None)
    a = ap.parse_args()

    spec = C.load_labels()
    n_classes = spec.n_loc + 1
    dataset304_model_dir = a.dataset304_model_dir or C.seg_model_dir(C.DS_ANEURYSM, C.TRAINER_LOC)

    if not a.expanded_ckpt.exists():
        raise SystemExit(f"[FAIL] expanded checkpoint not found at {a.expanded_ckpt} -- "
                         f"run expand_pretrained_channels.py first (--extra_channels 2)")

    model = build_model(a.expanded_ckpt, dataset304_model_dir, n_classes, torch.device("cpu"))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nPASS -- model built and warm-start loaded successfully "
         f"({n_params:,} parameters, {n_classes} output classes)")


if __name__ == "__main__":
    main()
