"""
Expand a trained nnU-Net checkpoint's first conv layer to accept extra input
channels, for use as -pretrained_weights on a dataset with more channels than
the original model had.

This fork's load_pretrained_weights() (nnUNet/nnunetv2/run/load_pretrained_weights.py)
only skips '.seg_layers.' keys -- every other key must match in *shape* or it
hard-asserts. Feeding Model 1's 1-channel checkpoint straight into Dataset307's
2-channel network trips exactly that assertion on the stem's first conv layer.

Fix: rewrite that layer's weight tensor from
    (out_ch, in_ch,      k, k, k)
to
    (out_ch, in_ch+extra, k, k, k)
keeping the original channel's trained weights exactly as-is and zero-filling
the new channel(s) -- so at the start of training the expanded network computes
the *identical* output Model 1 did (the new channel contributes nothing yet),
and the rest of the checkpoint (everything but the resized layer and the
already-skipped seg_layers) loads unchanged. Gradients still flow into the new
channel's weights normally during training.

The stem's first conv is registered under two different attribute paths in
this architecture (`...conv.weight` and `...all_modules.0.weight`, same
underlying weights, two names) -- the checkpoint stores both as separate
dict entries, and nnU-Net's loader can walk either name depending on the
network variant, so *both* aliases get widened, not just one.

    python -m topaneu_rsna.seg.expand_pretrained_channels \\
        --in_ckpt  nnUNet_results/Dataset301.../fold_all/checkpoint_final.pth \\
        --out_ckpt work/checkpoints/model1_2ch.pth \\
        --extra_channels 1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_ckpt", type=Path, required=True)
    ap.add_argument("--out_ckpt", type=Path, required=True)
    ap.add_argument("--extra_channels", type=int, default=1)
    ap.add_argument("--stem_key", nargs="+", default=[
        "encoder.stem.convs.0.conv.weight",
        "encoder.stem.convs.0.all_modules.0.weight",
    ], help="candidate key(s) for the first conv layer's weight -- every one "
            "that's present gets widened (they may be aliases of the same "
            "underlying weights); add more via the AssertionError message "
            "if training still fails on a different key name")
    a = ap.parse_args()

    ckpt = torch.load(a.in_ckpt, map_location="cpu", weights_only=False)
    weights = ckpt["network_weights"]

    found = [k for k in a.stem_key if k in weights]
    if not found:
        raise KeyError(f"None of {a.stem_key} found in checkpoint. Keys starting "
                       f"'encoder.stem': {[k for k in weights if k.startswith('encoder.stem')]}")

    for key in found:
        w = weights[key]
        out_ch, _, *k = w.shape
        pad = torch.zeros((out_ch, a.extra_channels, *k), dtype=w.dtype)
        weights[key] = torch.cat([w, pad], dim=1)
        print(f"{key}: {tuple(w.shape)} -> {tuple(weights[key].shape)}")

    a.out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, a.out_ckpt)
    print(f"wrote {a.out_ckpt}")


if __name__ == "__main__":
    main()
