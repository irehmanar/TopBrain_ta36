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

The stem's first conv is registered under multiple attribute paths in this
architecture -- `encoder.stem.convs.0.{conv,all_modules.0}.weight` (the same
underlying weights exposed two ways) *and*, because the decoder holds a
reference to the same encoder object for skip connections, that whole pair
again under a `decoder.encoder.` prefix. All are separate entries in the
checkpoint's state dict, and nnU-Net's loader can walk any of them depending
on the network wrapper -- so instead of hardcoding each alias as it's
discovered, every key matching the stem-conv pattern gets widened.

    python -m topaneu_rsna.seg.expand_pretrained_channels \\
        --in_ckpt  nnUNet_results/Dataset301.../fold_all/checkpoint_final.pth \\
        --out_ckpt work/checkpoints/model1_2ch.pth \\
        --extra_channels 1
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch

# Matches the stem's first conv weight under any prefix (`encoder.`,
# `decoder.encoder.`, or whatever else this architecture aliases it as) and
# either of its two attribute names (`.conv.weight` or `.all_modules.0.weight`).
STEM_CONV_PATTERN = re.compile(r"stem\.convs\.0\.(conv|all_modules\.0)\.weight$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_ckpt", type=Path, required=True)
    ap.add_argument("--out_ckpt", type=Path, required=True)
    ap.add_argument("--extra_channels", type=int, default=1)
    ap.add_argument("--stem_key", nargs="+", default=None,
                    help="override auto-detection with exact key(s) instead "
                         "(auto-detection matches any prefix ending in "
                         "'stem.convs.0.conv.weight' or "
                         "'stem.convs.0.all_modules.0.weight')")
    a = ap.parse_args()

    ckpt = torch.load(a.in_ckpt, map_location="cpu", weights_only=False)
    weights = ckpt["network_weights"]

    if a.stem_key:
        found = [k for k in a.stem_key if k in weights]
    else:
        found = [k for k in weights if STEM_CONV_PATTERN.search(k)]
    if not found:
        raise KeyError("No stem-conv key found in checkpoint (pattern "
                       f"{STEM_CONV_PATTERN.pattern!r}). Keys containing 'stem': "
                       f"{[k for k in weights if 'stem' in k]}")

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
