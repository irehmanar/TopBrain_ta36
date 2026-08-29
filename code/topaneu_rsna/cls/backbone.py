"""
Pretrained nnU-Net backbone with a truncated decoder and a sphere head.

Ablation from the author's write-up: removing this pretraining drops the score
from 0.902 to 0.794.  It is the single most important component of the whole
pipeline, so `pretrained=False` should only ever be used for a smoke test.

The weights come from the Model 2 checkpoint trained in stage 2 -- the same
network that produced the vessel masks, so its features already encode vessels.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn as nn

from nnunetv2.utilities.get_network_from_plans import get_network_from_plans


class NnUNetTruncatedBackbone(nn.Module):
    """
    Returns
      enc_feat      : (B, C_enc, d, h, w)  bottleneck features
      dec_feat      : (B, C_dec, D/2, H/2, W/2)  decoder features, one stage early
      logits_sphere : (B, 1, D/2, H/2, W/2)
    """

    def __init__(self, nnunet_model_dir, fold="all", pretrained=True,
                 checkpoint_name="checkpoint_final.pth", configuration="3d_fullres",
                 in_channels=1, num_truncate_stages=1, sphere_mid_channels=32):
        super().__init__()
        model_dir = Path(nnunet_model_dir)
        plans = json.loads((model_dir / "plans.json").read_text())
        arch = plans["configurations"][configuration]["architecture"]

        self.nnunet = get_network_from_plans(
            arch_class_name=arch["network_class_name"],
            arch_kwargs=arch["arch_kwargs"],
            arch_kwargs_req_import=arch["_kw_requires_import"],
            input_channels=int(in_channels),
            output_channels=1,
            allow_init=True,
            deep_supervision=False,
        )

        if pretrained:
            ckpt = model_dir / f"fold_{fold}" / checkpoint_name
            state = torch.load(ckpt, map_location="cpu", weights_only=False)
            weights = state.get("network_weights", state)
            # the seg head has a different class count -- drop it, keep everything else
            weights = {k: v for k, v in weights.items()
                       if not k.startswith("decoder.seg_layers.")}
            missing, unexpected = self.nnunet.load_state_dict(weights, strict=False)
            print(f"[backbone] loaded {ckpt}\n"
                  f"[backbone] missing={len(missing)} unexpected={len(unexpected)}")
            if len(missing) > 20:
                raise RuntimeError(
                    "Too many missing keys -- the checkpoint architecture does not "
                    "match plans.json. Check that --plans matches the trained model.")

        dec = self.nnunet.decoder
        n_stages = len(dec.stages)
        k = int(num_truncate_stages)
        if not 0 <= k <= n_stages:
            raise ValueError(f"num_truncate_stages must be in 0..{n_stages}")
        self._cutoff = n_stages - k

        if self._cutoff == n_stages:
            self.out_channels = int(self.nnunet.encoder.output_channels[0])
        else:
            self.out_channels = int(dec.transpconvs[self._cutoff].in_channels)
        self.encoder_channels = int(self.nnunet.encoder.output_channels[-1])

        self.head_sphere = nn.Sequential(
            nn.Conv3d(self.out_channels, sphere_mid_channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(sphere_mid_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(sphere_mid_channels, 1, 1, bias=True),
        )
        nn.init.zeros_(self.head_sphere[-1].weight)
        nn.init.constant_(self.head_sphere[-1].bias, -4.0)   # sparse target prior

    def _decode(self, skips):
        dec = self.nnunet.decoder
        x = skips[-1]
        for s in range(self._cutoff):
            x = dec.transpconvs[s](x)
            x = torch.cat((x, skips[-(s + 2)]), 1)
            x = dec.stages[s](x)
        return x

    def forward(self, x):
        skips = self.nnunet.encoder(x)
        dec_feat = self._decode(skips)
        return {"enc_feat": skips[-1], "dec_feat": dec_feat,
                "logits_sphere": self.head_sphere(dec_feat)}
