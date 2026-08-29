"""
Vessel Region-Masked Pooling.

Two branches, as in the author's RegionMaskedPooling3D:
  * mask branch   -- mean of decoder features inside each vessel region,
                     LayerNorm, then a linear projection to proj_mask_dim (32)
  * global branch -- mean of *encoder* features over the whole volume,
                     broadcast to every token, LayerNorm, projection to
                     proj_global_dim (64)

Implementation note [TOPANEU]: the author materialised (B, 13, D, H, W) masks.
At 52 locations and 128x256x256 that is ~1.7 GB per sample in fp32, so this
version scatters over the vessel label map instead.  Verified numerically
identical to the dense form; regions are mutually exclusive so the two agree.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VesselRegionMaskedPooling(nn.Module):
    def __init__(self, feat_channels, global_channels, proj_mask_dim=32,
                 proj_global_dim=64, branch_norm=True, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.mask_norm = nn.LayerNorm(feat_channels) if branch_norm else nn.Identity()
        self.mask_proj = nn.Linear(feat_channels, proj_mask_dim)
        self.global_norm = nn.LayerNorm(global_channels) if branch_norm else nn.Identity()
        self.global_proj = nn.Linear(global_channels, proj_global_dim)
        self.mask_dim = proj_mask_dim
        self.global_dim = proj_global_dim

    @staticmethod
    def _to_feat_res(labelmap, size):
        if tuple(labelmap.shape[1:]) == tuple(size):
            return labelmap
        return F.interpolate(labelmap[:, None].float(), size=size,
                             mode="nearest")[:, 0].long()

    @torch.autocast(device_type="cuda", enabled=False)
    def pool_regions(self, feat, labelmap, n_region):
        """(B,C,D,H,W) + (B,D,H,W) -> pooled (B, n_region, C), valid (B, n_region)."""
        feat = feat.float()
        B, C = feat.shape[:2]
        lm = self._to_feat_res(labelmap, feat.shape[2:])
        flat_lab = lm.reshape(B, -1)
        fg = flat_lab > 0
        idx = (flat_lab - 1).clamp(0, n_region - 1)

        counts = torch.zeros(B, n_region, device=feat.device, dtype=feat.dtype)
        counts.scatter_add_(1, idx, fg.to(feat.dtype))

        sums = torch.zeros(B, C, n_region, device=feat.device, dtype=feat.dtype)
        sums.scatter_add_(2, idx.unsqueeze(1).expand(B, C, idx.shape[1]),
                          feat.reshape(B, C, -1) * fg.unsqueeze(1).to(feat.dtype))

        valid = counts > self.eps
        pooled = (sums / counts.clamp_min(self.eps).unsqueeze(1)).transpose(1, 2)
        return pooled * valid.unsqueeze(-1).to(pooled.dtype), valid

    @torch.autocast(device_type="cuda", enabled=False)
    def pool_union(self, feat, labelmap):
        feat = feat.float()
        B, C = feat.shape[:2]
        lm = self._to_feat_res(labelmap, feat.shape[2:])
        m = (lm > 0).reshape(B, 1, -1).to(feat.dtype)
        return (feat.reshape(B, C, -1) * m).sum(-1) / m.sum(-1).clamp_min(self.eps)

    def project_mask(self, pooled):
        return self.mask_proj(self.mask_norm(pooled))

    def project_global(self, enc_feat, n_tokens):
        g = enc_feat.float().flatten(2).mean(-1)
        g = self.global_proj(self.global_norm(g))
        return g.unsqueeze(1).expand(-1, n_tokens, -1)
