"""
Losses for the ROI classifier.

  locations / presence / types : plain BCEWithLogits (each label independent --
      the author explicitly treats them as independent binary problems because
      per-location positives are very rare)
  auxiliary sphere             : BalancedBCE + FocalTversky++ (alpha 0.3,
      beta 0.7, gamma_pp 2.0, gamma_focal 1.33)

The sphere pair is what carries weight 1.0 in the total loss; the classification
terms carry 0.1 / 0.05.  The author's ablation shows setting all three to 1.0
costs about 1.8 AUC points through overfitting.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BalancedBCEWithLogitsLoss(nn.Module):
    def forward(self, pred, target):
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        t = (target > 0.01).float()
        pos = (loss * t).sum() / (t.sum() + 1e-6)
        neg = (loss * (1 - t)).sum() / ((1 - t).sum() + 1e-6)
        return pos + neg


class FocalTverskyPlusPlusLoss(nn.Module):
    """Tversky++ with focal modulation, sigmoid inputs, mean over batch."""

    def __init__(self, alpha=0.3, beta=0.7, gamma_pp=2.0, gamma_focal=1.33,
                 smooth_nr=1e-5, smooth_dr=1e-5):
        super().__init__()
        self.alpha, self.beta = alpha, beta
        self.gamma_pp, self.gamma_focal = gamma_pp, gamma_focal
        self.smooth_nr, self.smooth_dr = smooth_nr, smooth_dr

    def forward(self, logits, target):
        p = torch.sigmoid(logits.float())
        t = target.float()
        dims = tuple(range(1, p.dim()))
        tp = (p * t).sum(dims)
        # the "++" term raises the false-positive/negative products to gamma_pp,
        # which stops confident-but-wrong voxels from saturating the score
        fp = torch.pow((p * (1 - t)).clamp_min(0) + 1e-8, self.gamma_pp).sum(dims)
        fn = torch.pow(((1 - p) * t).clamp_min(0) + 1e-8, self.gamma_pp).sum(dims)
        ti = (tp + self.smooth_nr) / (tp + self.alpha * fp + self.beta * fn + self.smooth_dr)
        return torch.pow((1.0 - ti).clamp_min(0) + 1e-8, 1.0 / self.gamma_focal).mean()


class AneurysmLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.bce = nn.BCEWithLogitsLoss()
        # only used when cfg.loc_loss == "balanced_bce" (see config.py's
        # ClsConfig.loc_loss docstring for why the plain BCE + w_loc=0.1
        # default gives near-zero recall at a 0.5 threshold despite good AUC)
        self.loc_balanced_bce = BalancedBCEWithLogitsLoss()
        self.sphere_bce = BalancedBCEWithLogitsLoss()
        self.sphere_ft = FocalTverskyPlusPlusLoss(
            cfg.ft_alpha, cfg.ft_beta, cfg.ft_gamma_pp, cfg.ft_gamma_focal)

    def forward(self, out, batch):
        parts = {}
        loc_loss_fn = (self.loc_balanced_bce
                      if getattr(self.cfg, "loc_loss", "bce") == "balanced_bce"
                      else self.bce)
        loss_loc = loc_loss_fn(out["loc_logits"], batch["loc"])
        loss_ap = self.bce(out["ap_logit"], batch["ap"])
        parts["loc"] = loss_loc.detach()
        parts["ap"] = loss_ap.detach()
        total = self.cfg.w_loc * loss_loc + self.cfg.w_ap * loss_ap

        tgt = batch["sphere"]
        if tgt.shape[-3:] != out["logits_sphere"].shape[-3:]:
            tgt = F.interpolate(tgt.float(), size=out["logits_sphere"].shape[-3:],
                                mode="nearest")
        s_bce = self.sphere_bce(out["logits_sphere"].float(), tgt)
        has_gt = tgt.flatten(1).sum(1) > 0
        s_ft = (self.sphere_ft(out["logits_sphere"][has_gt], tgt[has_gt])
                if has_gt.any() else out["logits_sphere"].sum() * 0.0)
        parts["sph_bce"] = s_bce.detach()
        parts["sph_ft"] = s_ft.detach()
        total = total + self.cfg.w_sphere * (s_bce + s_ft)

        if out.get("type_logits") is not None and "typ" in batch:
            loss_t = self.bce(out["type_logits"], batch["typ"])
            parts["typ"] = loss_t.detach()
            total = total + self.cfg.w_type * loss_t

        return total, parts
