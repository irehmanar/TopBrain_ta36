"""Exponential moving average of model weights (author used decay 0.995)."""
from __future__ import annotations

import copy
import torch


class ModelEMA:
    def __init__(self, model, decay=0.995, update_after_step=100):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.update_after_step = update_after_step
        self.step = 0

    @torch.no_grad()
    def update(self, model):
        self.step += 1
        d = 0.0 if self.step <= self.update_after_step else self.decay
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if not v.dtype.is_floating_point:
                v.copy_(msd[k])
            else:
                v.mul_(d).add_(msd[k].detach(), alpha=1.0 - d)

    def state_dict(self):
        return self.ema.state_dict()
