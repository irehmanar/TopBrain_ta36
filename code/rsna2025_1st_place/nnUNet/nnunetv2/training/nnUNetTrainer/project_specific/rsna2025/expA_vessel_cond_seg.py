"""
[TOPANEU] Experiment A: Vessel-Aware Multi-Task Segmentation (Tapar, TopAneu
2026 -- 45_Vessel_Aware_Multi_Task_3D_.pdf), Task-2 segmentation half.

The paper trains one network with two heads (segmentation + classification)
and a single joint loss (Eq. 1: Dice + CE + TopK + Focal_seg + lambda_cls *
BCEFocal_cls). This pipeline replicates that as a DECOUPLED two-model design
instead (segmentation here, a separate small classifier in
seg/expA_train_classifier.py on frozen encoder features) -- every existing
custom trainer in this codebase only overrides loss/augmentation, none of
them touch nnU-Net's dataloader output shape or add a second output head, so
building the paper's true joint dual-head trainer would mean writing and
debugging untested nnU-Net internals (a custom dataloader yielding per-case
multi-label targets, a custom network wrapper, custom train/validation
steps, checkpoint I/O and TTA/ensemble aggregation for two heads) with zero
precedent to build on, on a 1-week deadline before this needs to run
unattended on Narval. The decoupled version needs none of that: this trainer
is a normal single-head nnU-Net trainer, and consensus fusion still happens
at inference by combining this model's segmentation with the separate
classifier's per-case probabilities (see seg/expA_consensus_fusion.py).

This file implements ONLY the paper's Stage-1 loss (Sect 2.3): Dice + CE +
TopK(10% hardest voxels) + Focal (gamma=2.0, alpha=0.25) -- the lambda_cls
term does not apply here since the classification head is a separate model.
No loss primitive for a per-voxel multi-class Focal term already existed in
this nnU-Net fork (only a Focal-*Tversky* variant does, a different,
Dice-side formula -- see class_balanced_focal.py /
focal_tversky_plusplus_loss.py), so FocalLoss below is new, built by mirroring
TopKLoss's own construction pattern immediately below it in
nnunetv2/training/loss/robust_ce_loss.py (same legacy
size_average=False/reduce=False trick to get an unreduced per-voxel CE this
class then reweights, rather than inventing a new reduction mechanism).

Base class: RSNA2025Trainer_moreDAv6_1_SkeletonRecallTverskyBeta07 (Model 2's
own trainer, TRAINER_M2/TRAINER_LOC in config.py) -- NOT because SkeletonRecall
is wanted here (it isn't; _build_loss below replaces it entirely), but because
it is the one trainer in this codebase already proven to run at exactly this
pipeline's anisotropic Task-2 patch shape (128x256x256 @ 0.80x0.45x0.44mm,
the same recipe used for Datasets304/305/306/307/308) without hitting
RSNA2025Trainer_moreDAv7's cube-only augmentation assertion (see
28_verify_m1cond.sbatch's own comment for that exact failure). The one thing
this inheritance requires: nnUNetTrainerSkeletonRecall's train_step/
validation_step call `self.loss(output, target, skel)` -- THREE positional
args, unconditionally, regardless of what the actual loss needs skel for.
DC_CE_TopK_Focal_loss.forward therefore accepts and silently ignores a third
`skel` argument so that call succeeds; the (wasted, but already-proven-cheap
in production on Datasets304/305/306/308) skeletonization dataloader/
augmentation machinery keeps running underneath, it just never influences
this trainer's actual loss value.

    nnUNetv2_train 313 3d_fullres all -p nnUNetResEncUNetMPlans \\
        -tr RSNA2025Trainer_ExpA_VesselCondSeg \\
        -pretrained_weights <expanded Dataset304 checkpoint>
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss, TopKLoss
from nnunetv2.training.nnUNetTrainer.project_specific.rsna2025.more_DAv6 import (
    RSNA2025Trainer_moreDAv6_1_SkeletonRecallTverskyBeta07,
)
from nnunetv2.utilities.helpers import softmax_helper_dim1


class FocalLoss(RobustCrossEntropyLoss):
    """
    Standard multi-class (softmax) focal loss (Lin et al., "Focal Loss for
    Dense Object Detection", ICCV 2017), gamma/alpha as scalars matching the
    paper's Sect 2.3 spec (gamma=2.0, alpha=0.25) -- no per-class alpha
    vector, the paper doesn't specify one.

    Built as a thin subclass of RobustCrossEntropyLoss exactly the way
    TopKLoss (same file, immediately above this class in the upstream
    source) is: passing size_average=False, reduce=False as legacy positional
    args gets torch's nn.CrossEntropyLoss to return one loss value per voxel
    (reduction='none') instead of a scalar, which RobustCrossEntropyLoss.
    forward() already handles (squeezing the channel-1 target dim, casting to
    long) -- so `super().forward(inp, target)` below is exactly the per-voxel
    CE this loss then reweights, no new target-handling logic needed.
    """

    def __init__(self, weight=None, ignore_index: int = -100, gamma: float = 2.0,
                alpha: float = 0.25, label_smoothing: float = 0):
        self.gamma = gamma
        self.alpha = alpha
        super().__init__(weight, False, ignore_index, reduce=False, label_smoothing=label_smoothing)

    def forward(self, inp, target):
        ce = super().forward(inp, target)          # per-voxel CE, no reduction
        pt = torch.exp(-ce)
        focal = self.alpha * (1 - pt) ** self.gamma * ce
        return focal.mean()


class DC_CE_TopK_Focal_loss(nn.Module):
    """
    The paper's Sect 2.3 Stage-1 loss, four independent additive terms (not a
    single fused compound the way DC_and_topk_loss fuses CE-via-TopK with
    Dice -- here Dice, plain CE, TopK-CE, and Focal-CE are each computed
    separately and summed with their own weights, matching Eq. 1 with the
    lambda_cls term dropped -- see this file's module docstring for why).
    Same ignore_label masking / conditional-skip-if-weight-zero structure as
    DC_and_CE_loss / DC_and_topk_loss elsewhere in this loss module, just
    extended to four terms instead of two.
    """

    def __init__(self, soft_dice_kwargs, ce_kwargs, topk_kwargs, focal_kwargs,
                weight_dice: float = 1.0, weight_ce: float = 1.0,
                weight_topk: float = 0.5, weight_focal: float = 0.5,
                ignore_label=None, dice_class=MemoryEfficientSoftDiceLoss):
        super().__init__()
        if ignore_label is not None:
            ce_kwargs = dict(ce_kwargs, ignore_index=ignore_label)
            topk_kwargs = dict(topk_kwargs, ignore_index=ignore_label)
            focal_kwargs = dict(focal_kwargs, ignore_index=ignore_label)

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_topk = weight_topk
        self.weight_focal = weight_focal
        self.ignore_label = ignore_label

        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.topk = TopKLoss(**topk_kwargs)
        self.focal = FocalLoss(**focal_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor, skel=None):
        if self.ignore_label is not None:
            assert target.shape[1] == 1, ("ignore label is not implemented for "
                                          "one-hot encoded target variables")
            mask = (target != self.ignore_label).bool()
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None
            num_fg = None

        has_fg = self.ignore_label is None or num_fg > 0
        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target) if self.weight_ce != 0 and has_fg else 0
        topk_loss = self.topk(net_output, target) if self.weight_topk != 0 and has_fg else 0
        focal_loss = self.focal(net_output, target) if self.weight_focal != 0 and has_fg else 0

        return (self.weight_dice * dc_loss + self.weight_ce * ce_loss
               + self.weight_topk * topk_loss + self.weight_focal * focal_loss)


class RSNA2025Trainer_ExpA_VesselCondSeg(RSNA2025Trainer_moreDAv6_1_SkeletonRecallTverskyBeta07):
    def __init__(self, plans: dict, configuration: str, fold, dataset_json: dict,
                device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        # Paper: 200-epoch joint Stage 2 on top of a 50-epoch Stage 1 warm-up.
        # 1-week budget, decoupled design (no Stage 2 classification term to
        # co-train), and a real pretrained-weight warm start (job 81, from
        # Dataset304) standing in for the paper's from-scratch Stage 1 --
        # shortened to a single ~80-epoch run per the task's own guidance.
        # Bump this (and resubmit as a NEW job, never edit one already run --
        # see this pipeline's standing convention) if job 82's loss is still
        # improving and epoch 10-15's health check (this trainer's own
        # foreground-ratio log, see train_step below) looked good.
        self.num_epochs = 80
        self.save_every = 5
        # Paper: "guaranteed-positive, aneurysm-centered patch sampling --
        # every training patch must contain a real lesion voxel." nnU-Net's
        # oversample_foreground_percent=1.0 is the documented way to force
        # that (default 0.33 only guarantees it for a third of each batch).
        self.oversample_foreground_percent = 1.0
        # Verifies the line above is actually taking effect (don't just trust
        # the config, per the task's own instruction) by logging the true
        # foreground-voxel fraction of the first N training batches.
        self._fg_log_batches_remaining = 300

    def _build_loss(self):
        loss = DC_CE_TopK_Focal_loss(
            soft_dice_kwargs={
                "batch_dice": self.configuration_manager.batch_dice,
                "smooth": 1e-5, "do_bg": False, "ddp": self.is_ddp,
            },
            ce_kwargs={},
            topk_kwargs={"k": 10},
            focal_kwargs={"gamma": 2.0, "alpha": 0.25},
            weight_dice=1.0, weight_ce=1.0, weight_topk=0.5, weight_focal=0.5,
            ignore_label=self.label_manager.ignore_label,
        )

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2**i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss

    def _log_foreground_ratio(self, target):
        t = target[0] if isinstance(target, list) else target
        frac = float((t > 0).float().mean().item())
        remaining = self._fg_log_batches_remaining
        self.print_to_log_file(
            f"[expA fg-check] batch (first {300 - remaining + 1}/300 logged): "
            f"foreground voxel fraction = {frac:.4f} "
            f"(oversample_foreground_percent={self.oversample_foreground_percent} "
            f"-- this should stay near 1.0 batch-to-batch if guaranteed-positive "
            f"sampling is really active; a low or wildly varying value here means "
            f"it isn't and the config value alone should not be trusted).")
        self._fg_log_batches_remaining -= 1

    def train_step(self, batch: dict) -> dict:
        if self._fg_log_batches_remaining > 0:
            self._log_foreground_ratio(batch["target"])
        return super().train_step(batch)
