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


class RSNA2025Trainer_ExpA_VesselCondSeg_ClassBalanced(RSNA2025Trainer_ExpA_VesselCondSeg):
    """
    RSNA2025Trainer_ExpA_VesselCondSeg's first 80-epoch run (jobs 3107515/
    3109070) collapsed to all-background: pseudo dice was exactly 0.0/nan for
    every one of the 52 location classes across the entire run, never once
    nonzero, and this held for even the most common classes, not just rare
    ones -- guaranteed-positive sampling (oversample_foreground_percent=1.0)
    only forces the sampled patch to CONTAIN some foreground voxel, it says
    nothing about which of the 52 classes gets seen across the training run
    as a whole, or how often. This is the exact collapse mode diagnosed for
    Dataset303 (see config.py's DS_LOCATION comment): with do_bg=False, most
    patches carry ground truth for at most one class, and MemoryEfficient
    SoftDiceLoss/CE/TopK/Focal all average across all 52 channels jointly, so
    a class that's rarely (or never) the one class present in a given batch
    gets almost no gradient signal to escape "predict background" with.

    class_balanced_focal.py's own postmortem on that exact collapse used TWO
    independent levers together, not one: guaranteed-ish positive sampling
    (there, only oversample_foreground_percent=0.66) AND case-level
    class-balanced sampling (weighting which CASE gets drawn so a case whose
    only lesion is a rare class isn't drowned out by cases with a common
    one). This trainer keeps this experiment's oversample_foreground_percent
    at 1.0 (per the task's own "every patch must contain a lesion" spec --
    an even stronger version of that file's first lever) and adds the SECOND
    lever verbatim from class_balanced_focal.py's own
    _class_balanced_case_weights()/get_dataloaders() (already proven code in
    this exact codebase, not reinvented here), rather than silently changing
    what RSNA2025Trainer_ExpA_VesselCondSeg means after two jobs already ran
    under that name.
    """

    def __init__(self, plans: dict, configuration: str, fold, dataset_json: dict,
                device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        # Job 87 (this exact class, 80 epochs) broke the all-background
        # collapse but only reached one class in that time (pseudo dice
        # nonzero for a single class by epoch 73-79, all other 51 classes
        # still 0.0/nan) -- 80 epochs simply wasn't enough gradient exposure
        # for the rest. Raised to 1000 (job 20/66's own precedent for "give
        # a full-res 3D run a generous budget when time allows," not picked
        # arbitrarily) so job 94 can resume via nnU-Net's own --c flag
        # (note: double-dash -- job 94's first attempt used single-dash -c,
        # which this fork's argparse never registers, so it silently trained
        # fresh from epoch 0 instead of resuming -- see run_training.py's own
        # `parser.add_argument("--c", ...)`) straight from job 87's epoch-80
        # checkpoint instead of restarting those 80 already-completed epochs.
        self.num_epochs = 1000
        # configure_optimizers() builds PolyLRScheduler(optimizer, initial_lr,
        # num_epochs) -- purely a function of current_epoch/num_epochs. Even
        # with --c correctly restoring current_epoch=80, computing that
        # fraction against the NEW num_epochs=1000 (only 8% through) pushes
        # the LR back up near its original peak (0.01), which would likely
        # wash out the one fragile, low-LR-dependent class job 87 had just
        # gotten to learn -- the same failure job 94's botched restart
        # actually demonstrated by accident. Lowering initial_lr to roughly
        # where job 87's own decay had already reached by epoch 80 (its last
        # printed LR was in the 1e-4-5e-4 range) keeps a resumed run in the
        # same low, stable regime instead of re-warming it, while still
        # decaying gently over the remaining ~920 epochs.
        self.initial_lr = 5e-4

    def _class_balanced_case_weights(self, dataset_tr) -> np.ndarray:
        from batchgenerators.utilities.file_and_folder_operations import join, load_pickle

        all_labels = set(self.label_manager.foreground_labels)
        case_classes = {}
        class_case_count = {c: 0 for c in all_labels}

        for key in dataset_tr.identifiers:
            props = load_pickle(join(dataset_tr.source_folder, key + ".pkl"))
            class_locations = props.get("class_locations", {}) or {}
            present = {c for c in all_labels if c in class_locations and len(class_locations[c]) > 0}
            case_classes[key] = present
            for c in present:
                class_case_count[c] += 1

        default_weight = 1.0 / max(1, len(dataset_tr.identifiers))
        weights = []
        for key in dataset_tr.identifiers:
            present = case_classes[key]
            weights.append(default_weight if not present else max(1.0 / class_case_count[c] for c in present))

        weights = np.asarray(weights, dtype=np.float64)
        weights = weights / weights.sum()

        n_classes_seen = sum(1 for c in all_labels if class_case_count[c] > 0)
        self.print_to_log_file(
            f"Class-balanced sampling: {n_classes_seen}/{len(all_labels)} of the location "
            f"classes appear at least once in this fold's training split. Case weights range "
            f"{weights.min():.2e} - {weights.max():.2e} (uniform would be {1.0 / len(weights):.2e})."
        )
        return weights

    def get_dataloaders(self):
        """
        Copy of nnUNetTrainerSkeletonRecall.get_dataloaders() (NOT the plain
        nnUNetTrainer base -- this trainer's inherited train_step/
        validation_step require the "skel" key only nnUNetDataLoader2DSkel/
        3DSkel's generate_train_batch() produces). The only change from that
        original, same as class_balanced_focal.py's own copy: the training
        loader gets class-balanced sampling_probabilities instead of None.
        """
        from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
        from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
        from nnunetv2.training.dataloading.data_loader_2d_skel import nnUNetDataLoader2DSkel
        from nnunetv2.training.dataloading.data_loader_3d_skel import nnUNetDataLoader3DSkel
        from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
        from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA

        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)
        deep_supervision_scales = self._get_deep_supervision_scales()

        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)

        val_transforms = self.get_validation_transforms(
            deep_supervision_scales, is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        train_sampling_probabilities = self._class_balanced_case_weights(dataset_tr)

        loader_cls = nnUNetDataLoader2DSkel if dim == 2 else nnUNetDataLoader3DSkel
        dl_tr = loader_cls(dataset_tr, self.batch_size,
                           initial_patch_size,
                           self.configuration_manager.patch_size,
                           self.label_manager,
                           oversample_foreground_percent=self.oversample_foreground_percent,
                           sampling_probabilities=train_sampling_probabilities, pad_sides=None,
                           transforms=tr_transforms)
        dl_val = loader_cls(dataset_val, self.batch_size,
                            self.configuration_manager.patch_size,
                            self.configuration_manager.patch_size,
                            self.label_manager,
                            oversample_foreground_percent=self.oversample_foreground_percent,
                            sampling_probabilities=None, pad_sides=None, transforms=val_transforms)

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(data_loader=dl_tr, transform=None,
                                                        num_processes=allowed_num_processes,
                                                        num_cached=max(6, allowed_num_processes // 2), seeds=None,
                                                        pin_memory=self.device.type == "cuda", wait_time=0.002)
            mt_gen_val = NonDetMultiThreadedAugmenter(data_loader=dl_val,
                                                      transform=None, num_processes=max(1, allowed_num_processes // 2),
                                                      num_cached=max(3, allowed_num_processes // 4), seeds=None,
                                                      pin_memory=self.device.type == "cuda",
                                                      wait_time=0.002)
        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        return mt_gen_train, mt_gen_val
