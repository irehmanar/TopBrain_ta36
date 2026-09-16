"""
[TOPANEU] Experiment C: a single shared 3D encoder with two heads -- binary
aneurysm segmentation (Branch A) and 52-class location classification
(Branch B) -- trained JOINTLY, unlike Experiment A/B's decoupled designs.
Genuinely higher engineering risk than either: this is the first trainer in
this codebase to override build_network_architecture / train_step /
validation_step, not just _build_loss / augmentation.

===========================================================================
INPUT / TARGET BOUNDARY (printed once at the start of training, not just
documented here -- see _confirm_io_boundary())
===========================================================================
forward() receives EXACTLY: data = [raw image, real Model2 vessel
prediction] (2 channels, Dataset313's own imagesTr -- reused as-is, no new
dataset build). Nothing else.

Two ground-truth items exist and BOTH are loss-only, never passed to
forward():
  A. target (Dataset313's own 53-class label: background + 52 locations) --
     Branch A's Dice/CE loss COLLAPSES this to binary foreground-vs-
     background (see BinaryCollapsedDiceCE below); Branch B's classification
     target is ALSO derived from this exact same tensor (its patch center
     voxel's class id, majority-nonzero as a fallback -- see
     _derive_cls_target()). No second label volume, no custom dataloader:
     Dataset313's raw 53-class label already carries everything both
     branches need, this trainer just reads it two different ways.
  B. Nothing else -- there is no separate "instance class" ground truth
     file; deriving it from `target` above IS the answer to "what should
     Branch B's target granularity be" this trainer's own design settled on
     (per-patch, matching Branch A's own patch granularity, not per-case or
     per-instance-after-detection -- see this file's own design discussion,
     not reproduced here).

===========================================================================
WHY REUSE DATASET313 RATHER THAN BUILD A NEW ONE
===========================================================================
Dataset313 (built for the now-superseded Experiment A) is EXACTLY the input
this needs: 2-channel (raw image + Model2's real, non-oracle, whole-head
vessel prediction), 53-class label (background + 52 locations), already
preprocessed (job 79). Reusing it outright, rather than building a
dedicated binary-labeled dataset, is what lets both branches share one
label tensor with zero new data-prep engineering -- the tradeoff is that
the network's segmentation head structurally outputs 53 channels (matching
label_manager.num_segmentation_heads, keeping nnU-Net's own deep-supervision/
pseudo-dice-tracking machinery internally consistent) even though Branch A's
LOSS only ever asks it to discriminate foreground-vs-background, never among
the 52 location sub-classes. The 52 raw sub-class channels nnU-Net's own
online pseudo-dice logger will print every epoch are therefore NOT a
meaningful metric here (the network was never given gradient to
differentiate them) -- watch the '[expC]' binary dice and classification
accuracy this trainer prints instead, not nnU-Net's own "Pseudo dice" line.

===========================================================================
CLASS IMBALANCE (Branch B) AND SIZE IMBALANCE (Branch A) -- BOTH HANDLED,
NOT JUST ONE
===========================================================================
- Class imbalance (Branch B, 52-way, some classes with single-digit or zero
  training examples): TWO levers, both already proven in this exact
  codebase for this exact failure mode (class_balanced_focal.py's own
  postmortem, and Experiment A's own job 81->87 fix) -- (1) case-level
  class-balanced sampling (get_dataloaders() below, copied from
  class_balanced_focal.py verbatim) so a case whose only lesion is a rare
  class isn't drowned out by common ones, AND (2) effective-number-of-
  samples loss (Cui et al., CVPR 2019) on top, not flat inverse-frequency
  (which already made a different classifier's results worse this week).
- Size imbalance (Branch A, binary Dice/CE): aneurysm size varies hugely
  (a few mm to 15mm+), and a pooled/batch-aggregated Dice would let a large
  lesion's raw voxel count dominate a batch's gradient over a tiny one in
  the same batch. BinaryCollapsedDiceCE computes Dice PER SAMPLE (never
  pooling intersection/union across the batch) then averages the resulting
  per-sample [0,1] scores -- so a 20-voxel lesion and a 2000-voxel lesion
  each contribute one equally-weighted term, regardless of their own size.
  This is the batch_dice=False equivalent, deliberately hardcoded rather
  than left to whatever Dataset313's plans.json default happens to be.

===========================================================================
LESSON CARRIED OVER FROM EXPERIMENT A'S JOB 81 COLLAPSE
===========================================================================
oversample_foreground_percent=1.0 mechanically guarantees a sampled patch
CONTAINS a foreground voxel -- it does NOT by itself fix per-class exposure
across the whole training run (that needed the case-level sampling lever
above too), and per-batch foreground-voxel-fraction logging is informational
only (a low fraction is NORMAL for a tiny aneurysm in a big patch, it is NOT
itself evidence guaranteed-positive sampling is broken -- an earlier
diagnostic comment in this codebase said otherwise and was wrong). Logged
here anyway, unconditionally, for the first 300 batches, per this
experiment's own instrumentation requirement -- the real signal to watch is
per-class validation classification accuracy trending up, not this fraction.
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast

from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.training.nnUNetTrainer.project_specific.rsna2025.more_DAv6 import (
    RSNA2025Trainer_moreDAv6_1_SkeletonRecallTverskyBeta07,
)

N_LOC = 52


def effective_number_weights(class_counts: np.ndarray, beta: float = 0.999) -> np.ndarray:
    """Cui et al., CVPR 2019 -- identical formula to expB_train_classifier.py's
    own copy, duplicated (not cross-imported) deliberately: this file lives
    under the nnUNet fork's own package tree, invoked via the nnUNetv2_train
    console script, which is not guaranteed to have topaneu_rsna importable
    on sys.path the way `python -m topaneu_rsna...` invocations are -- a
    six-line duplication is a far smaller risk than an untested cross-
    package import breaking trainer lookup for every trainer in this folder
    (see job 43/job 80's own precedent on how fragile that lookup is)."""
    counts = np.maximum(class_counts, 1)
    effective_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / effective_num
    weights = weights / weights.sum() * len(counts)
    return weights.astype(np.float32)


class DualHeadNetwork(nn.Module):
    """Wraps a stock nnU-Net segmentation network (unmodified decoder/deep
    supervision) with a small classification head reading the SAME encoder's
    bottleneck features. forward(x) returns ONLY the segmentation output
    (matching what nnUNetPredictor's stock sliding-window inference expects
    from self.network(x), a plain tensor/list -- not a tuple) so this
    network stays inference-pipeline-compatible; forward_with_cls(x) is what
    this trainer's own train_step/validation_step actually call, returning
    both outputs. The encoder runs twice per forward_with_cls call (once
    inside seg_network's own forward, once standalone for pooling) -- a
    deliberate, accepted compute cost in exchange for never touching the
    stock decoder's internals directly."""

    def __init__(self, seg_network: nn.Module, encoder_channels: int,
                n_cls_classes: int = N_LOC, dropout: float = 0.3):
        super().__init__()
        self.seg_network = seg_network
        self.cls_head = nn.Sequential(
            nn.Linear(encoder_channels, 512), nn.Dropout(dropout), nn.ReLU(inplace=True),
            nn.Linear(512, 256), nn.Dropout(dropout), nn.ReLU(inplace=True),
            nn.Linear(256, n_cls_classes),
        )

    def forward(self, x):
        return self.seg_network(x)

    def forward_with_cls(self, x):
        seg_out = self.seg_network(x)
        feat = self.seg_network.encoder(x)[-1]
        pooled = feat.mean(dim=(2, 3, 4))
        cls_logits = self.cls_head(pooled)
        return seg_out, cls_logits


class BinaryCollapsedDiceCE(nn.Module):
    """Standard Dice + CE (per the task's own 'well-proven elsewhere in this
    codebase' instruction), applied to a >2-channel softmax output collapsed
    to binary foreground-vs-background: fg = sum of all 52 location
    channels' softmax probability, bg = the background channel. Dice is
    computed PER SAMPLE (batch_dice=False, hardcoded, not left to the
    dataset's own plans default) and averaged across the batch afterward --
    this is the size-imbalance fix, see this module's own docstring."""

    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_bin = (target[:, 0] > 0).float()                    # (B, D, H, W)
        probs = torch.softmax(logits.float(), dim=1)
        fg_prob = probs[:, 1:].sum(1)                               # (B, D, H, W)
        bg_logit = logits[:, 0]
        fg_logit = torch.logsumexp(logits[:, 1:], dim=1)
        ce = F.cross_entropy(torch.stack([bg_logit, fg_logit], dim=1), target_bin.long())

        axes = tuple(range(1, fg_prob.ndim))
        inter = (fg_prob * target_bin).sum(axes)
        denom = fg_prob.sum(axes) + target_bin.sum(axes)
        dice_per_sample = (2 * inter + self.smooth) / (denom + self.smooth)
        dice_loss = 1.0 - dice_per_sample.mean()

        return dice_loss + ce


def _derive_cls_target(target_full: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """target_full: (B, 1, D, H, W) raw location-class ids 0..52 (Dataset313's
    own label, full resolution -- target[0] if deep supervision). Returns
    (cls_ids 0-indexed for CrossEntropyLoss, valid mask) -- per-patch target
    is the class at the patch's own center voxel (where guaranteed-positive
    sampling places the forced-in lesion), falling back to the majority
    nonzero voxel in the rare case augmentation has shifted the center off
    the lesion. `valid=False` for a sample with literally no foreground
    voxel at all (should not happen with oversample_foreground_percent=1.0,
    handled defensively rather than assumed impossible)."""
    B = target_full.shape[0]
    center = tuple(s // 2 for s in target_full.shape[2:])
    cls_ids = target_full[:, 0, center[0], center[1], center[2]].clone().long()
    valid = torch.ones(B, dtype=torch.bool, device=target_full.device)
    for b in range(B):
        if cls_ids[b].item() == 0:
            voxels = target_full[b, 0]
            nz = voxels[voxels > 0]
            if nz.numel() > 0:
                vals, counts = torch.unique(nz, return_counts=True)
                cls_ids[b] = vals[counts.argmax()]
            else:
                valid[b] = False
    return (cls_ids - 1).clamp(min=0), valid   # 0-indexed; invalid rows excluded by caller, not by clamping


class RSNA2025Trainer_ExpC_JointSegCls(RSNA2025Trainer_moreDAv6_1_SkeletonRecallTverskyBeta07):
    def __init__(self, plans: dict, configuration: str, fold, dataset_json: dict,
                device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        # Raised from 200 to 1000. Job 99 actually ran to full completion
        # (200/200 epochs, not killed early as originally planned) --
        # Branch A binary_dice converged to a real, stable 0.6-0.75, and
        # Branch B cls_accuracy improved from ~5.9% avg (epochs 0-27) to
        # ~10.5% avg (epochs 174-199), a genuine if modest upward trend,
        # not a plateau. Given that real progress, job 102 RESUMES from
        # job 99's completed checkpoint via --c rather than restarting from
        # epoch 0 (unlike the earlier plan, back when only 28/200 epochs
        # were expected to be sunk cost).
        self.num_epochs = 1000
        # Same fix Experiment A's job 87->100 needed: PolyLRScheduler
        # computes LR purely from current_epoch/num_epochs, so resuming a
        # FINISHED 200-epoch run (LR decayed to ~8e-05 by the end) against a
        # NEW num_epochs=1000 would see current_epoch=200 as only 20% through
        # and push LR back up toward ~0.008 -- large enough to disrupt
        # Branch A's already-converged 0.6-0.75 dice. Lowered to roughly the
        # same order of magnitude Experiment A's own resume used, well below
        # where job 99's schedule would otherwise re-warm to.
        self.initial_lr = 5e-4
        self.save_every = 5
        self.oversample_foreground_percent = 1.0
        self.weight_seg = 1.0
        self.weight_cls = 1.0
        self.loss_cls = None   # built in initialize(), needs per-class training counts first
        self._fg_log_batches_remaining = 300
        self._printed_io_confirmation = False

    @staticmethod
    def build_network_architecture(architecture_class_name, arch_init_kwargs,
                                   arch_init_kwargs_req_import, num_input_channels,
                                   num_output_channels, enable_deep_supervision: bool = True):
        seg_network = get_network_from_plans(
            arch_class_name=architecture_class_name, arch_kwargs=arch_init_kwargs,
            arch_kwargs_req_import=arch_init_kwargs_req_import,
            input_channels=num_input_channels, output_channels=num_output_channels,
            allow_init=True, deep_supervision=enable_deep_supervision)
        encoder_channels = int(seg_network.encoder.output_channels[-1])
        return DualHeadNetwork(seg_network, encoder_channels, n_cls_classes=N_LOC)

    def set_deep_supervision_enabled(self, enabled: bool):
        """Stock nnUNetTrainer's own version (its own docstring: 'this
        function is specific for the default architecture ... if you change
        the architecture, there are chances you need to change this as
        well') does `mod.decoder.deep_supervision = enabled` where `mod` is
        self.network directly -- true for the inner seg_network, not for
        DualHeadNetwork, which holds it as self.seg_network. Same DDP/
        torch.compile unwrapping as stock, just redirected one level in."""
        from torch._dynamo import OptimizedModule
        if self.is_ddp:
            mod = self.network.module
        else:
            mod = self.network
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod
        mod.seg_network.decoder.deep_supervision = enabled

    def _build_loss(self):
        loss_seg = BinaryCollapsedDiceCE()
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2**i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            loss_seg = DeepSupervisionWrapper(loss_seg, weights)
        return loss_seg

    def initialize(self):
        super().initialize()
        self._load_pretrained_seg_weights()

        class_counts = self._count_training_classes()
        weights = effective_number_weights(class_counts)
        self.loss_cls = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(self.device))
        self.print_to_log_file(
            f"[expC] classification class weights (effective-number, beta=0.999): "
            f"{class_counts.astype(int).tolist()} training-case counts -> "
            f"weight range {weights.min():.3f}-{weights.max():.3f}")

    def _load_pretrained_seg_weights(self):
        """Deliberately NOT using nnU-Net's own -pretrained_weights CLI flag:
        that mechanism (nnunetv2/run/load_pretrained_weights.py) hard-asserts
        every key in the network (besides .seg_layers.) already exists in the
        checkpoint -- correct for every other trainer in this codebase (same
        architecture, just a channel-count change), but cls_head is a
        genuinely NEW submodule with no counterpart in a segmentation-only
        checkpoint, so that assertion can never pass for this network. Loads
        directly into self.network.seg_network (the inner stock architecture,
        no 'seg_network.' key prefix needed here since we're addressing that
        submodule directly, not the outer wrapper) with strict=False instead
        -- the same lenient pattern cls/backbone.py's NnUNetTruncatedBackbone
        and expB_train_classifier.py's build_model() already use successfully
        for exactly this kind of partial-checkpoint warm start. cls_head
        stays randomly initialized, as it should -- it's new, not something
        this checkpoint could ever have an answer for."""
        ckpt_path = os.environ.get("EXPC_PRETRAINED_SEG_CKPT")
        if not ckpt_path:
            self.print_to_log_file("[expC] EXPC_PRETRAINED_SEG_CKPT not set -- "
                                   "training seg_network from random init, no warm start.")
            return
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        weights = state.get("network_weights", state)
        weights = {k: v for k, v in weights.items() if not k.startswith("decoder.seg_layers.")}
        missing, unexpected = self.network.seg_network.load_state_dict(weights, strict=False)
        self.print_to_log_file(
            f"[expC] warm-started seg_network from {ckpt_path}: "
            f"missing={len(missing)} unexpected={len(unexpected)} "
            f"(missing should be ~0 -- cls_head is a separate submodule, not part of "
            f"seg_network, so it never appears here at all)")
        if len(missing) > 20:
            raise RuntimeError("Too many missing keys warm-starting seg_network -- "
                               "checkpoint architecture doesn't match Dataset313's plans.json.")

    def _count_training_classes(self) -> np.ndarray:
        from batchgenerators.utilities.file_and_folder_operations import join, load_pickle
        dataset_tr, _ = self.get_tr_and_val_datasets()
        counts = np.zeros(N_LOC + 1)   # index 0 unused, class ids are 1..52
        for key in dataset_tr.identifiers:
            props = load_pickle(join(dataset_tr.source_folder, key + ".pkl"))
            class_locations = props.get("class_locations", {}) or {}
            for c, locs in class_locations.items():
                if 1 <= c <= N_LOC and len(locs) > 0:
                    counts[c] += 1
        return counts[1:]

    def _class_balanced_case_weights(self, dataset_tr) -> np.ndarray:
        """Verbatim from class_balanced_focal.py -- the case-level sampling
        lever, see this file's own module docstring for why both that lever
        and the loss-level one below are used together."""
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
            f"[expC] class-balanced sampling: {n_classes_seen}/{len(all_labels)} location "
            f"classes appear at least once in this training split.")
        return weights

    def get_dataloaders(self):
        """Verbatim from class_balanced_focal.py, adapted to this trainer's
        own name -- only change from the inherited SkeletonRecall version:
        training loader gets class-balanced sampling_probabilities."""
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
        (rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size,
        mirror_axes) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

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
        dl_tr = loader_cls(dataset_tr, self.batch_size, initial_patch_size,
                           self.configuration_manager.patch_size, self.label_manager,
                           oversample_foreground_percent=self.oversample_foreground_percent,
                           sampling_probabilities=train_sampling_probabilities, pad_sides=None,
                           transforms=tr_transforms)
        dl_val = loader_cls(dataset_val, self.batch_size, self.configuration_manager.patch_size,
                            self.configuration_manager.patch_size, self.label_manager,
                            oversample_foreground_percent=self.oversample_foreground_percent,
                            sampling_probabilities=None, pad_sides=None, transforms=val_transforms)

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(
                data_loader=dl_tr, transform=None, num_processes=allowed_num_processes,
                num_cached=max(6, allowed_num_processes // 2), seeds=None,
                pin_memory=self.device.type == "cuda", wait_time=0.002)
            mt_gen_val = NonDetMultiThreadedAugmenter(
                data_loader=dl_val, transform=None, num_processes=max(1, allowed_num_processes // 2),
                num_cached=max(3, allowed_num_processes // 4), seeds=None,
                pin_memory=self.device.type == "cuda", wait_time=0.002)
        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        return mt_gen_train, mt_gen_val

    def _confirm_io_boundary(self, data, target):
        if self._printed_io_confirmation:
            return
        t = target[0] if isinstance(target, list) else target
        self.print_to_log_file(
            "[expC I/O BOUNDARY CHECK] forward_with_cls() input tensor: `data` "
            f"shape={tuple(data.shape)} (2 channels: raw image + real vessel prediction). "
            f"LOSS-ONLY tensor: `target` shape={tuple(t.shape)} (Dataset313's 53-class "
            "label -- used to compute both the binary seg loss (collapsed) and the "
            "classification loss (center-voxel/majority class), NEVER passed to "
            "forward_with_cls). If this ever changes, that is a leakage bug.")
        self._printed_io_confirmation = True

    def _log_foreground_ratio(self, target):
        t = target[0] if isinstance(target, list) else target
        frac = float((t > 0).float().mean().item())
        self.print_to_log_file(
            f"[expC fg-check] foreground voxel fraction = {frac:.4f} -- informational "
            f"only, a small value is NORMAL for a tiny lesion in a large patch and is "
            f"NOT itself evidence guaranteed-positive sampling is broken.")
        self._fg_log_batches_remaining -= 1

    def train_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self._confirm_io_boundary(data, target)
        if self._fg_log_batches_remaining > 0:
            self._log_foreground_ratio(target)

        target_full = target[0] if isinstance(target, list) else target
        cls_ids, valid = _derive_cls_target(target_full)

        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            seg_out, cls_logits = self.network.forward_with_cls(data)
            loss_seg = self.loss(seg_out, target)
            loss_cls = (self.loss_cls(cls_logits[valid], cls_ids[valid])
                       if valid.any() else torch.zeros((), device=self.device))
            l = self.weight_seg * loss_seg + self.weight_cls * loss_cls

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        with torch.no_grad():
            cls_correct = int((cls_logits[valid].argmax(1) == cls_ids[valid]).sum()) if valid.any() else 0
            cls_total = int(valid.sum())

        return {"loss": l.detach().cpu().numpy(), "loss_seg": float(loss_seg.detach()),
               "loss_cls": float(loss_cls.detach()) if valid.any() else np.nan,
               "cls_correct": cls_correct, "cls_total": cls_total}

    def on_train_epoch_end(self, train_outputs):
        super().on_train_epoch_end(train_outputs)
        outputs = collate_outputs(train_outputs)
        cls_acc = np.sum(outputs["cls_correct"]) / max(1, np.sum(outputs["cls_total"]))
        self.print_to_log_file(
            f"[expC] train: mean_loss_seg={np.nanmean(outputs['loss_seg']):.4f} "
            f"mean_loss_cls={np.nanmean(outputs['loss_cls']):.4f} "
            f"cls_accuracy={cls_acc:.4f} ({int(np.sum(outputs['cls_total']))} patches)")

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        target_full = target[0] if isinstance(target, list) else target
        cls_ids, valid = _derive_cls_target(target_full)

        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            seg_out, cls_logits = self.network.forward_with_cls(data)
            loss_seg = self.loss(seg_out, target)
            loss_cls = (self.loss_cls(cls_logits[valid], cls_ids[valid])
                       if valid.any() else torch.zeros((), device=self.device))
            l = self.weight_seg * loss_seg + self.weight_cls * loss_cls

        if self.enable_deep_supervision:
            output = seg_out[0]
            target_for_dice = target[0]
        else:
            output = seg_out
            target_for_dice = target

        # standard nnU-Net hard-dice bookkeeping on the raw 53-channel output
        # (informational -- see this file's own module docstring on why the
        # per-sub-class breakdown isn't the metric that matters here)
        axes = [0] + list(range(2, output.ndim))
        output_seg = output.argmax(1)[:, None]
        predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
        predicted_segmentation_onehot.scatter_(1, output_seg, 1)
        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target_for_dice, axes=axes)
        tp_hard, fp_hard, fn_hard = tp.detach().cpu().numpy()[1:], fp.detach().cpu().numpy()[1:], fn.detach().cpu().numpy()[1:]

        # this trainer's own binary-collapsed dice (the metric that actually
        # reflects Branch A's real training objective)
        with torch.no_grad():
            probs = torch.softmax(output.float(), dim=1)
            fg_pred = (probs[:, 1:].sum(1) > 0.5).float()
            fg_gt = (target_for_dice[:, 0] > 0).float()
            inter = (fg_pred * fg_gt).sum()
            denom = fg_pred.sum() + fg_gt.sum()
            binary_dice = float((2 * inter / denom).item()) if denom > 0 else float("nan")
            cls_correct = int((cls_logits[valid].argmax(1) == cls_ids[valid]).sum()) if valid.any() else 0
            cls_total = int(valid.sum())

        return {"loss": l.detach().cpu().numpy(), "tp_hard": tp_hard, "fp_hard": fp_hard,
               "fn_hard": fn_hard, "binary_dice": binary_dice,
               "cls_correct": cls_correct, "cls_total": cls_total}

    def on_validation_epoch_end(self, val_outputs):
        super().on_validation_epoch_end(val_outputs)
        outputs = collate_outputs(val_outputs)
        cls_acc = np.sum(outputs["cls_correct"]) / max(1, np.sum(outputs["cls_total"]))
        mean_binary_dice = float(np.nanmean(outputs["binary_dice"]))
        self.print_to_log_file(
            f"[expC] val: BRANCH A binary_dice={mean_binary_dice:.4f}  "
            f"BRANCH B cls_accuracy={cls_acc:.4f} "
            f"({int(np.sum(outputs['cls_total']))} patches) -- these two numbers, not "
            f"nnU-Net's own 'Pseudo dice' line above, are this experiment's real metrics.")
