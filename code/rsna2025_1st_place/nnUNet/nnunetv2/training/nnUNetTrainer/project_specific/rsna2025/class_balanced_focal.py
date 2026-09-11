"""
[TOPANEU] Class-balanced, focal-loss variant of the 52-class location trainer,
aimed at Dataset303_TopAneuLocation's diagnosed collapse (see config.py's
DS_LOCATION comment): with do_bg=False, most training patches carry ground
truth for at most one of 52 classes, so the loss had too little per-class
signal to ever move logits off the trivial "predict background everywhere"
optimum. Separately, aneurysm size varies hugely across classes/types, so a
class that's usually large dominates a batch-averaged loss over one that's
usually tiny -- a second, independent imbalance on top of the class-frequency
one.

Two independent levers, stacked on top of RSNA2025Trainer_moreDAv6_1_SkeletonRecall
(SkeletonRecall + the thick-slice augmentation already used by the working
Task2 trainers) rather than replacing it:

  1. Class-balanced case sampling (get_dataloaders() below). nnU-Net's own
     foreground oversampling (oversample_foreground_percent) only guarantees
     a patch contains *some* foreground class present in whichever case was
     already drawn -- it does nothing about one class appearing in far more
     cases than another across the whole dataset (most cases have exactly one
     aneurysm, so most classes are absent from most cases). nnUNetDataLoader
     already accepts a sampling_probabilities constructor arg for exactly
     this, but every stock nnU-Net trainer always passes it as None. This
     computes per-case weights (inverse frequency of that case's rarest
     present class) from each case's class_locations -- already written to
     <case>.pkl by nnU-Net's own preprocessing for foreground oversampling,
     so no new preprocessing step is needed -- and passes them in for the
     *training* loader only; validation sampling is left uniform so val
     metrics still reflect the true class distribution.

  2. Focal Tversky++ instead of plain Tversky (_build_loss below). Reuses the
     existing DC_SkelREC_and_CE_loss wrapper (SkeletonRecall + CE) but swaps
     in MemoryEfficientFocalTverskyPlusPlusLoss (focal_tversky_plusplus_loss.py)
     as its dice_class -- the extra gamma_focal exponent further suppresses
     the loss contribution from classes/voxels the network already gets right
     easily (typically the larger, more common ones), concentrating gradient
     on the hard, small, rare cases this dataset is dominated by.

Also raises oversample_foreground_percent (0.33 -> 0.66) as a cheap third
lever in the same direction, and follows the _ep250/save_every=10 pattern
already used elsewhere in this pipeline for a faster diagnostic run instead
of the inherited 1000-epoch default.

    nnUNetv2_train 303 3d_fullres <fold> -p nnUNetResEncUNetMPlans \\
        -tr RSNA2025Trainer_moreDAv6_1_ClassBalancedFocalTversky_ep250
"""
from __future__ import annotations

import numpy as np
import torch
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import join, load_pickle

from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.project_specific.rsna2025.focal_tversky_plusplus_loss import (
    MemoryEfficientFocalTverskyPlusPlusLoss,
)
from nnunetv2.training.nnUNetTrainer.project_specific.rsna2025.more_DAv6 import (
    RSNA2025Trainer_moreDAv6_1_SkeletonRecall,
)
from nnunetv2.training.nnUNetTrainer.project_specific.rsna2025.skeleton_recall_loss import (
    DC_SkelREC_and_CE_loss,
)
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA


class RSNA2025Trainer_moreDAv6_1_ClassBalancedFocalTversky_ep250(RSNA2025Trainer_moreDAv6_1_SkeletonRecall):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250
        self.save_every = 10
        # nnU-Net's default (0.33) only guarantees ~1/3 of a batch touches
        # *some* foreground; raised here since with 52 sparsely-present
        # classes most non-forced patches would otherwise be pure background.
        self.oversample_foreground_percent = 0.66

    def _build_loss(self):
        loss = DC_SkelREC_and_CE_loss(
            soft_dice_kwargs={
                "batch_dice": self.configuration_manager.batch_dice,
                "do_bg": False,
                "ddp": self.is_ddp,
                "alpha": 0.3,
                "beta": 0.7,
                "gamma_pp": 2.0,
                "gamma_focal": 4.0 / 3.0,
                "smooth_nr": 1e-5,
                "smooth_dr": 1e-5,
            },
            soft_skelrec_kwargs={
                "batch_dice": self.configuration_manager.batch_dice,
                "smooth": 1e-5,
                "do_bg": False,
                "ddp": self.is_ddp,
            },
            ce_kwargs={},
            weight_ce=1,
            weight_dice=1,
            weight_srec=self.weight_srec,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientFocalTverskyPlusPlusLoss,
        )

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2**i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss

    def _class_balanced_case_weights(self, dataset_tr) -> np.ndarray:
        """
        Per-case sampling weight = 1 / (# training cases containing that
        case's rarest present class). A case with no annotated foreground at
        all (shouldn't happen for this dataset, but handled just in case)
        gets the uniform baseline weight instead of being excluded.
        """
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
        Copy of nnUNetTrainer.get_dataloaders() with exactly one change: the
        training nnUNetDataLoader gets class-balanced sampling_probabilities
        instead of None. There is no smaller extension point nnU-Net exposes
        for this -- sampling_probabilities is only ever set at construction.
        """
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

        patch_size = self.configuration_manager.patch_size
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

        dl_tr = nnUNetDataLoader(dataset_tr, self.batch_size,
                                 initial_patch_size,
                                 self.configuration_manager.patch_size,
                                 self.label_manager,
                                 oversample_foreground_percent=self.oversample_foreground_percent,
                                 sampling_probabilities=train_sampling_probabilities, pad_sides=None,
                                 transforms=tr_transforms,
                                 probabilistic_oversampling=self.probabilistic_oversampling)
        dl_val = nnUNetDataLoader(dataset_val, self.batch_size,
                                  self.configuration_manager.patch_size,
                                  self.configuration_manager.patch_size,
                                  self.label_manager,
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  sampling_probabilities=None, pad_sides=None, transforms=val_transforms,
                                  probabilistic_oversampling=self.probabilistic_oversampling)

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
