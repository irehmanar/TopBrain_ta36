"""
[TOPANEU] Class-balanced-sampling variant of Model 2's own trainer
(RSNA2025Trainer_moreDAv6_1_SkeletonRecallTverskyBeta07, Dataset302, 36
vessel classes), aimed at the two classes that never converge: job 08's full
1000-epoch log shows "3rd-A2" (index 14) at exactly 0.0 dice for every single
logged epoch, and "3rd-A3" (index 15) touching 0.08 once but ending at 0.0 --
these are the accessory/azygos third-A2/A3 anterior cerebral artery segment,
a genuine anatomical variant present in only a minority of cases, so most
training patches -- and likely most whole cases -- never contain it at all.

Reuses class_balanced_focal.py's _class_balanced_case_weights()/
get_dataloaders() VERBATIM (same mechanism already used for the ExpA
Dataset313 trainer): each training case is weighted by 1 / (number of cases
containing that case's rarest present class), so the handful of cases with
3rd-A2/A3 present get sampled far more often than uniform. Only the sampling
changes here -- _build_loss is inherited unchanged from
RSNA2025Trainer_moreDAv6_1_SkeletonRecallTverskyBeta07, so Model 2's own
already-validated loss (Tversky beta=0.7 + SkeletonRecall + CE, the
best-performing of the three loss variants tried in jobs 08/09) is untouched.
This isolates whether the two dead classes are a *sampling* problem (fixable)
or a genuine data-scarcity ceiling (job 44_prevalence_check should be read
first to know how many cases actually contain either class before expecting
this to work miracles -- if it's a handful of cases, class-balanced sampling
can make the model SEE them far more often, but cannot manufacture anatomical
diversity that doesn't exist in the data).

    nnUNetv2_train 302 3d_fullres all -p nnUNetResEncUNetMPlans \\
        -tr RSNA2025Trainer_moreDAv6_1_SkeletonRecallTverskyBeta07_ClassBalanced
"""
from __future__ import annotations

import numpy as np
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import join, load_pickle

from nnunetv2.training.dataloading.data_loader_2d_skel import nnUNetDataLoader2DSkel
from nnunetv2.training.dataloading.data_loader_3d_skel import nnUNetDataLoader3DSkel
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.nnUNetTrainer.project_specific.rsna2025.more_DAv6 import (
    RSNA2025Trainer_moreDAv6_1_SkeletonRecallTverskyBeta07,
)
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA


class RSNA2025Trainer_moreDAv6_1_SkeletonRecallTverskyBeta07_ClassBalanced(
    RSNA2025Trainer_moreDAv6_1_SkeletonRecallTverskyBeta07
):
    # _build_loss is intentionally NOT overridden -- inherits Model 2's own
    # proven Tversky(beta=0.7)+SkeletonRecall+CE loss unchanged.

    def _class_balanced_case_weights(self, dataset_tr) -> np.ndarray:
        """Verbatim copy of class_balanced_focal.py's own method (dataset-
        agnostic: reads self.label_manager.foreground_labels and each case's
        class_locations pickle, so it works for Dataset302's 36 classes the
        same way it already worked for Dataset313's 52)."""
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
            f"Class-balanced sampling: {n_classes_seen}/{len(all_labels)} of the vessel "
            f"classes appear at least once in this fold's training split. Case weights range "
            f"{weights.min():.2e} - {weights.max():.2e} (uniform would be {1.0 / len(weights):.2e})."
        )
        return weights

    def get_dataloaders(self):
        """Verbatim copy of class_balanced_focal.py's own get_dataloaders():
        same nnUNetTrainerSkeletonRecall-compatible Skel loaders (needed
        because SkeletonRecall's train_step() reads tmp["skel"]), only the
        training loader's sampling_probabilities changes from None to the
        class-balanced weights above; validation sampling stays uniform so
        val metrics still reflect the true class distribution."""
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
