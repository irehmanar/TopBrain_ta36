"""How many Dataset302 (fine vessel, 36 classes) training cases actually
contain the two classes job 08's full 1000-epoch log shows never converging:
"3rd-A2" (vessel index 14, label value 15) stuck at exactly 0.0 dice every
single epoch, "3rd-A3" (index 15, label value 16) touching 0.08 once then
also ending at 0.0. Both are the accessory/azygos third-A2/A3 anterior
cerebral artery segment -- a real anatomical variant, not obviously a data
bug -- so before spending a full retrain (job 165) on class-balanced sampling,
this checks how many cases the sampler would even have to work with.

Reads the SAME class_locations pickles nnU-Net's own preprocessing already
wrote (one per case, at nnUNet_preprocessed/<dataset>/<plans>_3d_fullres/) --
no re-reading of raw label volumes needed, and this is exactly what
vessel_class_balanced.py's _class_balanced_case_weights() itself reads at
training time, so the count here is guaranteed consistent with what the
trainer will actually see.

    python -m topaneu_rsna.seg.vessel_rare_class_prevalence
"""
from __future__ import annotations

import argparse
import json

from batchgenerators.utilities.file_and_folder_operations import join, load_pickle, subfiles

from topaneu_rsna import config as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=int, default=C.DS_VESSEL)
    ap.add_argument("--plans", default=C.PLANS_RESENC)
    a = ap.parse_args()

    ds_name = f"Dataset{a.dataset:03d}_{C.DS_NAMES[a.dataset]}"
    # nnU-Net names the preprocessed folder after the plans JSON's own
    # configurations.3d_fullres.data_identifier, NOT "<plans_file_name>_3d_
    # fullres" -- confirmed on disk for Dataset302: the plans FILE is
    # nnUNetResEncUNetMPlans.json but the preprocessed folder is
    # nnUNetPlans_3d_fullres (a legacy default identifier the ResEnc planner
    # keeps unless told otherwise). Read it directly rather than guess again.
    plans_json = C.nnUNet_preprocessed / ds_name / f"{a.plans}.json"
    plans = json.loads(plans_json.read_text())
    data_identifier = plans["configurations"]["3d_fullres"]["data_identifier"]
    preproc_dir = C.nnUNet_preprocessed / ds_name / data_identifier
    pkls = sorted(subfiles(str(preproc_dir), suffix=".pkl", join=False))
    print(f"{len(pkls)} cases in {preproc_dir}")

    spec = C.load_labels()
    n_vessel = spec.n_vessel
    class_case_count = {i + 1: 0 for i in range(n_vessel)}  # label value -> case count
    class_voxels = {i + 1: 0 for i in range(n_vessel)}
    cases_with = {i + 1: [] for i in range(n_vessel)}

    for pkl in pkls:
        props = load_pickle(join(str(preproc_dir), pkl))
        class_locations = props.get("class_locations", {}) or {}
        case = pkl[:-4]
        for label, locs in class_locations.items():
            if label in class_case_count and locs is not None and len(locs) > 0:
                class_case_count[label] += 1
                class_voxels[label] += len(locs)
                cases_with[label].append(case)

    print(f"\n{'idx':<4}{'vessel':<16}{'label':<6}{'n_cases':<8}{'%cases':<8}{'total_voxels(sampled)':<10}")
    for i, name in enumerate(spec.vessels):
        label = i + 1
        n = class_case_count[label]
        pct = 100.0 * n / max(1, len(pkls))
        print(f"{i:<4}{name:<16}{label:<6}{n:<8}{pct:<8.1f}{class_voxels[label]:<10}")

    print("\nDetail for the two dead classes:")
    for i in (14, 15):
        label = i + 1
        name = spec.vessels[i]
        print(f"  {name} (label {label}): present in {class_case_count[label]} of {len(pkls)} cases "
             f"-- {cases_with[label]}")


if __name__ == "__main__":
    main()
