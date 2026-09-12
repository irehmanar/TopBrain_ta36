"""
Task 2, 52-class location assignment: the rule-based alternative to a
directly-trained 52-way segmentation head (Dataset303/307/308/312, all of
which collapsed or stalled -- see config.py TRAINER_LOC comment). Follows the
TopAneu rule-based-localisation paper's decomposition: a binary aneurysm
segmenter finds *where* the lesion is, a vessel segmenter's own labels say
*what anatomy* surrounds it, and a small declared table -- no training --
reads the location off that anatomy, per lesion instance rather than per
voxel.

Simple first pass (see build_vessel_location_prior.py for why): for each
connected component of the binary aneurysm prediction,
  1. find its nearest/host vessel label within `--tau_mm` of the instance
     (utils.geometry.nearest_vessel_label -- touching labels win outright,
     otherwise nearest by Euclidean distance transform);
  2. look up that vessel's location(s) via labels.json's location_to_vessel
     (inverted here to vessel -> locations); vessels hosting exactly one
     location resolve directly; vessels hosting several (e.g. "BA" hosts 7)
     resolve to that vessel's cohort-majority location from
     vessel_location_prior.json;
  3. laterality falls out for free: vessel names are already lateralized
     (e.g. "L-PICA" vs "R-PICA"), so matching the correct-side vessel by
     geometry already gets the side right -- no separate midline fit.
An instance with no vessel label within tau is left unassigned (background in
the painted mask; scored as a miss for whatever its true class is, never a
false positive for any class).

Two vessel-channel sources:
  --vessel_source gt    ground-truth VESSEL_MASKS (36-class, whole-head,
                         native grid) -- an oracle/upper-bound pass, always
                         available for every training case, no new inference
                         needed. Use this first to check whether the rule
                         itself is sound before trusting a noisier real
                         vessel prediction.
  --vessel_source pred  a real (whole-head, native-grid) Model 2 prediction
                         directory, once one exists -- VESSEL_PRED_M2 today
                         is only ever computed on the coarse-ROI crop (jobs
                         05/06), not whole-head, so this needs a fresh
                         inference pass first; pass its output dir via
                         --vessel_pred_dir.

Binary-mask source is Dataset304 (whole-head, same native grid as
LOCATION_MASKS -- no crop/back-projection needed), pooled across its 5
folds' `validation/` predictions so every case is covered exactly once as
honest held-out data, same convention as seg/evaluate_location.py.

Scoring reports the same per-class Dice/VS/HD95/Precision/Recall/MCC as
evaluate_location.py, plus the paper's own headline unit: pooled accuracy
over ground-truth connected components (one component = one true lesion; a
component counts correct if the predicted instance overlapping it carries
its true location label).

    python -m topaneu_rsna.seg.assign_location_rule
    python -m topaneu_rsna.seg.assign_location_rule --vessel_source pred \\
        --vessel_pred_dir /path/to/whole_head_vessel_pred --write_masks
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import ndimage
from tqdm import tqdm

from topaneu_rsna import config as C
from topaneu_rsna.utils import geometry as geo
from topaneu_rsna.utils import io as uio
from topaneu_rsna.seg.evaluate_location import hd95

PRIOR_PATH = C.CODE_ROOT / "topaneu_rsna" / "vessel_location_prior.json"


def load_binary_pred_paths(dataset_id: int, trainer: str, plans: str, folds) -> dict:
    model_dir = (C.nnUNet_results / f"Dataset{dataset_id:03d}_{C.DS_NAMES[dataset_id]}"
                / f"{trainer}__{plans}__3d_fullres")
    out = {}
    for fold in folds:
        val_dir = model_dir / f"fold_{fold}" / "validation"
        if not val_dir.exists():
            continue
        for case in uio.list_cases(val_dir, C.LABEL_SUFFIX):
            out[case] = val_dir / f"{case}.nii.gz"
    return out


def build_vessel_to_locations(spec) -> dict[str, list[str]]:
    out = defaultdict(list)
    for loc, v in spec.location_to_vessel.items():
        out[v].append(loc)
    return dict(out)


def assign_case(binmask: np.ndarray, vessel_map: np.ndarray, spacing,
                vessel_names: list, vessel_to_locations: dict, majority: dict,
                loc_value: dict, tau_mm: float, min_voxels: int):
    """Returns (final_mask uint16 of location ids, list of per-instance dicts)."""
    lab, n = ndimage.label(binmask)
    final = np.zeros(binmask.shape, dtype=np.uint16)
    instances = []
    for i in range(1, n + 1):
        inst = lab == i
        size = int(inst.sum())
        if size < min_voxels:
            continue
        vessel_id, dist, touching = geo.nearest_vessel_label(
            inst, vessel_map, spacing, tau_mm=tau_mm)
        assigned = None
        vessel_name = None
        if vessel_id is not None:
            vessel_name = vessel_names[vessel_id - 1]
            locs = vessel_to_locations.get(vessel_name)
            if locs:
                assigned = locs[0] if len(locs) == 1 else majority.get(vessel_name, locs[0])
        instances.append(dict(size=size, vessel=vessel_name, dist=dist, assigned=assigned))
        if assigned is not None:
            final[inst] = loc_value[assigned]
    return final, instances


def score(cases: list[str], preds: dict, gts: dict, loc_value: dict, n_loc: int):
    names = list(loc_value)
    inter = np.zeros(n_loc); pred_sum = np.zeros(n_loc); gt_sum = np.zeros(n_loc)
    tp = np.zeros(n_loc); fp = np.zeros(n_loc); fn = np.zeros(n_loc); tn = np.zeros(n_loc)
    hd_sum = np.zeros(n_loc); hd_n = np.zeros(n_loc)

    comp_correct = comp_total = 0

    for case in cases:
        pred, spacing = preds[case]
        gt = gts[case]

        for c in range(1, n_loc + 1):
            pm, gm = pred == c, gt == c
            pn, gn = int(pm.sum()), int(gm.sum())
            it = int((pm & gm).sum())
            inter[c - 1] += it; pred_sum[c - 1] += pn; gt_sum[c - 1] += gn
            if gn > 0 and it > 0:
                tp[c - 1] += 1
            elif gn > 0:
                fn[c - 1] += 1
            elif pn > 0:
                fp[c - 1] += 1
            else:
                tn[c - 1] += 1
            if pn > 0 and gn > 0:
                h = hd95(pm, gm, spacing)
                if h is not None:
                    hd_sum[c - 1] += h; hd_n[c - 1] += 1

        gt_lab, n_gt = ndimage.label(gt > 0)
        for i in range(1, n_gt + 1):
            comp = gt_lab == i
            true_vals, true_counts = np.unique(gt[comp], return_counts=True)
            true_cls = int(true_vals[np.argmax(true_counts)])
            pred_vals, pred_counts = np.unique(pred[comp], return_counts=True)
            keep = pred_vals != 0
            pred_cls = (int(pred_vals[keep][np.argmax(pred_counts[keep])])
                       if keep.any() else 0)
            comp_total += 1
            comp_correct += int(pred_cls == true_cls)

    with np.errstate(invalid="ignore", divide="ignore"):
        dice = 2 * inter / (pred_sum + gt_sum)
        vs = 1 - np.abs(pred_sum - gt_sum) / (pred_sum + gt_sum)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        hd = hd_sum / hd_n

    per_class = dict(zip(names, zip(dice, vs, hd, precision, recall, mcc,
                                    tp.astype(int), fp.astype(int),
                                    fn.astype(int), tn.astype(int))))
    pooled_component_accuracy = comp_correct / comp_total if comp_total else float("nan")
    return per_class, pooled_component_accuracy, comp_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary_dataset", type=int, default=C.DS_ANEURYSM)
    ap.add_argument("--trainer", default=C.TRAINER_LOC)
    ap.add_argument("--plans", default=C.PLANS_RESENC)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--vessel_source", choices=["gt", "pred"], default="gt")
    ap.add_argument("--vessel_pred_dir", type=Path, default=None)
    ap.add_argument("--tau_mm", type=float, default=4.0)
    ap.add_argument("--min_voxels", type=int, default=3)
    ap.add_argument("--prior", type=Path, default=PRIOR_PATH)
    ap.add_argument("--write_masks", action="store_true")
    ap.add_argument("--out_dir", type=Path, default=C.WORK / "task2_rule_masks")
    ap.add_argument("--out_csv", type=Path,
                    default=C.LOG_ROOT / "task2_rule_assignment.csv")
    a = ap.parse_args()

    if a.vessel_source == "pred" and a.vessel_pred_dir is None:
        raise SystemExit("--vessel_source pred requires --vessel_pred_dir")
    if not a.prior.exists():
        raise SystemExit(f"{a.prior} missing -- run "
                         "python -m topaneu_rsna.seg.build_vessel_location_prior first")

    spec = C.load_labels()
    vessel_to_locations = build_vessel_to_locations(spec)
    majority = json.loads(a.prior.read_text())["majority"]
    loc_value = {loc: i + 1 for i, loc in enumerate(spec.locations)}

    bin_paths = load_binary_pred_paths(a.binary_dataset, a.trainer, a.plans, a.folds)
    print(f"{len(bin_paths)} held-out binary predictions "
         f"(Dataset{a.binary_dataset}, folds {a.folds})")

    preds, gts, all_instances = {}, {}, []
    for case, bp in tqdm(bin_paths.items(), desc="assigning"):
        binmask, meta = uio.read(bp)
        binmask = binmask > 0

        if a.vessel_source == "gt":
            vp = C.VESSEL_MASKS / f"{case}{C.LABEL_SUFFIX}"
        else:
            vp = a.vessel_pred_dir / f"{case}.nii.gz"
        if not vp.exists():
            continue
        vessel_map, _ = uio.read(vp)

        gt_p = C.LOCATION_MASKS / f"{case}{C.LABEL_SUFFIX}"
        if not gt_p.exists():
            continue
        gt, _ = uio.read(gt_p)

        final, instances = assign_case(binmask, vessel_map, meta["spacing"],
                                       spec.vessels, vessel_to_locations, majority,
                                       loc_value, a.tau_mm, a.min_voxels)
        for inst in instances:
            inst["case"] = case
        all_instances.extend(instances)

        preds[case] = (final, meta["spacing"])
        gts[case] = gt
        if a.write_masks:
            uio.write(final.astype(np.uint8), meta, a.out_dir / f"{case}.nii.gz")

    cases = sorted(set(preds) & set(gts))
    print(f"{len(cases)} cases scored "
         f"({sum(1 for i in all_instances if i['assigned'] is not None)}/"
         f"{len(all_instances)} instances assigned a location)")

    per_class, pooled_acc, n_components = score(cases, preds, gts, loc_value, spec.n_loc)

    a.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "dice", "vs", "hd95_mm", "precision", "recall", "mcc",
                    "tp", "fp", "fn", "tn"])
        for name, row in per_class.items():
            w.writerow([name, *row])

    def nanmean(key_idx):
        return float(np.nanmean([row[key_idx] for row in per_class.values()]))

    print(f"\npooled per-component accuracy: {pooled_acc:.4f} "
         f"over {n_components} ground-truth lesion instances")
    print(f"{'metric':<12}{'mean over classes':>20}")
    for i, k in enumerate(("dice", "vs", "hd95", "precision", "recall", "mcc")):
        print(f"{k:<12}{nanmean(i):>20.4f}")
    print(f"\nper-class detail written to {a.out_csv}")


if __name__ == "__main__":
    main()
