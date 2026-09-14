"""
Task 2, 52-class location assignment: the rule-based alternative to a
directly-trained 52-way segmentation head (Dataset303/307/308/312, all of
which collapsed or stalled -- see config.py TRAINER_LOC comment). Follows the
TopAneu rule-based-localisation paper's decomposition: a binary aneurysm
segmenter finds *where* the lesion is, a vessel segmenter's own labels say
*what anatomy* surrounds it, and a small declared table -- no training --
reads the location off that anatomy, per lesion instance rather than per
voxel.

For each connected component of the binary aneurysm prediction:
  1. find its nearest/host vessel label within `--tau_mm` of the instance
     (utils.geometry.nearest_vessel_label -- touching labels win outright,
     otherwise nearest by Euclidean distance transform);
  2. look up that vessel's location(s) via labels.json's location_to_vessel
     (inverted here to vessel -> locations). Vessels hosting exactly one
     location resolve directly. Vessels hosting several (e.g. "BA" hosts 7)
     go through resolve_shared_vessel, which computes a junction-object
     candidate and an arc-fraction candidate independently and lets them
     compete, rather than the junction check short-circuiting the arc one
     outright (an earlier, junction-first version did exactly that, and
     concrete evidence showed it actively wrong: two of BA's three true
     "1.10 BA tip" instances computed textbook-confident arc-fractions --
     0.988 and 1.000, against a 0.9847 bucket boundary -- but still lost to
     the junction check, because BA-SCA's branch point sits only 0.5-1.1mm
     away on this vessel, well inside any radius loose enough to catch
     genuine junction cases elsewhere):
       a. junction-object candidate (utils.vessel_skeleton.JUNCTION_BRANCH,
          Paper 1's treatment of junctions as a contact patch rather than a
          position): for each of the vessel's hosted locations that is
          junction/bifurcation/terminus-defined, measure the instance's
          distance to the voxel contact patch between the host vessel and
          that location's declared branch vessel; the closest one within
          `--junction_tau_mm` is this candidate (or none, if nothing
          qualifies);
       b. arc-fraction candidate (utils.vessel_skeleton): skeletonize this
          case's own copy of the host vessel, orient it via its declared
          proximal anchor, project the instance's centroid onto it, and look
          up which cohort-derived arc-fraction bucket
          (vessel_location_prior.json's "arc" table) it falls into, flagging
          whether the fraction sits within `--arc_ambiguous_margin` of the
          boundary to a neighbouring bucket (a close call) or confidently
          inside one. A bucket flagged "low_sample" (built from fewer than
          build_vessel_location_prior.py's --min_arc_samples training
          instances) counts as no candidate at all -- untrustworthy either
          way;
       c. the junction candidate wins only if the arc candidate doesn't
          exist, OR the junction distance is within the tighter
          `--junction_override_mm` AND the arc candidate is itself
          ambiguous -- i.e. a confident position reading should not lose
          just because some junction happens to be nearby; it only loses
          when the junction evidence is strong and the position evidence is
          itself equivocal. Otherwise the arc candidate wins;
       d. if neither produced anything usable (including the low_sample
          case, kept as its own "arc_low_sample_fallback" diagnostic
          category), fall back to the flat cohort-majority location.
  3. laterality mostly falls out for free: vessel names are already
     lateralized (e.g. "L-PICA" vs "R-PICA"), so matching the correct-side
     vessel by geometry already gets the side right for locations on a
     paired host vessel -- no midline fit needed there. The exception is a
     junction-type location shared by an *unpaired* midline host vessel
     (e.g. "BA" hosts both "R-1.9 BA-SCA junction" and "L-1.9 BA-SCA
     junction" in the same step-2a contact-patch loop above), where nothing
     upstream already forces the correct side; for those,
     utils.vessel_skeleton.reconcile_side applies Paper 1's side-
     reconciliation directly -- a per-case midline calibrated from the
     vessel map's own other paired R-/L- labels overrides the contact-patch
     winner's side if the lesion's own position disagrees with it.
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
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import ndimage
from tqdm import tqdm

from topaneu_rsna import config as C
from topaneu_rsna.utils import geometry as geo
from topaneu_rsna.utils import io as uio
from topaneu_rsna.utils import vessel_skeleton as vsk
from topaneu_rsna.seg.evaluate_location import hd95
from topaneu_rsna.seg.evaluate_official import official_score, print_official

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


def _junction_candidate(inst, vessel_map, spacing, vessel_id, locs,
                        name_to_id, junction_tau_mm):
    """Best junction-object candidate within `junction_tau_mm`, or
    (None, inf) if none qualifies. Does NOT decide anything by itself --
    see resolve_shared_vessel for how this competes against the arc
    candidate rather than winning outright just by existing."""
    best_loc, best_dist = None, np.inf
    for loc in locs:
        branch = vsk.JUNCTION_BRANCH.get(loc)
        if not branch:
            continue
        d = vsk.contact_patch_distance(inst, vessel_map, spacing, vessel_id,
                                       branch, name_to_id)
        if d is not None and d <= junction_tau_mm and d < best_dist:
            best_loc, best_dist = loc, d
    return best_loc, best_dist


def _arc_candidate(inst, vessel_map, spacing, vessel_name, vessel_id, prior,
                   name_to_id, arc_ambiguous_margin):
    """Best arc-fraction candidate. Returns (location_or_None,
    lesion-to-skeleton distance, is_ambiguous, is_low_sample):
      - unorientable / no table at all: (None, inf, False, False)
      - orientable but the winning bucket is low_sample-flagged: (None,
        skel_dist, False, True) -- untrustworthy, but distinct from the
        "couldn't compute a position at all" case so callers can still
        report it as its own diagnostic category
      - confident hit: (location, skel_dist, is_ambiguous, False), where
        `is_ambiguous` is True when the fraction sits within
        `arc_ambiguous_margin` of the boundary separating its bucket from a
        neighbour -- a close call between two adjacent locations rather
        than sitting confidently inside one."""
    arc_info = prior.get("arc", {}).get(vessel_name)
    if not arc_info or not arc_info["locations"]:
        return None, np.inf, False, False
    raw = vsk.extract_skeleton(vessel_map == vessel_id, spacing)
    if raw is None:
        return None, np.inf, False, False
    skel = vsk.orient_skeleton(raw, vessel_map, spacing,
                               vsk.PROXIMAL_ANCHOR.get(vessel_name, []), name_to_id)
    centroid_mm = np.argwhere(inst).mean(0) * np.asarray(spacing, np.float64)
    frac, skel_dist = vsk.arc_fraction(skel, centroid_mm)
    if frac is None:
        return None, np.inf, False, False

    bucketed = arc_info["locations"]
    boundaries = arc_info["boundaries"]
    idx = min(int(np.searchsorted(boundaries, frac)), len(bucketed) - 1)
    entry = bucketed[idx]
    if entry.get("low_sample"):
        return None, skel_dist, False, True

    nearby = [b for b in (boundaries[idx - 1] if idx > 0 else None,
                         boundaries[idx] if idx < len(boundaries) else None)
             if b is not None]
    ambiguous = bool(nearby) and min(abs(frac - b) for b in nearby) < arc_ambiguous_margin
    return entry["location"], skel_dist, ambiguous, False


def resolve_shared_vessel(inst: np.ndarray, vessel_map: np.ndarray, spacing,
                          vessel_name: str, vessel_id: int, locs: list,
                          prior: dict, name_to_id: dict, junction_tau_mm: float,
                          junction_override_mm: float,
                          arc_ambiguous_margin: float) -> tuple[str, str, float | None]:
    """Resolve which of a shared vessel's several locations one instance
    belongs to. Both the junction-object check and the arc-fraction check
    always compute their own best candidate independently (neither
    short-circuits the other) -- this replaced an earlier junction-first
    version after concrete evidence it was actively wrong: two of BA's three
    true "1.10 BA tip" instances computed textbook-confident arc-fractions
    (0.988 and 1.000, against a 0.9847 bucket boundary) but were still
    intercepted by the junction check, because BA-SCA's branch point sits
    only 0.5-1.1mm away -- well inside any threshold loose enough to catch
    genuine junction cases elsewhere on the same vessel. Tightening
    `--junction_tau_mm` further couldn't have fixed this: those distances
    were already small, the check was just being asked the wrong question
    (nearest vs. best-explains-the-evidence).

    The junction candidate now only overrides the arc candidate when it is
    BOTH close in absolute terms (`junction_override_mm`, meant to be
    tighter than `junction_tau_mm`) AND the arc call is itself ambiguous
    (`arc_ambiguous_margin` of a bucket boundary) rather than sitting
    confidently inside a bucket -- a confident arc position should not lose
    to a junction match just because one happens to be nearby; it should
    only lose when the junction evidence is strong AND the position
    evidence is itself equivocal.

    Returns (assigned_location, resolved_by, extra_dist_mm): resolved_by is
    one of "junction", "arc", "arc_low_sample_fallback", "majority";
    extra_dist_mm is the junction-contact or lesion-to-skeleton distance
    (whichever was used), or None for "majority"."""
    j_loc, j_dist = _junction_candidate(inst, vessel_map, spacing, vessel_id,
                                        locs, name_to_id, junction_tau_mm)
    a_loc, a_dist, a_ambiguous, a_low_sample = _arc_candidate(
        inst, vessel_map, spacing, vessel_name, vessel_id, prior, name_to_id,
        arc_ambiguous_margin)

    if j_loc is not None and (a_loc is None or (j_dist <= junction_override_mm
                                                and a_ambiguous)):
        # Side reconciliation (Paper 1): an unpaired midline host vessel like
        # "BA" hosts both "R-1.9 BA-SCA junction" and "L-1.9 BA-SCA junction"
        # among `locs`, so nothing upstream already forces the correct side
        # the way a paired host vessel (e.g. "L-PICA") does -- trust the
        # lesion's own position relative to a per-case calibrated midline
        # over a contact-patch distance that can be swayed by segmentation
        # noise between two closely-spaced bilateral branches.
        j_loc = vsk.reconcile_side(j_loc, inst, vessel_map, spacing, name_to_id, set(locs))
        return j_loc, "junction", j_dist

    if a_loc is not None:
        return a_loc, "arc", a_dist

    if a_low_sample:
        return prior["majority"].get(vessel_name, locs[0]), "arc_low_sample_fallback", a_dist

    return prior["majority"].get(vessel_name, locs[0]), "majority", None


def assign_case(binmask: np.ndarray, vessel_map: np.ndarray, spacing,
                vessel_names: list, vessel_to_locations: dict, prior: dict,
                name_to_id: dict, loc_value: dict, tau_mm: float,
                junction_tau_mm: float, junction_override_mm: float,
                arc_ambiguous_margin: float, min_voxels: int,
                gt: np.ndarray | None = None):
    """Returns (final_mask uint16 of location ids, list of per-instance dicts).
    `gt` (the case's ground-truth location mask), if given, adds `true_class`
    and `correct` diagnostic fields per instance -- for --instances_csv."""
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
        resolved_by = None
        extra_dist = None
        if vessel_id is not None:
            vessel_name = vessel_names[vessel_id - 1]
            locs = vessel_to_locations.get(vessel_name)
            if locs:
                if len(locs) == 1:
                    assigned, resolved_by = locs[0], "single"
                else:
                    assigned, resolved_by, extra_dist = resolve_shared_vessel(
                        inst, vessel_map, spacing, vessel_name, vessel_id,
                        locs, prior, name_to_id, junction_tau_mm,
                        junction_override_mm, arc_ambiguous_margin)

        rec = dict(instance_idx=i, size=size, vessel=vessel_name, dist=dist,
                  assigned=assigned, resolved_by=resolved_by, extra_dist=extra_dist)
        if gt is not None:
            vals, counts = np.unique(gt[inst], return_counts=True)
            nz = vals != 0
            true_cls = int(vals[nz][np.argmax(counts[nz])]) if nz.any() else 0
            rec["true_class"] = true_cls
            rec["correct"] = bool(assigned is not None and loc_value[assigned] == true_cls)
        instances.append(rec)
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
    ap.add_argument("--tau_mm", type=float, default=4.0,
                    help="host-vessel search radius (geometry.nearest_vessel_label)")
    ap.add_argument("--junction_tau_mm", type=float, default=4.0,
                    help="candidacy radius for the junction contact-patch check -- "
                         "how far a lesion may be from a branch contact patch and "
                         "still be considered a junction candidate at all (kept "
                         "independent of --tau_mm; see --junction_override_mm for "
                         "the separate, tighter bar for actually preferring it "
                         "over a competing arc-fraction candidate)")
    ap.add_argument("--junction_override_mm", type=float, default=1.5,
                    help="a junction candidate only overrides a competing, "
                         "non-ambiguous arc-fraction candidate when its distance "
                         "is within this (tighter than --junction_tau_mm) bound -- "
                         "see resolve_shared_vessel for why a merely-qualifying "
                         "junction distance isn't enough on its own")
    ap.add_argument("--arc_ambiguous_margin", type=float, default=0.03,
                    help="an arc-fraction candidate is 'ambiguous' (and so can "
                         "lose to a close junction candidate) when its fraction "
                         "sits within this margin, in [0,1] arc-length units, of "
                         "the boundary separating its bucket from a neighbour")
    ap.add_argument("--min_voxels", type=int, default=3)
    ap.add_argument("--prior", type=Path, default=PRIOR_PATH)
    ap.add_argument("--write_masks", action="store_true")
    ap.add_argument("--out_dir", type=Path, default=C.WORK / "task2_rule_masks")
    ap.add_argument("--out_csv", type=Path,
                    default=C.LOG_ROOT / "task2_rule_assignment.csv")
    ap.add_argument("--instances_csv", type=Path, default=None,
                    help="optional per-instance diagnostic dump (case, vessel, "
                         "resolved_by, distance, true/predicted class, correct) "
                         "-- e.g. to check the actual junction-contact distance "
                         "behind a given resolved_by=junction correct hit before "
                         "picking --junction_tau_mm")
    ap.add_argument("--official_metrics", action="store_true",
                    help="also score with evaluate_official.py's replica of the "
                         "TopAneu-26 grand-challenge Task 2 evaluator (harsher "
                         "Dice/HD95, looser presence-based classification metrics "
                         "-- a different convention from this script's own score(), "
                         "not a rerun of the rule itself)")
    ap.add_argument("--official_out_csv", type=Path,
                    default=C.LOG_ROOT / "task2_official_metrics.csv")
    a = ap.parse_args()

    if a.vessel_source == "pred" and a.vessel_pred_dir is None:
        raise SystemExit("--vessel_source pred requires --vessel_pred_dir")
    if not a.prior.exists():
        raise SystemExit(f"{a.prior} missing -- run "
                         "python -m topaneu_rsna.seg.build_vessel_location_prior first")

    spec = C.load_labels()
    vessel_to_locations = build_vessel_to_locations(spec)
    prior = json.loads(a.prior.read_text())
    loc_value = {loc: i + 1 for i, loc in enumerate(spec.locations)}
    name_to_id = {v: i + 1 for i, v in enumerate(spec.vessels)}

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
                                       spec.vessels, vessel_to_locations, prior,
                                       name_to_id, loc_value, a.tau_mm,
                                       a.junction_tau_mm, a.junction_override_mm,
                                       a.arc_ambiguous_margin, a.min_voxels, gt=gt)
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
    resolved_counts = Counter(i["resolved_by"] for i in all_instances)
    print(f"  resolution method: {dict(resolved_counts)}")

    if a.instances_csv is not None:
        id_to_loc = {v: k for k, v in loc_value.items()}
        a.instances_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(a.instances_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["case", "instance_idx", "vessel", "size", "host_dist_mm",
                       "resolved_by", "extra_dist_mm", "assigned", "true_class",
                       "correct"])
            for r in all_instances:
                w.writerow([r["case"], r["instance_idx"], r["vessel"], r["size"],
                           r["dist"], r["resolved_by"], r["extra_dist"], r["assigned"],
                           id_to_loc.get(r.get("true_class"), "background"),
                           r.get("correct")])
        print(f"  per-instance diagnostics written to {a.instances_csv}")

    per_class, pooled_acc, n_components = score(cases, preds, gts, loc_value, spec.n_loc)

    if a.official_metrics:
        off_per_class, off_avg = official_score(cases, preds, gts, loc_value, spec.n_loc)
        print_official(off_per_class, off_avg, len(cases), out_csv=a.official_out_csv)

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
