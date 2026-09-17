# TopAneu-26 Task 2 submission: rule-based location assignment

Implements `infer_ct`/`infer_mr` for the official submission template
(https://github.com/Bangulli/TopAneu-26, `templates/task2/`). `main.py` and
`Dockerfile` here are that template verbatim (fetched directly), plus the
additions marked `--- added for this pipeline ---` in the Dockerfile.

Per the challenge's own documented behavior, **Task 1 results are
auto-derived from this Task 2 mask** (class presence extraction) -- no
separate Task 1 container is being built.

## What you still need to do before this builds/runs

This directory is not yet complete or tested -- I don't have cluster access,
so these steps need to happen on your end:

### 1. Copy the vendored nnU-Net fork in
```bash
cp -r code/rsna2025_1st_place/nnUNet docker/task2_rule_based/nnUNet_src
```
Only the `nnunetv2/` package inside is actually needed at inference time (not
`documentation/`, tests, etc.) -- feel free to trim it down to keep the image
smaller, but copying the whole thing first and confirming it builds/runs is
safer than guessing what's safe to cut.

### 2. Copy the topaneu_rsna package in
```bash
cp -r code/topaneu_rsna docker/task2_rule_based/topaneu_rsna
```
This bundles `config.py`, `labels.json`, `vessel_location_prior.json`, and
`seg/assign_location_rule.py` (whose `assign_case()`/`build_vessel_to_locations()`
`inference.py` imports directly -- not reimplemented, so it can't silently
drift from the actual tuned rule). `config.py` will resolve
`TOPANEU_ROOT`/`SCRATCH_ROOT`/etc. to nonsense paths inside the container --
that's fine, nothing at inference time reads those, only `CODE_ROOT` (set via
the Dockerfile's `ENV`) and `LABELS_JSON`/the bundled `vessel_location_prior.json`
matter, both derived from `CODE_ROOT`.

### 3. Get the official `do_build.sh` / `do_test_run.sh` / `do_save.sh`
```bash
git clone https://github.com/Bangulli/TopAneu-26 /tmp/topaneu26
cp /tmp/topaneu26/templates/task2/do_*.sh docker/task2_rule_based/
```
I fetched `main.py` and `Dockerfile` verbatim to write the versions in this
directory, but not these three scripts in full -- pull them from the real
repo rather than trust a guessed version, since they define exactly how
`./test/input`/`./test/output` get wired up for local testing.

### 4. Get your trained model checkpoints onto whatever machine builds/tests this
Copy, from the cluster, the exact nnU-Net model-folder structure
`initialize_from_trained_model_folder` expects:
```
<local test model dir>/dataset304_binary_aneurysm/
    dataset.json
    plans.json
    fold_0/checkpoint_final.pth
    fold_1/checkpoint_final.pth
    fold_2/checkpoint_final.pth
    fold_3/checkpoint_final.pth
    fold_4/checkpoint_final.pth
<local test model dir>/dataset302_vessel/
    dataset.json
    plans.json
    fold_all/checkpoint_final.pth
```
i.e. exactly the `nnUNet_results/Dataset304_TopAneuAneurysm/RSNA2025Trainer_moreDAv6_1_SkeletonRecallTverskyBeta07__nnUNetResEncUNetMPlans__3d_fullres/`
and `nnUNet_results/Dataset302_TopAneuVessel/RSNA2025Trainer_moreDAv6_1_SkeletonRecallTverskyBeta07__nnUNetResEncUNetMPlans__3d_fullres/`
directories, renamed/copied to the two names above (`inference.py` reads
`$MODEL_ROOT/dataset304_binary_aneurysm` and `$MODEL_ROOT/dataset302_vessel`).

**These are NOT copied into the Docker image.** Grand-challenge's own
convention (confirmed by the official template's `/opt/ml/model` path) is
that model weights are uploaded as a *separate* resource and mounted into
the container at `/opt/ml/model` at runtime, decoupled from the 10GB image
size limit. For local testing, mount them the same way, e.g.:
```bash
docker run --rm -v /path/to/your/model/dir:/opt/ml/model:ro ...
```
(the official `do_test_run.sh` may already do something equivalent --
check it once you've pulled it in step 3, and adjust if it expects a
different local path convention).

## Before you trust this for a real submission -- open risks, not yet checked

1. **Never tested against real (non-oracle) vessel predictions.** Every
   pooled-accuracy number this pipeline has produced (0.4064, 0.4286, the
   bootstrap ensemble) used `--vessel_source gt`. Run
   `assign_location_rule.py --vessel_source pred --vessel_pred_dir
   $SCRATCH_ROOT/work/vessel_pred_m2_fullhead` on the cluster FIRST (see the
   earlier conversation) and see how much the real vessel channel's own
   errors degrade the rule's accuracy, before assuming this container
   reproduces anything close to 0.4286.
2. **12-minute per-case time limit, 5-fold ensemble chosen for Dataset304.**
   5-fold softmax averaging is roughly 5x a single fold's inference cost.
   Combined with Dataset302's own full sliding-window pass and the rule
   engine's own skeletonization work, this needs to be timed on a
   representative (large) case before trusting it fits in 12 minutes --
   especially since grand-challenge's T4 (16GB VRAM) is weaker than whatever
   GPU you've been developing on. If it's too slow, dropping to a single
   fold (e.g. fold 0) is the fallback, at the cost of losing the ensemble.
3. **Rule parameters (`TAU_MM`/`JUNCTION_TAU_MM`/etc. in `inference.py`) are
   copied from the oracle-vessel-tuned values (jobs 62/65), never re-tuned
   for a real, noisier vessel mask.** A noisier vessel channel may call for
   a looser `tau_mm` (more tolerance for the predicted vessel not touching
   the lesion exactly) -- worth a real check once you have a real-vessel
   evaluation number from risk #1, not assumed fine by default.
4. **`labels.json`'s "locations" list must be in the exact 1-52 order the
   challenge expects for its own 0-52 label convention.** This pipeline's
   own `loc_value` mapping (`{loc: i+1 for i, loc in enumerate(spec.locations)}`)
   assumes so -- confirm this ordering actually matches the official
   `eval/task2/README.md`'s label definitions before trusting the output
   mask's class IDs mean what the evaluator thinks they mean.
5. **Model 2 (vessel) was trained `fold_all`, no held-out CV split** -- not
   a blocker for a real test-set submission (every test case is equally
   unseen to it), but means you have no honest internal estimate of Model
   2's own real-world vessel-segmentation accuracy either, beyond whatever
   qualitative checks you've already done.

I'd resolve risk #1 before spending more time on #2-4 -- if the real-vessel
rule accuracy turns out to collapse, the container's exact timing/parameter
tuning matters much less than finding that out now rather than after using
one of your limited submission attempts.
