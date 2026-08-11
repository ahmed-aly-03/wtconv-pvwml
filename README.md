# wtconv-pvwml

Two training scripts for identifying periventricular white matter lesion
(PVWML) regions in FLAIR MRI, both built on the WTConvNeXt backbone from
[BGU-CS-VIL/WTConv](https://github.com/BGU-CS-VIL/WTConv) (vendored under
`src/wtconv*` so no separate clone is needed).

- **`train_multitask.py`** -- the joint encoder/decoder/classifier
  architecture from the block diagram: `Encoder -> BN (bottleneck) ->
  Decoder -> segmentation`, with the bottleneck also branching into
  `GP -> Linear -> classification`, and skip connections from three encoder
  stages into the decoder. Loss: `L = L_cls + lambda * L_seg`, `lambda`
  defaults to `0.3`.
- **`train_tdlf_finetune.py`** -- finetunes a standard `wtconvnext_tiny` /
  `wtconvnext_small` with the loss from **"Decoupling Representation
  Learning and Classifier for Long-Tailed Adversarial Training" (TDLF)**,
  *Pattern Recognition* 172 (2026) 112607: stage 1 finetunes the backbone
  with an adversarial supervised contrastive loss (PGD-generated views +
  adversarial weight perturbation), stage 2 freezes the backbone and trains
  a linear classifier with class-balanced sampling.

Both scripts print a `sklearn` classification report + confusion matrix at
the end and save them (plus per-epoch history and the best checkpoint) into
whatever `--output-dir` you pass on the CLI.

## Data preparation: `prepare_dataset.py`

Turns the full registered FLAIR volumes (`.../Data/{Database}_Registered/{Database}/vols/*.nii.gz`)
into the same train/val/test `ImageFolder` layout the training scripts already
expect, using `WML_Prevalence.xlsx`'s `New_Cohort` column for labels. It
does NOT touch `wmls`/`vents` (those look like real WML segmentation masks
and ventricle masks -- worth wiring into script 1's segmentation branch
later, but that's explicitly out of scope for now).

**Class grouping:**
- Non-vascular: `ADMCI, ALS, FTD, LBD, PD, SCI` -- the sheet has no `SC`
  value, `SCI` is what was almost certainly meant; check `New_Cohort` value
  counts yourself if that's wrong and edit `NON_VASCULAR_CODES` in the script.
- Vascular: `CVD`
- Control: `CN`, all ages included by default. Pass `--control-max-age N`
  to exclude CN rows above age N (and rows with missing Age, since the
  cutoff can't be verified for those).
- vMCIAD: `vMCIAD`

**Design choices, spelled out:**
- Splitting is done by **subject** (`ID` column), not by row/visit -- a
  subject's multiple visits never end up split across train/val/test.
- **All valid visits are used**, not baseline-only, since most subjects
  have multiple visits and dropping them would throw away most of the data
  (this can be changed if you actually wanted baseline-only).
- `ADNI` rows map to the `ANDI_Registered` folder (yes, that's a typo on
  disk, not in the code) -- override with `--adni-folder-name` if that gets
  fixed server-side.
- Slices: the middle 50% of axial slices per volume (`--slice-fraction`),
  matching the naming of the earlier `swml_vols_middle50_4class` dataset,
  intensity-normalized per-volume (not per-slice, to avoid brightness
  jitter between adjacent slices) and reoriented to a canonical orientation
  via `nibabel.as_closest_canonical` so slicing is consistent across the
  4 different source cohorts/scanners.

**Always run `--dry-run` first.** It does the Excel parsing, file-path
construction, existence checks, and subject-level splitting, and writes a
coverage report (`found/total` per Database x class) plus `manifest.csv` --
all without the slow part (extracting ~7,000 volumes). If `--adni-folder-name`
or `--data-root` is wrong, this is where you'll see it (near-zero coverage
for that Database), before wasting an hour on the full run.

```bash
python prepare_dataset.py \
  --excel-path /path/to/WML_Prevalance.xlsx \
  --data-root "/home/sharedFolder/Spatial WML/Data" \
  --output-dir /home/ra/aaly/WTConv/wtconvnext/swml_vols_full_4class \
  --dry-run
```

Check the printed coverage table and `swml_vols_full_4class/manifest.csv`.
Once it looks right, rerun the exact same command without `--dry-run`:

```bash
python prepare_dataset.py \
  --excel-path /path/to/WML_Prevalance.xlsx \
  --data-root "/home/sharedFolder/Spatial WML/Data" \
  --output-dir /home/ra/aaly/WTConv/wtconvnext/swml_vols_full_4class \
  --num-workers 16
```

**Never commit the Excel file or anything this script produces** (subject
IDs, ages, diagnoses) -- `.gitignore` already blocks `*.xlsx`, `manifest.csv`,
and `swml_vols*/`, but keep it that way; this repo is public.

## Classification loss: Asymmetric Loss (ASL) is now the default

A full TDLF run (SCL stage 1 + CE stage 2, class-balanced sampling) landed at
44.8% val accuracy with `vMCIAD` precision 0.12 -- recall was fine (0.59) but
precision collapsed, i.e. the model was flooding `vMCIAD` with false
positives from the majority classes. That's the textbook failure mode plain
cross-entropy has on severe imbalance: easy-negative gradients (a
confidently-not-vMCIAD `Non-vascular` scan) dominate training regardless of
class weighting, because weighting only rescales the loss, it doesn't change
which examples' gradients dominate.

Both scripts now default their classification loss (`L_cls` in script 1,
the stage-2 classifier loss in script 2) to **Asymmetric Loss** (Ridnik et
al., *"Asymmetric Loss for Multi-Label Classification"*, ICCV 2021),
attached for this exact reason. ASL treats each class as an independent
one-vs-rest binary decision and applies two mechanisms only to the negative
side of each: asymmetric focusing (`gamma_neg=4` vs `gamma_pos=0` --
down-weight easy negatives far more than positives) and a hard probability
margin (`clip=0.05` -- fully zero the gradient from very-easy negatives).
The paper explicitly validates this for single-label classification too,
not just multi-label. See `src/losses.py::AsymmetricLoss` for the full
rationale and the exact formula. Old behavior is still available via
`--cls-loss ce` (script 1) / `--classifier-loss ce` or `balanced_softmax`
(script 2).

ASL is not designed to be combined with static class weights or resampling
(the paper found weighting and focusing interact badly) -- script 2 now
exposes `--no-class-balanced-sampling` so you can try ASL on its own if ASL
+ sampling together turns out to over-correct.

## Full volumes now available (deferred)

You mentioned full NIfTI volumes are now available (not just the
middle-50-slice crops used so far), which should give the model more to
learn from. That dataset needs sorting into classes and train/val/test
splits before either script can use it -- intentionally not done yet, per
your call to worry about it later. Nothing in this repo depends on it
today.

## About the segmentation branch -- read this first

**There are no PVWML segmentation masks in the dataset** (`swml_vols_middle50_4class`
is a plain classification `ImageFolder` layout: `train/val/test` ->
`Control/Non-vascular/Vascular/vMCIAD`). `train_multitask.py` therefore:

- Builds the full encoder-decoder-classifier architecture from the diagram.
- Trains **classification-only** by default (`L = L_cls`, the segmentation
  head gets no gradient) -- it's fully usable today.
- Switches on real segmentation supervision (`L = L_cls + lambda * L_seg`,
  BCE + Dice) the moment you pass `--mask-dir` pointing at real per-image
  masks mirroring the data directory structure.

`generate_pseudo_masks.py` is included as a stopgap: it derives **heuristic,
unsupervised pseudo-masks** from the FLAIR slices themselves (periventricular
band via ventricle/CSF detection, intersected with hyperintense pixels).
**This is not clinical ground truth** -- it's a weak-label heuristic so the
segmentation branch has *something* to train on. Read the docstring in
`src/pseudo_masks.py` for exactly what it assumes and inspect the
`_previews/` folder it writes before trusting it for a real run. Prefer real
radiologist annotations the moment any exist.

## Repo layout

```
src/
  wtconv/, wtconvnext/     vendored from BGU-CS-VIL/WTConv (unmodified)
  datasets.py              dataloaders for both scripts (paired image+mask
                            transforms, class-balanced sampler, two-view
                            dataset for contrastive learning)
  pseudo_masks.py           heuristic PVWML pseudo-mask extraction
  multitask_model.py         script 1 model (encoder/BN/decoder/classifier)
  losses.py                 shared classification loss (AsymmetricLoss) +
                            script 1 losses (Dice+BCE seg, combined loss)
  tdlf_model.py              script 2 model wrappers (normalized backbone,
                            projection head, frozen-backbone classifier)
  tdlf_losses.py             script 2 losses (PGD view construction, AWP,
                            supervised contrastive loss)
train_multitask.py          script 1 entrypoint
train_tdlf_finetune.py       script 2 entrypoint
generate_pseudo_masks.py     optional pseudo-mask generator CLI
```

## Server setup

```bash
git clone https://github.com/ahmed-aly-03/wtconv-pvwml.git
cd wtconv-pvwml
conda create -n wtconv-pvwml python=3.12 -y
conda activate wtconv-pvwml

# Install torch matching your CUDA version FIRST (check with `nvidia-smi` /
# `nvcc --version`), e.g. for CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

### Pretrained WTConvNeXt weights (recommended for both scripts)

The WTConv authors release ImageNet-pretrained checkpoints:

| name | acc@1 | link |
|---|---|---|
| WTConvNeXt-T | 82.5 | https://drive.google.com/file/d/1wMgUUJBAs4Fz2dZoNS7QCk9kMB8MPMtC/view |
| WTConvNeXt-S | 83.6 | https://drive.google.com/file/d/1F5yo1nSbCvUH8lQXTM1pdK4T_W_2PmFQ/view |
| WTConvNeXt-B | 84.1 | https://drive.google.com/file/d/1snpt4L38NB8vIhKRcelylj0guGd0Q7q7/view |

Download the one matching `--model-name` onto the server and pass its path
via `--pretrained-encoder-path` (script 1) or `--pretrained-path` (script 2).
Both scripts work without it (random init), but starting from ImageNet
weights will train much faster and should generalize better given how small
this dataset is.

## Running script 1 (multitask encoder/decoder/classifier)

Classification-only (no masks) -- what you can run today:

```bash
python train_multitask.py \
  --data-dir /home/ra/aaly/WTConv/wtconvnext/swml_vols_middle50_4class \
  --model-name wtconvnext_tiny \
  --pretrained-encoder-path /path/to/WTConvNeXt_tiny_5_300e_ema.pth \
  --num-classes 4 \
  --batch-size 16 \
  --epochs 40 \
  --output-dir ./outputs_multitask
```

With segmentation supervision (real masks, or the heuristic pseudo-masks
below), add `--mask-dir` and `--seg-loss-weight` (defaults to `0.3`):

```bash
python train_multitask.py \
  --data-dir /home/ra/aaly/WTConv/wtconvnext/swml_vols_middle50_4class \
  --mask-dir /home/ra/aaly/WTConv/wtconvnext/swml_vols_middle50_4class_pseudomasks \
  --model-name wtconvnext_tiny \
  --pretrained-encoder-path /path/to/WTConvNeXt_tiny_5_300e_ema.pth \
  --num-classes 4 \
  --seg-loss-weight 0.3 \
  --output-dir ./outputs_multitask
```

To generate the heuristic pseudo-masks first:

```bash
python generate_pseudo_masks.py \
  --data-dir /home/ra/aaly/WTConv/wtconvnext/swml_vols_middle50_4class \
  --output-dir /home/ra/aaly/WTConv/wtconvnext/swml_vols_middle50_4class_pseudomasks \
  --preview-n 30
# then scp/rsync the _previews/ folder locally and actually look at it
# before trusting these for training.
```

Outputs land in `--output-dir`: `<model>_multitask.pth` (best checkpoint),
`..._epoch_history.csv`, and `..._validation_metrics.txt` (classification
report + confusion matrix).

`L_cls` defaults to Asymmetric Loss (see above); add `--cls-loss ce` to
compare against plain class-weighted cross-entropy, or tune
`--asl-gamma-neg` / `--asl-gamma-pos` / `--asl-clip`.

## Running script 2 (TDLF finetuning)

```bash
python train_tdlf_finetune.py \
  --data-dir /home/ra/aaly/WTConv/wtconvnext/swml_vols_middle50_4class \
  --model-name wtconvnext_tiny \
  --pretrained-path /path/to/WTConvNeXt_tiny_5_300e_ema.pth \
  --num-classes 4 \
  --batch-size 32 \
  --stage1-epochs 30 \
  --stage2-epochs 25 \
  --output-dir ./outputs_tdlf
```

Cost knobs -- stage 1 does a PGD inner loop (`--pgd-steps`, default 5 vs. the
paper's 10) plus an extra forward/backward for AWP (`--use-awp`/`--no-awp`)
on **every batch**, so it is meaningfully slower than plain finetuning.
On a single modern GPU with `wtconvnext_tiny` at `batch-size 32`, expect
roughly 3-6x a normal finetuning epoch. To cut cost: `--pgd-steps 3`,
`--no-awp`, or fewer `--stage1-epochs` (it's initializing from ImageNet
weights already, not training from scratch, so it needs far fewer epochs
than the paper's 600).

Outputs in `--output-dir`: `<model>_tdlf_stage1_backbone.pth` +
`..._stage1_history.csv` from stage 1; `<model>_tdlf_stage2_classifier.pth`,
`..._epoch_history.csv`, and `..._validation_metrics.txt` (classification
report + confusion matrix) from stage 2.

Stage 2's classifier loss defaults to Asymmetric Loss now (see above). To
reproduce the earlier CE run for comparison: `--classifier-loss ce`. To try
ASL without class-balanced sampling stacked on top: `--no-class-balanced-sampling`.

If you already have a finetuned backbone and only want to redo the
classifier stage, use `--skip-stage1 --pretrained-path /path/to/stage1_backbone.pth`.

## Running in the background on the server

```bash
nohup python train_multitask.py --data-dir ... --output-dir ./outputs_multitask \
  > outputs_multitask.log 2>&1 &
tail -f outputs_multitask.log
```

## Notes / deliberate scope choices

- `--num-classes 4` and the class folder names match the
  `Control / Non-vascular / Vascular / vMCIAD` layout shown in the dataset
  screenshot. Both scripts compute inverse-frequency class weights from
  `train/` automatically (`src/datasets.py::compute_class_weights`) instead
  of hardcoding them, so they stay correct regardless of exact split sizes.
- Segmentation loss (`src/losses.py::SegmentationLoss`) is BCE-with-logits +
  soft Dice, the standard combination for imbalanced pixel-wise medical
  segmentation, and is skipped gracefully (0 gradient contribution) for any
  batch with no masked samples in it.
- `lambda` (`--seg-loss-weight`) is a plain CLI float defaulting to `0.3`,
  per the diagram -- not a learned parameter.
- Adversarial perturbation and AWP in script 2 operate directly on raw
  `[0,1]` pixel tensors (normalization happens inside `TDLFBackbone`), so
  `--pgd-epsilon 8/255` stays meaningful in the same units the paper uses.
- ASL reference: Ridnik, Ben-Baruch, Zamir, Noy, Friedman, Protter &
  Zelnik-Manor, *"Asymmetric Loss for Multi-Label Classification"*, ICCV
  2021. Official implementation: https://github.com/Alibaba-MIIL/ASL
  (`src/losses.py::AsymmetricLoss` reimplements Eq. 7 from the paper).
