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
  losses.py                 script 1 losses (Dice+BCE seg, combined loss)
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
python3 -m venv .venv
source .venv/bin/activate

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
