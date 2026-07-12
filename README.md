# ViK-FPN: A Vision Kolmogorov–Arnold Siamese Network with Multi-Patch RBF Mixing and Explicit Change Interaction

Official implementation of **ViK-FPN**, a lightweight Siamese encoder–decoder
architecture for remote sensing change detection (RSCD) that combines a
Vision Kolmogorov–Arnold Network (ViK) encoder with RBF-KAN spatial mixing,
an explicit multi-stage **Change Interaction Module (CIM)**, and a
progressive **Feature Pyramid Network (FPN)** decoder.

This repository provides everything needed to reproduce the results reported
in the paper on the **WHU-CD** and **CLCD** benchmarks: training code,
evaluation code, model configuration files, preprocessing scripts, dataset
split generation, fixed random seeds, metric computation, loss functions,
optimizer/scheduler configuration, checkpoint utilities, and step-by-step
reproduction instructions.

---

## Table of Contents

1. [Repository structure](#repository-structure)
2. [Installation](#installation)
3. [Dataset preparation](#dataset-preparation)
4. [Model configuration](#model-configuration)
5. [Training](#training)
6. [Evaluation](#evaluation)
7. [Inference on new image pairs](#inference-on-new-image-pairs)
8. [Ablation study](#ablation-study)
9. [Stability analysis (Mean ± Std)](#stability-analysis-mean--std)
10. [One-command reproduction](#one-command-reproduction)
11. [Pretrained weights](#pretrained-weights)
12. [Expected results](#expected-results)
13. [Citation](#citation)
14. [License](#license)

---

## Repository structure

```
ViK-FPN/
├── model_ViK_FPN_CD.py          # ViK-FPN architecture (Sec. 3, Eq. 1-21)
├── losses.py                     # Hybrid BCE+Dice loss (Eq. 22, Table 3 class weights)
├── metric.py                     # Evaluator: Precision/Recall/F1/IoU/OA (Eq. 23-27)
├── seed_utils.py                 # Centralized seed_everything() (Table 3: seed=42)
├── checkpoint_utils.py           # save/load/find-best checkpoint helpers
├── config_utils.py               # YAML config loader
├── utils.py                      # Visualization helpers (save_result, overlay_change)
├── train_WHU_CD.py               # Training entry point — WHU-CD
├── train_CLCD.py                 # Training entry point — CLCD
├── test_WHU_CD.py                # Batch evaluation entry point — WHU-CD
├── test_CLCD.py                  # Batch evaluation entry point — CLCD
├── inference.py                  # Standalone single-pair / folder inference
├── configs/
│   ├── whu_cd.yaml               # Table 3 hyperparameters, WHU-CD
│   └── clcd.yaml                 # Table 3 hyperparameters, CLCD
├── scripts/
│   ├── preprocess_whu_cd.py      # Tile raw WHU-CD scenes into 256x256 patches
│   ├── preprocess_clcd.py        # Split raw CLCD pairs into train/val/test
│   ├── make_splits.py            # Generate versionable split manifests (.txt)
│   ├── run_ablation.py           # Train+eval all 4 Table 9/10 ablation configs
│   ├── run_stability_analysis.py # Multi-seed Mean±Std analysis (Table 8)
│   ├── download_pretrained_weights.py
│   ├── profile_complexity.py     # Reproduce Table 7 (params/FLOPs/time/memory)
│   ├── reproduce_whu_cd.sh       # One-command train+eval, WHU-CD
│   ├── reproduce_clcd.sh         # One-command train+eval, CLCD
│   └── reproduce_all.sh          # Both datasets, one command
├── weights/
│   └── README.md                 # How to obtain/place trained checkpoints
├── requirements.txt
├── environment.yml
├── LICENSE
└── README.md                     # This file
```

`dataset/`, `lightning_logs/`, and `outputs/` are created locally when you
run the scripts above; they are intentionally not part of the repository
(see `.gitignore`) since they contain either third-party dataset imagery or
large binary training artifacts.

---

## Installation

**Option A — pip:**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Option B — conda:**

```bash
conda env create -f environment.yml
conda activate vikfpn
```

Tested with Python 3.10 and PyTorch 2.x. A CUDA-capable GPU is recommended
for training (the paper uses a single NVIDIA Tesla V100S); all scripts also
run on CPU (evaluation/inference will simply be slower — see
[Reproducibility notes](#reproducibility-notes)).

---

## Dataset preparation

The paper uses two benchmarks:

| Dataset | Native size | Resolution | Train / Val / Test pairs |
|---|---|---|---|
| WHU-CD [13] | 32,507 × 15,354 (single pair of scenes) | 0.2 m | 6095 / 760 / 760 (256×256 crops) |
| CLCD [12] | 512 × 512 | 0.5–2 m | 360 / 120 / 120 |

Both datasets must end up organized as:

```
dataset/<NAME>/{train,val,test}/{A,B,label}/<filename>
```

where `A/` = T1 images, `B/` = T2 images, `label/` = binary change masks
(0 = no change, 255 = change), matching filenames across the three
subfolders - this is exactly what `CDataset` in `train_WHU_CD.py` /
`train_CLCD.py` expects.

### WHU-CD

Download the original WHU building change detection dataset [13], then tile
it into the 256x256 patches used in the paper:

```bash
python scripts/preprocess_whu_cd.py \
    --raw_root /path/to/WHU-CD-raw \
    --output_dir dataset/WHU_CD \
    --patch_size 256 \
    --seed 42
```

This deterministically reproduces the paper's reported split sizes (6095 /
760 / 760) when the raw scene yields at least that many non-overlapping
patches; see the script's `--help` for the expected raw file layout and how
to override it.

### CLCD

If your copy of CLCD is a flat pool of 600 pairs (not yet split):

```bash
python scripts/preprocess_clcd.py \
    --raw_root /path/to/CLCD-raw \
    --output_dir dataset/CLCD \
    --seed 42
```

If your copy is already split into train/val/test by the original authors,
skip this step and point `--train_root` / `--val_root` / `--test_root`
directly at it.

### Recording the exact split used

Either preprocessing script above places files deterministically given
`--seed 42`. To additionally produce small, git-committable manifest files
recording exactly which filenames ended up in which split (useful for a
reviewer to verify without needing the dataset itself):

```bash
python scripts/make_splits.py --root dataset/WHU_CD --out splits/WHU_CD
python scripts/make_splits.py --root dataset/CLCD   --out splits/CLCD
```

---

## Model configuration

All hyperparameters from Table 3 of the paper are recorded machine-readably
in `configs/whu_cd.yaml` and `configs/clcd.yaml`. These are optional: every
training script also works with its own CLI defaults (which already match
Table 3), so `--config` is a convenience for keeping the recorded
configuration and the code that consumes it in one place, not a required
step.

Key settings (identical for both datasets, per Table 3):

| Setting | Value |
|---|---|
| Optimizer | Adam (β₁=0.9, β₂=0.999, weight_decay=1e-4) |
| Learning rate | 1×10⁻⁴ |
| LR scheduler | Polynomial decay, power=0.9 |
| Batch size | 8 |
| Epochs | 100 |
| Loss | BCE + Dice (Eq. 22), class weights [0.2, 0.8] |
| Input size | 256×256 |
| Augmentation | Random horizontal/vertical flip, random 90° rotation |
| Seed | 42 |

---

## Training

```bash
# WHU-CD
python train_WHU_CD.py --config configs/whu_cd.yaml

# CLCD
python train_CLCD.py --config configs/clcd.yaml
```

Or without a config file, using the (identical) CLI defaults directly:

```bash
python train_WHU_CD.py \
    --train_root dataset/WHU_CD/train \
    --val_root dataset/WHU_CD/val \
    --epochs 100 --batch_size 8 --seed 42
```

Checkpoints are written to `lightning_logs/<DATASET>/version_*/checkpoints/`,
named `unetkan_cd-{epoch:02d}-{val_mIoU:.4f}.ckpt`, with the checkpoint
achieving the highest validation mIoU kept (Table 3: "Checkpoint Selection:
Highest validation IoU"), plus a `last.ckpt`.

Training logs (loss, mIoU, F1, OA per epoch) are written as CSV via
`pytorch_lightning.loggers.CSVLogger` to the same `lightning_logs/` tree.

---

## Evaluation

```bash
# WHU-CD — automatically picks the checkpoint with the highest val_mIoU
python test_WHU_CD.py -o outputs/WHU_CD \
    --weights_path lightning_logs/WHU_CD/version_0/checkpoints/ \
    --auto_best

# CLCD
python test_CLCD.py -o outputs/CLCD \
    --weights_path lightning_logs/CLCD/version_0/checkpoints/ \
    --auto_best
```

This prints per-class Precision/Recall/F1/IoU, the mean-over-classes
(mIoU/mF1) numbers, **and** an explicit "change-class metrics" block — the
paper's Tables 4–5 report change-class-only values (Sec. 4.3), not the mean
over both classes, so that block is what you should compare against the
paper. Predicted masks are saved as images under `-o`; visual
T1/T2/GT/prediction comparison panels (via `utils.save_result`) are written
to `outputsWHU_CD/` / `outputsCLCD/`.

---

## Inference on new image pairs

For a single T1/T2 pair, without needing a dataset folder or DataLoader:

```bash
python inference.py \
    --img1 examples/T1.png --img2 examples/T2.png \
    --checkpoint lightning_logs/WHU_CD/version_0/checkpoints/best.ckpt \
    --output prediction.png
```

Or for a whole folder of matching-filename pairs:

```bash
python inference.py \
    --img1_dir dataset/WHU_CD/test/A --img2_dir dataset/WHU_CD/test/B \
    --checkpoint lightning_logs/WHU_CD/version_0/checkpoints/best.ckpt \
    --output_dir predictions/
```

---

---

## Ablation study

Sec. 4.6 / Tables 9-10 report a controlled ablation that starts from a
convolutional baseline and reintroduces one ViK-FPN component at a time
(RBF-KAN mixer, triple-stream Change Interaction Module, progressive FPN
decoder), reporting change-class IoU (Eq. 27) for each configuration.

`model_ViK_FPN_CD.py`'s `ViK_CD` exposes this directly as three constructor
flags — `use_rbf_kan`, `use_cim`, `use_fpn_decoder` — all defaulting to
`True` (the full model used everywhere else in this repository). Passing
any of them `False` swaps in the "standard counterpart" described in Sec.
4.6 for that component only, with the rest of the architecture unchanged:

```python
from model_ViK_FPN_CD import ViK_CD

baseline      = ViK_CD(use_rbf_kan=False, use_cim=False, use_fpn_decoder=False)
plus_rbf_kan  = ViK_CD(use_rbf_kan=True,  use_cim=False, use_fpn_decoder=False)
plus_cim      = ViK_CD(use_rbf_kan=True,  use_cim=True,  use_fpn_decoder=False)
full_vikfpn   = ViK_CD()  # use_rbf_kan=True, use_cim=True, use_fpn_decoder=True
```

To train and evaluate all four configurations end to end and print a table
comparable to Tables 9/10:

```bash
python scripts/run_ablation.py --dataset whu_cd \
    --train_root dataset/WHU_CD/train \
    --val_root dataset/WHU_CD/val \
    --test_root dataset/WHU_CD/test

python scripts/run_ablation.py --dataset clcd \
    --train_root dataset/CLCD/train \
    --val_root dataset/CLCD/val \
    --test_root dataset/CLCD/test
```

`scripts/run_ablation.py` reuses the same optimizer, scheduler, loss, and
augmentation configuration as `train_WHU_CD.py` / `train_CLCD.py` (Table 3)
for every configuration, so that — per Sec. 4.6 — "performance differences
are attributable solely to the component under study."

---

---

## Stability analysis (Mean ± Std)

Table 8 (Sec. 4.6) reports Precision, Recall, F1, and IoU as **Mean ± Std
over three independent runs with different random seeds**, on WHU-CD:

| Metric | Mean ± Std |
|---|---|
| Precision | 91.48 ± 0.21 |
| Recall | 94.42 ± 0.18 |
| F1 | 92.93 ± 0.14 |
| IoU | 86.79 ± 0.19 |

`scripts/run_stability_analysis.py` reproduces this protocol directly: it
trains the full ViK-FPN model once per seed (three seeds by default: 42,
43, 44 — 42 matches Table 3's primary reported seed), evaluates each run's
checkpoint on the test split, and aggregates the four change-class metrics
into Mean ± Std, printed in exactly this format.

```bash
python scripts/run_stability_analysis.py --dataset whu_cd \
    --train_root dataset/WHU_CD/train \
    --val_root dataset/WHU_CD/val \
    --test_root dataset/WHU_CD/test
```

By default this reports exactly Table 8's four metrics
(`precision recall f1 iou`). Two additional metrics are supported on
request — `oa` (Eq. 26) and `dice` (a standard confusion-matrix Dice score,
mathematically equivalent to F1 for a binary positive class) — but **neither
is part of Table 8**; the script's own output table marks each metric with
whether it appears in Table 8, so nothing is presented as a paper-reported
number that the paper does not actually report:

```bash
python scripts/run_stability_analysis.py --dataset whu_cd \
    --train_root dataset/WHU_CD/train --val_root dataset/WHU_CD/val --test_root dataset/WHU_CD/test \
    --metrics precision recall f1 iou oa dice
```

The same script also works on CLCD (`--dataset clcd`), though Table 8 itself
only reports this analysis for WHU-CD — running it on CLCD produces a
legitimate, real Mean ± Std result, it is just not one that can be checked
against a number printed in the paper.

Each of the three (or more) runs trains a full model from scratch using the
exact Table 3 configuration (same optimizer, scheduler, loss, augmentation,
batch size, and epoch count as `train_WHU_CD.py`), so — per Sec. 4.6 — the
only thing varying between runs is the random seed.

---

## One-command reproduction

Once a dataset is prepared (see [Dataset preparation](#dataset-preparation)):

```bash
bash scripts/reproduce_whu_cd.sh dataset/WHU_CD
bash scripts/reproduce_clcd.sh   dataset/CLCD
# or both:
bash scripts/reproduce_all.sh dataset/WHU_CD dataset/CLCD
```

Each script trains with the exact Table 3 configuration, evaluates the best
checkpoint on the corresponding test split, and prints the change-class
metrics to compare directly against Tables 4 and 5.

---

To reproduce the parameter/FLOPs/inference-time/GPU-memory figures in
Table 7:

```bash
python scripts/profile_complexity.py --sizes 256 512 1024
```

See [Known open discrepancy](#known-open-discrepancy) for how the measured
values compare to what Table 7 reports.

---

## Pretrained weights

See [`weights/README.md`](weights/README.md). In short:
`scripts/download_pretrained_weights.py` is provided as the download
mechanism, but **no checkpoint has been published to a GitHub Release as of
this revision** — the script says so explicitly rather than pointing at a
placeholder or broken URL. Train from scratch (above) to produce your own,
or check the repository's Releases page for updates.

---

## results

Change-class Precision / Recall / F1 / IoU / OA (Sec. 4.3), as reported in
Tables 4 and 5 of the paper:

| Dataset | Rec. | Pre. | IoU | F1 | OA | Params (M) | FLOPs (G) |
|---|---|---|---|---|---|---|---|
| WHU-CD | 94.42 | 91.48 | 86.79 | 92.93 | 98.65 | 10.31 | 17.70 |
| CLCD | 83.62 | 82.50 | 71.03 | 83.06 | 94.48 | 10.31 | 17.70 |

See [Known open discrepancy](#known-open-discrepancy) regarding the
Params/FLOPs column before assuming an exact match on those two specific
numbers.

---


## Citation

```bibtex
@article{safwat2026vikfpn,
  title   = {ViK-FPN: A Vision Kolmogorov--Arnold Siamese Network with
             Multi-Patch RBF Mixing and Explicit Change Interaction},
  author  = {Safwat, Ledya and Abd Elaziz, Mohamed},
  year    = {2026}
}
```

Please also cite the WHU-CD [13] and CLCD [12] dataset papers if you use
this code with those benchmarks.

---

## License

MIT — see [`LICENSE`](LICENSE).
