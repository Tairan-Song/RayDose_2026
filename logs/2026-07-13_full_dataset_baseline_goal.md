# Full-Dataset Baseline Goal: 2026-07-13

## Objective

Complete the official full-dataset baseline for DoseRAD2026 Photon Task 1 using
a geometry-conditioned 3D U-Net.

The purpose of this run is to create the first reproducible benchmark that can
be compared against later architecture changes, preprocessing changes, mask
definitions, and energy-conditioning ablations. It must therefore use the full
training split and full validation split, not a subset or smoke-test sample.

This benchmark must use the fixed case-level split and fixed random seed:

```text
Split file: splits/photon_case_split.csv
Random seed: 20260628
Split ratio: 80:20
Train cases: 60
Validation cases: 15
```

The split must remain case-level. Individual dose/control-point files must not
be randomly split across train and validation, because that would allow one
patient/case to appear in both sets.

## Dataset Scale

```text
Cases total: 75
Beams per case: 3
Control points per beam: 180
Dose samples per case: 540
All dose samples: 40,500

Train samples: 60 cases x 540 = 32,400
Validation samples: 15 cases x 540 = 8,100
```

## Model

```text
Architecture: geometry-conditioned 3D U-Net
Image input: CT volume
Condition input:
  - beam index
  - control-point index
  - gantry angle
  - isocenter
  - MLC left/right leaf positions
Optional inputs:
  - HU-to-density CT representation
  - photon energy spectrum
Output:
  - Dose_Bx_CPy.mha
```

## Hardware And Parallel Execution Plan

Local hardware detected on 2026-07-13:

```text
GPU: NVIDIA GeForce GTX 1070
CUDA available: yes
CPU cores: 8
```

The full run should use GPU for model training/inference and multiple CPU
workers for data loading and preprocessing. The expected division of work is:

```text
GPU:
  - 3D U-Net forward pass
  - backpropagation
  - checkpoint evaluation inference
  - validation prediction export inference

CPU workers:
  - reading CT, dose, and mask MHA files
  - JSON parameter parsing
  - CT normalization or HU-to-density preprocessing
  - dose resampling/cropping
  - batch construction
  - exported prediction metric calculation
```

This run should not be configured as a single-threaded data pipeline unless
debugging a failure. Full-dataset training is expected to be I/O-heavy, so CPU
workers are part of the formal baseline setup.

Recommended initial worker setting:

```text
num_workers: 4
```

If data loading is stable and GPU utilization is low, increase to:

```text
num_workers: 6
```

## Formal Baseline Command

This command intentionally sets all sample limits to `0`. In the current dataset
implementation, `0` means no limit, i.e. use the complete split.

```powershell
F:\anaconda3\envs\myenvs3_9\python.exe scripts\python\run_baseline_experiment.py `
  --training-dir data\photon\training `
  --split-csv splits\photon_case_split.csv `
  --output-dir outputs\full_baseline_hu_no_energy_seed20260628 `
  --target-shape "64 64 64" `
  --ct-mode hu `
  --dose-mode global `
  --global-dose-scale 0.00015 `
  --max-train-samples 0 `
  --max-val-samples 0 `
  --eval-samples 0 `
  --export-samples 0 `
  --export-eval-samples 0 `
  --sample-strategy uniform `
  --sample-seed 20260628 `
  --seed 20260628 `
  --batch-size 1 `
  --epochs 10 `
  --steps-per-epoch 0 `
  --base-channels 8 `
  --lr 0.001 `
  --weight-decay 0.0001 `
  --mask-weight 1.0 `
  --num-workers 4 `
  --filename-style dose `
  --full-mode crop_insert `
  --print-every 200
```

## Checkpoints And Expected Outputs

### Checkpoint 0: Pre-run Validation

Expected duration:

```text
5-15 minutes
```

Purpose:

- verify split counts
- verify data paths
- verify MHA reading
- verify Python environment
- verify CUDA availability

Required evidence:

```text
cases = 75
train cases = 60
val cases = 15
train samples = 32,400
val samples = 8,100
CUDA available = True
```

### Checkpoint 1: Training Started

Expected duration to reach:

```text
within 10-30 minutes after launch
```

Required evidence:

```text
outputs/full_baseline_hu_no_energy_seed20260628/train/
outputs/full_baseline_hu_no_energy_seed20260628/train/metrics.csv
```

### Checkpoint 2: First Epoch Complete

Estimated duration:

```text
4-8 hours
```

Reasoning:

- full train epoch = 32,400 samples
- GTX 1070 is available but relatively limited for 3D volumes
- MHA loading and preprocessing may become a bottleneck

Required evidence:

```text
metrics.csv contains epoch 1
train_loss is finite
val_loss is finite
checkpoint last.pt exists
```

### Checkpoint 3: Training Complete

Estimated duration for 10 epochs:

```text
40-80 hours
```

Required evidence:

```text
outputs/full_baseline_hu_no_energy_seed20260628/train/checkpoints/best.pt
outputs/full_baseline_hu_no_energy_seed20260628/train/checkpoints/last.pt
outputs/full_baseline_hu_no_energy_seed20260628/train/metrics.csv
```

Expected `metrics.csv` rows:

```text
10 rows, one per epoch
```

### Checkpoint 4: Full Validation Checkpoint Evaluation Complete

Estimated duration:

```text
0.5-2 hours
```

Required evidence:

```text
outputs/full_baseline_hu_no_energy_seed20260628/evaluate/per_sample_metrics.csv
outputs/full_baseline_hu_no_energy_seed20260628/evaluate/summary_metrics.csv
```

Expected evaluated samples:

```text
8,100 validation samples
```

### Checkpoint 5: Full Validation Prediction Export Complete

Estimated duration:

```text
1-3 hours
```

Required evidence:

```text
outputs/full_baseline_hu_no_energy_seed20260628/dose_predictions/prediction_manifest.csv
```

Expected prediction files:

```text
8,100 Dose_Bx_CPy.mha files
```

### Checkpoint 6: Full Exported Prediction Evaluation Complete

Estimated duration:

```text
6-12 hours
```

Reasoning:

- full-volume exported evaluation includes gamma pass-rate calculations
- gamma evaluation is substantially slower than simple MAE/RMSE

Required evidence:

```text
outputs/full_baseline_hu_no_energy_seed20260628/evaluate_exported/exported_prediction_metrics.csv
outputs/full_baseline_hu_no_energy_seed20260628/evaluate_exported/exported_prediction_summary.csv
```

Expected evaluated samples:

```text
8,100 validation predictions
```

## Overall Estimated Runtime

Conservative estimate on the detected local machine:

```text
Total expected time: 2-4 days
```

Breakdown:

```text
Pre-run validation: 5-15 minutes
Training: 40-80 hours
Checkpoint evaluation: 0.5-2 hours
Prediction export: 1-3 hours
Exported full-volume evaluation with gamma: 6-12 hours
```

## Progress Check Command

Use this command while the run is in progress:

```powershell
F:\anaconda3\envs\myenvs3_9\python.exe scripts\python\check_full_baseline_progress.py `
  --output-dir outputs\full_baseline_hu_no_energy_seed20260628 `
  --expected-epochs 10
```

The checker reports:

```text
manifest status
train/validation sample counts
completed epochs
last epoch duration
estimated training time remaining
checkpoint completion status
prediction export count
exported evaluation count
```

The expected full-dataset values are:

```text
train_samples_recorded = 32400
val_samples_recorded = 8100
```

## Formal Completion Criteria

This full-dataset baseline is complete only when all of the following exist:

```text
outputs/full_baseline_hu_no_energy_seed20260628/train/checkpoints/best.pt
outputs/full_baseline_hu_no_energy_seed20260628/train/checkpoints/last.pt
outputs/full_baseline_hu_no_energy_seed20260628/train/metrics.csv
outputs/full_baseline_hu_no_energy_seed20260628/run_manifest.json
outputs/full_baseline_hu_no_energy_seed20260628/evaluate/per_sample_metrics.csv
outputs/full_baseline_hu_no_energy_seed20260628/evaluate/summary_metrics.csv
outputs/full_baseline_hu_no_energy_seed20260628/dose_predictions/prediction_manifest.csv
outputs/full_baseline_hu_no_energy_seed20260628/evaluate_exported/exported_prediction_metrics.csv
outputs/full_baseline_hu_no_energy_seed20260628/evaluate_exported/exported_prediction_summary.csv
```

And the following counts are verified:

```text
train samples used: 32,400
validation samples evaluated: 8,100
validation predictions exported: 8,100
exported predictions evaluated: 8,100
```

The `run_manifest.json` file must record:

```text
hardware settings
Python/PyTorch/CUDA metadata
fixed split file
train/validation sample counts
all stage commands
stage wall-clock times
total wall-clock time
output directory
```

## Public Push Policy

Do not push full baseline performance claims until the full validation set has
been evaluated.

Allowed before completion:

- code changes
- run plans
- reproducibility notes

Not allowed before completion:

- subset-only result summaries presented as baseline results
- smoke-test metrics presented as benchmark results
- incomplete validation metrics presented as final results
