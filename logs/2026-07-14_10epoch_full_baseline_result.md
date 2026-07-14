# DoseRAD2026 Photon Task 1: 10-Epoch Full-Dataset Baseline Result

Date: 2026-07-14

## Status

This is the completed 10-epoch full-dataset geometry-conditioned 3D U-Net
baseline for DoseRAD2026 Photon Task 1.

Formal completion audit:

```text
checks: 15
failed: 0
status: PASS
```

## Dataset Split

```text
Split file: splits/photon_case_split.csv
Random seed: 20260628
Split ratio: 80:20 by case
Training cases: 60
Validation cases: 15
Training dose samples: 32,400
Validation dose samples: 8,100
```

The split is case-level. A patient/case does not appear in both training and
validation.

## Model And Inputs

```text
Architecture: geometry-conditioned 3D U-Net
Target shape: 64 x 64 x 64
Input image: CT crop, HU mode
Condition vector:
  - beam index
  - control-point index
  - gantry angle
  - isocenter
  - MLC left leaf positions
  - MLC right leaf positions
Energy spectrum input: not included
Dose target scaling: global, scale = 0.00015
Loss: global L1 + masked L1 using dose_gt_1pct
```

## Hardware And Runtime

```text
GPU: NVIDIA GeForce GTX 1070
CUDA GPUs detected: 1
CPU count: 8
DataLoader workers: 4
Prefetch factor: 2
CT cache size: 4
Case-grouped sampling: enabled
cuDNN benchmark: enabled
```

Stage runtimes:

```text
Training: 24,316.385 s
Checkpoint evaluation: 390.194 s
Prediction export: 2,672.059 s
Exported prediction evaluation: 6,854.612 s
Total: 34,234.842 s
```

Total wall time was approximately 9 h 30 min.

## Training Result

Best validation epoch:

```text
epoch: 8
train_loss: 0.0429758914
val_loss: 0.0445761755
val_global_l1: 0.0042814882
val_masked_l1: 0.0402947441
```

Final epoch:

```text
epoch: 10
train_loss: 0.0421478599
val_loss: 0.0458424278
val_global_l1: 0.0044862214
val_masked_l1: 0.0413562357
```

The best checkpoint selected by validation loss was epoch 8, not epoch 10.

## Exported Full-Volume Prediction Evaluation

This is the primary result because it evaluates the exported task-style
prediction files listed in `prediction_manifest.csv`.

Validation prediction export:

```text
prediction manifest rows: 8,100
missing prediction files: 0
exported evaluation rows: 8,100
```

Primary exported prediction metrics:

```text
MAE mean: 2.2182961016e-07
RMSE mean: 2.1204919358e-06
Relative MAE mean: 0.3044251664 %
Relative RMSE mean: 2.9701784284 %
Masked MAE mean: 9.0364961658e-06
Masked RMSE mean: 1.5969898469e-05
Masked relative MAE mean: 12.9234403850 %
Masked relative RMSE mean: 22.8184573746 %
Gamma 3%/3mm pass-rate mean: 0.0003129931
Gamma 2%/2mm pass-rate mean: 0.0000833082
Prediction seconds mean: 0.1214567757
Write seconds mean: 0.2019697279
Total seconds mean: 0.3234279496
```

## Interpretation

This run is a valid completed full-dataset baseline, but the dose prediction
quality is weak.

The strongest warning signs are:

```text
mean predicted dose is much lower than mean target dose
mean predicted max dose is far lower than mean target max dose
gamma pass-rate is close to zero
high-dose-region errors are large
```

This suggests that the current minimal 3D U-Net mostly learns a coarse low-dose
pattern and does not yet reproduce high-dose structure or dose falloff well.
The result is still useful because it provides a reproducible reference point
for later architecture and preprocessing changes.

## Public Result Files

Small public result summaries:

```text
results/2026-07-14_10epoch_training_metrics.csv
results/2026-07-14_10epoch_checkpoint_eval_summary.csv
results/2026-07-14_10epoch_exported_prediction_summary.csv
results/2026-07-14_10epoch_completion_audit.csv
```

Large local artifacts are not committed to GitHub:

```text
outputs/full_baseline_hu_no_energy_optimized_seed20260628/train/checkpoints/best.pt
outputs/full_baseline_hu_no_energy_optimized_seed20260628/dose_predictions/
outputs/full_baseline_hu_no_energy_optimized_seed20260628/evaluate/
outputs/full_baseline_hu_no_energy_optimized_seed20260628/evaluate_exported/
```

## Recommended Next Experiments

```text
1. Add explicit beam geometry maps or fluence-like channels instead of using
   only a global condition vector.
2. Compare HU input against HU-to-density input.
3. Run the same split with energy spectrum included as an ablation.
4. Use a loss that better emphasizes high-dose and in-field voxels.
5. Evaluate whether full-volume or sliding-window prediction avoids crop-insert
   artifacts.
6. Increase model capacity after confirming preprocessing and target scaling.
```
