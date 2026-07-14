# DoseRAD2026 Photon Task 1: 5-Epoch Intermediate Result

Date: 2026-07-14

## Status

This is a 5-epoch intermediate result for the full-dataset
geometry-conditioned 3D U-Net baseline. It is not the final 10-epoch baseline.

The active 10-epoch run continued after this snapshot.

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

The split is case-level. A case does not appear in both training and validation.

## Model

```text
Architecture: geometry-conditioned 3D U-Net
Input image: CT crop
Condition vector:
  - beam index
  - control-point index
  - gantry angle
  - isocenter
  - MLC left leaf positions
  - MLC right leaf positions
Energy spectrum input: not included in this run
Output: per-control-point 3D dose crop
```

## Training Configuration

```text
Target shape: 64 x 64 x 64
CT mode: HU
Dose mode: global
Global dose scale: 0.00015
Mask for focused loss: dose_gt_1pct
Loss: global L1 + masked L1
Batch size: 1
Base channels: 8
Learning rate: 0.001
Weight decay: 0.0001
Mask weight: 1.0
Epoch snapshot: 5
```

## Hardware And Runtime

```text
GPU: NVIDIA GeForce GTX 1070
CUDA GPUs detected: 1
DataLoader workers: 4
Prefetch factor: 2
CT cache size: 4
Case-grouped sampling: enabled
cuDNN benchmark: enabled
```

The local machine has one CUDA GPU, so this run used single-GPU training with
multi-worker CPU data loading. Multi-GPU training support was added to the
scripts for future machines, but it was not active in this run.

## Epoch-5 Snapshot

The epoch-5 checkpoint was copied before epoch 6 could overwrite `last.pt`.

Local snapshot path:

```text
outputs/full_baseline_hu_no_energy_optimized_seed20260628/snapshots/epoch_005
```

Snapshot files:

```text
epoch_005_last.pt
epoch_005_best.pt
metrics_epoch_005_snapshot.csv
run_manifest_snapshot.json
```

Checkpoint verification:

```text
epoch_005_last.pt: epoch 5, val_loss 0.0465990789
epoch_005_best.pt: epoch 5, val_loss 0.0465990789
```

## Training Curve Through Epoch 5

```text
epoch  train_loss  val_loss
1      0.0499493   0.0479026
2      0.0475929   0.0489157
3      0.0466961   0.0482943
4      0.0458878   0.0467454
5      0.0452017   0.0465991
```

The epoch-5 validation loss is the best value observed through the first five
epochs. Validation was computed over the full validation split of 8,100 dose
samples.

## Interpretation

The 5-epoch result is suitable as an intermediate training milestone because it
uses the fixed full training split and full validation split. It should not be
reported as the final challenge baseline because the following tasks are still
pending:

```text
1. Complete epochs 6-10.
2. Evaluate the final checkpoint on all 8,100 validation samples.
3. Export all validation predictions as task-style Dose_Bx_CPy.mha files.
4. Evaluate exported predictions, including gamma pass-rate metrics.
5. Run the formal completion audit.
```

## Public Reporting Label

Recommended label:

```text
5epoch-intermediate
```

Recommended wording:

```text
5-epoch intermediate full-dataset training milestone, not final 10-epoch
baseline.
```
