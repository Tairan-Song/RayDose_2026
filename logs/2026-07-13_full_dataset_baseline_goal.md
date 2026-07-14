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

## Throughput Optimization Added After Launch

The command above records the full-dataset run that was already launched. After
launch, the training scripts were optimized for the next full-dataset run. A
future optimized full run should add:

```powershell
  --prefetch-factor 2 `
  --ct-cache-size 4 `
  --case-grouped-sampling `
  --cudnn-benchmark
```

Do not add `--amp` by default on GTX 1070 without a short benchmark, because
this GPU does not have Tensor Cores and mixed precision may not improve speed.

## Optimized Run Relaunch

The first full run in `outputs/full_baseline_hu_no_energy_seed20260628` stopped
before epoch 1. Cause: training code was edited while Windows DataLoader worker
processes were still active, so workers loaded a newer Dataset class than the
already-created main-process Dataset object.

That failed directory is not a formal result.

The optimized full run was relaunched at:

```text
outputs/full_baseline_hu_no_energy_optimized_seed20260628
```

Main PID:

```text
14808
```

Optimization flags included:

```text
--num-workers 4
--prefetch-factor 2
--ct-cache-size 4
--case-grouped-sampling
--cudnn-benchmark
```

Initial health check:

```text
train samples = 32400
val samples = 8100
GPU utilization ~= 77%
GPU memory ~= 3432 MiB
status = running
```

Second health check after script optimization:

```text
run elapsed ~= 8 minutes
completed epochs = 0 / 10
GPU utilization ~= 87%
GPU memory ~= 3431 MiB
status = running
```

Third health check:

```text
run elapsed ~= 9 minutes
completed epochs = 0 / 10
GPU utilization ~= 81%
GPU memory ~= 3443 MiB
stderr = empty
status = running
```

Fourth health check:

```text
run elapsed ~= 12 minutes
completed epochs = 0 / 10
GPU utilization ~= 94%
GPU memory ~= 4614 MiB
stderr = empty
status = running
```

Fifth health check:

```text
run elapsed ~= 15 minutes
completed epochs = 0 / 10
GPU utilization ~= 93%
GPU memory ~= 4555 MiB
stderr = empty
status = running
```

Sixth health check:

```text
run elapsed ~= 17 minutes
completed epochs = 0 / 10
GPU utilization ~= 94%
GPU memory ~= 4589 MiB
stderr = empty
status = running
```

Seventh health check:

```text
run elapsed ~= 19 minutes
completed epochs = 0 / 10
GPU utilization ~= 94%
GPU memory ~= 4583 MiB
stderr = empty
status = running
```

Eighth health check:

```text
run elapsed ~= 21 minutes
completed epochs = 0 / 10
GPU utilization ~= 94%
GPU memory ~= 4612 MiB
stderr = empty
status = running
```

Ninth health check:

```text
run elapsed ~= 25 minutes
completed epochs = 0 / 10
GPU utilization ~= 74%
GPU memory ~= 4502 MiB
stderr = empty
status = running
```

Tenth health check:

```text
run elapsed ~= 26 minutes
completed epochs = 0 / 10
GPU utilization ~= 93%
GPU memory ~= 4534 MiB
stderr = empty
status = running
```

Eleventh health check:

```text
run elapsed ~= 28 minutes
completed epochs = 0 / 10
GPU utilization ~= 94%
GPU memory ~= 4613 MiB
stderr = empty
status = running
```

Twelfth health check:

```text
run elapsed ~= 29 minutes
completed epochs = 0 / 10
GPU utilization ~= 83%
GPU memory ~= 4594 MiB
stderr = empty
status = running
```

Thirteenth health check:

```text
run elapsed ~= 31 minutes
completed epochs = 0 / 10
GPU utilization ~= 85%
GPU memory ~= 4541 MiB
stderr = empty
status = running
```

Fourteenth health check:

```text
run elapsed ~= 32 minutes
completed epochs = 0 / 10
GPU utilization ~= 93%
GPU memory ~= 4583 MiB
stderr = empty
status = running
```

Fifteenth health check:

```text
run elapsed ~= 34 minutes
completed epochs = 0 / 10
GPU utilization ~= 93%
GPU memory ~= 4586 MiB
stderr = empty
status = running
```

Sixteenth health check:

```text
run elapsed ~= 37 minutes
completed epochs = 0 / 10
epoch 1 validation started = true
first validation prediction = train/predictions/val_prediction_epoch_001.npz
first validation prediction modified = 2026-07-13 21:30:59
checkpoint 2a_first_epoch_validation_started = complete
checkpoint 2_first_epoch_complete = pending
status = running
```

Seventeenth health check:

```text
run elapsed ~= 38 minutes
completed epochs = 0 / 10
epoch 1 validation started = true
metrics.csv = missing
last.pt = missing
best.pt = missing
GPU utilization ~= 30%
GPU memory ~= 4648 MiB
stderr = empty
status = running, epoch 1 validation in progress
```

Eighteenth health check:

```text
run elapsed ~= 40 minutes
completed epochs = 0 / 10
epoch 1 validation started = true
metrics.csv = missing
last.pt = missing
best.pt = missing
GPU utilization ~= 34%
GPU memory ~= 4623 MiB
stderr = empty
status = running, epoch 1 validation in progress
```

Nineteenth health check:

```text
run elapsed ~= 42 minutes
completed epochs = 0 / 10
epoch 1 validation started = true
metrics.csv = missing
last.pt = missing
best.pt = missing
GPU utilization ~= 24%
GPU memory ~= 4550 MiB
stderr = empty
status = running, epoch 1 validation in progress
```

Disk-space check for validation export:

```text
F: total ~= 917.81 GB
F: free ~= 474.57 GB
average compressed CT size across 75 cases ~= 10.49 MB
expected validation predictions = 8,100 full-volume MHA files
status = enough free space for the planned full validation export
```

Split reproducibility check:

```text
split file = splits/photon_case_split.csv
train cases = 60
val cases = 15
seed = 20260628
git tracking status = allowed by .gitignore
```

The repository ignore rules were adjusted so the formal split file can be
tracked, while generated outputs remain ignored.

Important implementation note:

```text
The training subprocess was already launched with the optimized Dataset/training code.
Later export/evaluation script improvements will still apply after training completes,
because those stages spawn new Python processes that read the current scripts from disk.
```

Heartbeat note:

```text
Current running process was launched before --heartbeat-every was added.
Therefore epoch-internal batch progress is unavailable for this run.
Future resumed/restarted runs can use --heartbeat-every 500 or 1000 to write
train/heartbeat.json and print batch progress before a full epoch completes.
```

## Checkpoint Schedule For Current Optimized Run

Current formal output directory:

```text
outputs/full_baseline_hu_no_energy_optimized_seed20260628
```

Checkpoint status:

```text
0. Pre-run manifest and split verification: complete
1. Training process started with GPU + 4 CPU workers: complete
2. First epoch complete: pending
3. All 10 training epochs complete: pending
4. Full checkpoint evaluation on 8,100 validation samples: pending
5. Full validation prediction export, Dose_Bx_CPy.mha style: pending
6. Exported full-volume evaluation with MAE/RMSE/masked/high-dose/gamma metrics: pending
```

Expected time policy:

```text
Before epoch 1 completes:
  exact ETA is not reliable.

After epoch 1 completes:
  estimated training remaining = epoch_1_seconds * 9
  total training estimate = epoch_1_seconds * 10

Conservative current stage estimates:
  training: unknown until epoch 1; previous fallback estimate 40-80 hours
  checkpoint evaluation: 0.5-2 hours
  prediction export: 1-3 hours
  exported evaluation with gamma: 6-12 hours using 4 CPU workers
```

The next formal checkpoint is `train/metrics.csv` containing epoch 1.

## Completion Audit Command

After all training/evaluation/export stages complete, run:

```powershell
F:\anaconda3\envs\myenvs3_9\python.exe scripts\python\audit_full_baseline_completion.py `
  --output-dir outputs\full_baseline_hu_no_energy_optimized_seed20260628 `
  --split-csv splits\photon_case_split.csv `
  --expected-epochs 10
```

Current audit result while training is still in epoch 1:

```text
checks = 15
failed = 10
expected status = incomplete
```

This failure is expected until training, checkpoint evaluation, prediction
export, exported metrics, and gamma summaries finish.

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
outputs/full_baseline_hu_no_energy_optimized_seed20260628/train/
outputs/full_baseline_hu_no_energy_optimized_seed20260628/train/metrics.csv
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
outputs/full_baseline_hu_no_energy_optimized_seed20260628/train/checkpoints/best.pt
outputs/full_baseline_hu_no_energy_optimized_seed20260628/train/checkpoints/last.pt
outputs/full_baseline_hu_no_energy_optimized_seed20260628/train/metrics.csv
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
outputs/full_baseline_hu_no_energy_optimized_seed20260628/evaluate/per_sample_metrics.csv
outputs/full_baseline_hu_no_energy_optimized_seed20260628/evaluate/summary_metrics.csv
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
outputs/full_baseline_hu_no_energy_optimized_seed20260628/dose_predictions/prediction_manifest.csv
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
outputs/full_baseline_hu_no_energy_optimized_seed20260628/evaluate_exported/exported_prediction_metrics.csv
outputs/full_baseline_hu_no_energy_optimized_seed20260628/evaluate_exported/exported_prediction_summary.csv
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
Training: unknown until epoch 1 completes; previous conservative estimate was 40-80 hours
Checkpoint evaluation: 0.5-2 hours
Prediction export: 1-3 hours with DataLoader workers
Exported full-volume evaluation with gamma: 6-12 hours with 4 CPU workers
```

## Progress Check Command

Use this command while the run is in progress:

```powershell
F:\anaconda3\envs\myenvs3_9\python.exe scripts\python\check_full_baseline_progress.py `
  --output-dir outputs\full_baseline_hu_no_energy_optimized_seed20260628 `
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
outputs/full_baseline_hu_no_energy_optimized_seed20260628/train/checkpoints/best.pt
outputs/full_baseline_hu_no_energy_optimized_seed20260628/train/checkpoints/last.pt
outputs/full_baseline_hu_no_energy_optimized_seed20260628/train/metrics.csv
outputs/full_baseline_hu_no_energy_optimized_seed20260628/run_manifest.json
outputs/full_baseline_hu_no_energy_optimized_seed20260628/evaluate/per_sample_metrics.csv
outputs/full_baseline_hu_no_energy_optimized_seed20260628/evaluate/summary_metrics.csv
outputs/full_baseline_hu_no_energy_optimized_seed20260628/dose_predictions/prediction_manifest.csv
outputs/full_baseline_hu_no_energy_optimized_seed20260628/evaluate_exported/exported_prediction_metrics.csv
outputs/full_baseline_hu_no_energy_optimized_seed20260628/evaluate_exported/exported_prediction_summary.csv
outputs/full_baseline_hu_no_energy_optimized_seed20260628/evaluate_exported/evaluation_runtime.json
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

## 2026-07-13 Multi-CPU/GPU Optimization Update

The local machine currently exposes:

```text
CPU logical processors: 8
CUDA GPUs: 1
GPU model: NVIDIA GeForce GTX 1070
```

Therefore, this run can use one GPU plus multiple CPU DataLoader workers. True
multi-GPU training is not available on this machine because only one CUDA GPU is
detected.

Code updates made locally:

```text
scripts/python/train_3d_unet.py
scripts/python/run_baseline_experiment.py
scripts/python/evaluate_checkpoint.py
scripts/python/predict_3d_unet_batch.py
scripts/README.md
```

Optimization changes:

```text
--num-workers -1 now enables automatic worker selection: min(cpu_count - 2, 8)
--data-parallel now enables torch.nn.DataParallel on machines with >=2 CUDA GPUs
checkpoint saving now unwraps DataParallel so evaluation/export remain portable
evaluation/export loaders can read DataParallel-prefixed checkpoints if needed
README documents current single-GPU limitation and throughput flags
```

Validation of the optimization changes:

```text
py_compile: passed
run_baseline_experiment.py --dry-run with --num-workers -1 --data-parallel: passed
train_3d_unet.py tiny CPU smoke test with --num-workers -1: passed
```

The active formal full-dataset run was not interrupted. It was launched before
these latest code changes, so it continues with the already-started settings:

```text
num_workers: 4
prefetch_factor: 2
ct_cache_size: 4
case_grouped_sampling: enabled
cudnn_benchmark: enabled
data_parallel: not enabled
```

## 2026-07-13 First Epoch Formal Progress

The current formal run has completed epoch 1/10.

Output directory:

```text
outputs/full_baseline_hu_no_energy_optimized_seed20260628
```

Generated epoch-1 artifacts:

```text
train/metrics.csv
train/checkpoints/last.pt
train/checkpoints/best.pt
train/predictions/val_prediction_epoch_001.npz
train/predictions/val_prediction_epoch_001_pred.mha
```

Epoch-1 metrics:

```text
train_samples: 32400
val_samples: 8100
train_loss: 0.0499493033
train_global_l1: 0.0062723723
train_masked_l1: 0.0436769575
val_loss: 0.0479025692
val_global_l1: 0.0056649703
val_masked_l1: 0.0422376245
train_seconds: 2138.3191
val_seconds: 438.7266
epoch_seconds: 2577.0462
```

Estimated remaining training time after epoch 1:

```text
approximately 6h 26m for epochs 2-10
```

Current formal completion status:

```text
training: in progress, 1/10 epochs complete
checkpoint evaluation: pending
full validation prediction export: pending
exported prediction evaluation with gamma: pending
formal audit: pending
```

## 2026-07-13 Epoch-3 Progress

The current formal full-dataset run has completed epoch 3/10.

Latest metrics:

```text
epoch: 3
train_loss: 0.0466960967
train_global_l1: 0.0052420241
train_masked_l1: 0.0414542109
val_loss: 0.0482943021
val_global_l1: 0.0041572941
val_masked_l1: 0.0441370457
epoch_seconds: 2578.6193
```

Current status:

```text
training: in progress, 3/10 epochs complete
estimated remaining training time: approximately 5h
5epoch snapshot: pending
```

## 2026-07-14 Epoch-5 Intermediate Snapshot

The full-dataset run has completed epoch 5/10. The epoch-5 checkpoint was
copied before the active training process could overwrite `last.pt` during
later epochs.

Snapshot directory:

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

Epoch-5 metrics:

```text
train_loss: 0.0452016853
train_global_l1: 0.0050022546
train_masked_l1: 0.0401995070
val_loss: 0.0465990789
val_global_l1: 0.0051342561
val_masked_l1: 0.0414648019
train_samples: 32400
val_samples: 8100
```

Intermediate public summary files:

```text
logs/2026-07-14_5epoch_intermediate_result.md
results/2026-07-14_5epoch_intermediate_metrics.csv
```

This is a 5-epoch intermediate milestone, not the final 10-epoch baseline.

## 2026-07-13 Public GitHub And Deadline Plan

Release plan:

```text
After the full-dataset 10-epoch run finishes, complete all downstream formal
tasks and push the final code/log/results summary to the public GitHub
repository.
```

Public repository requirement:

```text
The GitHub repository is public. Logs and documentation are written for
external readers and collaborators. Public documentation should avoid private
conversation wording, temporary local-only claims, or unsupported performance
claims.
```

Required final local documentation after completion:

```text
1. Full task objective
2. Dataset split and random seed
3. Model input/output definition
4. Training configuration
5. Hardware/resource configuration
6. Runtime record
7. Evaluation metrics
8. Prediction export outputs
9. Known limitations
10. Next recommended experiments
```

Deadline target:

```text
Target completion time: before 2026-07-14 07:00 local time
```

Runtime risk:

```text
Training is expected to finish before the deadline if the process remains
stable. Full exported prediction evaluation, especially gamma pass-rate
calculation, may be the longest downstream stage. Results must not be pushed as
formal baseline results unless the full validation set and completion audit pass.
```

## 2026-07-13 Five-Epoch Intermediate Release Rule

Intermediate release plan:

```text
If the 5-epoch result is trustworthy enough, push that version first and mark it
as 5epoch, while continuing the complete 10-epoch run.
```

Documentation rule:

```text
The 5-epoch result may be shared only as an intermediate milestone, not as the
final full baseline. Public wording must explicitly say "5-epoch intermediate
result" and "not the final 10-epoch baseline".
```

Required action at epoch 5:

```text
1. Detect that metrics.csv contains epoch 5.
2. Copy train/checkpoints/last.pt to a stable 5-epoch snapshot path before epoch
   6 overwrites last.pt.
3. Copy metrics.csv to the same 5-epoch snapshot folder.
4. Run validation evaluation from the epoch-5 snapshot if time permits.
5. Push code/log/snapshot summary with a clear 5epoch label.
6. Keep the active 10-epoch process running.
```
