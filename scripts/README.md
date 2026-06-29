# Scripts

Use Python as the main workflow.

PowerShell scripts are optional Windows helper scripts. New algorithm and model
work should be in Python.

## Main Python Scripts

Path:

```text
scripts/python/
```

### 1. Create Case-Level Train/Validation Split

Script:

```text
scripts/python/make_case_split.py
```

Purpose:

- scan valid photon training cases
- split by case, not by dose/control-point file
- default split: 80% train / 20% validation
- fixed seed for reproducibility

Run:

```powershell
python scripts/python/make_case_split.py `
  --training-dir data/photon/training `
  --output-csv splits/photon_case_split.csv `
  --train-fraction 0.8 `
  --seed 20260628
```

The generated CSV has this format:

```text
case_id,split,seed,train_fraction
```

All CT, JSON, dose, and mask files from the same case must stay in the same
split.

### 2. Validate Baseline Inputs

Script:

```text
scripts/python/validate_baseline_data.py
```

Purpose:

- verify the split CSV references valid case folders
- check required CT, JSON, dose, and mask file paths
- validate basic beam/control-point JSON fields
- optionally read CT/dose/mask MHA files
- write a CSV validation report

Run a fast structure check:

```powershell
python scripts/python/validate_baseline_data.py `
  --training-dir data/photon/training `
  --split-csv splits/photon_case_split.csv `
  --report-csv outputs/data_validation/baseline_data_validation.csv
```

By default this checks all dose/mask file paths in the split.

Read a small number of MHA files as well:

```powershell
python scripts/python/validate_baseline_data.py `
  --training-dir data/photon/training `
  --split-csv splits/photon_case_split.csv `
  --report-csv outputs/data_validation/baseline_data_validation_mha.csv `
  --max-cases 5 `
  --max-dose-files-per-case 2 `
  --check-mha
```

### 3. Preprocess One Training Sample

Script:

```text
scripts/python/preprocess_training_sample.py
```

Purpose:

- read one case + beam + control point sample
- load CT, dose, and dose-support mask
- clip CT HU to `[-1000, 3000]`
- normalize CT to `[-1, 1]`
- optionally convert CT HU to density using `beam_parameters.json`
- normalize dose by the sample max dose
- crop/pad CT, dose, and mask to a fixed shape
- save a compressed `.npz` for Dataset/model debugging

Example:

```powershell
python scripts/python/preprocess_training_sample.py `
  --training-dir data/photon/training `
  --case-id 1ABB006 `
  --beam-idx 0 `
  --cp-idx 0 `
  --mask-name dose_gt_1pct `
  --target-shape "128 128 128" `
  --ct-mode hu `
  --output-npz outputs/preprocessing_smoke/1ABB006_B0_CP000.npz
```

Current assumptions:

- crop center is the case beam `iso_center` when it can be mapped to voxel space
- fallback crop center is the image center
- this script is a smoke test for one sample, not a full preprocessing pipeline
- `--ct-mode hu` uses clipped normalized HU
- `--ct-mode density` uses the HU-to-density table in `beam_parameters.json`

### 4. PyTorch Dataset And Baseline Smoke Test

Scripts:

```text
scripts/python/doserad_dataset.py
scripts/python/model_3d_unet.py
scripts/python/train_3d_unet.py
scripts/python/run_baseline_experiment.py
scripts/python/run_energy_ablation.py
scripts/python/run_pipeline_smoke.py
scripts/python/predict_3d_unet.py
scripts/python/predict_3d_unet_batch.py
scripts/python/evaluate_prediction.py
scripts/python/evaluate_checkpoint.py
scripts/python/train_3d_unet_smoke.py
scripts/python/estimate_dose_scale.py
```

Purpose:

- define one training sample as one case + beam + control point
- load CT, dose, and dose-support mask
- crop/pad samples to a fixed 3D shape
- encode beam/control-point geometry as a condition vector
- train a geometry-conditioned 3D U-Net baseline
- run a tiny code-path smoke test

Current model input:

```text
CT crop + condition vector
```

CT crop options:

```text
--ct-mode hu       clipped HU normalized to [-1, 1]
--ct-mode density  HU converted to density and scaled to [0, 1]
```

The condition vector contains:

```text
beam_idx, cp_idx, sin/cos gantry angle, isocenter, MLC left positions, MLC right positions
```

Optional energy-spectrum ablation:

```text
--include-energy
```

This appends the global 100-bin photon energy-spectrum weights from
`beam_parameters.json` to the condition vector. By default, energy is not
included because the current dataset stores it as a global machine parameter,
not as a per-control-point variable.

Current target:

```text
Dose_Bx_CPy.mha crop
```

Dose target scaling:

```text
--dose-mode global      recommended for direct dose prediction
--global-dose-scale     constant used to scale dose targets
--dose-mode sample_max  useful for debugging only; not ideal for test inference
```

Sample selection for limited-size runs:

```text
--sample-strategy uniform  default; spread samples across cases and control points
--sample-strategy random   deterministic random subset using --sample-seed
--sample-strategy first    legacy behavior; take the first sorted samples
```

Use `--max-train-samples 0` and `--max-val-samples 0` to use all available
samples in the selected split.

Reproducibility controls:

```text
--seed         model initialization and DataLoader shuffle seed
--sample-seed subset selection seed when --sample-strategy random is used
```

Estimate a global scale from existing training statistics:

```powershell
python scripts/python/estimate_dose_scale.py `
  --stats-csv data/photon/training/dose_mask_stats_gt_1pct.csv
```

Current local estimate:

```text
max dose_max ~= 1.49e-4
recommended --global-dose-scale 1.5e-4
```

Current loss:

```text
global L1 dose loss + masked L1 dose loss inside dose_gt_1pct
```

Baseline train/validation run:

```powershell
python scripts/python/train_3d_unet.py `
  --training-dir data/photon/training `
  --split-csv splits/photon_case_split.csv `
  --output-dir outputs/baseline_3d_unet `
  --target-shape "64 64 64" `
  --ct-mode hu `
  --dose-mode global `
  --global-dose-scale 1.5e-4 `
  --max-train-samples 32 `
  --max-val-samples 8 `
  --batch-size 1 `
  --epochs 5 `
  --base-channels 8
```

Train, evaluate, and export dose-style MHA predictions:

```powershell
python scripts/python/run_baseline_experiment.py `
  --training-dir data/photon/training `
  --split-csv splits/photon_case_split.csv `
  --make-split `
  --output-dir outputs/baseline_experiment `
  --target-shape "64 64 64" `
  --ct-mode hu `
  --dose-mode global `
  --global-dose-scale 1.5e-4 `
  --max-train-samples 32 `
  --max-val-samples 8 `
  --eval-samples 8 `
  --export-samples 8 `
  --batch-size 1 `
  --epochs 5 `
  --base-channels 8
```

This writes:

```text
outputs/baseline_experiment/train/
outputs/baseline_experiment/evaluate/
outputs/baseline_experiment/dose_predictions/
```

With energy-spectrum conditioning:

```powershell
python scripts/python/train_3d_unet.py `
  --training-dir data/photon/training `
  --split-csv splits/photon_case_split.csv `
  --output-dir outputs/baseline_3d_unet_energy `
  --target-shape "64 64 64" `
  --ct-mode hu `
  --include-energy `
  --dose-mode global `
  --global-dose-scale 1.5e-4 `
  --max-train-samples 32 `
  --max-val-samples 8 `
  --batch-size 1 `
  --epochs 5 `
  --base-channels 8
```

Run matched no-energy and with-energy experiments:

```powershell
python scripts/python/run_energy_ablation.py `
  --training-dir data/photon/training `
  --split-csv splits/photon_case_split.csv `
  --output-root outputs/energy_ablation `
  --target-shape "64 64 64" `
  --ct-mode hu `
  --dose-mode global `
  --global-dose-scale 1.5e-4 `
  --max-train-samples 32 `
  --max-val-samples 8 `
  --batch-size 1 `
  --epochs 5 `
  --base-channels 8
```

This writes:

```text
outputs/energy_ablation/no_energy/
outputs/energy_ablation/with_energy/
outputs/energy_ablation/ablation_summary.csv
```

Run a 2x2 CT-mode and energy ablation:

```powershell
python scripts/python/run_energy_ablation.py `
  --training-dir data/photon/training `
  --split-csv splits/photon_case_split.csv `
  --output-root outputs/baseline_conditioning_ablation `
  --ct-modes hu density `
  --target-shape "64 64 64" `
  --dose-mode global `
  --global-dose-scale 1.5e-4 `
  --max-train-samples 32 `
  --max-val-samples 8 `
  --batch-size 1 `
  --epochs 5 `
  --base-channels 8
```

This writes:

```text
outputs/baseline_conditioning_ablation/hu_no_energy/
outputs/baseline_conditioning_ablation/hu_with_energy/
outputs/baseline_conditioning_ablation/density_no_energy/
outputs/baseline_conditioning_ablation/density_with_energy/
outputs/baseline_conditioning_ablation/ablation_summary.csv
```

Outputs:

```text
outputs/baseline_3d_unet/metrics.csv
outputs/baseline_3d_unet/checkpoints/best.pt
outputs/baseline_3d_unet/checkpoints/last.pt
outputs/baseline_3d_unet/predictions/val_prediction_epoch_*.npz
outputs/baseline_3d_unet/predictions/val_prediction_epoch_*_pred.mha
```

The exported `.mha` prediction from training validation is crop-level. Use
`predict_3d_unet.py` or `predict_3d_unet_batch.py` to write full-volume MHA
predictions on the original CT grid.

Full-volume inference for one sample:

```powershell
python scripts/python/predict_3d_unet.py `
  --checkpoint outputs/baseline_3d_unet/checkpoints/best.pt `
  --split val `
  --output-dir outputs/baseline_3d_unet_inference
```

Optional specific sample:

```powershell
python scripts/python/predict_3d_unet.py `
  --checkpoint outputs/baseline_3d_unet/checkpoints/best.pt `
  --case-id 1ABB006 `
  --beam-idx 0 `
  --cp-idx 0 `
  --output-dir outputs/baseline_3d_unet_inference
```

Inference outputs:

```text
<case>_B<beam>_CP<cp>_pred_crop.mha
<case>_B<beam>_CP<cp>_pred_full.mha
<case>_B<beam>_CP<cp>_pred.npz
```

The full `.mha` has the same `DimSize`, `ElementSpacing`, and `Offset` as the
original CT. The current baseline places the predicted crop into the full
volume and fills outside-crop voxels with zero.

Full-volume inference for multiple samples:

```powershell
python scripts/python/predict_3d_unet_batch.py `
  --checkpoint outputs/baseline_3d_unet/checkpoints/best.pt `
  --split val `
  --output-dir outputs/baseline_3d_unet_batch_inference `
  --max-samples 64 `
  --filename-style pred `
  --no-npz
```

Batch inference writes one case subdirectory per case:

```text
outputs/baseline_3d_unet_batch_inference/
  prediction_manifest.csv
  <case>/
    <case>_B<beam>_CP<cp>_pred_crop.mha
    <case>_B<beam>_CP<cp>_pred_full.mha
```

Use task-style dose filenames:

```powershell
python scripts/python/predict_3d_unet_batch.py `
  --checkpoint outputs/baseline_3d_unet/checkpoints/best.pt `
  --split val `
  --output-dir outputs/baseline_3d_unet_dose_names `
  --max-samples 64 `
  --filename-style dose `
  --no-npz
```

This writes full-volume predictions named:

```text
<case>/Dose_B<beam>_CP<cp>.mha
```

Evaluate one prediction:

```powershell
python scripts/python/evaluate_prediction.py `
  --prediction outputs/baseline_3d_unet_inference/1ABB006_B0_CP000_pred_full.mha `
  --target data/photon/training/1ABB006/dose/Dose_B0_CP000.mha `
  --mask data/photon/training/1ABB006/label_masks/dose_gt_1pct/Dose_B0_CP000_mask.mha `
  --output-csv outputs/evaluation/1ABB006_B0_CP000_metrics.csv
```

Metrics:

```text
MAE, RMSE, max absolute error, masked MAE, masked RMSE, masked max absolute error
```

Evaluate a checkpoint on multiple validation samples:

```powershell
python scripts/python/evaluate_checkpoint.py `
  --checkpoint outputs/baseline_3d_unet/checkpoints/best.pt `
  --split val `
  --output-dir outputs/checkpoint_evaluation `
  --max-samples 64
```

This writes:

```text
outputs/checkpoint_evaluation/per_sample_metrics.csv
outputs/checkpoint_evaluation/summary_metrics.csv
```

Tiny smoke test:

```powershell
python scripts/python/train_3d_unet_smoke.py `
  --training-dir data/photon/training `
  --split-csv splits/photon_case_split.csv `
  --split train `
  --target-shape "32 32 32" `
  --max-samples 2 `
  --batch-size 1 `
  --steps 2 `
  --base-channels 4
```

The smoke test is a code-path test, not a performance experiment.

End-to-end pipeline smoke test:

```powershell
python scripts/python/run_pipeline_smoke.py `
  --training-dir data/photon/training `
  --output-dir outputs/pipeline_smoke
```

This runs:

```text
case split -> preprocessing -> 1-step training -> checkpoint evaluation -> batch full-volume inference
```

Expected final line:

```text
pipeline_smoke_passed output_dir=outputs/pipeline_smoke
```

### 5. Generate Dose-Support Labels

Script:

```text
scripts/python/generate_dose_support_masks.py
```

Purpose:

- read each photon dose `.mha`
- create a binary label mask
- default rule: `dose > 0.01 * max(dose)`
- write one mask per dose file
- write a CSV summary

Small test run:

```powershell
python scripts/python/generate_dose_support_masks.py `
  --training-dir data/photon/training `
  --stats-csv data/photon/training/dose_mask_stats_gt_1pct_py_test.csv `
  --max-files 2 `
  --force
```

Full run:

```powershell
python scripts/python/generate_dose_support_masks.py `
  --training-dir data/photon/training `
  --stats-csv data/photon/training/dose_mask_stats_gt_1pct_py.csv `
  --workers 6 `
  --force
```

Nonzero-dose baseline:

```powershell
python scripts/python/generate_dose_support_masks.py `
  --training-dir data/photon/training `
  --label-name dose_nonzero `
  --threshold-fraction 0 `
  --stats-csv data/photon/training/dose_mask_stats_nonzero.csv `
  --workers 6 `
  --force
```

### 6. Evaluate Alternative Mask Definitions

Script:

```text
scripts/python/evaluate_mask_definitions.py
```

Purpose:

- compare different mask definitions
- threshold masks: nonzero, 0.5%, 1%, 2%, 5%
- dilation masks around the 1% mask
- gradient masks for dose-edge regions

Small test run:

```powershell
python scripts/python/evaluate_mask_definitions.py `
  --training-dir data/photon/training `
  --stats-csv data/photon/training/aux_mask_definition_eval_py_test.csv `
  --max-files 2
```

### 7. MHA IO Helper

Script:

```text
scripts/python/mha_io.py
```

Purpose:

- read DoseRAD `.mha` dose files
- write binary `.mha` mask files

This file is a helper, not a script to run directly.

## Python Dependencies

Install:

```powershell
pip install -r scripts/python/requirements.txt
```

`numpy` is required for all Python scripts.

`scipy` is required only for dilation masks in
`evaluate_mask_definitions.py`.

## PowerShell Scripts

Path:

```text
scripts/powershell/
```

These are optional reproduction scripts:

```text
01_generate_mask_for_one_dose.ps1
02_generate_masks_for_one_case.ps1
03_generate_full_photon_1pct_masks.ps1
04_summarize_1pct_masks.ps1
05_evaluate_mask_definitions.ps1
06_summarize_mask_definition_eval.ps1
07_make_mask_recommendation_matrix.ps1
```

Use Python for new model code.
