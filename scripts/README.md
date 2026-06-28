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

### 2. Preprocess One Training Sample

Script:

```text
scripts/python/preprocess_training_sample.py
```

Purpose:

- read one case + beam + control point sample
- load CT, dose, and dose-support mask
- clip CT HU to `[-1000, 3000]`
- normalize CT to `[-1, 1]`
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
  --output-npz outputs/preprocessing_smoke/1ABB006_B0_CP000.npz
```

Current assumptions:

- crop center is the case beam `iso_center` when it can be mapped to voxel space
- fallback crop center is the image center
- this script is a smoke test for one sample, not a full preprocessing pipeline

### 3. PyTorch Dataset And Baseline Smoke Test

Scripts:

```text
scripts/python/doserad_dataset.py
scripts/python/model_3d_unet.py
scripts/python/train_3d_unet.py
scripts/python/train_3d_unet_smoke.py
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

The condition vector contains:

```text
beam_idx, cp_idx, sin/cos gantry angle, isocenter, MLC left positions, MLC right positions
```

Current target:

```text
Dose_Bx_CPy.mha crop
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
  --max-train-samples 32 `
  --max-val-samples 8 `
  --batch-size 1 `
  --epochs 5 `
  --base-channels 8
```

Outputs:

```text
outputs/baseline_3d_unet/metrics.csv
outputs/baseline_3d_unet/checkpoints/best.pt
outputs/baseline_3d_unet/checkpoints/last.pt
outputs/baseline_3d_unet/predictions/val_prediction_epoch_*.npz
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

### 4. Generate Dose-Support Labels

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

### 5. Evaluate Alternative Mask Definitions

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

### 6. MHA IO Helper

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
