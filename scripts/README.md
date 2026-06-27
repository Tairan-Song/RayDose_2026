# Scripts

Use Python as the main workflow.

PowerShell scripts are optional Windows helper scripts. New algorithm and model
work should be in Python.

## Main Python Scripts

Path:

```text
scripts/python/
```

### 1. Generate Dose-Support Labels

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

### 2. Evaluate Alternative Mask Definitions

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

### 3. MHA IO Helper

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
