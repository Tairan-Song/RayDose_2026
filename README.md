# RayDose_2026

[![Challenge](https://img.shields.io/badge/DoseRAD-2026-blue)](https://doserad2026.grand-challenge.org/)

CT-based photon dose prediction with spatial context learning and volumetric dose modeling.

RayDose_2026 is a research framework for three-dimensional photon dose prediction from planning CT images.

The project explores deep learning methods for efficient radiotherapy dose estimation, with a focus on spatial context learning, beam geometry representation, and anatomically informed dose prediction.

This repository is developed as part of our participation in the DoseRAD 2026 Challenge while serving as an independent research platform for AI-driven radiotherapy dose prediction.

## DoseRAD 2026 Challenge

This work is developed in the context of the DoseRAD 2026 Challenge:

https://doserad2026.grand-challenge.org/

### Current Focus

Task 1: Photon Therapy Dose Calculation on CT Images

The objective is to predict three-dimensional photon dose distributions from:

- CT images
- photon beam/control-point parameters
- beam geometry representations

The official Monte Carlo dose files are used as supervised labels. We do not reproduce Monte Carlo transport directly; the goal is to train a deep learning surrogate for fast dose prediction.

## Code Organization

Scripts are organized by language:

```text
scripts/
  README.md
  python/
    mha_io.py
    make_case_split.py
    preprocess_training_sample.py
    doserad_dataset.py
    model_3d_unet.py
    train_3d_unet_smoke.py
    generate_dose_support_masks.py
    evaluate_mask_definitions.py
    requirements.txt
  powershell/
    01_generate_mask_for_one_dose.ps1
    02_generate_masks_for_one_case.ps1
    03_generate_full_photon_1pct_masks.ps1
    04_summarize_1pct_masks.ps1
    05_evaluate_mask_definitions.ps1
    06_summarize_mask_definition_eval.ps1
    07_make_mask_recommendation_matrix.ps1
```

Use Python for new algorithm and training-pipeline work. The main Python entry
point is:

```text
scripts/python/generate_dose_support_masks.py
```

PowerShell scripts are optional Windows helper scripts. Python is the preferred
workflow for data preparation and model development.

See:

```text
scripts/README.md
```

## Dataset

The DoseRAD2026 dataset should be obtained from the official challenge data
source according to the challenge rules and license.

The scripts use the following default data path, which can be changed with
command-line arguments:

```text
data/photon/training
```

## Initial Mask Choices

Current working choices for auxiliary dose-mask experiments:

- Dose-support prediction: `dose_gt_1pct`
- Weighted dose regression: `dose_gt_1pct_dilate2`
- Edge-aware loss: `dose_gradient_top10pct`
- Baseline comparison: `dose_nonzero`
