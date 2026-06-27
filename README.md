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

Use PowerShell only to reproduce the already validated Windows preprocessing
outputs.

See:

```text
scripts/README.md
```

## Data Policy

Expected local data root:

```text
data/photon/training
```

Generated masks and CSV statistics are also under `data/` and are not committed.

## Current Auxiliary Mask Recommendation

Based on full photon stats-only evaluation:

- Dose-support prediction: `dose_gt_1pct`
- Weighted dose regression: `dose_gt_1pct_dilate2`
- Edge-aware loss: `dose_gradient_top10pct`
- Baseline comparison: `dose_nonzero`
