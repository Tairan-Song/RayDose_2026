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

## Included Scripts

### Dose-Support Mask Generation

- `scripts/New-DoseMaskLabel.ps1`
  - Generate a binary dose-support mask for one `Dose_Bx_CPy.mha` file.

- `scripts/New-DoseMaskLabelsForCase.ps1`
  - Generate masks for a limited subset of one case.

- `scripts/New-PhotonDoseMaskLabels.ps1`
  - Generate full photon `dose_gt_1pct` masks and detailed statistics.

- `scripts/Summarize-DoseMaskLabels.ps1`
  - Summarize `dose_gt_1pct` statistics by case.

### Auxiliary Mask Definition Evaluation

- `scripts/Evaluate-AuxiliaryMaskDefinitions.ps1`
  - Evaluate threshold, dilation, and gradient-based auxiliary mask definitions.

- `scripts/Summarize-AuxiliaryMaskDefinitions.ps1`
  - Summarize mask-definition statistics by definition.

- `scripts/New-AuxiliaryMaskRecommendationMatrix.ps1`
  - Generate a recommendation matrix for dose regression, dose-support prediction, and edge-aware loss.

## Data Policy

The full DoseRAD2026 data is intentionally ignored by git via `.gitignore`.

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

