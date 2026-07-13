# Project Logs

This folder contains public weekly progress records for the DoseRAD2026 photon
dose prediction baseline.

The logs are intended for collaborator review. They summarize:

- completed tasks
- implemented scripts
- local experiment settings
- known limitations
- next steps

Large generated artifacts are not stored here. The following remain local:

- DoseRAD raw training data
- full-volume prediction `.mha` files
- model checkpoints
- large masks and intermediate volumes
- subset-only experimental result summaries

## Entries

| Date | Topic |
| --- | --- |
| 2026-06-28 | Data preparation, mask generation, preprocessing, and baseline pipeline setup |
| 2026-07-10 | First functional 3D U-Net baseline pipeline and local subset sanity run |
| 2026-07-13 | Public GitHub sync policy and collaborator-facing repository organization |
| 2026-07-13 | Full-dataset baseline goal, fixed split, checkpoints, and runtime plan |

## Public Result Policy

Only full-dataset or otherwise explicitly justified benchmark results should be
published here. Subset runs, smoke tests, and debugging outputs should stay
local unless they are clearly marked as non-benchmark engineering checks.
