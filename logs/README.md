# Project Logs

This folder contains public weekly progress records for the DoseRAD2026 photon
dose prediction baseline.

The logs are intended for collaborator review. They summarize:

- completed tasks
- implemented scripts
- local experiment settings
- small result summaries
- known limitations
- next steps

Large generated artifacts are not stored here. The following remain local:

- DoseRAD raw training data
- full-volume prediction `.mha` files
- model checkpoints
- large masks and intermediate volumes

## Entries

| Date | Topic |
| --- | --- |
| 2026-06-28 | Data preparation, mask generation, preprocessing, and baseline pipeline setup |
| 2026-07-10 | First functional 3D U-Net baseline, performance metrics, and collaborator PPT |
| 2026-07-13 | Public GitHub sync log and collaborator-facing repository organization |

## Shareable Result Files

| File | Description |
| --- | --- |
| `results/baseline_hu_no_energy_v2_linear_exported_summary.csv` | Full-volume exported prediction metrics for the first functional baseline |
| `results/baseline_hu_no_energy_v2_linear_train_metrics.csv` | Training/validation loss curve for the first functional baseline |
| `presentations/DoseRAD_Baseline_Update_20260710.pptx` | Six-slide collaborator update deck |
