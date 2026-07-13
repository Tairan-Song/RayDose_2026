"""Evaluate exported full-volume MHA predictions listed in a manifest CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from evaluate_prediction import metrics, spacing_from_meta
from mha_io import read_mha


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def target_path(training_dir: Path, case_id: str, beam_idx: int, cp_idx: int) -> Path:
    return training_dir / case_id / "dose" / f"Dose_B{beam_idx}_CP{cp_idx:03d}.mha"


def mask_path(training_dir: Path, case_id: str, beam_idx: int, cp_idx: int, mask_name: str) -> Path:
    return training_dir / case_id / "label_masks" / mask_name / f"Dose_B{beam_idx}_CP{cp_idx:03d}_mask.mha"


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    metric_names = [
        "mae",
        "rmse",
        "mean_error",
        "max_abs_error",
        "target_mean",
        "pred_mean",
        "target_max",
        "pred_max",
        "relative_mae_percent",
        "relative_rmse_percent",
        "relative_max_abs_error_percent",
        "hd_10pct_mae",
        "hd_10pct_rmse",
        "hd_50pct_mae",
        "hd_50pct_rmse",
        "gamma_3pct_3mm_cutoff10pct_pass_rate",
        "gamma_2pct_2mm_cutoff10pct_pass_rate",
        "masked_mae",
        "masked_rmse",
        "masked_max_abs_error",
        "masked_relative_mae_percent",
        "masked_relative_rmse_percent",
        "prediction_seconds",
        "write_seconds",
        "total_seconds",
    ]
    summary_rows: list[dict[str, object]] = []
    for name in metric_names:
        numeric_values = []
        for row in rows:
            value = row.get(name, "")
            if value == "":
                continue
            try:
                numeric_values.append(float(value))
            except (TypeError, ValueError):
                continue
        values = np.asarray(numeric_values, dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        summary_rows.append(
            {
                "metric": name,
                "count": int(values.size),
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "p50": float(np.percentile(values, 50)),
                "p95": float(np.percentile(values, 95)),
                "max": float(values.max()),
            }
        )

    write_csv(path, summary_rows, ["metric", "count", "mean", "std", "min", "p50", "p95", "max"])


def evaluate_manifest(args: argparse.Namespace) -> None:
    training_dir = Path(args.training_dir)
    manifest = read_manifest(Path(args.manifest_csv))
    rows: list[dict[str, object]] = []

    for idx, item in enumerate(manifest):
        if args.max_samples > 0 and idx >= args.max_samples:
            break

        case_id = item["case_id"]
        beam_idx = int(item["beam_idx"])
        cp_idx = int(item["cp_idx"])
        pred_path = Path(item["full_mha"])
        tgt_path = target_path(training_dir, case_id, beam_idx, cp_idx)
        msk_path = mask_path(training_dir, case_id, beam_idx, cp_idx, args.mask_name) if args.mask_name else None

        pred = read_mha(pred_path).array
        target_img = read_mha(tgt_path)
        target = target_img.array
        mask = read_mha(msk_path).array if msk_path is not None else None

        row: dict[str, object] = {
            "sample_index": idx,
            "case_id": case_id,
            "beam_idx": beam_idx,
            "cp_idx": cp_idx,
            "full_mode": item.get("full_mode", ""),
            "prediction": str(pred_path),
            "target": str(tgt_path),
            "mask": "" if msk_path is None else str(msk_path),
            "prediction_seconds": item.get("prediction_seconds", ""),
            "write_seconds": item.get("write_seconds", ""),
            "total_seconds": item.get("total_seconds", ""),
        }
        row.update(
            metrics(
                pred,
                target,
                mask,
                spacing=spacing_from_meta(target_img.meta),
                gamma_cutoff_percent=args.gamma_cutoff_percent,
                skip_gamma=args.skip_gamma,
            )
        )
        rows.append(row)

        if args.print_every > 0 and (idx + 1) % args.print_every == 0:
            print(
                f"evaluated={idx + 1}",
                f"case={case_id}",
                f"B{beam_idx}",
                f"CP{cp_idx:03d}",
                f"mae={float(row['mae']):.6g}",
                f"rel_mae={float(row['relative_mae_percent']):.3g}%",
                f"masked_mae={float(row.get('masked_mae', 0.0)):.6g}",
                flush=True,
            )

    if not rows:
        raise RuntimeError(f"No rows evaluated from manifest: {args.manifest_csv}")

    output_dir = Path(args.output_dir)
    per_sample_csv = output_dir / "exported_prediction_metrics.csv"
    summary_csv = output_dir / "exported_prediction_summary.csv"
    fieldnames = [
        "sample_index",
        "case_id",
        "beam_idx",
        "cp_idx",
        "full_mode",
        "prediction",
        "target",
        "mask",
        "prediction_seconds",
        "write_seconds",
        "total_seconds",
        "mae",
        "rmse",
        "mean_error",
        "max_abs_error",
        "target_mean",
        "pred_mean",
        "target_max",
        "pred_max",
        "relative_mae_percent",
        "relative_rmse_percent",
        "relative_max_abs_error_percent",
        "hd_10pct_voxels",
        "hd_10pct_mae",
        "hd_10pct_rmse",
        "hd_50pct_voxels",
        "hd_50pct_mae",
        "hd_50pct_rmse",
        "gamma_3pct_3mm_cutoff10pct_pass_rate",
        "gamma_2pct_2mm_cutoff10pct_pass_rate",
        "mask_positive_voxels",
        "masked_mae",
        "masked_rmse",
        "masked_max_abs_error",
        "masked_relative_mae_percent",
        "masked_relative_rmse_percent",
    ]
    write_csv(per_sample_csv, rows, fieldnames)
    write_summary(summary_csv, rows)

    print(f"wrote_per_sample={per_sample_csv}", flush=True)
    print(f"wrote_summary={summary_csv}", flush=True)
    print(f"samples={len(rows)}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--training-dir", default="data/photon/training")
    parser.add_argument("--mask-name", default="dose_gt_1pct")
    parser.add_argument("--output-dir", default="outputs/exported_prediction_evaluation")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--gamma-cutoff-percent", type=float, default=10.0)
    parser.add_argument("--skip-gamma", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate_manifest(parse_args())
