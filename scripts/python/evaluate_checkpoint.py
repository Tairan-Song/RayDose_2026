"""Evaluate a trained 3D U-Net checkpoint on a dataset split."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from doserad_dataset import DoseRadControlPointDataset, condition_dim
from model_3d_unet import GeometryConditionedUNet3D


def load_checkpoint(path: str | Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def tensor_scalar(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def sample_metrics(
    pred_norm: torch.Tensor,
    dose_norm: torch.Tensor,
    mask: torch.Tensor,
    dose_scale: torch.Tensor,
) -> dict[str, float | int]:
    scale = dose_scale.view(-1, 1, 1, 1, 1)
    pred = pred_norm * scale
    dose = dose_norm * scale
    abs_err = torch.abs(pred - dose)
    sq_err = (pred - dose) ** 2
    mask_bool = mask > 0
    mask_count = int(mask_bool.sum().detach().cpu().item())

    metrics: dict[str, float | int] = {
        "mae": tensor_scalar(abs_err.mean()),
        "rmse": tensor_scalar(torch.sqrt(sq_err.mean())),
        "max_abs_error": tensor_scalar(abs_err.max()),
        "target_max": tensor_scalar(dose.max()),
        "pred_max": tensor_scalar(pred.max()),
        "mask_positive_voxels": mask_count,
    }

    if mask_count > 0:
        metrics.update(
            {
                "masked_mae": tensor_scalar(abs_err[mask_bool].mean()),
                "masked_rmse": tensor_scalar(torch.sqrt(sq_err[mask_bool].mean())),
                "masked_max_abs_error": tensor_scalar(abs_err[mask_bool].max()),
            }
        )
    else:
        metrics.update({"masked_mae": float("nan"), "masked_rmse": float("nan"), "masked_max_abs_error": float("nan")})

    return metrics


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
        "max_abs_error",
        "target_max",
        "pred_max",
        "masked_mae",
        "masked_rmse",
        "masked_max_abs_error",
    ]
    summary_rows: list[dict[str, object]] = []
    for name in metric_names:
        values = np.asarray([float(row[name]) for row in rows], dtype=np.float64)
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


def evaluate(args: argparse.Namespace) -> None:
    checkpoint = load_checkpoint(args.checkpoint)
    ckpt_args = checkpoint["args"]

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    training_dir = args.training_dir or ckpt_args["training_dir"]
    split_csv = args.split_csv or ckpt_args["split_csv"]
    include_energy = bool(ckpt_args.get("include_energy", False))

    dataset = DoseRadControlPointDataset(
        training_dir=training_dir,
        split_csv=split_csv,
        split=args.split,
        target_shape=ckpt_args["target_shape"],
        mask_name=ckpt_args.get("mask_name", "dose_gt_1pct"),
        max_samples=args.max_samples,
        ct_mode=ckpt_args.get("ct_mode", "hu"),
        include_energy=include_energy,
        dose_mode=ckpt_args.get("dose_mode", "global"),
        global_dose_scale=float(ckpt_args.get("global_dose_scale", 1.5e-4)),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    model = GeometryConditionedUNet3D(
        condition_dim=condition_dim(include_energy=include_energy),
        base_channels=int(ckpt_args.get("base_channels", 8)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            ct = batch["ct"].to(device)
            dose = batch["dose"].to(device)
            mask = batch["loss_mask"].to(device)
            condition = batch["condition"].to(device)
            dose_scale = batch["dose_scale"].to(device)

            pred = model(ct, condition)
            metrics = sample_metrics(pred, dose, mask, dose_scale)
            row = {
                "sample_index": idx,
                "case_id": batch["case_id"][0],
                "beam_idx": int(batch["beam_idx"][0]),
                "cp_idx": int(batch["cp_idx"][0]),
                **metrics,
            }
            rows.append(row)

            if args.print_every > 0 and (idx + 1) % args.print_every == 0:
                print(
                    f"evaluated={idx + 1}",
                    f"case={row['case_id']}",
                    f"B{row['beam_idx']}",
                    f"CP{int(row['cp_idx']):03d}",
                    f"mae={float(row['mae']):.6g}",
                    f"masked_mae={float(row['masked_mae']):.6g}",
                    flush=True,
                )

    output_dir = Path(args.output_dir)
    per_sample_csv = output_dir / "per_sample_metrics.csv"
    summary_csv = output_dir / "summary_metrics.csv"
    fieldnames = [
        "sample_index",
        "case_id",
        "beam_idx",
        "cp_idx",
        "mae",
        "rmse",
        "max_abs_error",
        "target_max",
        "pred_max",
        "mask_positive_voxels",
        "masked_mae",
        "masked_rmse",
        "masked_max_abs_error",
    ]
    write_csv(per_sample_csv, rows, fieldnames)
    write_summary(summary_csv, rows)

    print(f"wrote_per_sample={per_sample_csv}", flush=True)
    print(f"wrote_summary={summary_csv}", flush=True)
    print(f"samples={len(rows)} device={device}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--training-dir", default="")
    parser.add_argument("--split-csv", default="")
    parser.add_argument("--split", default="val")
    parser.add_argument("--output-dir", default="outputs/checkpoint_evaluation")
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
