"""Evaluate one predicted dose MHA against the corresponding ground-truth dose."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from mha_io import read_mha


def safe_percent(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return float("nan")
    return float(100.0 * numerator / denominator)


def spacing_from_meta(meta: dict[str, str]) -> tuple[float, float, float]:
    spacing_text = meta.get("ElementSpacing", "1 1 1")
    values = tuple(float(v) for v in spacing_text.split())
    if len(values) != 3:
        raise ValueError(f"Expected 3D ElementSpacing, got {spacing_text!r}")
    return values


def offset_slices(shape: tuple[int, int, int], offset: tuple[int, int, int]) -> tuple[tuple[slice, ...], tuple[slice, ...]]:
    target_slices = []
    pred_slices = []
    for axis, delta in enumerate(offset):
        if delta >= 0:
            target_start = 0
            target_end = shape[axis] - delta
            pred_start = delta
            pred_end = shape[axis]
        else:
            target_start = -delta
            target_end = shape[axis]
            pred_start = 0
            pred_end = shape[axis] + delta
        target_slices.append(slice(target_start, target_end))
        pred_slices.append(slice(pred_start, pred_end))
    return tuple(target_slices), tuple(pred_slices)


def gamma_pass_rate(
    pred: np.ndarray,
    target: np.ndarray,
    spacing: tuple[float, float, float],
    dose_percent: float,
    distance_mm: float,
    cutoff_percent: float,
) -> float:
    target_max = float(target.max())
    if target_max <= 0.0:
        return float("nan")

    dose_criterion = target_max * dose_percent / 100.0
    cutoff = target_max * cutoff_percent / 100.0
    eval_mask = target >= cutoff
    if not np.any(eval_mask):
        return float("nan")

    pred = pred.astype(np.float32, copy=False)
    target = target.astype(np.float32, copy=False)
    spacing_array = np.asarray(spacing, dtype=np.float32)
    radius_voxels = np.ceil(distance_mm / spacing_array).astype(int)
    gamma_sq_min = np.full(target.shape, np.inf, dtype=np.float32)

    for dx in range(-radius_voxels[0], radius_voxels[0] + 1):
        for dy in range(-radius_voxels[1], radius_voxels[1] + 1):
            for dz in range(-radius_voxels[2], radius_voxels[2] + 1):
                offset = np.asarray((dx, dy, dz), dtype=np.float32)
                distance = float(np.sqrt(np.sum((offset * spacing_array) ** 2)))
                if distance > distance_mm:
                    continue
                target_slices, pred_slices = offset_slices(target.shape, (dx, dy, dz))
                dose_term = (pred[pred_slices] - target[target_slices]) / dose_criterion
                gamma_sq = dose_term * dose_term + (distance / distance_mm) ** 2
                gamma_sq_min[target_slices] = np.minimum(gamma_sq_min[target_slices], gamma_sq)

    return float(np.mean(gamma_sq_min[eval_mask] <= 1.0))


def high_dose_metrics(diff: np.ndarray, abs_diff: np.ndarray, target: np.ndarray, fraction: float) -> dict[str, float]:
    target_max = float(target.max())
    high_mask = target >= target_max * fraction
    prefix = f"hd_{int(fraction * 100)}pct"
    count = int(high_mask.sum())
    if count == 0:
        return {
            f"{prefix}_voxels": 0.0,
            f"{prefix}_mae": float("nan"),
            f"{prefix}_rmse": float("nan"),
        }
    high_diff = diff[high_mask]
    return {
        f"{prefix}_voxels": float(count),
        f"{prefix}_mae": float(abs_diff[high_mask].mean()),
        f"{prefix}_rmse": float(np.sqrt(np.mean(high_diff * high_diff))),
    }


def metrics(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray | None = None,
    spacing: tuple[float, float, float] | None = None,
    gamma_cutoff_percent: float = 10.0,
    skip_gamma: bool = False,
) -> dict[str, float]:
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape}, target={target.shape}")

    diff = pred.astype(np.float32) - target.astype(np.float32)
    abs_diff = np.abs(diff)
    mae = float(abs_diff.mean())
    rmse = float(np.sqrt(np.mean(diff * diff)))
    max_abs_error = float(abs_diff.max())
    target_max = float(target.max())
    out = {
        "mae": mae,
        "rmse": rmse,
        "mean_error": float(diff.mean()),
        "max_abs_error": max_abs_error,
        "target_mean": float(target.mean()),
        "pred_mean": float(pred.mean()),
        "target_max": target_max,
        "pred_max": float(pred.max()),
        "relative_mae_percent": safe_percent(mae, target_max),
        "relative_rmse_percent": safe_percent(rmse, target_max),
        "relative_max_abs_error_percent": safe_percent(max_abs_error, target_max),
    }
    out.update(high_dose_metrics(diff, abs_diff, target, 0.1))
    out.update(high_dose_metrics(diff, abs_diff, target, 0.5))

    if mask is not None:
        if mask.shape != target.shape:
            raise ValueError(f"Mask shape mismatch: mask={mask.shape}, target={target.shape}")
        mask_bool = mask > 0
        positive = int(mask_bool.sum())
        out["mask_positive_voxels"] = float(positive)
        if positive > 0:
            masked_diff = diff[mask_bool]
            masked_abs = np.abs(masked_diff)
            masked_mae = float(masked_abs.mean())
            masked_rmse = float(np.sqrt(np.mean(masked_diff * masked_diff)))
            masked_max_abs_error = float(masked_abs.max())
            out["masked_mae"] = masked_mae
            out["masked_rmse"] = masked_rmse
            out["masked_max_abs_error"] = masked_max_abs_error
            out["masked_relative_mae_percent"] = safe_percent(masked_mae, target_max)
            out["masked_relative_rmse_percent"] = safe_percent(masked_rmse, target_max)
        else:
            out["masked_mae"] = 0.0
            out["masked_rmse"] = 0.0
            out["masked_max_abs_error"] = 0.0
            out["masked_relative_mae_percent"] = float("nan")
            out["masked_relative_rmse_percent"] = float("nan")

    if spacing is not None and not skip_gamma:
        out["gamma_3pct_3mm_cutoff10pct_pass_rate"] = gamma_pass_rate(
            pred,
            target,
            spacing=spacing,
            dose_percent=3.0,
            distance_mm=3.0,
            cutoff_percent=gamma_cutoff_percent,
        )
        out["gamma_2pct_2mm_cutoff10pct_pass_rate"] = gamma_pass_rate(
            pred,
            target,
            spacing=spacing,
            dose_percent=2.0,
            distance_mm=2.0,
            cutoff_percent=gamma_cutoff_percent,
        )
    return out


def write_row(path: Path, row: dict[str, str | float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--mask", default="")
    parser.add_argument("--output-csv", default="outputs/evaluation/prediction_metrics.csv")
    parser.add_argument("--gamma-cutoff-percent", type=float, default=10.0)
    parser.add_argument("--skip-gamma", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pred_path = Path(args.prediction)
    target_path = Path(args.target)
    mask_path = Path(args.mask) if args.mask else None

    pred = read_mha(pred_path).array
    target_img = read_mha(target_path)
    target = target_img.array
    mask = read_mha(mask_path).array if mask_path is not None else None

    row: dict[str, str | float] = {
        "prediction": str(pred_path),
        "target": str(target_path),
        "mask": "" if mask_path is None else str(mask_path),
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
    write_row(Path(args.output_csv), row)
    print("saved", args.output_csv)
    for key, value in row.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
