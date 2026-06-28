"""Evaluate one predicted dose MHA against the corresponding ground-truth dose."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from mha_io import read_mha


def metrics(pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape}, target={target.shape}")

    diff = pred.astype(np.float32) - target.astype(np.float32)
    abs_diff = np.abs(diff)
    out = {
        "mae": float(abs_diff.mean()),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "max_abs_error": float(abs_diff.max()),
        "target_max": float(target.max()),
        "pred_max": float(pred.max()),
    }

    if mask is not None:
        if mask.shape != target.shape:
            raise ValueError(f"Mask shape mismatch: mask={mask.shape}, target={target.shape}")
        mask_bool = mask > 0
        positive = int(mask_bool.sum())
        out["mask_positive_voxels"] = float(positive)
        if positive > 0:
            masked_diff = diff[mask_bool]
            masked_abs = np.abs(masked_diff)
            out["masked_mae"] = float(masked_abs.mean())
            out["masked_rmse"] = float(np.sqrt(np.mean(masked_diff * masked_diff)))
            out["masked_max_abs_error"] = float(masked_abs.max())
        else:
            out["masked_mae"] = 0.0
            out["masked_rmse"] = 0.0
            out["masked_max_abs_error"] = 0.0
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pred_path = Path(args.prediction)
    target_path = Path(args.target)
    mask_path = Path(args.mask) if args.mask else None

    pred = read_mha(pred_path).array
    target = read_mha(target_path).array
    mask = read_mha(mask_path).array if mask_path is not None else None

    row: dict[str, str | float] = {
        "prediction": str(pred_path),
        "target": str(target_path),
        "mask": "" if mask_path is None else str(mask_path),
    }
    row.update(metrics(pred, target, mask))
    write_row(Path(args.output_csv), row)
    print("saved", args.output_csv)
    for key, value in row.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
