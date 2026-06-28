"""Estimate a global dose scaling constant from dose-mask statistics CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats-csv", default="data/photon/training/dose_mask_stats_gt_1pct.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.stats_csv)
    values = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            values.append(float(row["dose_max"]))

    arr = np.asarray(values, dtype=np.float64)
    print(f"count={arr.size}")
    print(f"min={arr.min():.12g}")
    print(f"mean={arr.mean():.12g}")
    print(f"p95={np.quantile(arr, 0.95):.12g}")
    print(f"p99={np.quantile(arr, 0.99):.12g}")
    print(f"max={arr.max():.12g}")
    print(f"recommended_global_dose_scale={arr.max():.12g}")


if __name__ == "__main__":
    main()
