"""Generate dose-support binary masks from DoseRAD photon dose files.

Main use:
    python scripts/python/generate_dose_support_masks.py

Default label:
    dose_gt_1pct = dose > 0.01 * max(dose)
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from pathlib import Path

import numpy as np

from mha_io import read_mha, write_mask_mha


def list_dose_files(training_dir: Path) -> list[Path]:
    return sorted(training_dir.glob("*/dose/Dose_B*_CP*.mha"))


def select_files(files: list[Path], max_files: int) -> list[Path]:
    if max_files <= 0:
        return files
    return files[:max_files]


def parse_case_beam_cp(path: Path) -> tuple[str, int, int]:
    case_id = path.parents[1].name
    beam_part, cp_part = path.stem.replace("Dose_", "").split("_")
    return case_id, int(beam_part[1:]), int(cp_part[2:])


def output_mask_path(dose_path: Path, label_name: str) -> Path:
    case_dir = dose_path.parents[1]
    return case_dir / "label_masks" / label_name / f"{dose_path.stem}_mask.mha"


def process_dose_file(
    dose_path_str: str,
    label_name: str,
    threshold_fraction: float,
    stats_only: bool,
    force: bool,
) -> dict[str, object]:
    dose_path = Path(dose_path_str)
    case_id, beam_idx, cp_idx = parse_case_beam_cp(dose_path)
    mask_path = output_mask_path(dose_path, label_name)

    try:
        image = read_mha(dose_path)
        dose = image.array.astype(np.float32, copy=False)
        dose_max = float(dose.max())
        threshold = dose_max * threshold_fraction
        mask = dose > threshold
        positive = int(mask.sum())
        voxels = int(dose.size)

        if stats_only:
            status = "stats_only"
            notes = ""
        elif mask_path.exists() and not force:
            status = "skipped_exists"
            notes = "pass --force to overwrite"
        else:
            write_mask_mha(mask_path, mask, image.meta)
            status = "processed"
            notes = ""
    except Exception as exc:
        dose_max = 0.0
        threshold = 0.0
        voxels = 0
        positive = 0
        status = "error"
        notes = str(exc)

    return {
        "case_id": case_id,
        "beam_idx": beam_idx,
        "cp_idx": cp_idx,
        "dose_path": str(dose_path),
        "mask_path": "" if stats_only else str(mask_path),
        "dose_max": dose_max,
        "threshold_fraction": threshold_fraction,
        "threshold_value": threshold,
        "voxels": voxels,
        "positive_voxels": positive,
        "positive_percent": 0.0 if voxels == 0 else 100.0 * positive / voxels,
        "status": status,
        "notes": notes,
    }


def generate_masks(args: argparse.Namespace) -> None:
    training_dir = Path(args.training_dir)
    stats_csv = Path(args.stats_csv)
    label_name = args.label_name

    stats_csv.parent.mkdir(parents=True, exist_ok=True)
    dose_files = select_files(list_dose_files(training_dir), args.max_files)

    fields = [
        "case_id",
        "beam_idx",
        "cp_idx",
        "dose_path",
        "mask_path",
        "dose_max",
        "threshold_fraction",
        "threshold_value",
        "voxels",
        "positive_voxels",
        "positive_percent",
        "status",
        "notes",
    ]

    with stats_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        task_args = (
            (
                str(dose_path),
                label_name,
                args.threshold_fraction,
                args.stats_only,
                args.force,
            )
            for dose_path in dose_files
        )

        if args.workers <= 1:
            for item in task_args:
                writer.writerow(process_dose_file(*item))
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                for row in pool.map(process_dose_file_star, task_args, chunksize=args.chunksize):
                    writer.writerow(row)


def process_dose_file_star(args: tuple[str, str, float, bool, bool]) -> dict[str, object]:
    return process_dose_file(*args)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", default="data/photon/training")
    parser.add_argument("--label-name", default="dose_gt_1pct")
    parser.add_argument("--threshold-fraction", type=float, default=0.01)
    parser.add_argument("--stats-csv", default="data/photon/training/dose_mask_stats_gt_1pct_py.csv")
    parser.add_argument("--max-files", type=int, default=0, help="0 means all dose files")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--chunksize", type=int, default=8)
    parser.add_argument("--stats-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    generate_masks(parse_args())
