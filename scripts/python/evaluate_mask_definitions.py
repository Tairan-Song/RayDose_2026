"""Evaluate alternative dose-mask definitions for DoseRAD photon data.

This script is for analysis, not for model training directly. Use it to compare
threshold, dilation, and gradient masks before choosing a training label/loss.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Callable

import numpy as np

try:
    from scipy.ndimage import binary_dilation
except ImportError:  # pragma: no cover - handled at runtime
    binary_dilation = None

from mha_io import read_mha, write_mask_mha


MaskFn = Callable[[np.ndarray], tuple[np.ndarray, float, int, str]]


def list_dose_files(training_dir: Path) -> list[Path]:
    return sorted(training_dir.glob("*/dose/Dose_B*_CP*.mha"))


def select_files(files: list[Path], max_files: int, sample_mode: str) -> list[Path]:
    if max_files <= 0 or max_files >= len(files):
        return files
    if sample_mode == "first":
        return files[:max_files]
    indices = np.linspace(0, len(files) - 1, max_files).round().astype(int)
    return [files[i] for i in sorted(set(indices.tolist()))]


def parse_case_beam_cp(path: Path) -> tuple[str, int, int]:
    case_id = path.parents[1].name
    stem = path.stem
    # Dose_B0_CP133
    beam_part, cp_part = stem.replace("Dose_", "").split("_")
    return case_id, int(beam_part[1:]), int(cp_part[2:])


def threshold_mask(frac: float, name: str) -> tuple[str, MaskFn]:
    def make(dose: np.ndarray) -> tuple[np.ndarray, float, int, str]:
        threshold = 0.0 if frac == 0 else float(dose.max()) * frac
        mask = dose > threshold
        return mask, threshold, 0, f"mask = dose > {frac:g} * max(dose)"

    return name, make


def dilation_mask(radius: int) -> tuple[str, MaskFn]:
    def make(dose: np.ndarray) -> tuple[np.ndarray, float, int, str]:
        if binary_dilation is None:
            raise RuntimeError("scipy is required for dilation masks")
        threshold = float(dose.max()) * 0.01
        base = dose > threshold
        structure = np.ones((2 * radius + 1, 2 * radius + 1, 2 * radius + 1), dtype=bool)
        mask = binary_dilation(base, structure=structure)
        return mask, threshold, radius, f"1% mask cube dilation by {radius} voxel(s)"

    return f"dose_gt_1pct_dilate{radius}", make


def gradient_mask(top_fraction: float, name: str) -> tuple[str, MaskFn]:
    def make(dose: np.ndarray) -> tuple[np.ndarray, float, int, str]:
        gx, gy, gz = np.gradient(dose.astype(np.float32))
        grad = np.sqrt(gx * gx + gy * gy + gz * gz)
        nonzero = grad[grad > 0]
        if nonzero.size == 0:
            return np.zeros_like(dose, dtype=bool), 0.0, 0, "empty gradient"
        threshold = float(np.quantile(nonzero, 1.0 - top_fraction))
        mask = grad > threshold
        return mask, threshold, 0, f"top {top_fraction:.0%} nonzero |gradient(dose)| voxels"

    return name, make


def mask_definitions() -> list[tuple[str, MaskFn]]:
    return [
        threshold_mask(0.0, "dose_nonzero"),
        threshold_mask(0.005, "dose_gt_0p5pct"),
        threshold_mask(0.01, "dose_gt_1pct"),
        threshold_mask(0.02, "dose_gt_2pct"),
        threshold_mask(0.05, "dose_gt_5pct"),
        dilation_mask(1),
        dilation_mask(2),
        dilation_mask(3),
        gradient_mask(0.10, "dose_gradient_top10pct"),
        gradient_mask(0.05, "dose_gradient_top5pct"),
    ]


def mask_output_path(root: Path, definition: str, dose_path: Path) -> Path:
    case_id, beam_idx, cp_idx = parse_case_beam_cp(dose_path)
    return root / definition / case_id / f"Dose_B{beam_idx}_CP{cp_idx:03d}_mask.mha"


def evaluate(args: argparse.Namespace) -> None:
    training_dir = Path(args.training_dir)
    output_dir = Path(args.output_dir)
    stats_csv = Path(args.stats_csv)
    stats_csv.parent.mkdir(parents=True, exist_ok=True)

    dose_files = select_files(list_dose_files(training_dir), args.max_files, args.sample_mode)
    definitions = mask_definitions()

    with stats_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_id",
                "beam_idx",
                "cp_idx",
                "definition",
                "dose_path",
                "mask_path",
                "dim_size",
                "element_spacing",
                "dose_max",
                "threshold_value",
                "dilation_voxels",
                "voxels",
                "positive_voxels",
                "positive_percent",
                "status",
                "notes",
            ],
        )
        writer.writeheader()

        for dose_path in dose_files:
            case_id, beam_idx, cp_idx = parse_case_beam_cp(dose_path)
            image = read_mha(dose_path)
            dose = image.array.astype(np.float32, copy=False)
            voxels = int(dose.size)
            dim_size = image.meta.get("DimSize", "")
            spacing = image.meta.get("ElementSpacing", "")

            for definition, make_mask in definitions:
                mask_path = ""
                try:
                    mask, threshold, dilation_voxels, notes = make_mask(dose)
                    positive = int(mask.sum())
                    if args.write_masks:
                        out = mask_output_path(output_dir, definition, dose_path)
                        write_mask_mha(out, mask.astype(np.uint8), image.meta)
                        mask_path = str(out)
                    status = "processed" if args.write_masks else "stats_only"
                except Exception as exc:  # keep batch runs inspectable
                    positive = 0
                    threshold = 0.0
                    dilation_voxels = 0
                    notes = str(exc)
                    status = "error"

                writer.writerow(
                    {
                        "case_id": case_id,
                        "beam_idx": beam_idx,
                        "cp_idx": cp_idx,
                        "definition": definition,
                        "dose_path": str(dose_path),
                        "mask_path": mask_path,
                        "dim_size": dim_size,
                        "element_spacing": spacing,
                        "dose_max": float(dose.max()),
                        "threshold_value": threshold,
                        "dilation_voxels": dilation_voxels,
                        "voxels": voxels,
                        "positive_voxels": positive,
                        "positive_percent": 100.0 * positive / voxels,
                        "status": status,
                        "notes": notes,
                    }
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", default="data/photon/training")
    parser.add_argument("--output-dir", default="data/photon/training/aux_mask_definition_eval_py")
    parser.add_argument("--stats-csv", default="data/photon/training/aux_mask_definition_eval_py/mask_definition_stats.csv")
    parser.add_argument("--max-files", type=int, default=30)
    parser.add_argument("--sample-mode", choices=("first", "even"), default="even")
    parser.add_argument("--write-masks", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
