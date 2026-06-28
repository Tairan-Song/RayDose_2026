"""Preprocess one photon control-point sample for baseline model development.

This smoke-test script reads:

- CT image
- one per-control-point dose file
- optional dose-support mask
- beam/control-point parameters from the per-case JSON

It writes a compressed `.npz` with fixed-size arrays for quick Dataset/model
debugging. It is intentionally single-sample only.
"""

from __future__ import annotations

import argparse
import json
from math import cos, radians, sin
from pathlib import Path

import numpy as np

from mha_io import read_mha


def normalize_ct_hu(ct: np.ndarray, hu_min: float, hu_max: float) -> np.ndarray:
    ct = np.clip(ct.astype(np.float32), hu_min, hu_max)
    return 2.0 * (ct - hu_min) / (hu_max - hu_min) - 1.0


def load_hu_to_density_table(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = data["hu_to_density"]["entries"]
    hu = np.asarray([entry["hu"] for entry in entries], dtype=np.float32)
    density = np.asarray([entry["density_g_cm3"] for entry in entries], dtype=np.float32)
    order = np.argsort(hu)
    return hu[order], density[order]


def ct_to_density(ct: np.ndarray, hu: np.ndarray, density: np.ndarray) -> np.ndarray:
    return np.interp(ct.astype(np.float32), hu, density).astype(np.float32)


def normalize_density(density: np.ndarray, max_density: float = 4.0) -> np.ndarray:
    density = np.clip(density.astype(np.float32), 0.0, max_density)
    return density / max_density


def preprocess_ct(
    ct: np.ndarray,
    ct_mode: str,
    hu_min: float,
    hu_max: float,
    hu_density_table: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    if ct_mode == "hu":
        return normalize_ct_hu(ct, hu_min, hu_max)
    if ct_mode == "density":
        if hu_density_table is None:
            raise ValueError("hu_density_table is required when ct_mode='density'")
        return normalize_density(ct_to_density(ct, *hu_density_table))
    raise ValueError(f"Unsupported ct_mode: {ct_mode!r}")


def normalize_dose(dose: np.ndarray) -> tuple[np.ndarray, float]:
    dose = dose.astype(np.float32)
    dose_max = float(dose.max())
    if dose_max <= 0:
        return dose, dose_max
    return dose / dose_max, dose_max


def scale_dose(dose: np.ndarray, mode: str, global_scale: float) -> tuple[np.ndarray, float, float]:
    dose = dose.astype(np.float32)
    dose_max = float(dose.max())
    if mode == "sample_max":
        scale = dose_max if dose_max > 0 else 1.0
    elif mode == "global":
        if global_scale <= 0:
            raise ValueError("--global-dose-scale must be positive when dose mode is global")
        scale = global_scale
    elif mode == "raw":
        scale = 1.0
    else:
        raise ValueError(f"Unsupported dose mode: {mode!r}")
    return dose / scale, dose_max, float(scale)


def parse_int_tuple(text: str) -> tuple[int, int, int]:
    values = tuple(int(v) for v in text.replace(",", " ").split())
    if len(values) != 3:
        raise ValueError(f"Expected three integers, got {text!r}")
    return values


def voxel_index_from_physical(meta: dict[str, str], point_mm: list[float]) -> tuple[int, int, int] | None:
    try:
        offset = np.array([float(v) for v in meta["Offset"].split()], dtype=np.float32)
        spacing = np.array([float(v) for v in meta["ElementSpacing"].split()], dtype=np.float32)
        transform = np.array([float(v) for v in meta["TransformMatrix"].split()], dtype=np.float32).reshape(3, 3)
        if not np.allclose(transform, np.eye(3), atol=1e-5):
            return None
        index = np.rint((np.array(point_mm, dtype=np.float32) - offset) / spacing).astype(int)
        return tuple(int(v) for v in index)
    except Exception:
        return None


def crop_or_pad(array: np.ndarray, center: tuple[int, int, int], target_shape: tuple[int, int, int], pad_value: float) -> np.ndarray:
    output = np.full(target_shape, pad_value, dtype=array.dtype)

    src_slices = []
    dst_slices = []
    for axis, target in enumerate(target_shape):
        start = center[axis] - target // 2
        end = start + target

        src_start = max(start, 0)
        src_end = min(end, array.shape[axis])
        dst_start = src_start - start
        dst_end = dst_start + (src_end - src_start)

        src_slices.append(slice(src_start, src_end))
        dst_slices.append(slice(dst_start, dst_end))

    output[tuple(dst_slices)] = array[tuple(src_slices)]
    return output


def get_control_point(case_json: Path, beam_idx: int, cp_idx: int) -> tuple[dict, dict]:
    data = json.loads(case_json.read_text(encoding="utf-8"))
    for beam in data["beams"]:
        if int(beam["beam_idx"]) != beam_idx:
            continue
        for cp in beam["control_points"]:
            if int(cp["cp_idx"]) == cp_idx:
                return beam, cp
    raise ValueError(f"Could not find B{beam_idx} CP{cp_idx:03d} in {case_json}")


def preprocess(args: argparse.Namespace) -> None:
    training_dir = Path(args.training_dir)
    case_dir = training_dir / args.case_id
    case_json = case_dir / f"{args.case_id}.json"
    dose_path = case_dir / "dose" / f"Dose_B{args.beam_idx}_CP{args.cp_idx:03d}.mha"
    mask_path = case_dir / "label_masks" / args.mask_name / f"Dose_B{args.beam_idx}_CP{args.cp_idx:03d}_mask.mha"

    beam, cp = get_control_point(case_json, args.beam_idx, args.cp_idx)

    ct_img = read_mha(case_dir / "image" / "ct.mha")
    dose_img = read_mha(dose_path)
    mask_img = read_mha(mask_path)

    density_table = None
    if args.ct_mode == "density":
        density_table = load_hu_to_density_table(training_dir / "beam_parameters.json")

    ct = preprocess_ct(ct_img.array, args.ct_mode, args.hu_min, args.hu_max, density_table)
    dose, dose_max, dose_scale = scale_dose(dose_img.array, args.dose_mode, args.global_dose_scale)
    mask = (mask_img.array > 0).astype(np.float32)

    target_shape = parse_int_tuple(args.target_shape)
    center = voxel_index_from_physical(ct_img.meta, beam["iso_center"])
    center_source = "isocenter"
    if center is None:
        center = tuple(int(v // 2) for v in ct.shape)
        center_source = "volume_center"

    ct_crop = crop_or_pad(ct, center, target_shape, pad_value=-1.0)
    dose_crop = crop_or_pad(dose, center, target_shape, pad_value=0.0)
    mask_crop = crop_or_pad(mask, center, target_shape, pad_value=0.0)

    gantry_rad = radians(float(cp["gantry_angle"]))
    beam_vector = np.array(
        [
            float(args.beam_idx),
            float(args.cp_idx),
            sin(gantry_rad),
            cos(gantry_rad),
            *[float(v) for v in beam["iso_center"]],
        ],
        dtype=np.float32,
    )

    output_path = Path(args.output_npz)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        ct=ct_crop.astype(np.float32),
        dose=dose_crop.astype(np.float32),
        loss_mask=mask_crop.astype(np.float32),
        beam_vector=beam_vector,
        mlc_left=np.asarray(cp["mlc_left_int_mm"], dtype=np.float32),
        mlc_right=np.asarray(cp["mlc_right_int_mm"], dtype=np.float32),
        dose_max=np.asarray(dose_max, dtype=np.float32),
        dose_scale=np.asarray(dose_scale, dtype=np.float32),
        crop_center=np.asarray(center, dtype=np.int32),
        center_source=np.asarray(center_source),
        ct_mode=np.asarray(args.ct_mode),
    )

    print(
        "saved",
        output_path,
        "ct_shape",
        ct_crop.shape,
        "dose_shape",
        dose_crop.shape,
        "mask_shape",
        mask_crop.shape,
        "dose_max",
        dose_max,
        "dose_scale",
        dose_scale,
        "center",
        center,
        center_source,
        "ct_mode",
        args.ct_mode,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", default="data/photon/training")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--beam-idx", type=int, required=True)
    parser.add_argument("--cp-idx", type=int, required=True)
    parser.add_argument("--mask-name", default="dose_gt_1pct")
    parser.add_argument("--target-shape", default="128 128 128")
    parser.add_argument("--ct-mode", choices=("hu", "density"), default="hu")
    parser.add_argument("--dose-mode", choices=("sample_max", "global", "raw"), default="sample_max")
    parser.add_argument("--global-dose-scale", type=float, default=1.5e-4)
    parser.add_argument("--hu-min", type=float, default=-1000.0)
    parser.add_argument("--hu-max", type=float, default=3000.0)
    parser.add_argument("--output-npz", default="outputs/preprocessing_smoke/sample.npz")
    return parser.parse_args()


if __name__ == "__main__":
    preprocess(parse_args())
