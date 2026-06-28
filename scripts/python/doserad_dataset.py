"""PyTorch Dataset for per-control-point DoseRAD photon dose prediction."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from mha_io import read_mha
from preprocess_training_sample import (
    crop_or_pad,
    normalize_ct_hu,
    normalize_dose,
    parse_int_tuple,
    voxel_index_from_physical,
)


DOSE_NAME_RE = re.compile(r"Dose_B(\d+)_CP(\d+)\.mha$")
CONDITION_DIM = 167


def parse_dose_name(path: Path) -> tuple[int, int]:
    match = DOSE_NAME_RE.match(path.name)
    if not match:
        raise ValueError(f"Invalid dose filename: {path.name}")
    return int(match.group(1)), int(match.group(2))


def read_split_cases(split_csv: Path, split: str) -> list[str]:
    with split_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return sorted(row["case_id"] for row in rows if row["split"] == split)


def load_case_json(case_dir: Path) -> dict:
    return json.loads((case_dir / f"{case_dir.name}.json").read_text(encoding="utf-8"))


def find_control_point(case_data: dict, beam_idx: int, cp_idx: int) -> tuple[dict, dict]:
    for beam in case_data["beams"]:
        if int(beam["beam_idx"]) != beam_idx:
            continue
        for cp in beam["control_points"]:
            if int(cp["cp_idx"]) == cp_idx:
                return beam, cp
    raise ValueError(f"Could not find B{beam_idx} CP{cp_idx:03d}")


def geometry_condition_vector(beam: dict, cp: dict, beam_idx: int, cp_idx: int) -> np.ndarray:
    gantry = np.deg2rad(float(cp["gantry_angle"]))
    vector = [
        beam_idx / 2.0,
        cp_idx / 179.0,
        float(np.sin(gantry)),
        float(np.cos(gantry)),
        *[float(v) / 300.0 for v in beam["iso_center"]],
    ]
    vector.extend(float(v) / 200.0 for v in cp["mlc_left_int_mm"])
    vector.extend(float(v) / 200.0 for v in cp["mlc_right_int_mm"])
    out = np.asarray(vector, dtype=np.float32)
    if out.shape[0] != CONDITION_DIM:
        raise ValueError(f"Expected condition dim {CONDITION_DIM}, got {out.shape[0]}")
    return out


class DoseRadControlPointDataset(Dataset):
    """Return one sample per case/beam/control-point dose file."""

    def __init__(
        self,
        training_dir: str | Path,
        split_csv: str | Path,
        split: str,
        target_shape: str | tuple[int, int, int] = "128 128 128",
        mask_name: str = "dose_gt_1pct",
        max_samples: int = 0,
        hu_min: float = -1000.0,
        hu_max: float = 3000.0,
    ) -> None:
        self.training_dir = Path(training_dir)
        self.split_csv = Path(split_csv)
        self.split = split
        self.target_shape = parse_int_tuple(target_shape) if isinstance(target_shape, str) else target_shape
        self.mask_name = mask_name
        self.hu_min = hu_min
        self.hu_max = hu_max

        case_ids = read_split_cases(self.split_csv, split)
        samples: list[tuple[str, Path, int, int]] = []
        for case_id in case_ids:
            case_dir = self.training_dir / case_id
            for dose_path in sorted((case_dir / "dose").glob("Dose_B*_CP*.mha")):
                beam_idx, cp_idx = parse_dose_name(dose_path)
                samples.append((case_id, dose_path, beam_idx, cp_idx))

        self.samples = samples[:max_samples] if max_samples > 0 else samples
        if not self.samples:
            raise RuntimeError(f"No samples found for split={split!r}")

        self._case_json_cache: dict[str, dict] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def _case_data(self, case_id: str) -> dict:
        if case_id not in self._case_json_cache:
            self._case_json_cache[case_id] = load_case_json(self.training_dir / case_id)
        return self._case_json_cache[case_id]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        case_id, dose_path, beam_idx, cp_idx = self.samples[index]
        case_dir = self.training_dir / case_id
        beam, cp = find_control_point(self._case_data(case_id), beam_idx, cp_idx)

        ct_img = read_mha(case_dir / "image" / "ct.mha")
        dose_img = read_mha(dose_path)
        mask_img = read_mha(case_dir / "label_masks" / self.mask_name / f"Dose_B{beam_idx}_CP{cp_idx:03d}_mask.mha")

        center = voxel_index_from_physical(ct_img.meta, beam["iso_center"])
        if center is None:
            center = tuple(int(v // 2) for v in ct_img.array.shape)

        ct = normalize_ct_hu(ct_img.array, self.hu_min, self.hu_max)
        dose, dose_max = normalize_dose(dose_img.array)
        mask = (mask_img.array > 0).astype(np.float32)

        ct = crop_or_pad(ct, center, self.target_shape, pad_value=-1.0)
        dose = crop_or_pad(dose, center, self.target_shape, pad_value=0.0)
        mask = crop_or_pad(mask, center, self.target_shape, pad_value=0.0)
        condition = geometry_condition_vector(beam, cp, beam_idx, cp_idx)

        return {
            "ct": torch.from_numpy(ct[None].astype(np.float32)),
            "dose": torch.from_numpy(dose[None].astype(np.float32)),
            "loss_mask": torch.from_numpy(mask[None].astype(np.float32)),
            "condition": torch.from_numpy(condition),
            "dose_max": torch.tensor(dose_max, dtype=torch.float32),
            "case_id": case_id,
            "beam_idx": beam_idx,
            "cp_idx": cp_idx,
        }
