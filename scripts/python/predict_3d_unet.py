"""Run full-volume inference for one sample using a trained 3D U-Net checkpoint.

The model predicts a fixed-size crop. This script writes both:

- crop-level prediction MHA
- full-volume prediction MHA with the crop placed into the original CT grid
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from doserad_dataset import DoseRadControlPointDataset, condition_dim
from mha_io import read_mha, write_float_mha
from model_3d_unet import GeometryConditionedUNet3D


def load_checkpoint(path: str | Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def choose_sample_index(dataset: DoseRadControlPointDataset, case_id: str | None, beam_idx: int | None, cp_idx: int | None) -> int:
    if case_id is None:
        return 0
    if beam_idx is None or cp_idx is None:
        raise ValueError("--beam-idx and --cp-idx are required when --case-id is provided")
    for idx, (sample_case, _dose_path, sample_beam, sample_cp) in enumerate(dataset.samples):
        if sample_case == case_id and sample_beam == beam_idx and sample_cp == cp_idx:
            return idx
    raise ValueError(f"Sample not found: {case_id} B{beam_idx} CP{cp_idx:03d}")


def insert_crop(full_shape: tuple[int, int, int], crop: np.ndarray, start: np.ndarray) -> np.ndarray:
    full = np.zeros(full_shape, dtype=np.float32)
    src_slices = []
    dst_slices = []
    for axis in range(3):
        start_i = int(start[axis])
        end_i = start_i + crop.shape[axis]

        dst_start = max(start_i, 0)
        dst_end = min(end_i, full_shape[axis])
        src_start = dst_start - start_i
        src_end = src_start + (dst_end - dst_start)

        src_slices.append(slice(src_start, src_end))
        dst_slices.append(slice(dst_start, dst_end))

    full[tuple(dst_slices)] = crop[tuple(src_slices)]
    return full


def predict(args: argparse.Namespace) -> None:
    checkpoint = load_checkpoint(args.checkpoint)
    ckpt_args = checkpoint["args"]

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    training_dir = args.training_dir or ckpt_args["training_dir"]
    split_csv = args.split_csv or ckpt_args["split_csv"]
    target_shape = ckpt_args["target_shape"]
    include_energy = bool(ckpt_args.get("include_energy", False))

    dataset = DoseRadControlPointDataset(
        training_dir=training_dir,
        split_csv=split_csv,
        split=args.split,
        target_shape=target_shape,
        mask_name=ckpt_args.get("mask_name", "dose_gt_1pct"),
        max_samples=0,
        ct_mode=ckpt_args.get("ct_mode", "hu"),
        include_energy=include_energy,
        dose_mode=ckpt_args.get("dose_mode", "global"),
        global_dose_scale=float(ckpt_args.get("global_dose_scale", 1.5e-4)),
    )
    sample_idx = choose_sample_index(dataset, args.case_id, args.beam_idx, args.cp_idx)
    sample = dataset[sample_idx]

    model = GeometryConditionedUNet3D(
        condition_dim=condition_dim(include_energy=include_energy),
        base_channels=int(ckpt_args.get("base_channels", 8)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        ct = sample["ct"][None].to(device)
        condition = sample["condition"][None].to(device)
        pred_norm = model(ct, condition)[0, 0].detach().cpu().numpy().astype(np.float32)

    dose_scale = float(sample["dose_scale"])
    pred_abs_crop = np.clip(pred_norm * dose_scale, 0.0, None)
    crop_start = sample["crop_start"].numpy()
    original_shape = tuple(int(v) for v in sample["original_shape"].numpy())
    full_pred = insert_crop(original_shape, pred_abs_crop, crop_start)

    case_id = str(sample["case_id"])
    beam_idx = int(sample["beam_idx"])
    cp_idx = int(sample["cp_idx"])
    ct_img = read_mha(Path(training_dir) / case_id / "image" / "ct.mha")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{case_id}_B{beam_idx}_CP{cp_idx:03d}"

    crop_meta = dict(ct_img.meta)
    crop_meta["Offset"] = " ".join(f"{float(v):.6g}" for v in sample["crop_offset"].numpy())
    crop_meta["DimSize"] = " ".join(str(int(v)) for v in pred_abs_crop.shape)
    write_float_mha(
        output_dir / f"{stem}_pred_crop.mha",
        pred_abs_crop,
        crop_meta,
        offset=sample["crop_offset"].numpy(),
        dim_size=tuple(int(v) for v in pred_abs_crop.shape),
    )
    write_float_mha(
        output_dir / f"{stem}_pred_full.mha",
        full_pred,
        ct_img.meta,
        dim_size=original_shape,
    )
    np.savez_compressed(
        output_dir / f"{stem}_pred.npz",
        pred_crop=pred_abs_crop,
        pred_full=full_pred,
        crop_start=crop_start,
        dose_scale=np.asarray(dose_scale, dtype=np.float32),
        case_id=np.asarray(case_id),
        beam_idx=np.asarray(beam_idx),
        cp_idx=np.asarray(cp_idx),
    )

    print(
        "saved",
        output_dir,
        "case",
        case_id,
        "beam",
        beam_idx,
        "cp",
        cp_idx,
        "crop_shape",
        pred_abs_crop.shape,
        "full_shape",
        full_pred.shape,
        "device",
        device,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--training-dir", default="")
    parser.add_argument("--split-csv", default="")
    parser.add_argument("--split", default="val")
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--beam-idx", type=int, default=None)
    parser.add_argument("--cp-idx", type=int, default=None)
    parser.add_argument("--output-dir", default="outputs/baseline_3d_unet_inference")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    predict(parse_args())
