"""Run full-volume MHA inference for multiple control-point samples."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from doserad_dataset import DoseRadControlPointDataset, condition_dim
from mha_io import read_mha, write_float_mha
from model_3d_unet import GeometryConditionedUNet3D
from preprocess_training_sample import load_hu_to_density_table, preprocess_ct
from predict_3d_unet import insert_crop, load_checkpoint


def strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if state_dict and all(key.startswith("module.") for key in state_dict):
        return {key.removeprefix("module."): value for key, value in state_dict.items()}
    return state_dict


def load_model(checkpoint: dict, include_energy: bool, device: torch.device) -> GeometryConditionedUNet3D:
    ckpt_args = checkpoint["args"]
    model = GeometryConditionedUNet3D(
        condition_dim=condition_dim(include_energy=include_energy),
        base_channels=int(ckpt_args.get("base_channels", 8)),
    ).to(device)
    model.load_state_dict(strip_module_prefix(checkpoint["model_state_dict"]))
    model.eval()
    return model


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "sample_index",
        "case_id",
        "beam_idx",
        "cp_idx",
        "crop_mha",
        "full_mha",
        "npz",
        "crop_shape",
        "full_shape",
        "full_mode",
        "prediction_seconds",
        "write_seconds",
        "total_seconds",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def window_starts(full_dim: int, window_dim: int, stride: int) -> list[int]:
    if full_dim <= window_dim:
        return [0]
    starts = list(range(0, full_dim - window_dim + 1, stride))
    last = full_dim - window_dim
    if starts[-1] != last:
        starts.append(last)
    return starts


def extract_window(array: np.ndarray, start: tuple[int, int, int], shape: tuple[int, int, int], pad_value: float) -> np.ndarray:
    output = np.full(shape, pad_value, dtype=array.dtype)
    src_slices = []
    dst_slices = []
    for axis in range(3):
        src_start = max(int(start[axis]), 0)
        src_end = min(int(start[axis]) + shape[axis], array.shape[axis])
        dst_start = src_start - int(start[axis])
        dst_end = dst_start + (src_end - src_start)
        src_slices.append(slice(src_start, src_end))
        dst_slices.append(slice(dst_start, dst_end))
    output[tuple(dst_slices)] = array[tuple(src_slices)]
    return output


def add_window(target: np.ndarray, weights: np.ndarray, window: np.ndarray, start: tuple[int, int, int]) -> None:
    dst_slices = []
    src_slices = []
    for axis in range(3):
        dst_start = max(int(start[axis]), 0)
        dst_end = min(int(start[axis]) + window.shape[axis], target.shape[axis])
        src_start = dst_start - int(start[axis])
        src_end = src_start + (dst_end - dst_start)
        dst_slices.append(slice(dst_start, dst_end))
        src_slices.append(slice(src_start, src_end))
    target[tuple(dst_slices)] += window[tuple(src_slices)]
    weights[tuple(dst_slices)] += 1.0


def preprocess_full_ct(training_dir: str | Path, case_id: str, ct_mode: str) -> tuple[np.ndarray, dict[str, str]]:
    ct_img = read_mha(Path(training_dir) / case_id / "image" / "ct.mha")
    density_table = None
    if ct_mode == "density":
        density_table = load_hu_to_density_table(Path(training_dir) / "beam_parameters.json")
    return preprocess_ct(ct_img.array, ct_mode, -1000.0, 3000.0, density_table), ct_img.meta


def sliding_window_full_prediction(
    model: torch.nn.Module,
    full_ct: np.ndarray,
    condition: torch.Tensor,
    dose_scale: float,
    target_shape: tuple[int, int, int],
    device: torch.device,
    stride_fraction: float,
    max_windows: int,
) -> np.ndarray:
    if stride_fraction <= 0:
        raise ValueError("--sliding-stride-fraction must be positive")
    stride = tuple(max(1, int(dim * stride_fraction)) for dim in target_shape)
    starts = [
        (x, y, z)
        for x in window_starts(full_ct.shape[0], target_shape[0], stride[0])
        for y in window_starts(full_ct.shape[1], target_shape[1], stride[1])
        for z in window_starts(full_ct.shape[2], target_shape[2], stride[2])
    ]
    if max_windows > 0:
        starts = starts[:max_windows]

    full_sum = np.zeros(full_ct.shape, dtype=np.float32)
    full_weight = np.zeros(full_ct.shape, dtype=np.float32)
    condition_batch = condition[None].to(device)

    with torch.no_grad():
        for start in starts:
            ct_window = extract_window(full_ct, start, target_shape, pad_value=-1.0)
            ct_tensor = torch.from_numpy(ct_window[None, None].astype(np.float32)).to(device)
            pred = model(ct_tensor, condition_batch)[0, 0].detach().cpu().numpy().astype(np.float32)
            add_window(full_sum, full_weight, np.clip(pred * dose_scale, 0.0, None), start)

    return np.divide(full_sum, np.maximum(full_weight, 1.0), out=np.zeros_like(full_sum), where=full_weight > 0)


def predict_sample(
    sample: dict,
    model: torch.nn.Module,
    device: torch.device,
    training_dir: str | Path,
    output_dir: Path,
    sample_index: int,
    save_npz: bool,
    filename_style: str,
    full_mode: str,
    ckpt_args: dict,
    stride_fraction: float,
    max_sliding_windows: int,
) -> dict[str, object]:
    total_start = time.perf_counter()
    prediction_start = time.perf_counter()
    with torch.no_grad():
        ct = sample["ct"][None].to(device)
        condition = sample["condition"][None].to(device)
        pred_norm = model(ct, condition)[0, 0].detach().cpu().numpy().astype(np.float32)

    dose_scale = float(sample["dose_scale"])
    pred_abs_crop = np.clip(pred_norm * dose_scale, 0.0, None)
    crop_start = sample["crop_start"].numpy()
    original_shape = tuple(int(v) for v in sample["original_shape"].numpy())

    case_id = str(sample["case_id"])
    beam_idx = int(sample["beam_idx"])
    cp_idx = int(sample["cp_idx"])
    ct_img = read_mha(Path(training_dir) / case_id / "image" / "ct.mha")

    if full_mode == "crop_insert":
        full_pred = insert_crop(original_shape, pred_abs_crop, crop_start)
    elif full_mode == "sliding":
        full_ct, _ct_meta = preprocess_full_ct(training_dir, case_id, ckpt_args.get("ct_mode", "hu"))
        full_pred = sliding_window_full_prediction(
            model=model,
            full_ct=full_ct,
            condition=sample["condition"],
            dose_scale=dose_scale,
            target_shape=tuple(int(v) for v in pred_abs_crop.shape),
            device=device,
            stride_fraction=stride_fraction,
            max_windows=max_sliding_windows,
        )
    else:
        raise ValueError(f"Unsupported full mode: {full_mode}")
    prediction_seconds = time.perf_counter() - prediction_start

    write_start = time.perf_counter()
    sample_dir = output_dir / case_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{case_id}_B{beam_idx}_CP{cp_idx:03d}"
    if filename_style == "dose":
        crop_path = sample_dir / f"Dose_B{beam_idx}_CP{cp_idx:03d}_crop.mha"
        full_path = sample_dir / f"Dose_B{beam_idx}_CP{cp_idx:03d}.mha"
        npz_path = sample_dir / f"Dose_B{beam_idx}_CP{cp_idx:03d}.npz"
    elif filename_style == "pred":
        crop_path = sample_dir / f"{stem}_pred_crop.mha"
        full_path = sample_dir / f"{stem}_pred_full.mha"
        npz_path = sample_dir / f"{stem}_pred.npz"
    else:
        raise ValueError(f"Unsupported filename style: {filename_style}")

    crop_meta = dict(ct_img.meta)
    crop_meta["Offset"] = " ".join(f"{float(v):.6g}" for v in sample["crop_offset"].numpy())
    crop_meta["DimSize"] = " ".join(str(int(v)) for v in pred_abs_crop.shape)
    write_float_mha(
        crop_path,
        pred_abs_crop,
        crop_meta,
        offset=sample["crop_offset"].numpy(),
        dim_size=tuple(int(v) for v in pred_abs_crop.shape),
    )
    write_float_mha(full_path, full_pred, ct_img.meta, dim_size=original_shape)

    if save_npz:
        np.savez_compressed(
            npz_path,
            pred_crop=pred_abs_crop,
            pred_full=full_pred,
            crop_start=crop_start,
            dose_scale=np.asarray(dose_scale, dtype=np.float32),
            case_id=np.asarray(case_id),
            beam_idx=np.asarray(beam_idx),
            cp_idx=np.asarray(cp_idx),
        )
        npz_value = str(npz_path)
    else:
        npz_value = ""
    write_seconds = time.perf_counter() - write_start
    total_seconds = time.perf_counter() - total_start

    return {
        "sample_index": sample_index,
        "case_id": case_id,
        "beam_idx": beam_idx,
        "cp_idx": cp_idx,
        "crop_mha": str(crop_path),
        "full_mha": str(full_path),
        "npz": npz_value,
        "crop_shape": " ".join(str(int(v)) for v in pred_abs_crop.shape),
        "full_shape": " ".join(str(int(v)) for v in original_shape),
        "full_mode": full_mode,
        "prediction_seconds": f"{prediction_seconds:.6f}",
        "write_seconds": f"{write_seconds:.6f}",
        "total_seconds": f"{total_seconds:.6f}",
    }


def predict_batch(args: argparse.Namespace) -> None:
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
        sample_strategy=args.sample_strategy,
        sample_seed=args.sample_seed,
        ct_mode=ckpt_args.get("ct_mode", "hu"),
        include_energy=include_energy,
        dose_mode=ckpt_args.get("dose_mode", "global"),
        global_dose_scale=float(ckpt_args.get("global_dose_scale", 1.5e-4)),
        ct_cache_size=args.ct_cache_size,
    )
    model = load_model(checkpoint, include_energy, device)
    loader_kwargs = {
        "batch_size": None,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": False,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    loader = DataLoader(dataset, **loader_kwargs)

    output_dir = Path(args.output_dir)
    rows: list[dict[str, object]] = []
    for idx, sample in enumerate(loader):
        row = predict_sample(
            sample,
            model,
            device,
            training_dir,
            output_dir,
            idx,
            save_npz=not args.no_npz,
            filename_style=args.filename_style,
            full_mode=args.full_mode,
            ckpt_args=ckpt_args,
            stride_fraction=args.sliding_stride_fraction,
            max_sliding_windows=args.max_sliding_windows,
        )
        rows.append(row)
        if args.print_every > 0 and (idx + 1) % args.print_every == 0:
            print(
                f"predicted={idx + 1}",
                f"case={row['case_id']}",
                f"B{row['beam_idx']}",
                f"CP{int(row['cp_idx']):03d}",
                flush=True,
            )

    manifest = output_dir / "prediction_manifest.csv"
    write_manifest(manifest, rows)
    print(f"wrote_manifest={manifest}", flush=True)
    print(f"samples={len(rows)} device={device}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--training-dir", default="")
    parser.add_argument("--split-csv", default="")
    parser.add_argument("--split", default="val")
    parser.add_argument("--output-dir", default="outputs/baseline_3d_unet_batch_inference")
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--sample-strategy", choices=("uniform", "random", "first"), default="uniform")
    parser.add_argument("--sample-seed", type=int, default=20260628)
    parser.add_argument("--filename-style", choices=("pred", "dose"), default="pred")
    parser.add_argument("--full-mode", choices=("crop_insert", "sliding"), default="crop_insert")
    parser.add_argument("--sliding-stride-fraction", type=float, default=0.5)
    parser.add_argument("--max-sliding-windows", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--ct-cache-size", type=int, default=4)
    parser.add_argument("--no-npz", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    predict_batch(parse_args())
