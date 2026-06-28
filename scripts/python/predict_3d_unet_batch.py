"""Run full-volume MHA inference for multiple control-point samples."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from doserad_dataset import DoseRadControlPointDataset, condition_dim
from mha_io import read_mha, write_float_mha
from model_3d_unet import GeometryConditionedUNet3D
from predict_3d_unet import insert_crop, load_checkpoint


def load_model(checkpoint: dict, include_energy: bool, device: torch.device) -> GeometryConditionedUNet3D:
    ckpt_args = checkpoint["args"]
    model = GeometryConditionedUNet3D(
        condition_dim=condition_dim(include_energy=include_energy),
        base_channels=int(ckpt_args.get("base_channels", 8)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
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
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def predict_sample(
    sample: dict,
    model: torch.nn.Module,
    device: torch.device,
    training_dir: str | Path,
    output_dir: Path,
    sample_index: int,
    save_npz: bool,
) -> dict[str, object]:
    with torch.no_grad():
        ct = sample["ct"][None].to(device)
        condition = sample["condition"][None].to(device)
        pred_norm = model(ct, condition)[0, 0].detach().cpu().numpy().astype(np.float32)

    dose_scale = float(sample["dose_scale"])
    pred_abs_crop = pred_norm * dose_scale
    crop_start = sample["crop_start"].numpy()
    original_shape = tuple(int(v) for v in sample["original_shape"].numpy())
    full_pred = insert_crop(original_shape, pred_abs_crop, crop_start)

    case_id = str(sample["case_id"])
    beam_idx = int(sample["beam_idx"])
    cp_idx = int(sample["cp_idx"])
    ct_img = read_mha(Path(training_dir) / case_id / "image" / "ct.mha")

    sample_dir = output_dir / case_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{case_id}_B{beam_idx}_CP{cp_idx:03d}"
    crop_path = sample_dir / f"{stem}_pred_crop.mha"
    full_path = sample_dir / f"{stem}_pred_full.mha"
    npz_path = sample_dir / f"{stem}_pred.npz"

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
        ct_mode=ckpt_args.get("ct_mode", "hu"),
        include_energy=include_energy,
        dose_mode=ckpt_args.get("dose_mode", "global"),
        global_dose_scale=float(ckpt_args.get("global_dose_scale", 1.5e-4)),
    )
    model = load_model(checkpoint, include_energy, device)

    output_dir = Path(args.output_dir)
    rows: list[dict[str, object]] = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        row = predict_sample(sample, model, device, training_dir, output_dir, idx, save_npz=not args.no_npz)
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
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--no-npz", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    predict_batch(parse_args())
