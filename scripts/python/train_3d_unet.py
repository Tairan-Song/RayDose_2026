"""Train the geometry-conditioned 3D U-Net photon dose baseline."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from doserad_dataset import DoseRadControlPointDataset, condition_dim
from mha_io import write_float_mha
from model_3d_unet import GeometryConditionedUNet3D
from preprocess_training_sample import parse_int_tuple


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)


def load_checkpoint(path: str | Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum().clamp_min(1.0)
    return (torch.abs(pred - target) * mask).sum() / denom


def dose_loss(pred: torch.Tensor, dose: torch.Tensor, mask: torch.Tensor, mask_weight: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    global_loss = F.l1_loss(pred, dose)
    focus_loss = masked_l1(pred, dose, mask)
    return global_loss + mask_weight * focus_loss, global_loss, focus_loss


def make_loader(args: argparse.Namespace, split: str, max_samples: int, shuffle: bool) -> DataLoader:
    dataset = DoseRadControlPointDataset(
        training_dir=args.training_dir,
        split_csv=args.split_csv,
        split=split,
        target_shape=args.target_shape,
        mask_name=args.mask_name,
        max_samples=max_samples,
        sample_strategy=args.sample_strategy,
        sample_seed=args.sample_seed,
        ct_mode=args.ct_mode,
        include_energy=args.include_energy,
        dose_mode=args.dose_mode,
        global_dose_scale=args.global_dose_scale,
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed + (0 if split == "train" else 1))
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker if args.num_workers > 0 else None,
        generator=generator,
    )


def run_validation(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    mask_weight: float,
    export_path: Path | None = None,
) -> dict[str, float]:
    start_time = time.perf_counter()
    model.eval()
    total_loss = 0.0
    total_global = 0.0
    total_masked = 0.0
    batches = 0
    exported = False

    with torch.no_grad():
        for batch in loader:
            ct = batch["ct"].to(device)
            dose = batch["dose"].to(device)
            mask = batch["loss_mask"].to(device)
            condition = batch["condition"].to(device)

            pred = model(ct, condition)
            loss, global_loss, focus_loss = dose_loss(pred, dose, mask, mask_weight)

            total_loss += float(loss.cpu())
            total_global += float(global_loss.cpu())
            total_masked += float(focus_loss.cpu())
            batches += 1

            if export_path is not None and not exported:
                export_path.parent.mkdir(parents=True, exist_ok=True)
                dose_max = batch["dose_max"][0].detach().cpu().numpy().astype(np.float32)
                dose_scale = batch["dose_scale"][0].detach().cpu().numpy().astype(np.float32)
                pred_norm = pred[0, 0].detach().cpu().numpy().astype(np.float32)
                dose_norm = dose[0, 0].detach().cpu().numpy().astype(np.float32)
                pred_abs = np.clip(pred_norm * dose_scale, 0.0, None)
                dose_abs = dose_norm * dose_scale
                np.savez_compressed(
                    export_path,
                    pred=pred_norm,
                    dose=dose_norm,
                    pred_abs=pred_abs,
                    dose_abs=dose_abs,
                    loss_mask=mask[0, 0].detach().cpu().numpy().astype(np.float32),
                    dose_max=dose_max,
                    dose_scale=dose_scale,
                    crop_offset=batch["crop_offset"][0].detach().cpu().numpy().astype(np.float32),
                    element_spacing=batch["element_spacing"][0].detach().cpu().numpy().astype(np.float32),
                    case_id=np.asarray(batch["case_id"][0]),
                    beam_idx=np.asarray(int(batch["beam_idx"][0])),
                    cp_idx=np.asarray(int(batch["cp_idx"][0])),
                )
                mha_meta = {
                    "ObjectType": "Image",
                    "NDims": "3",
                    "BinaryData": "True",
                    "BinaryDataByteOrderMSB": "False",
                    "TransformMatrix": "1 0 0 0 1 0 0 0 1",
                    "Offset": " ".join(f"{float(v):.6g}" for v in batch["crop_offset"][0].detach().cpu().numpy()),
                    "CenterOfRotation": "0 0 0",
                    "AnatomicalOrientation": "RAI",
                    "ElementSpacing": " ".join(f"{float(v):.6g}" for v in batch["element_spacing"][0].detach().cpu().numpy()),
                    "DimSize": " ".join(str(int(v)) for v in pred_abs.shape),
                }
                write_float_mha(
                    export_path.with_name(export_path.stem + "_pred.mha"),
                    pred_abs,
                    mha_meta,
                    offset=batch["crop_offset"][0].detach().cpu().numpy(),
                    dim_size=tuple(int(v) for v in pred_abs.shape),
                )
                exported = True

    return {
        "val_loss": total_loss / max(batches, 1),
        "val_global_l1": total_global / max(batches, 1),
        "val_masked_l1": total_masked / max(batches, 1),
        "val_batches": batches,
        "val_seconds": time.perf_counter() - start_time,
    }


def save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, metrics: dict[str, float], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "args": vars(args),
        },
        path,
    )


def write_metrics(path: Path, rows: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
        "train_loss",
        "train_global_l1",
        "train_masked_l1",
        "train_batches",
        "train_seconds",
        "val_loss",
        "val_global_l1",
        "val_masked_l1",
        "val_batches",
        "val_seconds",
        "epoch_seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_metrics(path: Path) -> list[dict[str, float | int]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out: list[dict[str, float | int]] = []
    for row in rows:
        parsed: dict[str, float | int] = {"epoch": int(row["epoch"])}
        for key, value in row.items():
            if key == "epoch":
                continue
            parsed[key] = float(value)
        out.append(parsed)
    return out


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    target_shape = parse_int_tuple(args.target_shape)
    if any(dim % 4 != 0 for dim in target_shape):
        raise ValueError("--target-shape dimensions must be divisible by 4 for this U-Net")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    output_dir = Path(args.output_dir)

    train_loader = make_loader(args, "train", args.max_train_samples, shuffle=True)
    val_loader = make_loader(args, "val", args.max_val_samples, shuffle=False)

    model = GeometryConditionedUNet3D(
        condition_dim=condition_dim(include_energy=args.include_energy),
        base_channels=args.base_channels,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    rows: list[dict[str, float | int]] = []
    start_epoch = 1
    best_val = float("inf")
    if args.resume_checkpoint:
        checkpoint = load_checkpoint(args.resume_checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        move_optimizer_state_to_device(optimizer, device)
        start_epoch = int(checkpoint["epoch"]) + 1
        rows = read_metrics(output_dir / "metrics.csv")
        if rows:
            best_val = min(float(row["val_loss"]) for row in rows)
        else:
            metrics = checkpoint.get("metrics", {})
            best_val = float(metrics.get("val_loss", best_val))
        print(
            f"resumed_from={args.resume_checkpoint}",
            f"checkpoint_epoch={checkpoint['epoch']}",
            f"next_epoch={start_epoch}",
            flush=True,
        )

    if start_epoch > args.epochs:
        print(f"resume_checkpoint_epoch_already_reached target_epochs={args.epochs}")
        return

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_start = time.perf_counter()
        model.train()
        train_loss = 0.0
        train_global = 0.0
        train_masked = 0.0
        batches = 0

        for batch in train_loader:
            ct = batch["ct"].to(device)
            dose = batch["dose"].to(device)
            mask = batch["loss_mask"].to(device)
            condition = batch["condition"].to(device)

            pred = model(ct, condition)
            loss, global_loss, focus_loss = dose_loss(pred, dose, mask, args.mask_weight)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_loss += float(loss.detach().cpu())
            train_global += float(global_loss.detach().cpu())
            train_masked += float(focus_loss.detach().cpu())
            batches += 1

            if args.steps_per_epoch > 0 and batches >= args.steps_per_epoch:
                break

        train_metrics = {
            "train_loss": train_loss / max(batches, 1),
            "train_global_l1": train_global / max(batches, 1),
            "train_masked_l1": train_masked / max(batches, 1),
            "train_batches": batches,
            "train_seconds": time.perf_counter() - train_start,
        }
        val_metrics = run_validation(
            model,
            val_loader,
            device,
            args.mask_weight,
            export_path=output_dir / "predictions" / f"val_prediction_epoch_{epoch:03d}.npz",
        )
        row = {"epoch": epoch, **train_metrics, **val_metrics, "epoch_seconds": time.perf_counter() - epoch_start}
        rows.append(row)
        write_metrics(output_dir / "metrics.csv", rows)

        save_checkpoint(output_dir / "checkpoints" / "last.pt", model, optimizer, epoch, row, args)
        if val_metrics["val_loss"] < best_val:
            best_val = val_metrics["val_loss"]
            save_checkpoint(output_dir / "checkpoints" / "best.pt", model, optimizer, epoch, row, args)

        print(
            f"epoch={epoch}",
            f"train_loss={train_metrics['train_loss']:.6f}",
            f"val_loss={val_metrics['val_loss']:.6f}",
            f"epoch_seconds={row['epoch_seconds']:.1f}",
            f"device={device}",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", default="data/photon/training")
    parser.add_argument("--split-csv", default="splits/photon_case_split.csv")
    parser.add_argument("--output-dir", default="outputs/baseline_3d_unet")
    parser.add_argument("--target-shape", default="64 64 64")
    parser.add_argument("--mask-name", default="dose_gt_1pct")
    parser.add_argument("--ct-mode", choices=("hu", "density"), default="hu")
    parser.add_argument("--include-energy", action="store_true")
    parser.add_argument("--dose-mode", choices=("sample_max", "global", "raw"), default="global")
    parser.add_argument("--global-dose-scale", type=float, default=1.5e-4)
    parser.add_argument("--max-train-samples", type=int, default=32)
    parser.add_argument("--max-val-samples", type=int, default=8)
    parser.add_argument("--sample-strategy", choices=("uniform", "random", "first"), default="uniform")
    parser.add_argument("--sample-seed", type=int, default=20260628)
    parser.add_argument("--seed", type=int, default=20260628)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--steps-per-epoch", type=int, default=0)
    parser.add_argument("--base-channels", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mask-weight", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resume-checkpoint", default="")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
