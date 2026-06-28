"""Tiny training smoke test for the geometry-conditioned 3D U-Net baseline."""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from doserad_dataset import CONDITION_DIM, DoseRadControlPointDataset
from model_3d_unet import GeometryConditionedUNet3D


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum().clamp_min(1.0)
    return (torch.abs(pred - target) * mask).sum() / denom


def train_smoke(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    dataset = DoseRadControlPointDataset(
        training_dir=args.training_dir,
        split_csv=args.split_csv,
        split=args.split,
        target_shape=args.target_shape,
        mask_name=args.mask_name,
        max_samples=args.max_samples,
        ct_mode=args.ct_mode,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = GeometryConditionedUNet3D(condition_dim=CONDITION_DIM, base_channels=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    step = 0
    last_loss = None
    for epoch in range(args.epochs):
        for batch in loader:
            ct = batch["ct"].to(device)
            dose = batch["dose"].to(device)
            mask = batch["loss_mask"].to(device)
            condition = batch["condition"].to(device)

            pred = model(ct, condition)
            global_loss = F.l1_loss(pred, dose)
            focus_loss = masked_l1(pred, dose, mask)
            loss = global_loss + args.mask_weight * focus_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            step += 1
            last_loss = float(loss.detach().cpu())
            print(
                f"step={step}",
                f"loss={last_loss:.6f}",
                f"global_l1={float(global_loss.detach().cpu()):.6f}",
                f"masked_l1={float(focus_loss.detach().cpu()):.6f}",
                f"device={device}",
            )
            if step >= args.steps:
                print("smoke_test_passed", f"steps={step}", f"last_loss={last_loss:.6f}")
                return

    print("smoke_test_passed", f"steps={step}", f"last_loss={last_loss:.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", default="data/photon/training")
    parser.add_argument("--split-csv", default="splits/photon_case_split.csv")
    parser.add_argument("--split", default="train")
    parser.add_argument("--target-shape", default="32 32 32")
    parser.add_argument("--mask-name", default="dose_gt_1pct")
    parser.add_argument("--ct-mode", choices=("hu", "density"), default="hu")
    parser.add_argument("--max-samples", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--base-channels", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mask-weight", type=float, default=1.0)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train_smoke(parse_args())
