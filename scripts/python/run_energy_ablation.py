"""Run paired baseline ablations for CT mode and energy conditioning."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def bool_flag(cmd: list[str], enabled: bool, flag: str) -> None:
    if enabled:
        cmd.append(flag)


def build_train_command(args: argparse.Namespace, output_dir: Path, ct_mode: str, include_energy: bool) -> list[str]:
    train_script = Path(__file__).with_name("train_3d_unet.py")
    cmd = [
        sys.executable,
        str(train_script),
        "--training-dir",
        args.training_dir,
        "--split-csv",
        args.split_csv,
        "--output-dir",
        str(output_dir),
        "--target-shape",
        args.target_shape,
        "--mask-name",
        args.mask_name,
        "--ct-mode",
        ct_mode,
        "--dose-mode",
        args.dose_mode,
        "--global-dose-scale",
        str(args.global_dose_scale),
        "--max-train-samples",
        str(args.max_train_samples),
        "--max-val-samples",
        str(args.max_val_samples),
        "--sample-strategy",
        args.sample_strategy,
        "--sample-seed",
        str(args.sample_seed),
        "--seed",
        str(args.seed),
        "--batch-size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--steps-per-epoch",
        str(args.steps_per_epoch),
        "--base-channels",
        str(args.base_channels),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--mask-weight",
        str(args.mask_weight),
        "--num-workers",
        str(args.num_workers),
    ]
    bool_flag(cmd, include_energy, "--include-energy")
    bool_flag(cmd, args.cpu, "--cpu")
    return cmd


def read_best_metrics(metrics_csv: Path) -> dict[str, str]:
    if not metrics_csv.exists():
        return {}

    with metrics_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return {}

    best = min(rows, key=lambda row: float(row["val_loss"]))
    last = rows[-1]
    return {
        "best_epoch": best["epoch"],
        "best_val_loss": best["val_loss"],
        "best_val_global_l1": best["val_global_l1"],
        "best_val_masked_l1": best["val_masked_l1"],
        "last_epoch": last["epoch"],
        "last_train_loss": last["train_loss"],
        "last_val_loss": last["val_loss"],
    }


def write_summary(output_root: Path, experiments: list[dict[str, object]]) -> None:
    rows: list[dict[str, object]] = []
    for experiment in experiments:
        output_dir = Path(experiment["output_dir"])
        metrics = read_best_metrics(output_dir / "metrics.csv")
        if metrics:
            rows.append({**experiment, **metrics})

    if not rows:
        return

    summary_csv = output_root / "ablation_summary.csv"
    fieldnames = [
        "experiment",
        "ct_mode",
        "include_energy",
        "output_dir",
        "best_epoch",
        "best_val_loss",
        "best_val_global_l1",
        "best_val_masked_l1",
        "last_epoch",
        "last_train_loss",
        "last_val_loss",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote_summary={summary_csv}", flush=True)


def run_experiment(name: str, args: argparse.Namespace, ct_mode: str, include_energy: bool) -> dict[str, object]:
    output_dir = Path(args.output_root) / name
    cmd = build_train_command(args, output_dir, ct_mode, include_energy)
    print("running=" + " ".join(f'"{part}"' if " " in part else part for part in cmd), flush=True)
    if not args.dry_run:
        subprocess.run(cmd, check=True)
    return {
        "experiment": name,
        "ct_mode": ct_mode,
        "include_energy": int(include_energy),
        "output_dir": str(output_dir),
    }


def selected_ct_modes(args: argparse.Namespace) -> list[str]:
    return args.ct_modes if args.ct_modes else [args.ct_mode]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", default="data/photon/training")
    parser.add_argument("--split-csv", default="splits/photon_case_split.csv")
    parser.add_argument("--output-root", default="outputs/energy_ablation")
    parser.add_argument("--target-shape", default="64 64 64")
    parser.add_argument("--mask-name", default="dose_gt_1pct")
    parser.add_argument("--ct-mode", choices=("hu", "density"), default="hu")
    parser.add_argument("--ct-modes", nargs="+", choices=("hu", "density"), default=None)
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
    parser.add_argument("--only", choices=("both", "no_energy", "with_energy"), default="both")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    ct_modes = selected_ct_modes(args)
    multi_ct_mode = len(ct_modes) > 1
    experiments: list[dict[str, object]] = []
    for ct_mode in ct_modes:
        prefix = f"{ct_mode}_" if multi_ct_mode else ""
        if args.only in ("both", "no_energy"):
            experiments.append(run_experiment(f"{prefix}no_energy", args, ct_mode, include_energy=False))
        if args.only in ("both", "with_energy"):
            experiments.append(run_experiment(f"{prefix}with_energy", args, ct_mode, include_energy=True))

    if not args.dry_run:
        write_summary(output_root, experiments)


if __name__ == "__main__":
    main()
