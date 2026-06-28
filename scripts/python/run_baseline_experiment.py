"""Run a train/evaluate/export baseline experiment for photon dose prediction."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def bool_flag(cmd: list[str], enabled: bool, flag: str) -> None:
    if enabled:
        cmd.append(flag)


def run_command(cmd: list[str], dry_run: bool) -> None:
    print("running=" + " ".join(f'"{part}"' if " " in part else part for part in cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def require_path(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


def script_path(name: str) -> str:
    return str(Path(__file__).with_name(name))


def train_command(args: argparse.Namespace, python_exe: str, train_dir: Path) -> list[str]:
    cmd = [
        python_exe,
        script_path("train_3d_unet.py"),
        "--training-dir",
        args.training_dir,
        "--split-csv",
        args.split_csv,
        "--output-dir",
        str(train_dir),
        "--target-shape",
        args.target_shape,
        "--mask-name",
        args.mask_name,
        "--ct-mode",
        args.ct_mode,
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
    bool_flag(cmd, args.include_energy, "--include-energy")
    bool_flag(cmd, args.cpu, "--cpu")
    return cmd


def evaluate_command(args: argparse.Namespace, python_exe: str, checkpoint: Path, eval_dir: Path) -> list[str]:
    cmd = [
        python_exe,
        script_path("evaluate_checkpoint.py"),
        "--checkpoint",
        str(checkpoint),
        "--training-dir",
        args.training_dir,
        "--split-csv",
        args.split_csv,
        "--split",
        args.eval_split,
        "--output-dir",
        str(eval_dir),
        "--max-samples",
        str(args.eval_samples),
        "--sample-strategy",
        args.sample_strategy,
        "--sample-seed",
        str(args.sample_seed),
        "--num-workers",
        str(args.num_workers),
        "--print-every",
        str(args.print_every),
    ]
    bool_flag(cmd, args.cpu, "--cpu")
    return cmd


def export_command(args: argparse.Namespace, python_exe: str, checkpoint: Path, export_dir: Path) -> list[str]:
    cmd = [
        python_exe,
        script_path("predict_3d_unet_batch.py"),
        "--checkpoint",
        str(checkpoint),
        "--training-dir",
        args.training_dir,
        "--split-csv",
        args.split_csv,
        "--split",
        args.export_split,
        "--output-dir",
        str(export_dir),
        "--max-samples",
        str(args.export_samples),
        "--sample-strategy",
        args.sample_strategy,
        "--sample-seed",
        str(args.sample_seed),
        "--filename-style",
        args.filename_style,
        "--print-every",
        str(args.print_every),
    ]
    bool_flag(cmd, not args.save_npz, "--no-npz")
    bool_flag(cmd, args.cpu, "--cpu")
    return cmd


def run_baseline(args: argparse.Namespace) -> None:
    python_exe = args.python_exe or sys.executable
    output_dir = Path(args.output_dir)
    train_dir = output_dir / "train"
    eval_dir = output_dir / "evaluate"
    export_dir = output_dir / "dose_predictions"
    checkpoint = train_dir / "checkpoints" / "best.pt"

    if not Path(args.split_csv).exists():
        raise FileNotFoundError(f"Split CSV not found: {args.split_csv}. Run make_case_split.py first.")

    run_command(train_command(args, python_exe, train_dir), args.dry_run)
    run_command(evaluate_command(args, python_exe, checkpoint, eval_dir), args.dry_run)
    run_command(export_command(args, python_exe, checkpoint, export_dir), args.dry_run)

    if not args.dry_run:
        require_path(checkpoint, "best checkpoint")
        require_path(train_dir / "metrics.csv", "training metrics CSV")
        require_path(eval_dir / "summary_metrics.csv", "evaluation summary CSV")
        require_path(export_dir / "prediction_manifest.csv", "prediction manifest")
        print(f"baseline_experiment_done output_dir={output_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", default="data/photon/training")
    parser.add_argument("--split-csv", default="splits/photon_case_split.csv")
    parser.add_argument("--output-dir", default="outputs/baseline_experiment")
    parser.add_argument("--python-exe", default="")
    parser.add_argument("--target-shape", default="64 64 64")
    parser.add_argument("--mask-name", default="dose_gt_1pct")
    parser.add_argument("--ct-mode", choices=("hu", "density"), default="hu")
    parser.add_argument("--include-energy", action="store_true")
    parser.add_argument("--dose-mode", choices=("sample_max", "global", "raw"), default="global")
    parser.add_argument("--global-dose-scale", type=float, default=1.5e-4)
    parser.add_argument("--max-train-samples", type=int, default=32)
    parser.add_argument("--max-val-samples", type=int, default=8)
    parser.add_argument("--eval-samples", type=int, default=8)
    parser.add_argument("--export-samples", type=int, default=8)
    parser.add_argument("--eval-split", default="val")
    parser.add_argument("--export-split", default="val")
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
    parser.add_argument("--filename-style", choices=("pred", "dose"), default="dose")
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--save-npz", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_baseline(parse_args())
