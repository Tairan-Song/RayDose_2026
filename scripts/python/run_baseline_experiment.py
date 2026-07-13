"""Run a train/evaluate/export baseline experiment for photon dose prediction."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


def bool_flag(cmd: list[str], enabled: bool, flag: str) -> None:
    if enabled:
        cmd.append(flag)


def run_command(cmd: list[str], dry_run: bool) -> dict[str, object]:
    print("running=" + " ".join(f'"{part}"' if " " in part else part for part in cmd), flush=True)
    start = time.perf_counter()
    if dry_run:
        return {"command": cmd, "dry_run": True, "returncode": 0, "seconds": 0.0}
    completed = subprocess.run(cmd, check=True)
    return {"command": cmd, "dry_run": False, "returncode": completed.returncode, "seconds": time.perf_counter() - start}


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def hardware_info(cpu: bool) -> dict[str, object]:
    info: dict[str, object] = {
        "platform": platform.platform(),
        "python": sys.version.replace("\n", " "),
        "cpu_count": os.cpu_count(),
        "force_cpu": cpu,
    }
    try:
        import torch

        info.update(
            {
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count(),
                "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
            }
        )
    except Exception as exc:  # pragma: no cover - metadata best effort
        info["torch_error"] = str(exc)
    return info


def split_and_sample_counts(training_dir: str | Path, split_csv: str | Path) -> dict[str, object]:
    training_dir = Path(training_dir)
    split_csv = Path(split_csv)
    split_cases: dict[str, list[str]] = {}
    with split_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            split_cases.setdefault(row["split"], []).append(row["case_id"])

    out: dict[str, object] = {"split_csv": str(split_csv), "splits": {}}
    split_out: dict[str, object] = {}
    for split, case_ids in sorted(split_cases.items()):
        dose_files = 0
        for case_id in case_ids:
            dose_files += len(list((training_dir / case_id / "dose").glob("Dose_B*_CP*.mha")))
        split_out[split] = {"cases": len(case_ids), "dose_samples": dose_files}
    out["splits"] = split_out
    return out


def write_manifest(path: Path, manifest: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def require_path(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


def script_path(name: str) -> str:
    return str(Path(__file__).with_name(name))


def make_split_command(args: argparse.Namespace, python_exe: str, split_csv: Path) -> list[str]:
    return [
        python_exe,
        script_path("make_case_split.py"),
        "--training-dir",
        args.training_dir,
        "--output-csv",
        str(split_csv),
        "--train-fraction",
        str(args.train_fraction),
        "--seed",
        str(args.split_seed),
    ]


def train_command(args: argparse.Namespace, python_exe: str, train_dir: Path) -> list[str]:
    cmd = [
        python_exe,
        script_path("train_3d_unet.py"),
        "--training-dir",
        args.training_dir,
        "--split-csv",
        str(args.resolved_split_csv),
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
    if args.resume_checkpoint:
        cmd.extend(["--resume-checkpoint", args.resume_checkpoint])
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
        str(args.resolved_split_csv),
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
        str(args.resolved_split_csv),
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
        "--full-mode",
        args.full_mode,
        "--sliding-stride-fraction",
        str(args.sliding_stride_fraction),
        "--max-sliding-windows",
        str(args.max_sliding_windows),
        "--print-every",
        str(args.print_every),
    ]
    bool_flag(cmd, not args.save_npz, "--no-npz")
    bool_flag(cmd, args.cpu, "--cpu")
    return cmd


def exported_eval_command(args: argparse.Namespace, python_exe: str, export_dir: Path, exported_eval_dir: Path) -> list[str]:
    return [
        python_exe,
        script_path("evaluate_exported_predictions.py"),
        "--manifest-csv",
        str(export_dir / "prediction_manifest.csv"),
        "--training-dir",
        args.training_dir,
        "--mask-name",
        args.mask_name,
        "--output-dir",
        str(exported_eval_dir),
        "--max-samples",
        str(args.export_eval_samples),
        "--print-every",
        str(args.print_every),
    ]


def run_baseline(args: argparse.Namespace) -> None:
    run_start = time.perf_counter()
    python_exe = args.python_exe or sys.executable
    output_dir = Path(args.output_dir)
    train_dir = output_dir / "train"
    eval_dir = output_dir / "evaluate"
    export_dir = output_dir / "dose_predictions"
    exported_eval_dir = output_dir / "evaluate_exported"
    checkpoint = train_dir / "checkpoints" / "best.pt"
    split_csv = Path(args.split_csv)
    args.resolved_split_csv = split_csv
    manifest_path = output_dir / "run_manifest.json"
    stage_records: list[dict[str, object]] = []
    manifest: dict[str, object] = {
        "created_utc": iso_now(),
        "status": "started",
        "output_dir": str(output_dir),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "hardware": hardware_info(args.cpu),
        "data": split_and_sample_counts(args.training_dir, split_csv) if split_csv.exists() else {"split_csv": str(split_csv), "missing": True},
        "stages": stage_records,
    }
    write_manifest(manifest_path, manifest)

    if args.make_split and (args.overwrite_split or not split_csv.exists()):
        record = run_command(make_split_command(args, python_exe, split_csv), args.dry_run)
        record["stage"] = "make_split"
        stage_records.append(record)
        manifest["data"] = split_and_sample_counts(args.training_dir, split_csv) if split_csv.exists() else manifest["data"]
        write_manifest(manifest_path, manifest)
    elif not split_csv.exists():
        raise FileNotFoundError(f"Split CSV not found: {split_csv}. Run make_case_split.py first or pass --make-split.")

    for stage, command in [
        ("train", train_command(args, python_exe, train_dir)),
        ("evaluate_checkpoint", evaluate_command(args, python_exe, checkpoint, eval_dir)),
        ("export_predictions", export_command(args, python_exe, checkpoint, export_dir)),
    ]:
        record = run_command(command, args.dry_run)
        record["stage"] = stage
        stage_records.append(record)
        write_manifest(manifest_path, manifest)
    if not args.skip_export_eval:
        record = run_command(exported_eval_command(args, python_exe, export_dir, exported_eval_dir), args.dry_run)
        record["stage"] = "evaluate_exported_predictions"
        stage_records.append(record)
        write_manifest(manifest_path, manifest)

    if not args.dry_run:
        require_path(checkpoint, "best checkpoint")
        require_path(train_dir / "metrics.csv", "training metrics CSV")
        require_path(eval_dir / "summary_metrics.csv", "evaluation summary CSV")
        require_path(export_dir / "prediction_manifest.csv", "prediction manifest")
        if not args.skip_export_eval:
            require_path(exported_eval_dir / "exported_prediction_summary.csv", "exported prediction summary CSV")
        manifest["status"] = "complete"
        manifest["completed_utc"] = iso_now()
        manifest["total_seconds"] = time.perf_counter() - run_start
        write_manifest(manifest_path, manifest)
        print(f"baseline_experiment_done output_dir={output_dir}", flush=True)
    else:
        manifest["status"] = "dry_run_complete"
        manifest["completed_utc"] = iso_now()
        manifest["total_seconds"] = time.perf_counter() - run_start
        write_manifest(manifest_path, manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", default="data/photon/training")
    parser.add_argument("--split-csv", default="splits/photon_case_split.csv")
    parser.add_argument("--make-split", action="store_true")
    parser.add_argument("--overwrite-split", action="store_true")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--split-seed", type=int, default=20260628)
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
    parser.add_argument("--export-eval-samples", type=int, default=0)
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
    parser.add_argument("--resume-checkpoint", default="")
    parser.add_argument("--filename-style", choices=("pred", "dose"), default="dose")
    parser.add_argument("--full-mode", choices=("crop_insert", "sliding"), default="crop_insert")
    parser.add_argument("--sliding-stride-fraction", type=float, default=0.5)
    parser.add_argument("--max-sliding-windows", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--save-npz", action="store_true")
    parser.add_argument("--skip-export-eval", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_baseline(parse_args())
