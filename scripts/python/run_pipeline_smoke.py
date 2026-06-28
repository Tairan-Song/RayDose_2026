"""Run an end-to-end smoke test for the photon baseline pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str]) -> None:
    print("running=" + " ".join(f'"{part}"' if " " in part else part for part in cmd), flush=True)
    subprocess.run(cmd, check=True)


def require_path(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


def script_path(name: str) -> str:
    return str(Path(__file__).with_name(name))


def first_full_prediction(output_dir: Path) -> Path:
    matches = sorted(output_dir.glob("*/*_pred_full.mha"))
    if not matches:
        raise FileNotFoundError(f"No full-volume prediction MHA found under {output_dir}")
    return matches[0]


def run_pipeline(args: argparse.Namespace) -> None:
    python_exe = args.python_exe or sys.executable
    output_dir = Path(args.output_dir)
    split_csv = output_dir / "photon_case_split.csv"
    preprocess_npz = output_dir / "preprocess" / f"{args.case_id}_B{args.beam_idx}_CP{args.cp_idx:03d}.npz"
    train_dir = output_dir / "train"
    checkpoint = train_dir / "checkpoints" / "best.pt"
    eval_dir = output_dir / "evaluate"
    inference_dir = output_dir / "batch_inference"

    run_command(
        [
            python_exe,
            script_path("make_case_split.py"),
            "--training-dir",
            args.training_dir,
            "--output-csv",
            str(split_csv),
            "--train-fraction",
            str(args.train_fraction),
            "--seed",
            str(args.seed),
        ]
    )

    run_command(
        [
            python_exe,
            script_path("preprocess_training_sample.py"),
            "--training-dir",
            args.training_dir,
            "--case-id",
            args.case_id,
            "--beam-idx",
            str(args.beam_idx),
            "--cp-idx",
            str(args.cp_idx),
            "--target-shape",
            args.target_shape,
            "--ct-mode",
            args.ct_mode,
            "--dose-mode",
            args.dose_mode,
            "--global-dose-scale",
            str(args.global_dose_scale),
            "--output-npz",
            str(preprocess_npz),
        ]
    )

    train_cmd = [
        python_exe,
        script_path("train_3d_unet.py"),
        "--training-dir",
        args.training_dir,
        "--split-csv",
        str(split_csv),
        "--output-dir",
        str(train_dir),
        "--target-shape",
        args.target_shape,
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
        str(args.seed),
        "--seed",
        str(args.seed),
        "--batch-size",
        "1",
        "--epochs",
        "1",
        "--steps-per-epoch",
        "1",
        "--base-channels",
        str(args.base_channels),
    ]
    if args.include_energy:
        train_cmd.append("--include-energy")
    run_command(train_cmd)

    run_command(
        [
            python_exe,
            script_path("evaluate_checkpoint.py"),
            "--checkpoint",
            str(checkpoint),
            "--training-dir",
            args.training_dir,
            "--split-csv",
            str(split_csv),
            "--output-dir",
            str(eval_dir),
            "--max-samples",
            str(args.max_val_samples),
            "--sample-strategy",
            args.sample_strategy,
            "--sample-seed",
            str(args.seed),
            "--print-every",
            "1",
        ]
    )

    run_command(
        [
            python_exe,
            script_path("predict_3d_unet_batch.py"),
            "--checkpoint",
            str(checkpoint),
            "--training-dir",
            args.training_dir,
            "--split-csv",
            str(split_csv),
            "--output-dir",
            str(inference_dir),
            "--max-samples",
            str(args.max_val_samples),
            "--sample-strategy",
            args.sample_strategy,
            "--sample-seed",
            str(args.seed),
            "--no-npz",
        ]
    )

    required_outputs = [
        (split_csv, "case split CSV"),
        (preprocess_npz, "preprocessed NPZ"),
        (train_dir / "metrics.csv", "training metrics CSV"),
        (checkpoint, "best checkpoint"),
        (eval_dir / "per_sample_metrics.csv", "evaluation per-sample CSV"),
        (eval_dir / "summary_metrics.csv", "evaluation summary CSV"),
        (inference_dir / "prediction_manifest.csv", "batch inference manifest"),
    ]
    for path, description in required_outputs:
        require_path(path, description)

    full_prediction = first_full_prediction(inference_dir)
    print(f"full_prediction={full_prediction}", flush=True)
    print(f"pipeline_smoke_passed output_dir={output_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", default="data/photon/training")
    parser.add_argument("--output-dir", default="outputs/pipeline_smoke")
    parser.add_argument("--python-exe", default="")
    parser.add_argument("--case-id", default="1ABB006")
    parser.add_argument("--beam-idx", type=int, default=0)
    parser.add_argument("--cp-idx", type=int, default=0)
    parser.add_argument("--target-shape", default="32 32 32")
    parser.add_argument("--ct-mode", choices=("hu", "density"), default="hu")
    parser.add_argument("--include-energy", action="store_true")
    parser.add_argument("--dose-mode", choices=("sample_max", "global", "raw"), default="global")
    parser.add_argument("--global-dose-scale", type=float, default=1.5e-4)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--max-train-samples", type=int, default=2)
    parser.add_argument("--max-val-samples", type=int, default=1)
    parser.add_argument("--sample-strategy", choices=("uniform", "random", "first"), default="uniform")
    parser.add_argument("--base-channels", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260628)
    return parser.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())
