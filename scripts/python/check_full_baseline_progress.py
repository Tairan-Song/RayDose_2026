"""Check progress for a full-dataset DoseRAD photon baseline run."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_TRAIN_SAMPLES = 32400
EXPECTED_VAL_SAMPLES = 8100


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_pid(path: Path) -> int | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def pid_is_running(pid: int | None) -> bool:
    if pid is None:
        return False
    if os.name == "nt":
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            check=False,
            capture_output=True,
            text=True,
        )
        return str(pid) in result.stdout
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def fmt_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    if hours > 0:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def parse_utc_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def checkpoint_status(name: str, complete: bool, detail: str) -> dict[str, str]:
    return {
        "checkpoint": name,
        "status": "complete" if complete else "pending",
        "detail": detail,
    }


def count_manifest_predictions(path: Path) -> int:
    rows = read_csv_rows(path)
    return len(rows)


def output_report(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    manifest = read_json(output_dir / "run_manifest.json")
    train_metrics = read_csv_rows(output_dir / "train" / "metrics.csv")
    eval_rows = read_csv_rows(output_dir / "evaluate" / "per_sample_metrics.csv")
    prediction_manifest = output_dir / "dose_predictions" / "prediction_manifest.csv"
    prediction_rows = read_csv_rows(prediction_manifest)
    exported_rows = read_csv_rows(output_dir / "evaluate_exported" / "exported_prediction_metrics.csv")
    active_pid = read_pid(output_dir / "full_run_python.pid")
    active_pid_running = pid_is_running(active_pid)

    stages = manifest.get("stages", [])
    stage_seconds = {stage.get("stage", ""): float(stage.get("seconds", 0.0)) for stage in stages}
    created_utc = parse_utc_datetime(manifest.get("created_utc") if manifest else None)
    elapsed_seconds = None
    if created_utc is not None and manifest.get("status") == "started":
        elapsed_seconds = (datetime.now(timezone.utc) - created_utc).total_seconds()

    last_epoch_seconds = None
    completed_epochs = len(train_metrics)
    if train_metrics:
        last = train_metrics[-1]
        try:
            last_epoch_seconds = float(last.get("epoch_seconds", ""))
        except ValueError:
            last_epoch_seconds = None

    target_epochs = int(args.expected_epochs)
    estimated_train_remaining = None
    if last_epoch_seconds is not None and completed_epochs < target_epochs:
        estimated_train_remaining = (target_epochs - completed_epochs) * last_epoch_seconds

    statuses = [
        checkpoint_status(
            "0_pre_run_manifest",
            bool(manifest),
            f"run_manifest.json {'found' if manifest else 'missing'}",
        ),
        checkpoint_status(
            "1_training_started",
            active_pid_running or (output_dir / "train" / "metrics.csv").exists(),
            f"epochs_recorded={completed_epochs}/{target_epochs}; active_pid={active_pid}; active_pid_running={active_pid_running}",
        ),
        checkpoint_status(
            "2_first_epoch_complete",
            completed_epochs >= 1,
            f"last_epoch_seconds={fmt_seconds(last_epoch_seconds)}",
        ),
        checkpoint_status(
            "3_training_complete",
            completed_epochs >= target_epochs
            and (output_dir / "train" / "checkpoints" / "best.pt").exists()
            and (output_dir / "train" / "checkpoints" / "last.pt").exists(),
            f"epochs_recorded={completed_epochs}/{target_epochs}; estimated_training_remaining={fmt_seconds(estimated_train_remaining)}",
        ),
        checkpoint_status(
            "4_full_checkpoint_evaluation_complete",
            len(eval_rows) >= EXPECTED_VAL_SAMPLES,
            f"evaluated_samples={len(eval_rows)}/{EXPECTED_VAL_SAMPLES}",
        ),
        checkpoint_status(
            "5_full_prediction_export_complete",
            len(prediction_rows) >= EXPECTED_VAL_SAMPLES,
            f"exported_predictions={len(prediction_rows)}/{EXPECTED_VAL_SAMPLES}",
        ),
        checkpoint_status(
            "6_full_exported_evaluation_complete",
            len(exported_rows) >= EXPECTED_VAL_SAMPLES,
            f"exported_eval_samples={len(exported_rows)}/{EXPECTED_VAL_SAMPLES}",
        ),
    ]

    data = manifest.get("data", {})
    splits = data.get("splits", {}) if isinstance(data, dict) else {}
    train_count = splits.get("train", {}).get("dose_samples", "unknown") if isinstance(splits, dict) else "unknown"
    val_count = splits.get("val", {}).get("dose_samples", "unknown") if isinstance(splits, dict) else "unknown"

    print(f"output_dir={output_dir}")
    print(f"manifest_status={manifest.get('status', 'missing') if manifest else 'missing'}")
    print(f"train_samples_recorded={train_count}")
    print(f"val_samples_recorded={val_count}")
    print(f"completed_epochs={completed_epochs}/{target_epochs}")
    print(f"active_training_pid={active_pid if active_pid is not None else 'unknown'}")
    print(f"active_training_pid_running={active_pid_running}")
    print(f"run_elapsed={fmt_seconds(elapsed_seconds)}")
    print(f"last_epoch_seconds={fmt_seconds(last_epoch_seconds)}")
    print(f"estimated_training_remaining={fmt_seconds(estimated_train_remaining)}")
    if stage_seconds:
        for stage, seconds in stage_seconds.items():
            print(f"stage_seconds.{stage}={seconds:.3f}")
    print("")
    print("checkpoint,status,detail")
    for item in statuses:
        print(f"{item['checkpoint']},{item['status']},{item['detail']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/full_baseline_hu_no_energy_seed20260628")
    parser.add_argument("--expected-epochs", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    output_report(parse_args())
