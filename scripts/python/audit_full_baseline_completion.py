"""Audit whether a full-dataset photon baseline run is formally complete."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REQUIRED_SUMMARY_METRICS = {
    "mae",
    "rmse",
    "masked_mae",
    "masked_rmse",
    "masked_relative_mae_percent",
    "hd_10pct_mae",
    "hd_10pct_rmse",
    "hd_50pct_mae",
    "hd_50pct_rmse",
    "gamma_3pct_3mm_cutoff10pct_pass_rate",
    "gamma_2pct_2mm_cutoff10pct_pass_rate",
    "prediction_seconds",
    "total_seconds",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def check(condition: bool, name: str, detail: str, rows: list[dict[str, str]]) -> None:
    rows.append({"status": "PASS" if condition else "FAIL", "check": name, "detail": detail})


def split_counts(split_csv: Path) -> dict[str, int]:
    rows = read_csv(split_csv)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["split"]] = counts.get(row["split"], 0) + 1
    return counts


def audit(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    train_dir = output_dir / "train"
    evaluate_dir = output_dir / "evaluate"
    prediction_dir = output_dir / "dose_predictions"
    exported_eval_dir = output_dir / "evaluate_exported"

    manifest_path = output_dir / "run_manifest.json"
    train_metrics = read_csv(train_dir / "metrics.csv")
    checkpoint_eval = read_csv(evaluate_dir / "per_sample_metrics.csv")
    checkpoint_summary = read_csv(evaluate_dir / "summary_metrics.csv")
    prediction_manifest = read_csv(prediction_dir / "prediction_manifest.csv")
    exported_metrics = read_csv(exported_eval_dir / "exported_prediction_metrics.csv")
    exported_summary = read_csv(exported_eval_dir / "exported_prediction_summary.csv")

    audit_rows: list[dict[str, str]] = []
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    split = split_counts(Path(args.split_csv))
    check(split.get("train") == args.expected_train_cases, "train_case_count", str(split), audit_rows)
    check(split.get("val") == args.expected_val_cases, "val_case_count", str(split), audit_rows)

    data = manifest.get("data", {}).get("splits", {}) if manifest else {}
    train_samples = data.get("train", {}).get("dose_samples")
    val_samples = data.get("val", {}).get("dose_samples")
    check(train_samples == args.expected_train_samples, "manifest_train_samples", str(train_samples), audit_rows)
    check(val_samples == args.expected_val_samples, "manifest_val_samples", str(val_samples), audit_rows)
    check(manifest.get("status") == "complete", "manifest_status_complete", str(manifest.get("status")), audit_rows)

    check(len(train_metrics) >= args.expected_epochs, "train_epoch_rows", str(len(train_metrics)), audit_rows)
    check((train_dir / "checkpoints" / "best.pt").exists(), "best_checkpoint_exists", "best.pt", audit_rows)
    check((train_dir / "checkpoints" / "last.pt").exists(), "last_checkpoint_exists", "last.pt", audit_rows)

    check(len(checkpoint_eval) == args.expected_val_samples, "checkpoint_eval_rows", str(len(checkpoint_eval)), audit_rows)
    check(bool(checkpoint_summary), "checkpoint_summary_exists", str(len(checkpoint_summary)), audit_rows)

    check(len(prediction_manifest) == args.expected_val_samples, "prediction_manifest_rows", str(len(prediction_manifest)), audit_rows)
    missing_predictions = [
        row.get("full_mha", "")
        for row in prediction_manifest
        if not row.get("full_mha") or not Path(row["full_mha"]).exists()
    ]
    check(not missing_predictions, "prediction_files_exist", f"missing={len(missing_predictions)}", audit_rows)

    check(len(exported_metrics) == args.expected_val_samples, "exported_metric_rows", str(len(exported_metrics)), audit_rows)
    summary_metrics = {row.get("metric", "") for row in exported_summary}
    missing_summary = sorted(REQUIRED_SUMMARY_METRICS - summary_metrics)
    check(not missing_summary, "required_exported_summary_metrics", "missing=" + ",".join(missing_summary), audit_rows)
    check((exported_eval_dir / "evaluation_runtime.json").exists(), "exported_eval_runtime_exists", "evaluation_runtime.json", audit_rows)

    output_csv = output_dir / "completion_audit.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["status", "check", "detail"])
        writer.writeheader()
        writer.writerows(audit_rows)

    failed = [row for row in audit_rows if row["status"] != "PASS"]
    print(f"wrote_audit={output_csv}", flush=True)
    print(f"checks={len(audit_rows)} failed={len(failed)}", flush=True)
    if failed:
        for row in failed:
            print(f"FAIL {row['check']} {row['detail']}", flush=True)
        raise SystemExit(1)
    print("full_baseline_completion=PASS", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-csv", default="splits/photon_case_split.csv")
    parser.add_argument("--expected-epochs", type=int, default=10)
    parser.add_argument("--expected-train-cases", type=int, default=60)
    parser.add_argument("--expected-val-cases", type=int, default=15)
    parser.add_argument("--expected-train-samples", type=int, default=32400)
    parser.add_argument("--expected-val-samples", type=int, default=8100)
    return parser.parse_args()


if __name__ == "__main__":
    audit(parse_args())
