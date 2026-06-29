"""Validate required inputs for the photon baseline training pipeline."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import TypeVar

from doserad_dataset import parse_dose_name
from mha_io import read_mha


T = TypeVar("T")

REQUIRED_CP_FIELDS = [
    "cp_idx",
    "gantry_angle",
    "mlc_left_int_mm",
    "mlc_right_int_mm",
]


def read_split_rows(split_csv: Path) -> list[dict[str, str]]:
    with split_csv.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def limited_items(items: list[T], limit: int) -> list[T]:
    return items if limit <= 0 else items[:limit]


def validate_case_json(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"json_read_error:{exc}"]

    beams = data.get("beams")
    if not isinstance(beams, list) or not beams:
        return ["missing_or_empty_beams"]

    for beam in beams:
        if "beam_idx" not in beam:
            errors.append("beam_missing_beam_idx")
        if "iso_center" not in beam or len(beam["iso_center"]) != 3:
            errors.append(f"beam_{beam.get('beam_idx', '?')}_invalid_iso_center")
        cps = beam.get("control_points")
        if not isinstance(cps, list) or not cps:
            errors.append(f"beam_{beam.get('beam_idx', '?')}_missing_control_points")
            continue
        for cp in cps[:3]:
            for field in REQUIRED_CP_FIELDS:
                if field not in cp:
                    errors.append(f"beam_{beam.get('beam_idx', '?')}_cp_missing_{field}")
            if "mlc_left_int_mm" in cp and len(cp["mlc_left_int_mm"]) != 80:
                errors.append(f"beam_{beam.get('beam_idx', '?')}_cp_{cp.get('cp_idx', '?')}_left_mlc_not_80")
            if "mlc_right_int_mm" in cp and len(cp["mlc_right_int_mm"]) != 80:
                errors.append(f"beam_{beam.get('beam_idx', '?')}_cp_{cp.get('cp_idx', '?')}_right_mlc_not_80")

    return errors


def maybe_read_mha(path: Path, check_mha: bool) -> tuple[str, str, str]:
    if not check_mha:
        return "", "", ""
    try:
        image = read_mha(path)
        return " ".join(str(int(v)) for v in image.array.shape), image.meta.get("ElementSpacing", ""), ""
    except Exception as exc:
        return "", "", f"mha_read_error:{exc}"


def validate_case(
    training_dir: Path,
    case_id: str,
    split: str,
    mask_name: str,
    max_dose_files_per_case: int,
    check_mha: bool,
) -> dict[str, object]:
    case_dir = training_dir / case_id
    image_dir = case_dir / "image"
    dose_dir = case_dir / "dose"
    mask_dir = case_dir / "label_masks" / mask_name
    ct_path = image_dir / "ct.mha"
    json_path = case_dir / f"{case_id}.json"

    errors: list[str] = []
    ct_shape = ""
    ct_spacing = ""

    if not case_dir.exists():
        errors.append("missing_case_dir")
    if not ct_path.exists():
        errors.append("missing_ct")
    else:
        ct_shape, ct_spacing, ct_error = maybe_read_mha(ct_path, check_mha)
        if ct_error:
            errors.append(f"ct_{ct_error}")
    if not json_path.exists():
        errors.append("missing_case_json")
    else:
        errors.extend(validate_case_json(json_path))

    dose_files = sorted(dose_dir.glob("Dose_B*_CP*.mha")) if dose_dir.exists() else []
    if not dose_dir.exists():
        errors.append("missing_dose_dir")
    if not dose_files:
        errors.append("no_dose_files")
    if not mask_dir.exists():
        errors.append(f"missing_mask_dir:{mask_name}")

    checked = 0
    for dose_path in limited_items(dose_files, max_dose_files_per_case):
        checked += 1
        try:
            beam_idx, cp_idx = parse_dose_name(dose_path)
        except Exception as exc:
            errors.append(f"invalid_dose_name:{dose_path.name}:{exc}")
            continue
        mask_path = mask_dir / f"Dose_B{beam_idx}_CP{cp_idx:03d}_mask.mha"
        if not mask_path.exists():
            errors.append(f"missing_mask:{mask_path.name}")
        if check_mha:
            _shape, _spacing, dose_error = maybe_read_mha(dose_path, check_mha=True)
            if dose_error:
                errors.append(f"dose_{dose_path.name}_{dose_error}")
            if mask_path.exists():
                _mask_shape, _mask_spacing, mask_error = maybe_read_mha(mask_path, check_mha=True)
                if mask_error:
                    errors.append(f"mask_{mask_path.name}_{mask_error}")

    return {
        "case_id": case_id,
        "split": split,
        "ct_shape": ct_shape,
        "ct_spacing": ct_spacing,
        "dose_files": len(dose_files),
        "checked_dose_files": checked,
        "errors": ";".join(errors),
    }


def write_report(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["case_id", "split", "ct_shape", "ct_spacing", "dose_files", "checked_dose_files", "errors"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def validate(args: argparse.Namespace) -> None:
    training_dir = Path(args.training_dir)
    split_csv = Path(args.split_csv)
    report_csv = Path(args.report_csv)

    errors: list[str] = []
    if not training_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {training_dir}")
    if not split_csv.exists():
        raise FileNotFoundError(f"Split CSV not found: {split_csv}")
    if not (training_dir / "beam_parameters.json").exists():
        errors.append("missing_beam_parameters_json")

    split_rows = read_split_rows(split_csv)
    case_rows = [row for row in split_rows if row.get("split") in {"train", "val"}]
    case_rows = limited_items(case_rows, args.max_cases)
    seen_cases: set[str] = set()
    report_rows: list[dict[str, object]] = []

    for row in case_rows:
        case_id = row["case_id"]
        if case_id in seen_cases:
            errors.append(f"duplicate_case_in_split:{case_id}")
        seen_cases.add(case_id)
        report_rows.append(
            validate_case(
                training_dir=training_dir,
                case_id=case_id,
                split=row["split"],
                mask_name=args.mask_name,
                max_dose_files_per_case=args.max_dose_files_per_case,
                check_mha=args.check_mha,
            )
        )

    write_report(report_csv, report_rows)
    case_errors = [row for row in report_rows if row["errors"]]
    total_dose_files = sum(int(row["dose_files"]) for row in report_rows)
    checked_dose_files = sum(int(row["checked_dose_files"]) for row in report_rows)

    print(
        f"cases={len(report_rows)}",
        f"case_errors={len(case_errors)}",
        f"dose_files={total_dose_files}",
        f"checked_dose_files={checked_dose_files}",
        f"report={report_csv}",
        flush=True,
    )

    if errors or case_errors:
        for error in errors:
            print(f"error={error}", flush=True)
        raise RuntimeError(f"Baseline data validation failed. See {report_csv}")

    print("baseline_data_validation_passed", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", default="data/photon/training")
    parser.add_argument("--split-csv", default="splits/photon_case_split.csv")
    parser.add_argument("--mask-name", default="dose_gt_1pct")
    parser.add_argument("--report-csv", default="outputs/data_validation/baseline_data_validation.csv")
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--max-dose-files-per-case", type=int, default=0)
    parser.add_argument("--check-mha", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    validate(parse_args())
