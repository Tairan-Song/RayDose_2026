"""Create a case-level train/validation split for DoseRAD photon data.

The split is performed by case, not by dose/control-point file. This prevents
the same patient CT from appearing in both training and validation.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def list_cases(training_dir: Path) -> list[str]:
    case_ids: list[str] = []
    for case_dir in sorted(p for p in training_dir.iterdir() if p.is_dir()):
        case_id = case_dir.name
        if (case_dir / "image" / "ct.mha").exists() and (case_dir / f"{case_id}.json").exists():
            case_ids.append(case_id)
    return case_ids


def make_split(case_ids: list[str], train_fraction: float, seed: int) -> list[tuple[str, str]]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("--train-fraction must be between 0 and 1")

    shuffled = case_ids[:]
    random.Random(seed).shuffle(shuffled)

    train_count = round(len(shuffled) * train_fraction)
    train_cases = set(shuffled[:train_count])

    return [(case_id, "train" if case_id in train_cases else "val") for case_id in sorted(case_ids)]


def write_split(path: Path, rows: list[tuple[str, str]], seed: int, train_fraction: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "split", "seed", "train_fraction"])
        writer.writeheader()
        for case_id, split in rows:
            writer.writerow(
                {
                    "case_id": case_id,
                    "split": split,
                    "seed": seed,
                    "train_fraction": train_fraction,
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", default="data/photon/training")
    parser.add_argument("--output-csv", default="splits/photon_case_split.csv")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=20260628)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    case_ids = list_cases(Path(args.training_dir))
    if not case_ids:
        raise RuntimeError(f"No valid cases found under {args.training_dir}")

    rows = make_split(case_ids, train_fraction=args.train_fraction, seed=args.seed)
    write_split(Path(args.output_csv), rows, seed=args.seed, train_fraction=args.train_fraction)

    train_count = sum(split == "train" for _, split in rows)
    val_count = sum(split == "val" for _, split in rows)
    print(f"cases={len(rows)} train={train_count} val={val_count} output={args.output_csv}")


if __name__ == "__main__":
    main()
