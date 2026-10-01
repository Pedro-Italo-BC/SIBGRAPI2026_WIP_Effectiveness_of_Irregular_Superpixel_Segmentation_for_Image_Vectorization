#!/usr/bin/env python3
"""Compute standard deviation of benchmark metrics for each configuration.

This script scans all CSV files under metrics/results and computes the standard
deviation of the numeric metric columns for each configuration file.

Output is written to a single CSV file with columns:
- file
- method
- superpixels
- metric
- stddev
- count

Example:
    python3 metrics/compute_metrics_stddev.py
    python3 metrics/compute_metrics_stddev.py --input metrics/results --output metrics/results_stddev.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import pstdev
from typing import Any


NUMERIC_METRICS = (
    "mse",
    "psnr",
    "ssim",
    "lpips",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute standard deviation of benchmark metrics for each CSV configuration."
    )
    parser.add_argument(
        "--input",
        default="metrics/results",
        help="Directory containing metric CSV files. Defaults to metrics/results.",
    )
    parser.add_argument(
        "--output",
        default="metrics/results_stddev.csv",
        help="Output CSV file path. Defaults to metrics/results_stddev.csv.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return [row for row in reader if row.get("image") != "__mean__"]


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.strip()
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def compute_file_stddev(path: Path) -> list[dict[str, Any]]:
    rows = read_csv(path)
    if not rows:
        return []

    config = rows[0]
    method = config.get("method", "")
    superpixels = config.get("superpixels", "")

    results: list[dict[str, Any]] = []
    for metric in NUMERIC_METRICS:
        values = [
            parse_float(row.get(metric))
            for row in rows
            if row.get("status", "") == "ok"
        ]
        values = [value for value in values if value is not None]
        if not values:
            stddev = ""
            count = 0
        else:
            stddev = pstdev(values)
            count = len(values)
        results.append(
            {
                "file": str(path.name),
                "method": method,
                "superpixels": superpixels,
                "metric": metric,
                "stddev": f"{stddev:.8f}" if stddev != "" else "",
                "count": count,
            }
        )
    return results


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = ["file", "method", "superpixels", "metric", "stddev", "count"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input)
    output_file = Path(args.output)

    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        raise SystemExit(f"No CSV files found in {input_dir}")

    summary_rows: list[dict[str, Any]] = []
    for csv_path in csv_files:
        summary_rows.extend(compute_file_stddev(csv_path))

    write_summary(output_file, summary_rows)
    print(f"Standard deviation summary written to: {output_file}")


if __name__ == "__main__":
    main()
