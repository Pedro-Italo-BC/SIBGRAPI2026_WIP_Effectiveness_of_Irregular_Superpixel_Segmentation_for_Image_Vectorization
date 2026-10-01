#!/usr/bin/env python3
"""Aggregate effective SLIC superpixel counts by target configuration.

This script reads metrics/effective_superpixels.csv and computes the average
number of effective superpixels for each target superpixel configuration.

It writes a summary CSV containing:
- target_superpixels
- count
- mean_effective_superpixels
- min_effective_superpixels
- max_effective_superpixels
- stddev_effective_superpixels

Usage:
    python3 metrics/aggregate_effective_superpixels.py
    python3 metrics/aggregate_effective_superpixels.py --input metrics/effective_superpixels.csv --output metrics/effective_superpixels_summary.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate effective superpixel counts by target configuration."
    )
    parser.add_argument(
        "--input",
        default="metrics/effective_superpixels.csv",
        help="Input CSV file with effective superpixel counts. Defaults to metrics/effective_superpixels.csv.",
    )
    parser.add_argument(
        "--output",
        default="metrics/effective_superpixels_summary.csv",
        help="Output CSV file for the aggregated summary. Defaults to metrics/effective_superpixels_summary.csv.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return [row for row in reader]


def aggregate(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        try:
            target = int(row["target_superpixels"])
            effective = float(row["effective_superpixels"])
        except (KeyError, ValueError):
            continue
        groups[target].append(effective)

    summary: list[dict[str, Any]] = []
    for target in sorted(groups):
        values = groups[target]
        if not values:
            continue
        summary.append({
            "target_superpixels": target,
            "count": len(values),
            "mean_effective_superpixels": f"{mean(values):.6f}",
            "min_effective_superpixels": f"{min(values):.6f}",
            "max_effective_superpixels": f"{max(values):.6f}",
            "stddev_effective_superpixels": f"{pstdev(values):.6f}",
        })
    return summary


def write_summary(path: Path, summary: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "target_superpixels",
        "count",
        "mean_effective_superpixels",
        "min_effective_superpixels",
        "max_effective_superpixels",
        "stddev_effective_superpixels",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.is_file():
        raise SystemExit(f"Input file does not exist: {input_path}")

    rows = read_rows(input_path)
    summary = aggregate(rows)
    write_summary(output_path, summary)
    print(f"Aggregated summary written to: {output_path}")


if __name__ == "__main__":
    main()
