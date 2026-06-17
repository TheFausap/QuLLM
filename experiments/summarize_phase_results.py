"""Summarize phase relation scaling CSV output without extra dependencies."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", nargs="?", default="runs/phase_relation_scaling.csv")
    args = parser.parse_args()

    path = Path(args.csv_path)
    rows_by_size: dict[int, dict[str, float]] = defaultdict(dict)
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows_by_size[int(row["train_size"])][row["model"]] = float(row["accuracy"])

    print(f"results: {path}")
    print()
    for train_size in sorted(rows_by_size):
        results = rows_by_size[train_size]
        reference_name = next(
            (name for name in ("phase_unitary", "phase_margin", "phase_feature") if name in results),
            None,
        )
        reference = results.get(reference_name) if reference_name else None
        print(f"train_size={train_size}")
        for model, accuracy in sorted(results.items(), key=lambda item: item[0]):
            gap = ""
            if reference is not None and model != reference_name:
                gap = f"  {reference_name}_gap={reference - accuracy:+.4f}"
            print(f"  {model:16s} accuracy={accuracy:.4f}{gap}")
        print()


if __name__ == "__main__":
    main()
