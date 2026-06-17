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
    rows_by_size: dict[int, dict[str, dict[str, str]]] = defaultdict(dict)
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows_by_size[int(row["train_size"])][row["model"]] = row

    print(f"results: {path}")
    print()
    for train_size in sorted(rows_by_size):
        results = rows_by_size[train_size]
        reference_name = next(
            (
                name
                for name in ("phase_unitary", "phase_margin_fixed", "phase_margin", "phase_feature")
                if name in results
            ),
            None,
        )
        reference = float(results[reference_name]["accuracy"]) if reference_name else None
        print(f"train_size={train_size}")
        for model, row in sorted(results.items(), key=lambda item: item[0]):
            accuracy = float(row["accuracy"])
            gap = ""
            if reference is not None and model != reference_name:
                gap = f"  {reference_name}_gap={reference - accuracy:+.4f}"
            params = row.get("trainable_params", "")
            param_text = f" params={params}" if params else ""
            phase_mean_error = row.get("phase_mean_error", "")
            phase_max_error = row.get("phase_max_error", "")
            phase_text = ""
            if phase_mean_error:
                phase_text = f" phase_mean_err={float(phase_mean_error):.3f}"
                if phase_max_error:
                    phase_text += f" phase_max_err={float(phase_max_error):.3f}"
            phase_rule_accuracy = row.get("phase_rule_accuracy", "")
            rule_text = f" phase_rule_acc={float(phase_rule_accuracy):.4f}" if phase_rule_accuracy else ""
            breakdown = ""
            if row.get("positive_accuracy", ""):
                breakdown = (
                    f" pos={float(row['positive_accuracy']):.4f}"
                    f" same_neg={float(row['same_group_negative_accuracy']):.4f}"
                    f" diff_neg={float(row['diff_group_negative_accuracy']):.4f}"
                )
            print(f"  {model:18s} accuracy={accuracy:.4f}{gap}{param_text}{phase_text}{rule_text}{breakdown}")
        print()


if __name__ == "__main__":
    main()
