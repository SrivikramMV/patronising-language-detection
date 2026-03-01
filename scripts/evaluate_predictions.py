#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from pcl_evaluation.metrics import compute_binary_metrics  # noqa: E402


def read_vector(path: Path, column: str | None) -> np.ndarray:
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        if column is None:
            if len(frame.columns) != 1:
                raise ValueError("CSV inputs with multiple columns require --label-column")
            column = str(frame.columns[0])
        if column not in frame.columns:
            raise ValueError(f"column '{column}' was not found in {path}")
        return frame[column].to_numpy(dtype=np.int64)

    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return np.asarray([int(value) for value in values], dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate binary PCL predictions.")
    parser.add_argument("--labels", type=Path, required=True, help="One label per line or a CSV file.")
    parser.add_argument("--predictions", type=Path, required=True, help="One binary prediction per line.")
    parser.add_argument("--label-column", default=None, help="Label column when --labels is a CSV file.")
    args = parser.parse_args()

    labels = read_vector(args.labels, args.label_column)
    predictions = read_vector(args.predictions, None)
    metrics = compute_binary_metrics(labels, predictions)
    print(json.dumps(metrics.to_dict(), indent=2))


if __name__ == "__main__":
    main()
