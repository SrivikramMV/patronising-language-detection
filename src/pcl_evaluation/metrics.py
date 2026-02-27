from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class BinaryClassificationMetrics:
    precision: float
    recall: float
    f1: float
    accuracy: float
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _as_binary_vector(values: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.int64).reshape(-1)
    if not np.isin(vector, [0, 1]).all():
        raise ValueError(f"{name} must contain only binary values")
    return vector


def compute_binary_metrics(
    true_labels: np.ndarray,
    predictions: np.ndarray,
) -> BinaryClassificationMetrics:
    true_labels = _as_binary_vector(true_labels, "true_labels")
    predictions = _as_binary_vector(predictions, "predictions")
    if true_labels.size != predictions.size:
        raise ValueError("true_labels and predictions must have equal length")
    if true_labels.size == 0:
        raise ValueError("at least one prediction is required")

    true_positives = int(np.sum((true_labels == 1) & (predictions == 1)))
    true_negatives = int(np.sum((true_labels == 0) & (predictions == 0)))
    false_positives = int(np.sum((true_labels == 0) & (predictions == 1)))
    false_negatives = int(np.sum((true_labels == 1) & (predictions == 0)))

    precision_denominator = true_positives + false_positives
    recall_denominator = true_positives + false_negatives
    precision = true_positives / precision_denominator if precision_denominator else 0.0
    recall = true_positives / recall_denominator if recall_denominator else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (true_positives + true_negatives) / true_labels.size

    return BinaryClassificationMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=accuracy,
        true_negatives=true_negatives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        true_positives=true_positives,
    )
