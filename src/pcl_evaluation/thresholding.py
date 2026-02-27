from dataclasses import dataclass

import numpy as np

from .metrics import compute_binary_metrics


@dataclass(frozen=True)
class ThresholdSearchResult:
    threshold: float
    f1: float
    precision: float
    recall: float


def find_best_threshold(
    true_labels: np.ndarray,
    positive_probabilities: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> ThresholdSearchResult:
    labels = np.asarray(true_labels, dtype=np.int64).reshape(-1)
    probabilities = np.asarray(positive_probabilities, dtype=np.float64).reshape(-1)
    if labels.size != probabilities.size:
        raise ValueError("true_labels and positive_probabilities must have equal length")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("positive_probabilities must be finite values between zero and one")

    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    candidates = np.asarray(thresholds, dtype=np.float64).reshape(-1)
    if candidates.size == 0:
        raise ValueError("at least one threshold is required")

    best: ThresholdSearchResult | None = None
    for threshold in candidates:
        metrics = compute_binary_metrics(labels, (probabilities >= threshold).astype(np.int64))
        result = ThresholdSearchResult(
            threshold=float(threshold),
            f1=metrics.f1,
            precision=metrics.precision,
            recall=metrics.recall,
        )
        if best is None or (result.f1, result.recall, -result.threshold) > (
            best.f1,
            best.recall,
            -best.threshold,
        ):
            best = result

    assert best is not None
    return best
