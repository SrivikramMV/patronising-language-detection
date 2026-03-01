import numpy as np
import pytest

from pcl_evaluation import (
    compute_binary_metrics,
    find_best_threshold,
    weighted_average_probabilities,
)


def test_binary_metrics_match_expected_confusion_counts() -> None:
    labels = np.array([0, 0, 0, 1, 1, 1])
    predictions = np.array([0, 1, 0, 1, 0, 1])

    metrics = compute_binary_metrics(labels, predictions)

    assert metrics.true_negatives == 2
    assert metrics.false_positives == 1
    assert metrics.false_negatives == 1
    assert metrics.true_positives == 2
    assert metrics.precision == pytest.approx(2 / 3)
    assert metrics.recall == pytest.approx(2 / 3)
    assert metrics.f1 == pytest.approx(2 / 3)


def test_threshold_search_selects_the_best_candidate() -> None:
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.05, 0.35, 0.40, 0.90])

    result = find_best_threshold(labels, probabilities, np.array([0.30, 0.40, 0.50]))

    assert result.threshold == pytest.approx(0.40)
    assert result.f1 == pytest.approx(1.0)


def test_weighted_average_probabilities_normalises_weights() -> None:
    probabilities = np.array([[0.1, 0.8], [0.3, 0.4]])

    ensemble = weighted_average_probabilities(probabilities, np.array([3.0, 1.0]))

    np.testing.assert_allclose(ensemble, np.array([0.15, 0.70]))


def test_ensemble_rejects_invalid_probabilities() -> None:
    with pytest.raises(ValueError):
        weighted_average_probabilities(np.array([[0.2, 1.2]]))
