import numpy as np


def weighted_average_probabilities(
    probability_matrix: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    probabilities = np.asarray(probability_matrix, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[0] == 0:
        raise ValueError("probability_matrix must have shape [models, examples]")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("all probabilities must be finite values between zero and one")

    if weights is None:
        normalised_weights = np.full(probabilities.shape[0], 1.0 / probabilities.shape[0])
    else:
        model_weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        if model_weights.size != probabilities.shape[0]:
            raise ValueError("one weight is required for each model")
        if not np.isfinite(model_weights).all() or np.any(model_weights < 0):
            raise ValueError("weights must be finite and non-negative")
        weight_sum = model_weights.sum()
        if weight_sum <= 0:
            raise ValueError("weights must have a positive sum")
        normalised_weights = model_weights / weight_sum

    return np.average(probabilities, axis=0, weights=normalised_weights)
