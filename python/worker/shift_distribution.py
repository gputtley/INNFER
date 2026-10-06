"""Analytic shift probabilities and histogram corrections for preprocessing."""

import numpy as np
from scipy.special import ndtr, ndtri


def shift_cdf(values, shift):
    """CDF of a uniform shift or a flat plateau with Gaussian tails."""
    values = np.asarray(values, dtype=float)
    low, high = shift["range"]
    if high <= low:
        raise ValueError("Shift range must have positive width")
    if shift["type"] == "continuous":
        return np.clip((values - low) / (high - low), 0.0, 1.0)
    if shift["type"] != "flat_top":
        raise ValueError(f"Unsupported shift type: {shift['type']}")
    sigma = shift.get("other", {}).get("sigma_out", 0.1 * (high - low))
    if sigma <= 0:
        raise ValueError("Flat-top tail width must be positive")
    tail_area = sigma * np.sqrt(2.0 * np.pi)
    area = high - low + tail_area
    return np.where(
        values < low,
        tail_area * ndtr((values - low) / sigma) / area,
        np.where(
            values > high,
            1.0 - tail_area * ndtr((high - values) / sigma) / area,
            (tail_area / 2.0 + values - low) / area,
        ),
    )


def shift_quantiles(probabilities, shift):
    probabilities = np.asarray(probabilities, dtype=float)
    low, high = shift["range"]
    if shift["type"] == "continuous":
        return low + probabilities * (high - low)
    sigma = shift.get("other", {}).get("sigma_out", 0.1 * (high - low))
    tail_area = sigma * np.sqrt(2.0 * np.pi)
    area = high - low + tail_area
    tail_probability = tail_area / (2.0 * area)
    result = low + probabilities * area - tail_area / 2.0
    lower = probabilities < tail_probability
    upper = probabilities > 1.0 - tail_probability
    result[lower] = low + sigma * ndtri(probabilities[lower] * area / tail_area)
    result[upper] = high - sigma * ndtri((1.0 - probabilities[upper]) * area / tail_area)
    return result


def bin_factors(values, edges, factors):
    """Evaluate a binwise correction, including the final right edge."""
    indices = np.searchsorted(edges, values, side="right") - 1
    result = factors[np.clip(indices, 0, len(factors) - 1)]
    return np.where((values >= edges[0]) & (values <= edges[-1]), result, 0.0)
