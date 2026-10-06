"""Uniform dequantisation within floating-point rounding cells."""
import numpy as np


def dequantise_float_cells(values, rng, dtype="float16", bounds=None):
    """Smooth exact quantised values without crossing their rounding-cell edges."""
    if dtype not in ("float16", "float32"):
        raise ValueError("Dequantisation dtype must be float16 or float32")
    values = np.asarray(values, dtype=np.float64)
    quantised = values.astype(dtype)
    if not np.all(np.isfinite(values)) or not np.all(values == quantised.astype(np.float64)):
        raise ValueError("Dequantisation requires finite values exactly on the configured precision grid")
    previous = np.nextafter(quantised, np.array(-np.inf, dtype=dtype)).astype(np.float64)
    following = np.nextafter(quantised, np.array(np.inf, dtype=dtype)).astype(np.float64)
    lower = 0.5 * previous + 0.5 * values
    upper = 0.5 * following + 0.5 * values
    if bounds is not None:
        if len(bounds) != 2 or not bounds[0] < bounds[1]:
            raise ValueError("Dequantisation bounds must be [minimum, maximum]")
        if np.any(values < bounds[0]) or np.any(values > bounds[1]):
            raise ValueError("Dequantisation input lies outside physical bounds")
        lower = np.maximum(lower, bounds[0])
        upper = np.minimum(upper, bounds[1])
    if not np.all(np.isfinite(lower) & np.isfinite(upper) & (upper > lower)):
        raise ValueError("Dequantisation rounding cells must have finite, positive width")
    return lower + rng.random(values.shape) * (upper - lower)


def get_dequantisation_options(preprocess):
    """Read dataset-wide settings, accepting the original density-only key."""
    return preprocess.get("dequantisation", preprocess.get("density_dequantisation", {}))


class Dequantisation:
    """Reproducible batch transform applied once when final datasets are built."""

    def __init__(self, options, context=""):
        import hashlib
        self.options = options
        context_seed = int.from_bytes(hashlib.sha256(context.encode()).digest()[:8], "little")
        self.streams = {
            col: np.random.default_rng(np.random.SeedSequence([settings.get("seed", 42), context_seed]))
            for col, settings in options.items()
        }

    def __call__(self, df):
        for col, settings in self.options.items():
            if col not in df.columns:
                continue
            df[col] = dequantise_float_cells(
                df[col].to_numpy(), self.streams[col],
                dtype=settings.get("dtype", "float16"), bounds=settings.get("bounds"),
            )
        return df
