import numpy as np
from scipy.interpolate import PchipInterpolator

class FastPchipInverse:
    """
    Vectorized inverse of a strictly increasing scalar PchipInterpolator.

    Uses safeguarded Newton iterations followed by vectorized bisection.
    """

    def __init__(
        self,
        forward: PchipInterpolator,
        *,
        newton_iter: int = 12,
        bisection_iter: int = 40,
        bounds_error: bool = True,
    ):
        self.forward = forward
        self.x_nodes = np.asarray(forward.x, dtype=float)
        self.y_nodes = np.asarray(forward(self.x_nodes), dtype=float)

        if self.y_nodes.ndim != 1:
            raise ValueError("Only scalar-valued PCHIP objects are supported.")

        if not np.all(np.diff(self.y_nodes) > 0):
            raise ValueError(
                "The PCHIP must be strictly increasing. "
                "Flat sections do not have a unique inverse."
            )

        self.coefficients = np.asarray(forward.c, dtype=float)

        if self.coefficients.shape[0] != 4:
            raise ValueError("Expected a piecewise cubic polynomial.")

        self.newton_iter = newton_iter
        self.bisection_iter = bisection_iter
        self.bounds_error = bounds_error

    def __call__(self, y):
        y = np.asarray(y, dtype=float)
        scalar_input = y.ndim == 0
        shape = y.shape
        yq = y.reshape(-1)

        ymin = self.y_nodes[0]
        ymax = self.y_nodes[-1]

        if self.bounds_error:
            invalid = ~np.isfinite(yq) | (yq < ymin) | (yq > ymax)
            if np.any(invalid):
                raise ValueError(
                    f"Input lies outside the invertible range [{ymin}, {ymax}]."
                )
        else:
            yq = np.clip(yq, ymin, ymax)

        # Locate the monotonic output interval.
        interval = np.searchsorted(
            self.y_nodes,
            yq,
            side="right",
        ) - 1

        interval = np.clip(
            interval,
            0,
            len(self.x_nodes) - 2,
        )

        x0 = self.x_nodes[interval]
        h = self.x_nodes[interval + 1] - x0

        y0 = self.y_nodes[interval]
        y1 = self.y_nodes[interval + 1]

        a = self.coefficients[0, interval]
        b = self.coefficients[1, interval]
        c = self.coefficients[2, interval]
        d = self.coefficients[3, interval] - yq

        # Local coordinate t = x - x0.
        lo = np.zeros_like(yq)
        hi = h.copy()

        # Linear interpolation gives a good initial estimate.
        t = h * (yq - y0) / (y1 - y0)
        t = np.clip(t, lo, hi)

        def polynomial(t_value):
            return ((a * t_value + b) * t_value + c) * t_value + d

        def derivative(t_value):
            return (3.0 * a * t_value + 2.0 * b) * t_value + c

        # Safeguarded Newton steps.
        for _ in range(self.newton_iter):
            residual = polynomial(t)

            above = residual > 0.0
            hi = np.where(above, t, hi)
            lo = np.where(above, lo, t)

            slope = derivative(t)

            with np.errstate(divide="ignore", invalid="ignore"):
                proposal = t - residual / slope

            valid_newton = (
                np.isfinite(proposal)
                & (proposal > lo)
                & (proposal < hi)
                & (slope > 0.0)
            )

            midpoint = 0.5 * (lo + hi)
            t = np.where(valid_newton, proposal, midpoint)

        # Guaranteed refinement.
        for _ in range(self.bisection_iter):
            midpoint = 0.5 * (lo + hi)
            residual = polynomial(midpoint)

            above = residual > 0.0
            hi = np.where(above, midpoint, hi)
            lo = np.where(above, lo, midpoint)

        t = 0.5 * (lo + hi)
        result = (x0 + t).reshape(shape)

        if scalar_input:
            return result.item()

        return result