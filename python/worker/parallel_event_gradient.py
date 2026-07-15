import numpy as np
from typing import Callable, Optional, Tuple, Union


class ParallelEventGradient:
    """
    Numerical gradients for independent events.

    The function is always called with a 2D array:

        input:  (n_events, n_variables)
        output: (n_events,) or (n_events, 1)

    Events are vectorised. Finite-difference variables and step sizes
    are evaluated sequentially.
    """

    def __init__(
        self,
        num_steps: int = 15,
        step_ratio: float = 2.0,
        relative_step: Optional[float] = None,
        return_error: bool = False,
    ) -> None:
        if num_steps < 1:
            raise ValueError("num_steps must be at least 1")

        if step_ratio <= 1.0:
            raise ValueError("step_ratio must be greater than 1")

        if relative_step is not None and relative_step <= 0.0:
            raise ValueError("relative_step must be positive")

        self.num_steps = int(num_steps)
        self.step_ratio = float(step_ratio)
        self.relative_step = relative_step
        self.return_error = bool(return_error)

    def __call__(
        self,
        func: Callable[[np.ndarray], np.ndarray],
        x: np.ndarray,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        x = np.asarray(x, dtype=float)

        if x.ndim != 2:
            raise ValueError(
                "x must have shape (n_events, n_variables)"
            )

        n_events, n_variables = x.shape

        # Optional validation. This calls func once at the original point.
        self._evaluate(func, x, n_events)

        base_step = self._make_base_step(x)

        gradient = np.empty_like(x)
        error = np.empty_like(x)

        for variable in range(n_variables):
            # Shape: (num_steps, n_events)
            estimates = np.empty(
                (self.num_steps, n_events),
                dtype=float,
            )

            for step_index in range(self.num_steps):
                scale = self.step_ratio ** step_index
                h = base_step[:, variable] / scale

                x_plus = x.copy()
                x_minus = x.copy()

                # Perturb this variable for all events at once.
                x_plus[:, variable] += h
                x_minus[:, variable] -= h

                y_plus = self._evaluate(func, x_plus, n_events)
                y_minus = self._evaluate(func, x_minus, n_events)

                estimates[step_index] = (
                    y_plus - y_minus
                ) / (2.0 * h)

            derivative, derivative_error = (
                self._richardson_extrapolate(estimates)
            )

            gradient[:, variable] = derivative
            error[:, variable] = derivative_error

        if self.return_error:
            return gradient, error

        return gradient

    def _make_base_step(self, x: np.ndarray) -> np.ndarray:
        if self.relative_step is None:
            epsilon = np.finfo(x.dtype).eps

            # Appropriate scale for central differences followed by
            # num_steps levels of Richardson extrapolation.
            relative_step = epsilon ** (
                1.0 / (2 * self.num_steps + 1)
            )
        else:
            relative_step = self.relative_step

        return relative_step * np.maximum(1.0, np.abs(x))

    @staticmethod
    def _evaluate(
        func: Callable[[np.ndarray], np.ndarray],
        x: np.ndarray,
        n_events: int,
    ) -> np.ndarray:
        """
        Evaluate func while accepting either:
            (n_events,)
        or:
            (n_events, 1)
        """
        output = np.asarray(func(x))

        if output.shape == (n_events,):
            return output

        if output.shape == (n_events, 1):
            return output[:, 0]

        raise ValueError(
            "func must return one scalar per event. "
            f"For input shape {x.shape}, expected "
            f"{(n_events,)} or {(n_events, 1)}, "
            f"but received {output.shape}."
        )

    def _richardson_extrapolate(
        self,
        estimates: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Richardson extrapolation for second-order central differences.
        """
        if self.num_steps == 1:
            derivative = estimates[0]
            error = np.full_like(derivative, np.nan)
            return derivative, error

        table = estimates.copy()
        previous_best = table[-1].copy()

        for level in range(1, self.num_steps):
            previous_best = table[-1].copy()

            factor = self.step_ratio ** (2 * level)

            table = (
                factor * table[1:] - table[:-1]
            ) / (factor - 1.0)

        derivative = table[-1]
        error = np.abs(derivative - previous_best)

        return derivative, error