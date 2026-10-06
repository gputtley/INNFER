import numpy as np
from typing import Callable, Optional, Tuple, Union


class ParallelEventGradient:
    """
    Numerical gradients for independent, vectorised events.

    func input:  (n_events, n_variables)
    func output: (n_events,) or (n_events, 1)
    """

    def __init__(
        self,
        num_steps: int = 6,
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
        x = np.asarray(x, dtype=np.float64)

        if x.ndim != 2:
            raise ValueError(
                "x must have shape (n_events, n_variables)"
            )

        n_events, n_variables = x.shape
        self._evaluate(func, x, n_events)

        base_step = self._make_base_step(x)

        gradient = np.empty_like(x)
        error = np.empty_like(x)

        for variable in range(n_variables):
            estimates = np.empty(
                (self.num_steps, n_events),
                dtype=np.float64,
            )
            roundoff = np.empty_like(estimates)

            for step_index in range(self.num_steps):
                h = (
                    base_step[:, variable]
                    / self.step_ratio**step_index
                )

                x_plus = x.copy()
                x_minus = x.copy()

                x_plus[:, variable] += h
                x_minus[:, variable] -= h

                y_plus, eps_plus = self._evaluate(
                    func, x_plus, n_events
                )
                y_minus, eps_minus = self._evaluate(
                    func, x_minus, n_events
                )

                # Use the actual representable displacement.
                delta = (
                    x_plus[:, variable]
                    - x_minus[:, variable]
                )

                valid = delta != 0.0

                estimates[step_index] = np.divide(
                    y_plus - y_minus,
                    delta,
                    out=np.full(n_events, np.nan),
                    where=valid,
                )

                evaluation_epsilon = max(
                    eps_plus,
                    eps_minus,
                    np.finfo(np.float64).eps,
                )

                # Approximate lower bound caused by evaluation round-off.
                roundoff[step_index] = np.divide(
                    evaluation_epsilon
                    * (np.abs(y_plus) + np.abs(y_minus)),
                    np.abs(delta),
                    out=np.full(n_events, np.inf),
                    where=valid,
                )

            derivative, derivative_error = (
                self._richardson_extrapolate(
                    estimates,
                    roundoff,
                )
            )

            gradient[:, variable] = derivative
            error[:, variable] = derivative_error

        if self.return_error:
            return gradient, error

        return gradient

    def _make_base_step(self, x: np.ndarray) -> np.ndarray:
        if self.relative_step is None:
            # Fixed initial step, independent of num_steps.
            #
            # Richardson extrapolation will reduce this step and
            # automatically select the best result.
            relative_step = 1.0e-2
        else:
            relative_step = self.relative_step

        return relative_step * np.maximum(1.0, np.abs(x))

    @staticmethod
    def _evaluate(
        func: Callable[[np.ndarray], np.ndarray],
        x: np.ndarray,
        n_events: int,
    ) -> Tuple[np.ndarray, float]:
        output = np.asarray(func(x))

        if not np.issubdtype(output.dtype, np.floating):
            raise TypeError(
                "func must return floating-point values"
            )

        evaluation_epsilon = float(
            np.finfo(output.dtype).eps
        )

        if output.shape == (n_events,):
            values = output
        elif output.shape == (n_events, 1):
            values = output[:, 0]
        else:
            raise ValueError(
                "func must return one scalar per event. "
                f"For input shape {x.shape}, expected "
                f"{(n_events,)} or {(n_events, 1)}, "
                f"but received {output.shape}."
            )

        return (
            values.astype(np.float64, copy=False),
            evaluation_epsilon,
        )

    def _richardson_extrapolate(
        self,
        estimates: np.ndarray,
        roundoff: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build a Richardson table and select the entry with the
        smallest estimated error separately for every event.
        """
        n_steps, n_events = estimates.shape

        if n_steps == 1:
            return (
                estimates[0].copy(),
                np.full(n_events, np.nan),
            )

        best = estimates[0].copy()
        best_error = np.full(n_events, np.inf)

        # Also consider the unextrapolated central differences.
        for index in range(1, n_steps):
            candidate_error = np.maximum(
                np.abs(
                    estimates[index]
                    - estimates[index - 1]
                ),
                roundoff[index],
            )

            use = (
                np.isfinite(estimates[index])
                & np.isfinite(candidate_error)
                & (candidate_error < best_error)
            )

            best[use] = estimates[index, use]
            best_error[use] = candidate_error[use]

        table = estimates.copy()
        noise_table = roundoff.copy()

        for level in range(1, n_steps):
            factor = self.step_ratio ** (2 * level)

            coarse = table[:-1]
            fine = table[1:]

            coarse_noise = noise_table[:-1]
            fine_noise = noise_table[1:]

            # Algebraically equivalent to
            # (factor*fine - coarse)/(factor - 1),
            # but less susceptible to overflow.
            next_table = (
                fine
                + (fine - coarse) / (factor - 1.0)
            )

            next_noise = (
                factor * fine_noise + coarse_noise
            ) / (factor - 1.0)

            candidate_error = np.maximum.reduce(
                (
                    np.abs(next_table - fine),
                    np.abs(next_table - coarse),
                    next_noise,
                )
            )

            for row in range(next_table.shape[0]):
                use = (
                    np.isfinite(next_table[row])
                    & np.isfinite(candidate_error[row])
                    & (candidate_error[row] < best_error)
                )

                best[use] = next_table[row, use]
                best_error[use] = candidate_error[row, use]

            table = next_table
            noise_table = next_noise

        # Do not silently turn failed derivatives into zero.
        failed = ~np.isfinite(best_error)
        best[failed] = np.nan
        best_error[failed] = np.nan

        return best, best_error