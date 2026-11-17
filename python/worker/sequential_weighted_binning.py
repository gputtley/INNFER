"""Balanced sequential adaptive binning for weighted pandas data.

The public function, :func:`sequential_weighted_binning`, starts from a
regular lattice in every requested dimension and merges adjacent occupied
bins until the requested weighted statistical precision is reached.

Unlike a target-first greedy merger, the merge policy considers every
adjacent pair touching an under-precise bin and merges the smallest combined
pair.  This prevents an early bin from repeatedly absorbing its neighbours
when many initial bins have similar precision.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


Number = int | float | np.number
PerDimension = Number | Sequence[Number] | Mapping[str, Number]


@dataclass
class _Bin:
    first: int
    last: int
    sum_w: float
    sum_w2: float
    n_rows: int

    def merged(self, other: "_Bin") -> "_Bin":
        return _Bin(
            first=min(self.first, other.first),
            last=max(self.last, other.last),
            sum_w=self.sum_w + other.sum_w,
            sum_w2=self.sum_w2 + other.sum_w2,
            n_rows=self.n_rows + other.n_rows,
        )


def _expand_parameter(
    value: PerDimension, columns: list[str], name: str
) -> list[float]:
    if isinstance(value, Mapping):
        missing = [column for column in columns if column not in value]
        if missing:
            raise ValueError(f"{name} is missing values for {missing}")
        result = [float(value[column]) for column in columns]
    elif np.isscalar(value):
        result = [float(value)] * len(columns)
    else:
        result = [float(item) for item in value]
        if len(result) != len(columns):
            raise ValueError(
                f"{name} must have one value per column; "
                f"expected {len(columns)}, got {len(result)}"
            )
    return result


def _fractional_uncertainty(sum_w: float, sum_w2: float) -> float:
    """Return sqrt(sum(w**2)) / abs(sum(w))."""
    if not np.isfinite(sum_w) or not np.isfinite(sum_w2) or sum_w2 < 0:
        return np.inf
    if sum_w == 0.0:
        return np.inf
    return float(np.sqrt(sum_w2) / abs(sum_w))


def _merge_bins(bins: list[_Bin], threshold: float) -> list[_Bin]:
    """Merge adjacent bins with a size-balanced agglomerative policy.

    Every candidate pair must touch at least one bin whose fractional
    uncertainty exceeds ``threshold``.  Of those candidates, the pair with
    the fewest total rows is merged first.  Ties prefer the smaller physical
    span, then the lower post-merge uncertainty, then the leftmost pair.

    Selecting globally among pairs is important: after two small bins merge,
    two untouched small bins are normally merged before either joins the
    already-larger bin.  This avoids the left-edge snowballing produced by a
    target-first greedy algorithm.
    """
    bins = list(bins)

    while len(bins) > 1:
        uncertainties = [
            _fractional_uncertainty(item.sum_w, item.sum_w2) for item in bins
        ]
        failing = [uncertainty > threshold for uncertainty in uncertainties]
        if not any(failing):
            break

        candidates: list[tuple[tuple[float, ...], int]] = []
        for left in range(len(bins) - 1):
            right = left + 1
            if not (failing[left] or failing[right]):
                continue

            merged = bins[left].merged(bins[right])
            score = (
                float(merged.n_rows),
                float(merged.last - merged.first + 1),
                _fractional_uncertainty(merged.sum_w, merged.sum_w2),
                float(left),
            )
            candidates.append((score, left))

        # If a failing bin exists and there is more than one bin, at least one
        # adjacent pair touches a failing bin.
        _, left = min(candidates, key=lambda candidate: candidate[0])
        bins[left] = bins[left].merged(bins[left + 1])
        del bins[left + 1]

    return bins


def sequential_weighted_binning(
    df: pd.DataFrame,
    columns: Sequence[str],
    weight: str,
    initial_spacing: PerDimension,
    max_fractional_uncertainty: PerDimension,
    *,
    origins: PerDimension | None = None,
    output_prefix: str = "bin",
    include_bounds: bool = True,
    on_unmet: Literal["keep", "raise"] = "keep",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create hierarchical adaptive bins in ``columns`` order.

    Parameters
    ----------
    df:
        Input data. Its index and row order are preserved.
    columns:
        Ordered binning dimensions. Binning in each later dimension is done
        independently inside every bin path formed by the earlier dimensions.
    weight:
        Name of the event-weight column.
    initial_spacing:
        Initial regular-bin width. Supply one scalar, a sequence aligned with
        ``columns``, or ``{column: width}``.
    max_fractional_uncertainty:
        Maximum allowed ``sqrt(sum(w**2)) / abs(sum(w))``. This can also be a
        scalar, aligned sequence, or column mapping.
    origins:
        Optional origins of the regular lattices. By default, each origin is
        ``floor(min(column) / spacing) * spacing``.
    output_prefix:
        Prefix for output columns, e.g. ``bin_x`` and ``bin_x_left``.
    include_bounds:
        Add the locally adaptive left and right bin edges to every row.
    on_unmet:
        If an entire parent bin still fails the threshold, ``"keep"`` retains
        that single bin and marks it in the summary; ``"raise"`` raises.

    Returns
    -------
    binned, summary:
        ``binned`` is a copy of ``df`` with one nullable integer bin column per
        dimension (and optionally its local edges). Bin numbers are local to
        the preceding parent path. ``summary`` contains weights, uncertainties,
        bounds, effective counts, and a ``meets_threshold`` diagnostic.

    Notes
    -----
    Bins are half-open intervals ``[left, right)``. Rows with non-finite weight,
    a non-finite value in the current dimension, or an invalid earlier bin get
    ``NA`` from that dimension onward. Signed weights are supported, but weight
    cancellation can make the requested precision impossible; inspect
    ``meets_threshold`` (or use ``on_unmet="raise"``).

    Empty lattice cells are not represented as bins. Consequently, adjacent
    occupied bins can span an empty region when merged.
    """
    columns = list(columns)
    if not columns:
        raise ValueError("columns must contain at least one column")
    if len(set(columns)) != len(columns):
        raise ValueError("columns must not contain duplicates")
    missing = [name for name in [*columns, weight] if name not in df.columns]
    if missing:
        raise KeyError(f"DataFrame is missing columns: {missing}")
    if on_unmet not in {"keep", "raise"}:
        raise ValueError("on_unmet must be 'keep' or 'raise'")

    spacings = _expand_parameter(initial_spacing, columns, "initial_spacing")
    thresholds = _expand_parameter(
        max_fractional_uncertainty, columns, "max_fractional_uncertainty"
    )
    if any(not np.isfinite(value) or value <= 0 for value in spacings):
        raise ValueError("every initial spacing must be finite and > 0")
    if any(not np.isfinite(value) or value <= 0 for value in thresholds):
        raise ValueError("every uncertainty threshold must be finite and > 0")

    if origins is None:
        lattice_origins: list[float | None] = [None] * len(columns)
    else:
        lattice_origins = _expand_parameter(origins, columns, "origins")
        if any(not np.isfinite(value) for value in lattice_origins):
            raise ValueError("every origin must be finite")

    try:
        values = [
            df[column].to_numpy(dtype=float, na_value=np.nan) for column in columns
        ]
        weights = df[weight].to_numpy(dtype=float, na_value=np.nan)
    except (TypeError, ValueError) as error:
        raise TypeError("binning columns and weight must be numeric") from error

    n_rows = len(df)
    n_dimensions = len(columns)
    assignments = np.full((n_rows, n_dimensions), -1, dtype=np.int64)
    left_edges = np.full((n_rows, n_dimensions), np.nan, dtype=float)
    right_edges = np.full((n_rows, n_dimensions), np.nan, dtype=float)
    valid_weight = np.isfinite(weights)
    summary_rows: list[dict[str, object]] = []

    for dimension, (column, x, spacing, threshold) in enumerate(
        zip(columns, values, spacings, thresholds, strict=True)
    ):
        finite_x = np.isfinite(x)
        if not finite_x.any():
            raise ValueError(f"column {column!r} contains no finite values")

        origin = lattice_origins[dimension]
        if origin is None:
            origin = float(np.floor(np.min(x[finite_x]) / spacing) * spacing)

        eligible = valid_weight & finite_x
        if dimension:
            eligible &= np.all(assignments[:, :dimension] >= 0, axis=1)
        eligible_positions = np.flatnonzero(eligible)
        if not len(eligible_positions):
            continue

        scaled = (x[eligible_positions] - origin) / spacing
        # nextafter makes exact intended boundaries robust to tiny round-off.
        atomic = np.floor(np.nextafter(scaled, np.inf)).astype(np.int64)

        if dimension == 0:
            grouped_positions = [((), eligible_positions, atomic)]
        else:
            parents = assignments[eligible_positions, :dimension]
            parent_keys, inverse = np.unique(parents, axis=0, return_inverse=True)
            grouped_positions = []
            for group_id, key_array in enumerate(parent_keys):
                in_group = inverse == group_id
                grouped_positions.append(
                    (
                        tuple(int(v) for v in key_array),
                        eligible_positions[in_group],
                        atomic[in_group],
                    )
                )

        for parent_key, positions, atomic_ids in grouped_positions:
            occupied, inverse = np.unique(atomic_ids, return_inverse=True)
            sum_w = np.bincount(inverse, weights=weights[positions])
            sum_w2 = np.bincount(inverse, weights=np.square(weights[positions]))
            counts = np.bincount(inverse)
            initial_bins = [
                _Bin(
                    first=int(atom),
                    last=int(atom),
                    sum_w=float(sum_w[i]),
                    sum_w2=float(sum_w2[i]),
                    n_rows=int(counts[i]),
                )
                for i, atom in enumerate(occupied)
            ]
            final_bins = _merge_bins(initial_bins, threshold)

            for local_bin, item in enumerate(final_bins):
                in_bin = (atomic_ids >= item.first) & (atomic_ids <= item.last)
                bin_positions = positions[in_bin]
                left = float(origin + item.first * spacing)
                right = float(origin + (item.last + 1) * spacing)
                frac = _fractional_uncertainty(item.sum_w, item.sum_w2)
                meets = bool(frac <= threshold)

                assignments[bin_positions, dimension] = local_bin
                left_edges[bin_positions, dimension] = left
                right_edges[bin_positions, dimension] = right

                row: dict[str, object] = {
                    "dimension": dimension,
                    "column": column,
                    "bin": local_bin,
                    "left": left,
                    "right": right,
                    "n_rows": item.n_rows,
                    "sum_w": item.sum_w,
                    "sum_w2": item.sum_w2,
                    "uncertainty": float(np.sqrt(item.sum_w2)),
                    "fractional_uncertainty": frac,
                    "effective_count": (
                        item.sum_w * item.sum_w / item.sum_w2
                        if item.sum_w2 > 0
                        else 0.0
                    ),
                    "threshold": threshold,
                    "meets_threshold": meets,
                }
                row.update(
                    {
                        f"parent_bin_{columns[i]}": parent_key[i]
                        for i in range(dimension)
                    }
                )
                summary_rows.append(row)

                if not meets and on_unmet == "raise":
                    raise ValueError(
                        f"threshold cannot be met for {column!r} inside parent "
                        f"{parent_key}: fractional uncertainty is {frac:.6g} "
                        f"after merging all available bins"
                    )

    result = df.copy()
    for dimension, column in enumerate(columns):
        valid = assignments[:, dimension] >= 0
        bin_values = pd.array(
            np.where(valid, assignments[:, dimension], 0), dtype="Int64"
        )
        bin_values[~valid] = pd.NA
        result[f"{output_prefix}_{column}"] = bin_values
        if include_bounds:
            result[f"{output_prefix}_{column}_left"] = left_edges[:, dimension]
            result[f"{output_prefix}_{column}_right"] = right_edges[:, dimension]

    summary = pd.DataFrame(summary_rows)
    return result, summary