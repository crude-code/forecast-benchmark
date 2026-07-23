"""Error metrics. Every function here takes (actual, forecast) arrays of
equal length and returns a float, or None if the inputs don't support the
metric (e.g. all-zero actuals for a percent-error metric) — never a fake
number, never a silently-dropped NaN that corrupts an aggregate later.
"""
from __future__ import annotations

import numpy as np


def _clean_pair(actual: np.ndarray, forecast: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Drop months where actual is NaN (not reported). Returns None if
    nothing is left to score."""
    mask = ~np.isnan(actual)
    a, f = actual[mask], forecast[mask]
    if len(a) == 0:
        return None
    return a, f


def mape(actual: np.ndarray, forecast: np.ndarray) -> float | None:
    """Mean Absolute Percent Error. None if every actual is zero (undefined)."""
    pair = _clean_pair(actual, forecast)
    if pair is None:
        return None
    a, f = pair
    nonzero = a != 0
    if not np.any(nonzero):
        return None
    return float(np.mean(np.abs((f[nonzero] - a[nonzero]) / a[nonzero])) * 100.0)


def smape(actual: np.ndarray, forecast: np.ndarray) -> float | None:
    """Symmetric MAPE — bounded [0, 200], handles near-zero actuals better
    than plain MAPE. None if every (|a|+|f|) pair is zero."""
    pair = _clean_pair(actual, forecast)
    if pair is None:
        return None
    a, f = pair
    denom = np.abs(a) + np.abs(f)
    nonzero = denom != 0
    if not np.any(nonzero):
        return None
    return float(np.mean(2.0 * np.abs(f[nonzero] - a[nonzero]) / denom[nonzero]) * 100.0)


def mae(actual: np.ndarray, forecast: np.ndarray) -> float | None:
    """Mean Absolute Error, in the phase's native units (bbl or mcf)."""
    pair = _clean_pair(actual, forecast)
    if pair is None:
        return None
    a, f = pair
    return float(np.mean(np.abs(f - a)))


def bias(actual: np.ndarray, forecast: np.ndarray) -> float | None:
    """Signed mean percent error. Positive = over-forecast. This is the
    single most important number for a valuation shop: MAPE says how noisy
    you are, bias says which direction you're wrong in, on average."""
    pair = _clean_pair(actual, forecast)
    if pair is None:
        return None
    a, f = pair
    nonzero = a != 0
    if not np.any(nonzero):
        return None
    return float(np.mean((f[nonzero] - a[nonzero]) / a[nonzero]) * 100.0)


def log_error(actual: np.ndarray, forecast: np.ndarray) -> np.ndarray | None:
    """Per-month log error: ln(forecast / actual). This is the building
    block for SPEE-bake-off-style scoring (median |log error| + stdev of
    log error) — kept as a separate primitive from bias/MAPE since it's a
    different question (symmetric over/under-forecast treatment) and the
    bake-off explicitly scores this way. Returns None if no valid pairs
    (actual <= 0 or forecast <= 0 rows are dropped, not zeroed)."""
    pair = _clean_pair(actual, forecast)
    if pair is None:
        return None
    a, f = pair
    valid = (a > 0) & (f > 0)
    if not np.any(valid):
        return None
    return np.log(f[valid] / a[valid])


def spee_score(actual: np.ndarray, forecast: np.ndarray) -> float | None:
    """2/3 * |median(log_error)| + 1/3 * stdev(log_error) — the exact
    weighting SPEE used in the 2024 Software Symposium bake-off (see
    docs/METHODOLOGY.md). Lower is better. None if log_error is undefined."""
    le = log_error(actual, forecast)
    if le is None or len(le) < 2:
        return None
    return float((2.0 / 3.0) * abs(np.median(le)) + (1.0 / 3.0) * np.std(le))


ALL_METRICS = {
    "mape": mape,
    "smape": smape,
    "mae": mae,
    "bias": bias,
    "spee_score": spee_score,
}
