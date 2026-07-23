"""Forecast models. Each model is a plain function: (months_of_history,
production_array) -> forecast_fn(n_future_months) -> np.ndarray.

Kept intentionally dumb for v0 — no analogs, no cohort borrowing, no
crudecode dependency. The point of this repo is to know whether a given
change actually helps, measured, before it's allowed to get clever.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.optimize import curve_fit

ForecastFn = Callable[[int], np.ndarray]


def naive_last3(q: np.ndarray) -> ForecastFn:
    """Flat-line the average of the last 3 reported months. The dumbest
    possible baseline — anything we ship should beat this or it's not
    earning its complexity."""
    tail = q[~np.isnan(q)][-3:]
    level = float(np.mean(tail)) if len(tail) else 0.0

    def _forecast(n_future: int) -> np.ndarray:
        return np.full(n_future, level)

    return _forecast


def _hyperbolic_q(t: np.ndarray, qi: float, di: float, b: float) -> np.ndarray:
    if b < 1e-6:
        return qi * np.exp(-di * t)
    return qi / np.power(1.0 + b * di * t, 1.0 / b)


def arps_exponential(q: np.ndarray) -> ForecastFn:
    """Exponential decline (b=0) fit from peak forward. Two-parameter fit —
    the most stable option on short/noisy histories."""
    return _fit_fixed_b(q, b=0.0)


def arps_hyperbolic_bounded_b(q: np.ndarray, *, b_grid: tuple[float, ...] | None = None) -> ForecastFn:
    """Best-fit b via grid search over a bounded range, min-SSE selection.
    Same shape as crudecode's fit_curve_best_b, reimplemented standalone so
    this repo has zero import dependency on crudecode."""
    grid = b_grid or tuple(round(0.3 + 0.05 * i, 2) for i in range(21))  # 0.30..1.30
    mask = ~np.isnan(q)
    q_clean = q[mask]
    if len(q_clean) < 3:
        return naive_last3(q)

    peak_idx = int(np.argmax(q_clean))
    q_fit = q_clean[peak_idx:]
    t_fit = np.arange(len(q_fit), dtype=float)

    best_params, best_sse = None, float("inf")
    for b in grid:
        params = _fit_qi_di(t_fit, q_fit, b)
        if params is None:
            continue
        qi, di = params
        pred = _hyperbolic_q(t_fit, qi, di, b)
        sse = float(np.sum((pred - q_fit) ** 2))
        if sse < best_sse:
            best_sse, best_params = sse, (qi, di, b)

    if best_params is None:
        return naive_last3(q)

    qi, di, b = best_params
    n_history = len(q_fit)

    def _forecast(n_future: int) -> np.ndarray:
        t_future = np.arange(n_history, n_history + n_future, dtype=float)
        return _hyperbolic_q(t_future, qi, di, b)

    return _forecast


def _fit_fixed_b(q: np.ndarray, *, b: float) -> ForecastFn:
    mask = ~np.isnan(q)
    q_clean = q[mask]
    if len(q_clean) < 3:
        return naive_last3(q)
    peak_idx = int(np.argmax(q_clean))
    q_fit = q_clean[peak_idx:]
    t_fit = np.arange(len(q_fit), dtype=float)
    params = _fit_qi_di(t_fit, q_fit, b)
    if params is None:
        return naive_last3(q)
    qi, di = params
    n_history = len(q_fit)

    def _forecast(n_future: int) -> np.ndarray:
        t_future = np.arange(n_history, n_history + n_future, dtype=float)
        return _hyperbolic_q(t_future, qi, di, b)

    return _forecast


def _fit_qi_di(t: np.ndarray, q: np.ndarray, b: float) -> tuple[float, float] | None:
    if q[1:].sum() <= 0.0:
        return None

    def _model(t, qi, di):
        return _hyperbolic_q(t, qi, di, b)

    try:
        popt, _ = curve_fit(
            _model, t, q,
            p0=[float(q[0]), 0.05],
            bounds=([0.0, 0.0], [1e8, 1.0]),
            maxfev=5000,
        )
    except RuntimeError:
        return None
    return float(popt[0]), float(popt[1])
