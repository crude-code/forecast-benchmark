"""The thing this repo actually exists to test: one forecast function for
every well, with no separate PUD-vs-PDP branch.

This is deliberately naive right now — it's a strawman to benchmark against
the branching models in arps.py, not a claimed improvement. Bill's intuition
is that treating "a well with 2 months of history" and "a well with 40
months of history" as the SAME function (just conditioned on however much
data happens to exist) might generalize better than routing them through
different code paths with different assumptions baked in. That's an
empirical question — this file is the thing we point the benchmark at to
find out, nothing more.

No claim of superiority lives in this file. Only the benchmark results do.
"""
from __future__ import annotations

import numpy as np

from forecast_benchmark.arps import ForecastFn, _fit_qi_di, _hyperbolic_q, naive_last3

_DEFAULT_B = 0.8
_MIN_MONTHS_FOR_OWN_FIT = 3


def unified_forecast(q: np.ndarray, *, b_grid: tuple[float, ...] | None = None) -> ForecastFn:
    """One function, any amount of history. No HISTORY/THIN_PEAKED/CLIMBING/
    NO_HISTORY branching, no analog blending — just: fit what you can from
    whatever's there, fall back to a flat default b when there isn't enough
    signal to fit b at all.

    This treats "0 months of history" and "40 months of history" as two
    points on the same continuum (amount of conditioning data) rather than
    categorically different problems requiring different code paths.
    """
    grid = b_grid or tuple(round(0.3 + 0.05 * i, 2) for i in range(21))
    mask = ~np.isnan(q)
    q_clean = q[mask]

    if len(q_clean) == 0:
        # No history at all: nothing to condition on. Return a zero curve
        # rather than fabricate a number — same principle as crudecode's
        # zero-stream guard, ported here because it's a correctness issue,
        # not a style choice this benchmark should re-litigate.
        def _zero(n_future: int) -> np.ndarray:
            return np.zeros(n_future)
        return _zero

    if len(q_clean) < _MIN_MONTHS_FOR_OWN_FIT:
        return naive_last3(q)

    peak_idx = int(np.argmax(q_clean))
    q_fit = q_clean[peak_idx:]
    t_fit = np.arange(len(q_fit), dtype=float)

    if len(q_fit) < 2 or q_fit[1:].sum() <= 0.0:
        return naive_last3(q)

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
