"""Parameter-level forecasting: one common output for every method.

The rest of this repo scores a *forecast curve*. This module scores the
*judgment behind it*. Every method here — the deterministic fitters and, later,
an LLM — emits the SAME thing: Arps hyperbolic parameters (qi, Di, b) anchored
at the cutoff. Because the output space is shared, you can line the methods up
parameter by parameter ("arps picked Di=0.18, the LLM picked Di=0.12, the
held-out year implied ~0.14") and tie forecast error back to the specific
parameter that caused it.

Pinned convention — the whole point is that everyone speaks the same units:

- The anchor is the cutoff: the last training month, index ``cutoff-1`` of the
  full series. ``t = 0`` sits there.
- ``qi`` is the rate at the anchor, in stream units per month. It is the
  forecast's starting volume, never "peak-anything" (mirrors SKILL.md).
- ``Di`` is nominal monthly decline at the anchor.
- Held-out month k (k = 1..horizon) is ``q(t = k)`` — one month past the anchor
  is the first scored month, matching the base benchmark's holdout window.

The deterministic baselines in ``arps.py`` fit qi/Di at the *peak*. Arps
hyperbolic re-anchors losslessly under a time shift, so ``reanchor`` moves those
peak-anchored fits to the cutoff without changing the curve they imply. That
keeps this layer consistent with the base benchmark: a re-anchored parameter set
reproduces the original model's holdout forecast exactly (see the tests).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from forecast_benchmark.arps import _fit_qi_di, _hyperbolic_q

_B_EPS = 1e-6


@dataclass(frozen=True)
class HyperbolicParams:
    """Arps hyperbolic parameters anchored at the cutoff (t=0 = last train month).

    ``forecast`` evaluates the curve over the holdout window using the pinned
    convention above, so it plugs straight into the existing metrics.
    """

    qi: float  # rate at the anchor, stream units/month
    di: float  # nominal monthly decline at the anchor
    b: float   # Arps exponent

    def forecast(self, horizon: int) -> np.ndarray:
        t = np.arange(1, horizon + 1, dtype=float)
        return _hyperbolic_q(t, self.qi, self.di, self.b)


def reanchor(qi: float, di: float, b: float, delta: float) -> HyperbolicParams:
    """Shift a hyperbolic curve's anchor forward by ``delta`` months.

    Given (qi, Di, b) defined at some earlier time (e.g. the peak), return the
    equivalent parameters defined ``delta`` months later. The curve is
    unchanged: q'(t) == q(delta + t) for all t. b is invariant; Di decays as
    Di/(1+b·Di·delta) and qi follows the decline over the shift.
    """
    if delta <= 0:
        return HyperbolicParams(qi=float(qi), di=float(di), b=float(b))
    if b < _B_EPS:
        return HyperbolicParams(qi=float(qi * np.exp(-di * delta)), di=float(di), b=float(b))
    factor = 1.0 + b * di * delta
    return HyperbolicParams(
        qi=float(qi / factor ** (1.0 / b)),
        di=float(di / factor),
        b=float(b),
    )


# --- Deterministic methods, re-expressed as (qi, Di, b) at the cutoff --------
#
# These reuse the exact fit internals from arps.py (peak selection, curve_fit,
# grid search) and only add the re-anchoring step, so they agree with the base
# benchmark's forecasts by construction rather than by a second implementation.


def _clean_from_peak(q: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """The clean, peak-forward window the arps baselines fit against, plus its
    time axis. None if there isn't enough signal to fit."""
    q_clean = q[~np.isnan(q)]
    if len(q_clean) < 3:
        return None
    peak_idx = int(np.argmax(q_clean))
    q_fit = q_clean[peak_idx:]
    t_fit = np.arange(len(q_fit), dtype=float)
    return q_fit, t_fit


def params_naive_last3(q: np.ndarray) -> HyperbolicParams:
    """Flat line at the mean of the last 3 reported months: qi=level, Di=0."""
    tail = q[~np.isnan(q)][-3:]
    level = float(np.mean(tail)) if len(tail) else 0.0
    return HyperbolicParams(qi=level, di=0.0, b=0.0)


def params_arps_exponential(q: np.ndarray) -> HyperbolicParams:
    """Exponential decline (b=0) fit from peak, re-anchored to the cutoff."""
    window = _clean_from_peak(q)
    if window is None:
        return params_naive_last3(q)
    q_fit, t_fit = window
    fit = _fit_qi_di(t_fit, q_fit, 0.0)
    if fit is None:
        return params_naive_last3(q)
    qi, di = fit
    return reanchor(qi, di, 0.0, delta=len(q_fit) - 1)


def params_arps_bounded_b(
    q: np.ndarray, *, b_grid: tuple[float, ...] | None = None
) -> HyperbolicParams:
    """Best-fit b via bounded grid search (min SSE), re-anchored to the cutoff.

    Same selection as ``arps.arps_hyperbolic_bounded_b``; returns the winning
    (qi, Di, b) instead of a closure over them.
    """
    window = _clean_from_peak(q)
    if window is None:
        return params_naive_last3(q)
    q_fit, t_fit = window
    grid = b_grid or tuple(round(0.3 + 0.05 * i, 2) for i in range(21))  # 0.30..1.30

    best_params, best_sse = None, float("inf")
    for b in grid:
        fit = _fit_qi_di(t_fit, q_fit, b)
        if fit is None:
            continue
        qi, di = fit
        sse = float(np.sum((_hyperbolic_q(t_fit, qi, di, b) - q_fit) ** 2))
        if sse < best_sse:
            best_sse, best_params = sse, (qi, di, b)

    if best_params is None:
        return params_naive_last3(q)
    qi, di, b = best_params
    return reanchor(qi, di, b, delta=len(q_fit) - 1)


# A method that emits parameters instead of a forecast closure.
ParamModelFn = Callable[[np.ndarray], HyperbolicParams]

DETERMINISTIC_PARAM_MODELS: dict[str, ParamModelFn] = {
    "naive_last3": params_naive_last3,
    "arps_exponential": params_arps_exponential,
    "arps_bounded_b": params_arps_bounded_b,
}
