"""Secondary-phase forecasting by RATIO, not by independent fit
(OPEN_QUESTIONS.md Gaps 1 & 2).

The one architectural idea behind this whole module: never fit the secondary
stream on its own. Instead, forecast the *ratio* (GOR on oil wells, CGR on
gas wells, WOR/WGR for water) and derive the secondary phase from the primary
forecast you already trust:

    secondary_rate(t) = ratio(t) x primary_rate(t)

This is why the SPEE committee found everyone got secondary phases wrong in
the same direction. A single-phase Arps curve fit to the gas stream of an oil
well *declines*, but the physics says gas should RISE relative to oil as the
well depletes below bubble point and solution gas comes out (the four-stage
GOR trend: flat -> rising -> plateau). An independent fit structurally cannot
represent that. A rising ratio model can.

Two ratio shapes are implemented:

  1. Power-law yield, intentionally matching the same family as petbox-dca's
     `PLYield`: ratio(t) = c * t^m, optionally clamped. Simple, monotonic,
     good for the rising GOR / falling CGR regimes. This module currently uses
     a self-contained numpy implementation so the public benchmark has no
     optional runtime dependency. A future petbox-backed wrapper can be added
     as a separate model and compared head-to-head.

  2. Logistic (sigmoid) yield: ratio(t) = lo + (hi - lo) / (1 + exp(-k(t-t0))).
     Captures the flat -> rising -> plateau S-curve of GOR directly (the
     "Asymmetrical Sigmoid Model" idea from SPE-LRBC 2019), which the power
     law can't (power law has no plateau).

Both are fit to the historical ratio with robust (soft-L1) least squares,
because ratio data is even noisier than rate data.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.optimize import least_squares

RatioFn = Callable[[np.ndarray], np.ndarray]  # months -> ratio at each month


# --------------------------------------------------------------------------
# Ratio shapes
# --------------------------------------------------------------------------

def _power_law_ratio(t: np.ndarray, c: float, m: float, t0: float,
                     lo: float | None, hi: float | None) -> np.ndarray:
    """ratio(t) = c * ((t + t0) / t0) ^ m, optionally clamped to [lo, hi].

    Same functional family as petbox-dca PLYield. `t0` shifts the origin so
    t=0 gives a finite ratio (c) rather than 0 or infinity. m > 0 rises
    (GOR on oil wells), m < 0 falls (CGR on gas wells)."""
    base = np.maximum((t + t0) / max(t0, 1e-9), 1e-9)
    r = c * np.power(base, m)
    if lo is not None:
        r = np.maximum(r, lo)
    if hi is not None:
        r = np.minimum(r, hi)
    return r


def _logistic_ratio(t: np.ndarray, lo: float, hi: float, k: float, t0: float) -> np.ndarray:
    """Flat -> rising -> plateau S-curve. lo is the initial (solution) ratio,
    hi the plateau, t0 the inflection month (roughly the bubble-point
    transition), k the steepness."""
    return lo + (hi - lo) / (1.0 + np.exp(-k * (t - t0)))


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------

def fit_ratio(
    primary: np.ndarray,
    secondary: np.ndarray,
    *,
    shape: str = "power_law",
    rising: bool | None = None,
) -> RatioFn | None:
    """Fit a ratio model to historical secondary/primary. Returns a RatioFn
    mapping month-index -> ratio, or None if there isn't enough clean data.

    `rising`:
      - True  -> constrain the ratio to rise (GOR on oil wells)
      - False -> constrain it to fall (CGR on gas wells)
      - None  -> let the data decide (unconstrained sign on the exponent)
    Constraining the direction is the single most important guardrail here:
    it's what stops a noisy 8-month history from fitting a *falling* GOR on an
    oil well when the physics guarantees a rise is coming.
    """
    p = np.asarray(primary, dtype=float)
    s = np.asarray(secondary, dtype=float)
    mask = np.isfinite(p) & np.isfinite(s) & (p > 0)
    p, s = p[mask], s[mask]
    if len(p) < 3:
        return None

    ratio_hist = s / p
    t = np.arange(len(ratio_hist), dtype=float)
    # guard against all-zero / degenerate secondary
    if not np.any(ratio_hist > 0):
        return None

    if shape == "logistic":
        return _fit_logistic(t, ratio_hist, rising=rising)
    return _fit_power_law(t, ratio_hist, rising=rising)


def _fit_power_law(t: np.ndarray, ratio: np.ndarray, *, rising: bool | None) -> RatioFn | None:
    c0 = float(np.median(ratio[: max(1, len(ratio) // 4)]))  # early-life level
    c0 = max(c0, 1e-6)
    t0 = 3.0  # months; shift so early ratio is finite

    # exponent bounds encode the physical direction
    if rising is True:
        m_lo, m_hi = 0.0, 3.0
    elif rising is False:
        m_lo, m_hi = -3.0, 0.0
    else:
        m_lo, m_hi = -3.0, 3.0

    def resid(params):
        c, m = params
        pred = _power_law_ratio(t, c, m, t0, lo=None, hi=None)
        # fit in log space — ratios span orders of magnitude
        return np.log(np.maximum(pred, 1e-9)) - np.log(np.maximum(ratio, 1e-9))

    try:
        opt = least_squares(
            resid, x0=[c0, 0.1 if rising is not False else -0.1],
            bounds=([1e-6, m_lo], [1e9, m_hi]),
            loss="soft_l1", f_scale=0.5, max_nfev=2000,
        )
    except Exception:
        return None

    c, m = float(opt.x[0]), float(opt.x[1])
    # clamp the extrapolated ratio to a sane multiple of the observed range so
    # a steep early slope can't run away to absurd values 20 years out
    obs_max = float(np.max(ratio))
    obs_min = float(np.min(ratio[ratio > 0]))
    hi = obs_max * 10.0
    lo = obs_min * 0.1

    def _fn(t_eval: np.ndarray) -> np.ndarray:
        return _power_law_ratio(np.asarray(t_eval, dtype=float), c, m, t0, lo=lo, hi=hi)

    return _fn


def _fit_logistic(t: np.ndarray, ratio: np.ndarray, *, rising: bool | None) -> RatioFn | None:
    lo0 = float(np.median(ratio[: max(1, len(ratio) // 4)]))
    hi0 = float(np.max(ratio)) * (2.0 if rising is not False else 1.0)
    t0_0 = float(len(ratio))  # inflection near/after end of history by default
    k0 = 0.1

    def resid(params):
        lo, hi, k, t0 = params
        pred = _logistic_ratio(t, lo, hi, k, t0)
        return np.log(np.maximum(pred, 1e-9)) - np.log(np.maximum(ratio, 1e-9))

    # for a rising GOR, hi >= lo; for falling CGR, hi <= lo
    if rising is True:
        bounds = ([1e-6, lo0, 0.0, 0.0], [np.inf, np.inf, 2.0, len(ratio) * 4])
    elif rising is False:
        bounds = ([1e-6, 1e-6, -2.0, 0.0], [np.inf, lo0 * 1.5, 0.0, len(ratio) * 4])
    else:
        bounds = ([1e-6, 1e-6, -2.0, 0.0], [np.inf, np.inf, 2.0, len(ratio) * 4])

    try:
        opt = least_squares(
            resid, x0=[max(lo0, 1e-6), max(hi0, lo0 + 1e-6), k0, t0_0],
            bounds=bounds, loss="soft_l1", f_scale=0.5, max_nfev=3000,
        )
    except Exception:
        return None

    lo, hi, k, t0 = (float(x) for x in opt.x)

    def _fn(t_eval: np.ndarray) -> np.ndarray:
        return _logistic_ratio(np.asarray(t_eval, dtype=float), lo, hi, k, t0)

    return _fn


# --------------------------------------------------------------------------
# The secondary-phase forecast: ratio x primary
# --------------------------------------------------------------------------

def secondary_from_primary(
    primary_history: np.ndarray,
    secondary_history: np.ndarray,
    primary_forecast: np.ndarray,
    *,
    shape: str = "power_law",
    rising: bool | None = None,
) -> np.ndarray | None:
    """The whole point of the module, in one call: fit the historical ratio,
    then ride it on the primary forecast to get the secondary forecast.

    `primary_forecast` is the already-produced forecast of the primary phase
    over the holdout window. Returns the secondary-phase forecast over that
    same window, or None if the ratio couldn't be fit (caller decides whether
    to fall back to a flat trailing ratio).
    """
    ratio_fn = fit_ratio(primary_history, secondary_history, shape=shape, rising=rising)
    if ratio_fn is None:
        return None

    n_hist = int(np.sum(np.isfinite(primary_history) & (np.asarray(primary_history) > 0)))
    horizon = len(primary_forecast)
    t_future = np.arange(n_hist, n_hist + horizon, dtype=float)
    ratio_future = ratio_fn(t_future)
    return ratio_future * np.asarray(primary_forecast, dtype=float)


def flat_ratio_fallback(
    primary_history: np.ndarray,
    secondary_history: np.ndarray,
    primary_forecast: np.ndarray,
) -> np.ndarray:
    """Honest fallback when a ratio trend can't be fit: hold the trailing
    ratio flat and ride it on the primary. Not great, but it's what a careful
    engineer does on a well too noisy/short to trend — and it's explicitly
    better than fitting an independent secondary curve, which we never do."""
    p = np.asarray(primary_history, dtype=float)
    s = np.asarray(secondary_history, dtype=float)
    mask = np.isfinite(p) & np.isfinite(s) & (p > 0)
    if not np.any(mask):
        return np.zeros(len(primary_forecast))
    tail = min(3, int(np.sum(mask)))
    ratio_tail = (s[mask] / p[mask])[-tail:]
    flat = float(np.median(ratio_tail))
    return flat * np.asarray(primary_forecast, dtype=float)
