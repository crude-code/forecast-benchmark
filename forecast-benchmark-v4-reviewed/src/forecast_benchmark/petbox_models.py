"""petbox-dca-backed forecast models, as SEPARATE benchmark models.

Per the repo's own rule (OPEN_QUESTIONS.md / METHODOLOGY): "do not silently
swap model forms without benchmarking the change." So these wrappers are
ADDITIVE — new models that produce a `ForecastFn` in the same shape as
`arps.py`, to be compared head-to-head against the self-contained baselines,
never a replacement for them.

Two models are wrapped:

  1. THM (Transient Hyperbolic Model, Fulford & Blasingame SPE-167242) — the
     physically-grounded primary-phase decline where b transitions from a
     transient value (bi, often ~2) to a boundary-dominated value (bf, <1) over
     a characteristic time telf. This is the committee-authored answer to the
     b-factor instability our own `own_b`/bounded-b work approaches empirically.

  2. PLYield secondary/water phase — petbox's power-law yield ratio model,
     ridden on a petbox primary. The committee-authored version of the ratio
     idea in `ratios.py`.

THE UNIT BRIDGE (the thing most likely to silently corrupt a comparison):
  - The benchmark speaks MONTHLY: arrays indexed by month, values = volume
    produced that month. Month index 0,1,2,...
  - petbox speaks DAYS and PER-DAY RATES: time in days, rate() in vol/day, with
    a `monthly_vol(t_days)` helper returning the volume produced in the ~30.44-
    day month ending at each t.
  - So: month index m maps to day (m+1) * DAYS_PER_MONTH (end-of-month), and we
    score using monthly_vol at those day values, NOT rate(). Getting this wrong
    (e.g. using rate() as if it were monthly volume) would inflate everything by
    ~30x and quietly ruin the benchmark — hence it's isolated here and tested.

If petbox-dca is not installed, importing this module raises ImportError with
a clear message. The core benchmark never imports it unless asked, so the
public repo still has no hard dependency.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

try:
    from petbox import dca as _dca
except Exception as e:  # pragma: no cover - exercised only without petbox
    raise ImportError(
        "petbox_models requires petbox-dca. Install with `pip install petbox-dca`. "
        "The core benchmark does not need it; this module is an optional add-on."
    ) from e

from scipy.optimize import least_squares

ForecastFn = Callable[[int], np.ndarray]

DAYS_PER_MONTH = float(_dca.DAYS_PER_MONTH)
DAYS_PER_YEAR = float(_dca.DAYS_PER_YEAR)


def _month_end_days(month_indices: np.ndarray) -> np.ndarray:
    """Month index m (0-based) -> day at the END of that month. Month 0 spans
    days (0, 30.44], so its end is 1 * DAYS_PER_MONTH. This is the t we hand
    petbox's monthly_vol so the returned volume aligns with our month-indexed
    actuals."""
    return (np.asarray(month_indices, dtype=float) + 1.0) * DAYS_PER_MONTH


def _clean_monthly(q: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (month_index, volume) from peak forward, dropping NaN/pre-peak.
    Mirrors the peak-forward convention used by arps.py so the two model
    families are fit on the same footing."""
    q = np.asarray(q, dtype=float)
    mask = np.isfinite(q)
    if mask.sum() < 3:
        return None
    q_clean = q[mask]
    idx_clean = np.nonzero(mask)[0].astype(float)
    peak = int(np.argmax(q_clean))
    q_fit = q_clean[peak:]
    # re-index so the peak month is month 0 (petbox expects decline from t~0)
    m_fit = np.arange(len(q_fit), dtype=float)
    if q_fit[1:].sum() <= 0:
        return None
    return m_fit, q_fit


def thm_model(q: np.ndarray, *, bi: float = 2.0) -> ForecastFn:
    """Fit petbox THM to a monthly primary-phase stream, return a ForecastFn
    that emits monthly volumes for the next n months.

    Fitted params: qi, Di, bf, telf (bi is fixed at the transient default 2.0,
    matching petbox's own example; bterm/tterm left at 0 = no terminal segment,
    since the benchmark's holdouts are short enough that terminal decline rarely
    binds and adding it adds two more fit params). Robust soft-L1 loss in log
    space, same as our ratio fits.
    """
    cleaned = _clean_monthly(q)
    if cleaned is None:
        return _flat_fallback(q)
    m_fit, q_fit = cleaned
    days_fit = _month_end_days(m_fit)
    # monthly_vol at month-end days should reproduce the monthly volumes
    qi0 = float(q_fit[0]) / DAYS_PER_MONTH * 1.5  # rough per-day rate guess

    def residuals(params):
        qi, Di, bf, telf = params
        try:
            model = _dca.THM(qi=qi, Di=Di, bi=bi, bf=bf, telf=telf, bterm=0.0, tterm=0.0)
            pred = model.monthly_vol(days_fit)
        except Exception:
            return np.full_like(q_fit, 1e6)
        return np.log(np.maximum(pred, 1e-9)) - np.log(np.maximum(q_fit, 1e-9))

    # bounds: qi/day > 0; Di in (0, ~2/yr]; bf in [0.1, 1.0]; telf in [1, 400] days
    bounds = ([1e-3, 1e-4, 0.1, 1.0], [1e7, 2.0, 1.0, 400.0])
    x0 = [max(qi0, 1e-2), 0.5, 0.6, 30.0]
    try:
        opt = least_squares(
            residuals, x0=x0, bounds=bounds,
            loss="soft_l1", f_scale=0.35, max_nfev=4000,
        )
        qi, Di, bf, telf = (float(v) for v in opt.x)
        model = _dca.THM(qi=qi, Di=Di, bi=bi, bf=bf, telf=telf, bterm=0.0, tterm=0.0)
    except Exception:
        return _flat_fallback(q)

    n_hist = len(m_fit)

    def _forecast(n_future: int) -> np.ndarray:
        future_months = np.arange(n_hist, n_hist + n_future, dtype=float)
        return model.monthly_vol(_month_end_days(future_months))

    return _forecast


def _flat_fallback(q: np.ndarray) -> ForecastFn:
    q = np.asarray(q, dtype=float)
    tail = q[np.isfinite(q)][-3:]
    level = float(np.mean(tail)) if len(tail) else 0.0

    def _f(n_future: int) -> np.ndarray:
        return np.full(n_future, level)

    return _f


def thm_secondary_from_primary(
    primary_history: np.ndarray,
    secondary_history: np.ndarray,
    primary_forecast_months: int,
    *,
    is_water: bool = False,
) -> np.ndarray | None:
    """petbox-native secondary-phase forecast: fit a THM primary, attach a
    fitted PLYield, and read the secondary monthly volumes directly from petbox
    (so the primary AND the ratio both come from committee-authored math).

    Returns monthly secondary volumes for the holdout, or None if it couldn't
    fit — caller falls back exactly as with the self-contained ratio model.

    Note this differs from `ratios.secondary_from_primary`, which rides a
    self-fit ratio on whatever primary forecast it's given. Here the whole
    stack is petbox. That's intentional: it's a genuinely different model to
    benchmark, not the same one with a different curve library underneath.
    """
    cleaned = _clean_monthly(primary_history)
    if cleaned is None:
        return None
    m_fit, prim_fit = cleaned
    days_fit = _month_end_days(m_fit)

    # align secondary to the same peak-forward window
    p = np.asarray(primary_history, dtype=float)
    s = np.asarray(secondary_history, dtype=float)
    mask = np.isfinite(p)
    peak = int(np.argmax(p[mask]))
    s_clean = s[mask][peak:]
    if len(s_clean) != len(prim_fit):
        return None
    ratio_hist = np.where(prim_fit > 0, s_clean / prim_fit, np.nan)
    if not np.any(np.isfinite(ratio_hist) & (ratio_hist > 0)):
        return None

    # fit THM primary
    prim_fn = thm_model(primary_history)

    # fit a PLYield to the historical ratio: yield(t) = c * (t/t0)^m form.
    # petbox PLYield(c, m0, m, t0, min, max): ratio = c at t<=t0 region then
    # power-law; we fit c (level), m (slope), t0 (onset) in log space.
    valid = np.isfinite(ratio_hist) & (ratio_hist > 0)
    t_valid = days_fit[valid]
    r_valid = ratio_hist[valid]
    c0 = float(np.median(r_valid[: max(1, len(r_valid) // 4)]))

    def resid(params):
        c, m = params
        try:
            yld = _dca.PLYield(c=c, m0=0.0, m=m, t0=float(DAYS_PER_MONTH * 3), min=None, max=None)
            pred = yld.gor(t_valid) if not is_water else yld.wor(t_valid)
        except Exception:
            return np.full_like(r_valid, 1e6)
        return np.log(np.maximum(pred, 1e-12)) - np.log(np.maximum(r_valid, 1e-12))

    try:
        opt = least_squares(
            resid, x0=[max(c0, 1e-6), 0.1],
            bounds=([1e-9, -3.0], [1e9, 3.0]),
            loss="soft_l1", f_scale=0.5, max_nfev=2000,
        )
        c, m = float(opt.x[0]), float(opt.x[1])
    except Exception:
        return None

    n_hist = len(m_fit)
    future_months = np.arange(n_hist, n_hist + primary_forecast_months, dtype=float)
    future_days = _month_end_days(future_months)
    try:
        yld = _dca.PLYield(c=c, m0=0.0, m=m, t0=float(DAYS_PER_MONTH * 3), min=None, max=None)
        ratio_future = yld.gor(future_days) if not is_water else yld.wor(future_days)
    except Exception:
        return None

    prim_future = prim_fn(primary_forecast_months)
    return ratio_future * prim_future
