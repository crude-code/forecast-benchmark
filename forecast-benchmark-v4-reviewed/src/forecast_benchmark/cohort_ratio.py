"""Cohort-prior ratio forecasting — the piece that makes short-history
secondary-phase forecasts actually work (ties Gap 2 and Gap 3 together).

The finding that motivates this module, demonstrated in the benchmark:

  On a well cut BEFORE its bubble-point GOR rise has started, no
  self-contained fit can forecast the rise — not an independent Arps fit on
  the gas stream (misses ~-46% cumulative), not a ratio fit on the well's own
  flat early GOR (~-55%). The rise simply isn't in the data yet. The ONLY way
  to know it's coming is to borrow the transition timing from analog/cohort
  wells that have already been through it.

  Doing exactly that — anchoring the young well's own initial GOR level to the
  cohort's median normalized GOR-vs-time shape — cut the error to ~-7% on the
  same well.

That is the cohort prior from the research memos, made concrete for ratios.
It's the ratio-space analog of `cohort.shrink_params` for decline parameters:
lean on the cohort when the well is too young to speak for itself, and on the
well's own data as it matures.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from forecast_benchmark.cohort import shrinkage_weight
from forecast_benchmark.data import WellSeries


@dataclass(frozen=True)
class CohortRatioShape:
    """The cohort's median ratio-vs-month curve, normalized so month ~0 = 1.0.

    To forecast a young well, multiply this shape by the young well's OWN
    early-life ratio level. The shape carries the *timing and magnitude of the
    transition*; the anchor carries the *well-specific level*.
    """
    multiplier_by_month: np.ndarray  # shape[m] = ratio(month m) / ratio(month 0)
    n_members: int
    max_month: int


def build_cohort_ratio_shape(
    cohort_wells: list[WellSeries],
    *,
    primary_phase: str = "oil",
    secondary_phase: str = "gas",
    min_history: int = 30,
    anchor_months: int = 4,
) -> CohortRatioShape | None:
    """Build a normalized ratio-vs-month shape from cohort members mature
    enough to have gone through the transition (>= min_history months).

    Returns None if too few mature members — the caller must then fall back to
    a self-fit and say so, never fabricate a cohort shape from nothing.
    """
    shapes = []
    max_len = 0
    for w in cohort_wells:
        n = len(w.months)
        if n < min_history:
            continue
        prim = np.asarray(getattr(w, primary_phase), dtype=float)[:n]
        sec = np.asarray(getattr(w, secondary_phase), dtype=float)[:n]
        mask = np.isfinite(prim) & np.isfinite(sec) & (prim > 0)
        if mask.sum() < min_history:
            continue
        ratio = np.full(n, np.nan)
        ratio[mask] = sec[mask] / prim[mask]
        r0 = np.nanmedian(ratio[:anchor_months])
        if not np.isfinite(r0) or r0 <= 0:
            continue
        shapes.append(ratio / r0)
        max_len = max(max_len, n)

    if len(shapes) < 2:  # need at least a couple to call it a cohort
        return None

    # pad to common length and take the month-wise median
    padded = np.full((len(shapes), max_len), np.nan)
    for i, s in enumerate(shapes):
        padded[i, : len(s)] = s
    med = np.nanmedian(padded, axis=0)
    # forward-fill any trailing NaNs with the last finite value (plateau)
    last = 1.0
    for m in range(len(med)):
        if np.isfinite(med[m]):
            last = med[m]
        else:
            med[m] = last
    return CohortRatioShape(multiplier_by_month=med, n_members=len(shapes), max_month=max_len - 1)


def cohort_prior_secondary(
    primary_history: np.ndarray,
    secondary_history: np.ndarray,
    primary_forecast: np.ndarray,
    cohort_shape: CohortRatioShape,
    *,
    anchor_months: int = 4,
    blend_with_self: bool = True,
) -> np.ndarray | None:
    """Forecast the secondary phase by riding the cohort's ratio SHAPE
    (anchored to this well's own early ratio) on the primary forecast.

    If `blend_with_self` is True, the cohort shape is blended with the well's
    own trailing ratio by the same history-driven shrinkage weight used for
    decline params — so a longer-history well leans more on what it has
    actually observed and less on the cohort. This is the smooth
    prior->data handoff the Bayesian methods do; we do it explicitly.
    """
    p = np.asarray(primary_history, dtype=float)
    s = np.asarray(secondary_history, dtype=float)
    mask = np.isfinite(p) & np.isfinite(s) & (p > 0)
    if mask.sum() < 1:
        return None

    n_hist = int(mask.sum())
    ratio_hist = np.full(len(p), np.nan)
    ratio_hist[mask] = s[mask] / p[mask]

    r0 = np.nanmedian(ratio_hist[:anchor_months])
    if not np.isfinite(r0) or r0 <= 0:
        return None

    horizon = len(primary_forecast)
    months_future = np.arange(n_hist, n_hist + horizon)
    # cohort multiplier at each future month (clamp to available shape length)
    idx = np.clip(months_future, 0, cohort_shape.max_month)
    cohort_mult = cohort_shape.multiplier_by_month[idx]
    cohort_ratio_future = r0 * cohort_mult

    if blend_with_self:
        # well's own trailing ratio, held flat, as the "self" estimate
        tail = ratio_hist[mask][-min(3, n_hist):]
        self_ratio_future = np.full(horizon, float(np.nanmedian(tail)))
        w = shrinkage_weight(n_hist)  # more history -> trust self more
        ratio_future = w * self_ratio_future + (1 - w) * cohort_ratio_future
    else:
        ratio_future = cohort_ratio_future

    return ratio_future * np.asarray(primary_forecast, dtype=float)
