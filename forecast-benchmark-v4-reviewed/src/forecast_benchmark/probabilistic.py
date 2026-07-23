"""Probabilistic scoring — for models that output uncertainty bands
(P10/P50/P90), not just a point forecast.

WHY THIS MODULE EXISTS
Every metric in `metrics.py` scores a single point forecast (effectively the
P50). If a model outputs a distribution — which is the whole reason to reach
for a Bayesian approach — grading only its P50 throws away the thing that
makes it worth the complexity: whether its stated uncertainty is *honest*.

The motivating concern (raised on the team, and correct): single-well actuals
are contaminated by allocation error, undocumented downtime, lift changes, and
reporting noise. No fitter can recover information that isn't in the data. So
the useful thing a probabilistic model can do on a messy well is not a better
point forecast — it's to say "wide band, low confidence, don't trust this
one." That claim is only worth anything if the bands are *calibrated*: if the
P90 genuinely contains the truth ~90% of the time. This module measures
exactly that, so a probabilistic model can be tested on its real advantage
rather than unfairly graded on its P50 alone.

TWO QUESTIONS, TWO METRICS
1. Calibration / coverage: of the outcomes that fell at or below the model's
   stated quantile q, was it actually a q-fraction of them? A P90 band that
   only contains truth 60% of the time is overconfident; one that contains it
   99% of the time is uselessly wide. Measured by empirical coverage at each
   quantile, and summarized as a calibration error.
2. Sharpness-given-calibration: among models that ARE calibrated, tighter
   bands are better. Measured by the pinball (quantile) loss, which is the
   proper scoring rule for quantile forecasts — it rewards being both
   well-centered and appropriately narrow, and can't be gamed by just widening
   bands (unlike coverage alone).

A model is only "good" if it's calibrated AND sharp. Coverage alone is gameable
(predict [-inf, +inf] and you have perfect coverage, useless sharpness);
pinball loss alone can be beaten by a lucky point forecast. Report both.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _clean(actual: np.ndarray, quantile_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Drop months where actual is NaN. quantile_pred is aligned to actual."""
    a = np.asarray(actual, dtype=float)
    q = np.asarray(quantile_pred, dtype=float)
    mask = np.isfinite(a) & np.isfinite(q)
    if not np.any(mask):
        return None
    return a[mask], q[mask]


def pinball_loss(actual: np.ndarray, quantile_pred: np.ndarray, q: float) -> float | None:
    """Pinball (quantile) loss for a single quantile level q in (0, 1).

    For each observation, penalizes under-prediction by q * error and
    over-prediction by (1 - q) * error. This is the proper scoring rule for a
    quantile forecast: minimized in expectation exactly when quantile_pred is
    the true q-quantile of the outcome distribution. Lower is better.

    Example: at q=0.9, being below the actual (band too low) is penalized 0.9x,
    being above (band too high) only 0.1x — so the P90 is pushed up until ~90%
    of outcomes fall below it, which is precisely what calibration means.
    """
    if not (0.0 < q < 1.0):
        raise ValueError(f"quantile q must be in (0,1), got {q}")
    pair = _clean(actual, quantile_pred)
    if pair is None:
        return None
    a, pred = pair
    diff = a - pred
    loss = np.where(diff >= 0, q * diff, (q - 1.0) * diff)
    return float(np.mean(loss))


def empirical_coverage(actual: np.ndarray, quantile_pred: np.ndarray, q: float) -> float | None:
    """Fraction of actuals at or below the stated q-quantile prediction.

    A well-calibrated q-quantile has empirical coverage ~= q. Returns the
    observed fraction in [0, 1], to be compared against the nominal q.
    """
    pair = _clean(actual, quantile_pred)
    if pair is None:
        return None
    a, pred = pair
    return float(np.mean(a <= pred))


@dataclass
class CalibrationResult:
    """Coverage at each nominal quantile, plus summary calibration error.

    `by_quantile` maps nominal q -> (empirical_coverage, n). `calibration_error`
    is the mean absolute gap between nominal and empirical coverage across
    quantiles — 0.0 is perfect, and (say) 0.15 means bands are off by 15
    coverage points on average. `interval_coverage` reports, for symmetric
    central intervals (e.g. P10-P90 -> nominal 80%), how often truth landed
    inside — the most intuitive "is my band honest" number.
    """
    by_quantile: dict[float, tuple[float, int]]
    calibration_error: float
    interval_coverage: dict[str, tuple[float, float]] = field(default_factory=dict)
    # interval label -> (nominal, empirical)


def assess_calibration(
    actual: np.ndarray,
    quantile_preds: dict[float, np.ndarray],
) -> CalibrationResult | None:
    """Given actuals and a dict of {quantile_level: prediction_array}, report
    coverage at each level and central-interval coverage.

    `quantile_preds` e.g. {0.1: p10_array, 0.5: p50_array, 0.9: p90_array},
    each aligned to `actual`. Levels can be any subset of (0,1); intervals are
    inferred from symmetric pairs present (0.1&0.9 -> 80%, 0.05&0.95 -> 90%).
    """
    if not quantile_preds:
        return None
    by_q: dict[float, tuple[float, int]] = {}
    errs = []
    for q in sorted(quantile_preds):
        cov = empirical_coverage(actual, quantile_preds[q], q)
        if cov is None:
            continue
        pair = _clean(actual, quantile_preds[q])
        n = len(pair[0]) if pair else 0
        by_q[q] = (cov, n)
        errs.append(abs(cov - q))

    if not by_q:
        return None

    interval_cov: dict[str, tuple[float, float]] = {}
    levels = set(by_q)
    for lo in sorted(l for l in levels if l < 0.5):
        hi = round(1.0 - lo, 10)
        if hi in levels:
            nominal = round(hi - lo, 10)
            # fraction of actuals within [P_lo, P_hi]
            pair_lo = _clean(actual, quantile_preds[lo])
            pair_hi = _clean(actual, quantile_preds[hi])
            if pair_lo and pair_hi:
                a = np.asarray(actual, dtype=float)
                m = np.isfinite(a) & np.isfinite(quantile_preds[lo]) & np.isfinite(quantile_preds[hi])
                inside = np.mean(
                    (a[m] >= np.asarray(quantile_preds[lo])[m])
                    & (a[m] <= np.asarray(quantile_preds[hi])[m])
                )
                interval_cov[f"{int(nominal*100)}%"] = (float(nominal), float(inside))

    return CalibrationResult(
        by_quantile=by_q,
        calibration_error=float(np.mean(errs)),
        interval_coverage=interval_cov,
    )


def mean_pinball(
    actual: np.ndarray,
    quantile_preds: dict[float, np.ndarray],
) -> float | None:
    """Average pinball loss across all provided quantile levels — a single
    number for overall probabilistic forecast quality (lower is better).
    This is the headline "sharpness given calibration" score: a proper rule,
    so it can't be gamed by just widening the bands."""
    if not quantile_preds:
        return None
    losses = []
    for q, pred in quantile_preds.items():
        pl = pinball_loss(actual, pred, q)
        if pl is not None:
            losses.append(pl)
    return float(np.mean(losses)) if losses else None


# Convenience: the standard reporting bundle for a probabilistic forecast.
def score_probabilistic(
    actual: np.ndarray,
    quantile_preds: dict[float, np.ndarray],
) -> dict:
    """One call -> the full probabilistic scorecard: mean pinball loss (proper
    accuracy), calibration error, and central-interval coverage. This is what a
    benchmark row for a probabilistic model should carry, alongside the usual
    point metrics computed on its P50."""
    calib = assess_calibration(actual, quantile_preds)
    return {
        "mean_pinball": mean_pinball(actual, quantile_preds),
        "calibration_error": calib.calibration_error if calib else None,
        "interval_coverage": calib.interval_coverage if calib else {},
        "by_quantile_coverage": calib.by_quantile if calib else {},
    }
