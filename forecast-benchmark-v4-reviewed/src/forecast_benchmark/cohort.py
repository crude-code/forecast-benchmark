"""Cohort / analog shrinkage — the cheap, non-Bayesian version of the
short-history fix (OPEN_QUESTIONS.md Gap 3).

The idea, in one sentence: a well with 4 months of history shouldn't be
trusted to fit its own decline shape, so shrink its fitted parameters toward
the parameters of its cohort (basin/reservoir peers), by an amount that
depends on how much history the well actually has.

This is the principled thing the literature (Gong et al. SPE-147588; Lee &
Mallick's hierarchical model) does with a full Bayesian sampler. We do the
shrinkage explicitly instead of via MCMC, which captures most of the benefit
for a fraction of the machinery — and, unlike our production `own_b` fix,
this generalizes past just the b parameter.

Why this is the FOUNDATION and gets built first:
  - It's what makes a "unified vs. routing" comparison a fair fight — right
    now the benchmark can only compare on wells with enough history to fit
    independently (per OPEN_QUESTIONS.md). A cohort prior is exactly the
    thing a routing engine uses for its thin/no-history wells.
  - Both ratio models (GOR/CGR and WOR) lean on it: on a 6-month well the
    GOR rise often hasn't started yet, so the ratio model has to borrow the
    cohort's transition timing. Same shrinkage machinery.

No petbox-dca or Bayesian dependency — pure numpy. Shrinkage is a weighted
average in parameter space, with the weight driven by history length.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class CohortStats:
    """Summary of a cohort's fitted parameters. `n_members` is how many wells
    actually contributed a usable fit (not just how many were in the cohort —
    a cohort of 50 young wells might yield only 8 fittable members)."""
    qi_median: float
    di_median: float
    b_median: float
    n_members: int
    # spread is kept for diagnostics / future Bayesian upgrade, not used in
    # the point-shrinkage math itself
    b_mad: float = 0.0  # median absolute deviation of b across the cohort


def build_cohort_stats(fits: list[dict]) -> CohortStats | None:
    """Aggregate a list of per-well fit dicts (each with qi/di/b) into cohort
    medians. Returns None if no usable fits — the caller must then fall back
    to a hard-coded default prior, and say so, rather than invent a cohort.

    Medians (not means) throughout: decline parameters are heavy-tailed and a
    single bound-riding fit shouldn't drag the cohort. This mirrors the SPEE
    committee's own choice to score on median log error.
    """
    qis = [f["qi"] for f in fits if _usable(f)]
    dis = [f["di"] for f in fits if _usable(f)]
    bs = [f["b"] for f in fits if _usable(f)]
    if not bs:
        return None
    b_arr = np.array(bs)
    b_med = float(np.median(b_arr))
    b_mad = float(np.median(np.abs(b_arr - b_med)))
    return CohortStats(
        qi_median=float(np.median(qis)),
        di_median=float(np.median(dis)),
        b_median=b_med,
        n_members=len(bs),
        b_mad=b_mad,
    )


def _usable(fit: dict) -> bool:
    return (
        fit is not None
        and np.isfinite(fit.get("qi", np.nan))
        and np.isfinite(fit.get("di", np.nan))
        and np.isfinite(fit.get("b", np.nan))
    )


def shrinkage_weight(n_months: int, *, half_trust_months: float = 24.0) -> float:
    """How much to trust the well's OWN fit vs. the cohort prior, in [0, 1].

    0.0 = trust the cohort entirely (no usable history).
    1.0 = trust the well's own fit entirely (long, mature history).

    The functional form is `n / (n + half_trust_months)` — a smooth, bounded
    curve that hits 0.5 exactly at `half_trust_months`. We default the
    half-trust point to 24 months because that's the code's own identifiability
    threshold and is conservative relative to the Younk & Hoffman ~4-year
    finding for *fully* trusting a free b fit. So:

        6 mo  -> weight 0.20  (cohort dominates — right for a young well)
       24 mo  -> weight 0.50  (evenly balanced)
       48 mo  -> weight 0.67  (own fit leads, cohort still tempers)
       96 mo  -> weight 0.80  (own fit strongly leads)

    This is the smooth version of our production engine's hard gate ("borrow
    cohort b under 24 months, fit your own over 30"). Smooth avoids the cliff
    where a well crossing 24 months suddenly swings its whole forecast.
    """
    n = max(0, int(n_months))
    if half_trust_months <= 0:
        return 1.0
    return n / (n + half_trust_months)


def shrink_params(
    own: dict | None,
    cohort: CohortStats | None,
    n_months: int,
    *,
    default_b: float = 0.8,
    half_trust_months: float = 24.0,
    b_clamp: tuple[float, float] = (0.3, 1.3),
) -> dict:
    """Blend a well's own fit with its cohort prior by the history-driven
    weight. Returns a dict with shrunk qi/di/b plus provenance fields so the
    forecast can explain where each parameter came from (the "show your work"
    principle — industry write-ups on automated forecasting consistently find
    that forecast *review*, not generation, is where the real time goes, so
    provenance that speeds review is worth carrying).

    Fallback ladder, explicit and honest:
      - no own fit AND no cohort  -> default b, cohort-less: nothing to lean on
      - no own fit, have cohort   -> cohort params outright (weight forced 0)
      - have own fit, no cohort   -> own params, but b clamped as a backstop
      - both                      -> weighted blend
    """
    w = shrinkage_weight(n_months, half_trust_months=half_trust_months)

    own_ok = _usable(own or {})
    have_cohort = cohort is not None

    if not own_ok and not have_cohort:
        return {
            "qi": np.nan, "di": np.nan, "b": default_b,
            "weight_own": 0.0, "b_source": "default_no_data",
        }
    if not own_ok and have_cohort:
        return {
            "qi": cohort.qi_median, "di": cohort.di_median, "b": cohort.b_median,
            "weight_own": 0.0, "b_source": "cohort_only",
        }
    if own_ok and not have_cohort:
        b = float(np.clip(own["b"], *b_clamp))
        return {
            "qi": own["qi"], "di": own["di"], "b": b,
            "weight_own": 1.0, "b_source": "own_only_clamped",
        }

    # both present: weighted blend in parameter space
    qi = w * own["qi"] + (1 - w) * cohort.qi_median
    di = w * own["di"] + (1 - w) * cohort.di_median
    b = w * own["b"] + (1 - w) * cohort.b_median
    b = float(np.clip(b, *b_clamp))
    return {
        "qi": float(qi), "di": float(di), "b": b,
        "weight_own": round(w, 3), "b_source": "shrunk",
    }
