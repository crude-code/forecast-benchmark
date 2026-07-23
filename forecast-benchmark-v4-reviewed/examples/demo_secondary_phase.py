"""Runnable demo: reproduces the headline result from the v2 build.

    python examples/demo_secondary_phase.py

Shows, on the bubble-point fixture, that:
  1. On wells with enough history, ratio and independent methods are close.
  2. On a young well cut BEFORE its GOR rise, neither self-contained method
     can see the rise coming...
  3. ...but borrowing the cohort's transition timing recovers it.

This is the empirical case for the "forecast a ratio + borrow a cohort prior"
architecture — not an assertion, a measured result you can re-run.
"""
import os
import sys
import warnings

# Let this demo run directly from a fresh checkout without requiring an
# editable install first. Installed package imports still win normally.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import numpy as np

warnings.filterwarnings("ignore")

from forecast_benchmark.arps import arps_hyperbolic_bounded_b
from forecast_benchmark.cohort_ratio import build_cohort_ratio_shape, cohort_prior_secondary
from forecast_benchmark.data import load_csv
from forecast_benchmark.ratios import secondary_from_primary
from forecast_benchmark.secondary_benchmark import run_secondary_benchmark

HERE = os.path.dirname(__file__)


def cum_err(forecast, actual):
    return 100 * (np.nansum(forecast) - np.nansum(actual)) / np.nansum(actual)


def main():
    wells = load_csv(os.path.join(HERE, "bubblepoint_wells.csv"))
    print("=" * 68)
    print("Secondary-phase forecasting demo (bubble-point GOR fixture)")
    print("=" * 68)

    print("\n[1] Aggregate ratio-vs-independent, cut at 18 mo (moderate history):")
    r = run_secondary_benchmark(wells, plan="gas_on_oil", cutoff_month=18, horizon=18)
    ind, rat = r.summary["independent"], r.summary["ratio"]
    print(f"    independent  bias {ind['bias_median']:+6.1f}%   mape {ind['mape_median']:5.1f}%")
    print(f"    ratio        bias {rat['bias_median']:+6.1f}%   mape {rat['mape_median']:5.1f}%")

    print("\n[2] One young well (BP-03), cut at 10 mo = BEFORE its GOR rise (~mo 20):")
    target = [w for w in wells if w.well_id == "BP-03"][0]
    cut, end = 10, 30
    horizon = end - cut
    oil_h, gas_h = target.oil[:cut], target.gas[:cut]
    oil_fc = arps_hyperbolic_bounded_b(oil_h)(horizon)
    gas_actual = np.asarray(target.gas[cut:end], dtype=float)

    indep = arps_hyperbolic_bounded_b(gas_h)(horizon)
    self_ratio = secondary_from_primary(oil_h, gas_h, oil_fc, shape="power_law", rising=True)

    cohort = [w for w in wells if w.well_id != "BP-03"]
    shape = build_cohort_ratio_shape(cohort, min_history=30)
    cohort_gas = cohort_prior_secondary(oil_h, gas_h, oil_fc, shape, blend_with_self=True)

    print(f"    independent fit    : {cum_err(indep, gas_actual):+5.0f}% cumulative gas error")
    print(f"    self-ratio (flat)  : {cum_err(self_ratio, gas_actual):+5.0f}% cumulative gas error")
    print(f"    COHORT-PRIOR ratio : {cum_err(cohort_gas, gas_actual):+5.0f}% cumulative gas error  <-- borrows timing")

    print("\n[3] Why: cohort's median GOR multiplier vs. month (the transition it borrows):")
    m = shape.multiplier_by_month
    cells = "  ".join(f"mo{mo}:{m[mo]:.1f}x" for mo in [0, 6, 12, 18, 24] if mo <= shape.max_month)
    print(f"    {cells}")
    print("\nThe young well can't see its own rise yet — the cohort already did.")
    print("=" * 68)


if __name__ == "__main__":
    main()
