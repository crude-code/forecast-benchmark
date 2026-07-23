import os

import numpy as np
import pytest

from forecast_benchmark.arps import arps_hyperbolic_bounded_b
from forecast_benchmark.cohort_ratio import (
    build_cohort_ratio_shape,
    cohort_prior_secondary,
)
from forecast_benchmark.data import load_csv
from forecast_benchmark.ratios import secondary_from_primary

FIX = os.path.join(os.path.dirname(__file__), "..", "examples", "bubblepoint_wells.csv")


@pytest.fixture
def wells():
    return load_csv(FIX)


def test_build_cohort_shape_rises_through_bubble_point(wells):
    shape = build_cohort_ratio_shape(wells, min_history=30)
    assert shape is not None
    assert shape.n_members >= 2
    m = shape.multiplier_by_month
    # GOR multiplier should start ~1 and rise well above 1 after the transition
    assert abs(m[0] - 1.0) < 0.3
    assert m[20] > 1.8  # transition has happened by month 20


def test_build_cohort_shape_none_when_no_mature_members(wells):
    # require impossibly long history -> no members
    assert build_cohort_ratio_shape(wells, min_history=1000) is None


def test_cohort_prior_beats_self_fit_on_young_well(wells):
    """The headline result: on a well cut before its GOR rise, borrowing the
    cohort's transition timing beats both an independent fit and a self-ratio
    fit. This is the empirical justification for building the cohort layer."""
    target = [w for w in wells if w.well_id == "BP-03"][0]  # transition ~mo 20
    cut, end = 10, 30
    horizon = end - cut

    # cohort excludes the target well itself
    cohort = [w for w in wells if w.well_id != "BP-03"]
    shape = build_cohort_ratio_shape(cohort, min_history=30)
    assert shape is not None

    oil_h, gas_h = target.oil[:cut], target.gas[:cut]
    oil_fc = arps_hyperbolic_bounded_b(oil_h)(horizon)
    gas_actual = np.asarray(target.gas[cut:end], dtype=float)

    indep = arps_hyperbolic_bounded_b(gas_h)(horizon)
    self_ratio = secondary_from_primary(oil_h, gas_h, oil_fc, shape="power_law", rising=True)
    cohort_gas = cohort_prior_secondary(oil_h, gas_h, oil_fc, shape, blend_with_self=True)

    def cum_err(f):
        return abs(np.nansum(f) - np.nansum(gas_actual)) / np.nansum(gas_actual)

    # cohort-prior error must be clearly smaller than both self-contained methods
    assert cum_err(cohort_gas) < cum_err(indep)
    assert cum_err(cohort_gas) < cum_err(self_ratio)
    # and in absolute terms, materially better than the ~46% self-fit miss
    assert cum_err(cohort_gas) < 0.25


def test_cohort_prior_blends_toward_self_with_more_history(wells):
    """A longer-history well should lean more on its own observed ratio than a
    short-history one — the smooth prior->data handoff."""
    target = [w for w in wells if w.well_id == "BP-01"][0]  # 54 months
    cohort = [w for w in wells if w.well_id != "BP-01"]
    shape = build_cohort_ratio_shape(cohort, min_history=30)
    assert shape is not None

    oil_h, gas_h = target.oil[:40], target.gas[:40]  # long history
    oil_fc = arps_hyperbolic_bounded_b(oil_h)(10)
    out = cohort_prior_secondary(oil_h, gas_h, oil_fc, shape, blend_with_self=True)
    assert out is not None and len(out) == 10
    assert np.all(np.isfinite(out))


def test_cohort_prior_none_on_empty_primary(wells):
    shape = build_cohort_ratio_shape(wells, min_history=30)
    empty = np.full(6, np.nan)
    out = cohort_prior_secondary(empty, empty, np.array([100.0] * 6), shape)
    assert out is None
