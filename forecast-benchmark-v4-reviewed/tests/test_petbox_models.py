"""Tests for the optional petbox-backed models. Skipped cleanly if petbox-dca
isn't installed, so the core suite stays green without the dependency."""
import os

import numpy as np
import pytest

petbox = pytest.importorskip("petbox.dca", reason="petbox-dca not installed")

from forecast_benchmark.data import load_csv  # noqa: E402
from forecast_benchmark.petbox_models import (  # noqa: E402
    DAYS_PER_MONTH,
    _month_end_days,
    thm_model,
    thm_secondary_from_primary,
)

FIX = os.path.join(os.path.dirname(__file__), "..", "examples", "bubblepoint_wells.csv")


@pytest.fixture
def wells():
    return load_csv(FIX)


def test_month_end_days_convention():
    # month 0 ends at 1 * DAYS_PER_MONTH; month 11 ends at 12 * DAYS_PER_MONTH
    d = _month_end_days(np.array([0, 1, 11]))
    assert abs(d[0] - DAYS_PER_MONTH) < 1e-9
    assert abs(d[2] - 12 * DAYS_PER_MONTH) < 1e-9


def test_thm_forecast_is_monthly_volume_not_daily_rate(wells):
    """THE critical test: a THM forecast must be the same order of magnitude as
    the monthly actuals. If the unit bridge were wrong (rate/day vs monthly
    vol), this would be off by ~30x in one direction or the other."""
    w = [x for x in wells if x.well_id == "BP-01"][0]
    oil = np.asarray(w.oil, dtype=float)
    cut, hor = 24, 12
    fc = thm_model(oil[:cut])(hor)
    actual = oil[cut:cut + hor]
    ratio = np.sum(fc) / np.sum(actual)
    # generous band, but nowhere near 30x or 1/30x
    assert 0.5 < ratio < 2.0, f"unit bridge suspect: cum ratio {ratio:.2f}"


def test_thm_forecast_declines(wells):
    w = [x for x in wells if x.well_id == "BP-03"][0]
    fc = thm_model(np.asarray(w.oil, dtype=float)[:24])(12)
    assert len(fc) == 12
    assert np.all(np.isfinite(fc))
    assert fc[-1] <= fc[0]  # declining


def test_thm_short_history_falls_back_gracefully():
    fc = thm_model(np.array([100.0, 90.0]))(6)  # too short to fit
    assert len(fc) == 6
    assert np.all(np.isfinite(fc))


def test_thm_all_nan_history_is_flat_zero():
    fc = thm_model(np.full(10, np.nan))(4)
    assert np.allclose(fc, 0.0)


def test_thm_secondary_returns_monthly_gas_volume(wells):
    w = [x for x in wells if x.well_id == "BP-01"][0]
    oil = np.asarray(w.oil, dtype=float)
    gas = np.asarray(w.gas, dtype=float)
    cut, hor = 24, 12
    gas_fc = thm_secondary_from_primary(oil[:cut], gas[:cut], hor)
    assert gas_fc is not None
    assert len(gas_fc) == hor
    assert np.all(gas_fc > 0)
    # gas volume should be same magnitude as actual gas, not ~30x off
    gas_actual = gas[cut:cut + hor]
    ratio = np.sum(gas_fc) / np.sum(gas_actual)
    assert 0.3 < ratio < 3.0, f"secondary unit/scale suspect: {ratio:.2f}"


def test_thm_secondary_none_on_too_short():
    out = thm_secondary_from_primary(np.array([100.0, 90.0]), np.array([80.0, 70.0]), 6)
    assert out is None
