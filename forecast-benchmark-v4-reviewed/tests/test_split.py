import numpy as np
import pytest

from forecast_benchmark.data import WellSeries
from forecast_benchmark.split import make_split, make_splits


def _well(n_months: int, well_id: str = "W1") -> WellSeries:
    months = [f"2021-{(i % 12) + 1:02d}-01" for i in range(n_months)]
    # fix year rollover properly for >12 months
    months = []
    y, m = 2021, 1
    for _ in range(n_months):
        months.append(f"{y}-{m:02d}-01")
        m += 1
        if m > 12:
            m = 1
            y += 1
    oil = np.linspace(1000, 500, n_months)
    gas = oil * 2
    water = np.full(n_months, np.nan)
    return WellSeries(well_id=well_id, months=months, oil=oil, gas=gas, water=water)


def test_make_split_basic():
    well = _well(24)
    split = make_split(well, cutoff_month=12, horizon=6)
    assert split is not None
    assert len(split.train.months) == 12
    assert len(split.holdout_months) == 6
    assert split.holdout_actuals["oil"].shape == (6,)


def test_make_split_insufficient_history_returns_none():
    well = _well(10)
    split = make_split(well, cutoff_month=12, horizon=6)
    assert split is None


def test_make_split_exact_boundary():
    well = _well(18)
    split = make_split(well, cutoff_month=12, horizon=6)
    assert split is not None
    assert len(split.train.months) == 12


def test_make_splits_reports_skips():
    wells = [_well(24, "W1"), _well(10, "W2"), _well(30, "W3")]
    splits, skipped = make_splits(wells, cutoff_month=12, horizon=6)
    assert len(splits) == 2
    assert skipped == ["W2"]


def test_make_split_no_train_peek_past_cutoff():
    well = _well(24)
    split = make_split(well, cutoff_month=12, horizon=6)
    # train never contains holdout months
    assert not set(split.train.months) & set(split.holdout_months)
