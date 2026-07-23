"""Train/holdout split for hindcasting. Point-in-time discipline: a model
under test only ever sees months up to the cutoff — never the holdout.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from forecast_benchmark.data import PHASES, WellSeries


@dataclass(frozen=True)
class Split:
    well_id: str
    train: WellSeries          # months[:cutoff]
    holdout_months: list[str]  # months[cutoff:cutoff+horizon]
    holdout_actuals: dict[str, np.ndarray]  # phase -> actual values over holdout_months


def make_split(well: WellSeries, *, cutoff_month: int, horizon: int) -> Split | None:
    """Split one well at cutoff_month, scoring the next `horizon` months.

    Returns None (not an exception) if the well doesn't have enough total
    history to support both the cutoff and the full holdout — a benchmark
    should skip ineligible wells loudly in aggregate stats, not crash or
    silently truncate the holdout window shorter than requested.
    """
    total = len(well.months)
    if cutoff_month < 1 or cutoff_month + horizon > total:
        return None

    train = well.truncate(cutoff_month)
    holdout_months = well.months[cutoff_month:cutoff_month + horizon]
    holdout_actuals = {
        phase: getattr(well, phase)[cutoff_month:cutoff_month + horizon]
        for phase in PHASES
    }
    return Split(
        well_id=well.well_id,
        train=train,
        holdout_months=holdout_months,
        holdout_actuals=holdout_actuals,
    )


def make_splits(
    wells: list[WellSeries], *, cutoff_month: int, horizon: int
) -> tuple[list[Split], list[str]]:
    """Split every well; returns (splits, skipped_well_ids). Skips are a
    first-class output, not a silent filter — a benchmark run should always
    report how many wells it could and couldn't score."""
    splits, skipped = [], []
    for well in wells:
        s = make_split(well, cutoff_month=cutoff_month, horizon=horizon)
        if s is None:
            skipped.append(well.well_id)
        else:
            splits.append(s)
    return splits, skipped
