"""The simplest honest test: cumulative oil over the hidden window.

Each method produces a monthly oil forecast for the next `horizon` months. We
sum it, sum the hidden actuals over the same months, and compare. One number
per method per well: how far off was the 12-month cum.

No parameters, no anchoring, no curve shape. A method forecasts however it likes
(fit from peak, fit from the tail, an LLM, whatever) — this harness only looks
at the volume it delivered against the volume that actually showed up.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from forecast_benchmark.data import WellSeries
from forecast_benchmark.split import make_splits

ModelFn = Callable[[np.ndarray], Callable[[int], np.ndarray]]


@dataclass
class CumResult:
    well_id: str
    model: str
    pred_cum: float
    actual_cum: float

    @property
    def error(self) -> float:
        """Signed error, predicted minus actual. Positive = over-forecast."""
        return self.pred_cum - self.actual_cum

    @property
    def pct_error(self) -> float:
        """Signed percent error vs actual. NaN if the actual cum is zero."""
        return self.error / self.actual_cum if self.actual_cum else float("nan")


def run_cum_benchmark(
    wells: list[WellSeries],
    forecasters: dict[str, ModelFn],
    *,
    cutoff_month: int,
    horizon: int = 12,
    phase: str = "oil",
) -> tuple[list[CumResult], list[str]]:
    """Score every method on the cumulative `phase` volume over the hidden
    window. Returns (results, skipped_well_ids).

    A well with no reported actuals over the window (all NaN) is skipped for
    scoring — there is nothing to compare against.
    """
    splits, skipped = make_splits(wells, cutoff_month=cutoff_month, horizon=horizon)

    results: list[CumResult] = []
    for split in splits:
        actual = split.holdout_actuals[phase]
        if not np.any(~np.isnan(actual)):
            skipped.append(split.well_id)
            continue
        actual_cum = float(np.nansum(actual))

        train_arr = getattr(split.train, phase)
        for model_name, model_fn in forecasters.items():
            forecast = model_fn(train_arr)(horizon)
            pred_cum = float(np.nansum(forecast))
            results.append(CumResult(
                well_id=split.well_id, model=model_name,
                pred_cum=pred_cum, actual_cum=actual_cum,
            ))
    return results, skipped


def format_cum_report(
    results: list[CumResult], *, horizon: int = 12, phase: str = "oil"
) -> str:
    """Per-method summary: median absolute percent error on the cum, and the
    median signed percent error (bias). Lower abs is better; signed shows lean.
    """
    models = sorted({r.model for r in results})
    lines = [f"Cumulative {phase} over next {horizon} months — predicted vs hidden actual"]
    lines.append(f"{'model':<20}{'n':>5}{'median_abs_%err':>18}{'median_signed_%err':>20}")
    rows = []
    for model in models:
        errs = [r.pct_error for r in results if r.model == model and not np.isnan(r.pct_error)]
        if not errs:
            rows.append((model, 0, None, None))
            continue
        med_abs = float(np.median(np.abs(errs)))
        med_signed = float(np.median(errs))
        rows.append((model, len(errs), med_abs, med_signed))
    for model, n, med_abs, med_signed in sorted(rows, key=lambda x: (x[2] is None, x[2] or 0.0)):
        if med_abs is None:
            lines.append(f"{model:<20}{n:>5}{'-':>18}{'-':>20}")
        else:
            lines.append(f"{model:<20}{n:>5}{med_abs*100:>17.1f}%{med_signed*100:>19.1f}%")
    return "\n".join(lines)
