"""Ties data -> split -> model -> metrics together into one benchmark run.

A model in this repo is any callable: (WellSeries.train_phase_array) ->
ForecastFn. See arps.py and unified.py for the model set.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from forecast_benchmark.data import PHASES, WellSeries
from forecast_benchmark.metrics import ALL_METRICS
from forecast_benchmark.split import Split, make_splits

ModelFn = Callable[[np.ndarray], "ForecastFn"]  # noqa: F821 (ForecastFn from arps.py)


@dataclass
class WellResult:
    well_id: str
    phase: str
    phase_available: bool
    scores: dict[str, float | None]


@dataclass
class BenchmarkResult:
    model_name: str
    cutoff_month: int
    horizon: int
    n_wells_scored: int
    n_wells_skipped: int
    skipped_well_ids: list[str]
    per_well: list[WellResult]
    summary: dict[str, dict[str, float | None]]  # phase -> metric -> aggregate


def run_benchmark(
    wells: list[WellSeries],
    model_fn: ModelFn,
    *,
    model_name: str,
    cutoff_month: int,
    horizon: int,
) -> BenchmarkResult:
    """Run one model against one train/holdout configuration across all
    wells and phases. Phases with no reported data for a well are scored
    as phase_available=False with scores={} — never silently forecast as
    zero and never silently dropped from the well count."""
    splits, skipped = make_splits(wells, cutoff_month=cutoff_month, horizon=horizon)

    per_well: list[WellResult] = []
    for split in splits:
        for phase in PHASES:
            train_arr = getattr(split.train, phase)
            available = split.train.phase_available(phase)
            if not available:
                per_well.append(WellResult(
                    well_id=split.well_id, phase=phase,
                    phase_available=False, scores={},
                ))
                continue

            forecast_fn = model_fn(train_arr)
            forecast = forecast_fn(horizon)
            actual = split.holdout_actuals[phase]

            scores = {name: fn(actual, forecast) for name, fn in ALL_METRICS.items()}
            per_well.append(WellResult(
                well_id=split.well_id, phase=phase,
                phase_available=True, scores=scores,
            ))

    summary = _summarize(per_well)

    return BenchmarkResult(
        model_name=model_name,
        cutoff_month=cutoff_month,
        horizon=horizon,
        n_wells_scored=len(splits),
        n_wells_skipped=len(skipped),
        skipped_well_ids=skipped,
        per_well=per_well,
        summary=summary,
    )


def _summarize(per_well: list[WellResult]) -> dict[str, dict[str, float | None]]:
    summary: dict[str, dict[str, float | None]] = {}
    for phase in PHASES:
        phase_results = [r for r in per_well if r.phase == phase and r.phase_available]
        phase_summary: dict[str, float | None] = {
            "n_scored": len(phase_results),
            "n_phase_unavailable": sum(
                1 for r in per_well if r.phase == phase and not r.phase_available
            ),
        }
        for metric_name in ALL_METRICS:
            values = [
                r.scores[metric_name] for r in phase_results
                if r.scores.get(metric_name) is not None
            ]
            if values:
                phase_summary[f"{metric_name}_median"] = float(np.median(values))
                phase_summary[f"{metric_name}_mean"] = float(np.mean(values))
            else:
                phase_summary[f"{metric_name}_median"] = None
                phase_summary[f"{metric_name}_mean"] = None
        summary[phase] = phase_summary
    return summary
