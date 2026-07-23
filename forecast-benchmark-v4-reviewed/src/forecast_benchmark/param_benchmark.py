"""Parameter-level benchmark: hide the last months, ask every method for
(qi, Di, b), score the curve those parameters imply on the held-out window, and
record the parameters next to the error.

This is the base benchmark's discipline (point-in-time split, phase-aware
scoring, loud skips) applied to methods that share one output space. The result
is a tidy table — one row per (well, phase, method) — carrying both the chosen
parameters and the resulting error, so error can be attributed to the judgment
that produced it.
"""
from __future__ import annotations

import csv as csv_mod
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from forecast_benchmark.data import PHASES, WellSeries
from forecast_benchmark.metrics import ALL_METRICS
from forecast_benchmark.params import HyperbolicParams, ParamModelFn
from forecast_benchmark.split import make_splits

# Column order for the flat table / CSV export.
PARAM_COLUMNS = ("qi", "di", "b")
_ROW_FIELDS = ("well_id", "phase", "model", "phase_available", *PARAM_COLUMNS, *ALL_METRICS)


@dataclass
class ParamWellResult:
    well_id: str
    phase: str
    model: str
    phase_available: bool
    params: HyperbolicParams | None
    scores: dict[str, float | None] = field(default_factory=dict)

    def row(self) -> dict[str, object]:
        """Flatten to a plain dict keyed by _ROW_FIELDS (params + metrics)."""
        r: dict[str, object] = {
            "well_id": self.well_id,
            "phase": self.phase,
            "model": self.model,
            "phase_available": self.phase_available,
        }
        for col in PARAM_COLUMNS:
            r[col] = getattr(self.params, col) if self.params is not None else None
        for metric in ALL_METRICS:
            r[metric] = self.scores.get(metric)
        return r


@dataclass
class ParamBenchmarkResult:
    cutoff_month: int
    horizon: int
    model_names: list[str]
    n_wells_scored: int
    n_wells_skipped: int
    skipped_well_ids: list[str]
    per_well: list[ParamWellResult]

    def rows(self) -> list[dict[str, object]]:
        return [r.row() for r in self.per_well]

    def write_csv(self, path: str) -> None:
        with open(path, "w", newline="") as f:
            writer = csv_mod.DictWriter(f, fieldnames=list(_ROW_FIELDS))
            writer.writeheader()
            for r in self.per_well:
                writer.writerow(r.row())


def run_param_benchmark(
    wells: list[WellSeries],
    param_models: dict[str, ParamModelFn],
    *,
    cutoff_month: int,
    horizon: int,
) -> ParamBenchmarkResult:
    """Score every (well, phase, method) at one cutoff/horizon.

    Each method sees only ``train`` (months up to the cutoff) for the phase it is
    forecasting — the point-in-time guarantee is inherited from ``make_splits``.
    A phase with no reported training data is recorded phase_available=False with
    no parameters and no scores; it is never forecast as zero and never dropped
    from the count.
    """
    splits, skipped = make_splits(wells, cutoff_month=cutoff_month, horizon=horizon)

    per_well: list[ParamWellResult] = []
    for split in splits:
        for phase in PHASES:
            train_arr = getattr(split.train, phase)
            available = split.train.phase_available(phase)
            actual = split.holdout_actuals[phase]
            for model_name, model_fn in param_models.items():
                if not available:
                    per_well.append(ParamWellResult(
                        well_id=split.well_id, phase=phase, model=model_name,
                        phase_available=False, params=None, scores={},
                    ))
                    continue
                params = model_fn(train_arr)
                forecast = params.forecast(horizon)
                scores = {name: fn(actual, forecast) for name, fn in ALL_METRICS.items()}
                per_well.append(ParamWellResult(
                    well_id=split.well_id, phase=phase, model=model_name,
                    phase_available=True, params=params, scores=scores,
                ))

    return ParamBenchmarkResult(
        cutoff_month=cutoff_month,
        horizon=horizon,
        model_names=list(param_models),
        n_wells_scored=len(splits),
        n_wells_skipped=len(skipped),
        skipped_well_ids=skipped,
        per_well=per_well,
    )


def summarize_by_model(
    result: ParamBenchmarkResult, *, phase: str = "oil", metric: str = "spee_score"
) -> dict[str, dict[str, float | None]]:
    """Median/mean of one metric per method for one phase, plus median params.

    A compact leaderboard for chat: which method won, and the typical parameters
    each one chose. Returns {model: {metric_median, metric_mean, qi_median,
    di_median, b_median, n}}.
    """
    out: dict[str, dict[str, float | None]] = {}
    for model_name in result.model_names:
        rows = [
            r for r in result.per_well
            if r.model == model_name and r.phase == phase and r.phase_available
        ]
        vals = [r.scores.get(metric) for r in rows if r.scores.get(metric) is not None]
        stats: dict[str, float | None] = {
            "n": float(len(rows)),
            f"{metric}_median": float(np.median(vals)) if vals else None,
            f"{metric}_mean": float(np.mean(vals)) if vals else None,
        }
        for col in PARAM_COLUMNS:
            pv = [getattr(r.params, col) for r in rows if r.params is not None]
            stats[f"{col}_median"] = float(np.median(pv)) if pv else None
        out[model_name] = stats
    return out


def format_param_leaderboard(
    result: ParamBenchmarkResult, *, phase: str = "oil", metric: str = "spee_score"
) -> str:
    """Plain-text leaderboard: method, median metric, and median (qi, Di, b)."""
    summary = summarize_by_model(result, phase=phase, metric=metric)
    lines = [
        f"Parameter benchmark — phase={phase}, metric={metric} "
        f"(cutoff={result.cutoff_month}, horizon={result.horizon}, "
        f"n_scored={result.n_wells_scored}, n_skipped={result.n_wells_skipped})",
        f"{'model':<20}{metric+'_median':>16}{'qi_median':>14}{'di_median':>12}{'b_median':>10}",
    ]

    def _fmt(v: float | None, width: int, prec: int) -> str:
        return (f"{v:>{width}.{prec}f}" if isinstance(v, float) else f"{'-':>{width}}")

    ranked = sorted(
        summary.items(),
        key=lambda kv: (kv[1].get(f"{metric}_median") is None, kv[1].get(f"{metric}_median") or 0.0),
    )
    for model_name, stats in ranked:
        lines.append(
            f"{model_name:<20}"
            + _fmt(stats.get(f"{metric}_median"), 16, 4)
            + _fmt(stats.get("qi_median"), 14, 1)
            + _fmt(stats.get("di_median"), 12, 4)
            + _fmt(stats.get("b_median"), 10, 2)
        )
    return "\n".join(lines)
