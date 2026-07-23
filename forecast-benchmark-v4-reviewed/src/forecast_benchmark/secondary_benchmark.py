"""Head-to-head benchmark: ratio method vs. independent fit, for secondary
phases (gas on oil wells, condensate on gas wells, water).

This is the module that actually *proves the claim* the research memos make:
that forecasting a secondary phase by riding a fitted ratio on the primary
beats fitting the secondary stream independently. It's a separate function
from the primary-phase `run_benchmark` so the existing harness and its tests
are untouched — this adds capability, it doesn't rewire what works.

For each well it runs two competing methods on the SAME split and scores both:

  A. INDEPENDENT  — fit the secondary stream on its own with the standard
     bounded-b Arps model (what the field does by default, and what SPEE
     found gets the direction wrong).
  B. RATIO        — forecast the primary, fit the historical ratio, ride the
     ratio on the primary forecast (ratios.secondary_from_primary), with a
     flat-ratio fallback when the trend won't fit.

The output is a paired comparison: same wells, same holdout, two methods,
so the difference is the method and nothing else.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from forecast_benchmark.arps import arps_hyperbolic_bounded_b
from forecast_benchmark.data import WellSeries
from forecast_benchmark.metrics import ALL_METRICS, bias, spee_score
from forecast_benchmark.ratios import flat_ratio_fallback, secondary_from_primary
from forecast_benchmark.split import make_splits

# which secondary phase rides on which primary, and whether its ratio rises
_SECONDARY_PLAN = {
    # (primary, secondary): rising?  (None lets the fit decide)
    "gas_on_oil": {"primary": "oil", "secondary": "gas", "rising": True},
    "cond_on_gas": {"primary": "gas", "secondary": "oil", "rising": False},  # CGR uses oil col as condensate
    "water_on_oil": {"primary": "oil", "secondary": "water", "rising": None},
}


@dataclass
class PairedResult:
    well_id: str
    plan: str
    n_hist: int
    independent_scores: dict[str, float | None]
    ratio_scores: dict[str, float | None]
    ratio_used_fallback: bool


@dataclass
class SecondaryBenchmarkResult:
    plan: str
    primary: str
    secondary: str
    cutoff_month: int
    horizon: int
    n_scored: int
    per_well: list[PairedResult]
    summary: dict[str, dict[str, float | None]]  # method -> metric -> aggregate


def run_secondary_benchmark(
    wells: list[WellSeries],
    *,
    plan: str = "gas_on_oil",
    cutoff_month: int,
    horizon: int,
    ratio_shape: str = "power_law",
) -> SecondaryBenchmarkResult:
    cfg = _SECONDARY_PLAN[plan]
    primary_phase, secondary_phase, rising = cfg["primary"], cfg["secondary"], cfg["rising"]

    splits, _ = make_splits(wells, cutoff_month=cutoff_month, horizon=horizon)
    per_well: list[PairedResult] = []

    for split in splits:
        prim_hist = getattr(split.train, primary_phase)
        sec_hist = getattr(split.train, secondary_phase)
        sec_actual = split.holdout_actuals[secondary_phase]

        # need both phases present in history and a real secondary holdout
        if not (split.train.phase_available(primary_phase)
                and split.train.phase_available(secondary_phase)):
            continue
        if not np.any(np.isfinite(sec_actual)):
            continue

        # --- Method A: INDEPENDENT fit of the secondary stream ---
        indep_fn = arps_hyperbolic_bounded_b(sec_hist)
        indep_forecast = indep_fn(horizon)
        indep_scores = {name: fn(sec_actual, indep_forecast) for name, fn in ALL_METRICS.items()}

        # --- Method B: RATIO ridden on the primary forecast ---
        prim_fn = arps_hyperbolic_bounded_b(prim_hist)
        prim_forecast = prim_fn(horizon)
        ratio_forecast = secondary_from_primary(
            prim_hist, sec_hist, prim_forecast, shape=ratio_shape, rising=rising
        )
        used_fallback = ratio_forecast is None
        if used_fallback:
            ratio_forecast = flat_ratio_fallback(prim_hist, sec_hist, prim_forecast)
        ratio_scores = {name: fn(sec_actual, ratio_forecast) for name, fn in ALL_METRICS.items()}

        n_hist = int(np.sum(np.isfinite(prim_hist) & (np.asarray(prim_hist) > 0)))
        per_well.append(PairedResult(
            well_id=split.well_id, plan=plan, n_hist=n_hist,
            independent_scores=indep_scores, ratio_scores=ratio_scores,
            ratio_used_fallback=used_fallback,
        ))

    summary = _summarize_paired(per_well)
    return SecondaryBenchmarkResult(
        plan=plan, primary=primary_phase, secondary=secondary_phase,
        cutoff_month=cutoff_month, horizon=horizon,
        n_scored=len(per_well), per_well=per_well, summary=summary,
    )


def _summarize_paired(per_well: list[PairedResult]) -> dict[str, dict[str, float | None]]:
    out: dict[str, dict[str, float | None]] = {"independent": {}, "ratio": {}}
    for method, key in (("independent", "independent_scores"), ("ratio", "ratio_scores")):
        for metric in ALL_METRICS:
            vals = [getattr(r, key)[metric] for r in per_well
                    if getattr(r, key).get(metric) is not None]
            out[method][f"{metric}_median"] = float(np.median(vals)) if vals else None
            out[method][f"{metric}_mean"] = float(np.mean(vals)) if vals else None
        out[method]["n"] = len(per_well)
    out["_meta"] = {
        "n_fallback": sum(1 for r in per_well if r.ratio_used_fallback),
    }
    return out
