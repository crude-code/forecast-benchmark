# Methodology

## Train/holdout split

Every score in this repo comes from a point-in-time split (`split.py`):
a well's history is truncated at `cutoff_month`, a model forecasts forward,
and the forecast is scored against the real, already-known months from
`cutoff_month` to `cutoff_month + horizon`. The model never sees the
holdout months — this is the only way a benchmark result means anything.

Wells with fewer than `cutoff_month + horizon` total months of history are
**skipped**, not truncated to a shorter holdout and not silently dropped —
`BenchmarkResult.n_wells_skipped` and `.skipped_well_ids` report this every
run, because a benchmark that quietly loses wells is lying about its
coverage.

## Metrics (`metrics.py`)

Every metric operates on `(actual, forecast)` arrays over the holdout
window, for one phase (oil/gas/water), for one well. Actual months that are
`NaN` (not reported) are dropped before scoring — not treated as zero.

- **MAPE** — mean absolute percent error. Undefined (`None`) when every
  actual in the window is zero.
- **sMAPE** — symmetric MAPE, bounded [0, 200]. More stable than MAPE when
  actuals are near zero (e.g. late-life stripper wells).
- **MAE** — mean absolute error in native units (bbl/mcf). Not comparable
  across wells of different size, but useful for sanity-checking a single
  well's forecast against its own scale.
- **bias** — signed mean percent error. Positive = the model
  over-forecasts on average; negative = under-forecasts. This is the
  number that matters most for a valuation use case: MAPE says how noisy
  a model is, bias says which direction it's wrong in.
- **spee_score** — `2/3 * |median(log_error)| + 1/3 * stdev(log_error)`,
  where `log_error = ln(forecast / actual)`. This is the exact scoring
  formula SPEE used in the 2024 Software Symposium bake-off (see the
  archived roadshow deck). It's included here specifically so results from
  this repo are directly comparable to bake-off-style scoring, if that
  path is ever pursued. Lower is better; a perfect forecast scores 0.

## Why log-error, separately from bias/MAPE

`log_error` treats a 20%-high forecast and a 20%-low forecast asymmetrically
under plain percent error (over-forecasting by 25% and under-forecasting by
20% both correspond to the same log-error magnitude in opposite directions).
SPEE's own conclusions slide noted vendors were "just as likely to be 10%
too high as 10% too low" when scored this way — that symmetry only shows up
correctly under log error, not plain MAPE. Both are reported so a change
that helps on one but hurts on the other doesn't slip through unnoticed.

## Phase availability

A well's `oil`/`gas`/`water` array is `NaN`-filled, not zero-filled, when
that phase was never reported. `WellSeries.phase_available(phase)` is the
single source of truth for "does this well have any signal on this phase
at all" — the benchmark runner uses it to decide whether to score a
phase for a well (`phase_available=True`, real forecast attempted) or
mark it out of scope (`phase_available=False`, `scores={}`, still counted
in the run's totals). A model is never asked to forecast a phase with zero
historical signal, and a missing phase is never invisible in the output.

## What "unified" means here

`unified.py`'s `unified_forecast` is not claimed to be better than the
routing-based approach in `arps.py` / CrudeCode's `routing.py` — it's the
strawman this repo exists to test. It fits whatever history is available
through one code path (down to a flat zero-curve at zero history, and a
naive last-3-month average below a minimum fit threshold) rather than
branching on a well's classified maturity state. Whether that generalizes
better is an empirical question this repo is built to answer, not an
assumption it starts from.
