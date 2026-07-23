# forecast-benchmark

A small, honest benchmark harness for oil & gas decline-curve forecasting
methods. This is a research repo, not a product. It exists to answer one
question at a time:

> Does a forecasting idea actually improve accuracy against held-out production,
> or does it just feel more intuitive?

No CrudeCode dependency. No private data. No licensed exports. The included
fixtures are synthetic.

## Public-safe scope

This repo is safe to keep public because it contains only standalone benchmark
code, docs, tests, and synthetic fixtures. Do not add proprietary well data,
licensed exports, private CrudeCode internals, customer files, production ETL,
credentials, or infrastructure config.

## What v2 adds

v1 proved the basic harness: point-in-time split, Arps baselines, a unified
forecast strawman, and metrics.

v2 adds the first serious pieces for the SPEE-style weak spots:

- `cohort.py` — history-weighted shrinkage toward cohort parameter medians.
- `ratios.py` — secondary-phase forecasting by ratio, not independent fit.
- `cohort_ratio.py` — borrows cohort ratio-shape timing for short-history wells.
- `secondary_benchmark.py` — paired independent-vs-ratio secondary-phase scoring.
- `examples/bubblepoint_wells.csv` — synthetic GOR transition fixture.
- `examples/ratio_wells.csv` — synthetic rising/falling ratio fixture.
- `examples/demo_secondary_phase.py` — reproducible demo of the v2 result.

The key result is intentionally nuanced: a plain ratio model is **not** always
better. On the synthetic bubble-point fixture, self-contained ratio fitting can
underperform. The cohort-prior ratio is what helps on the hard short-history
case because the young well cannot observe its own transition yet.

## What v3 adds

v3 adds `petbox-dca` as an **optional** model-form dependency, not a hard
requirement. The new wrappers live in `petbox_models.py` and compete as
separate benchmark models instead of silently replacing the self-contained
baselines.

Added pieces:

- `petbox_models.py` — optional petbox-backed THM primary model and PLYield
  secondary/water wrapper.
- `examples/demo_petbox_vs_baseline.py` — head-to-head THM vs bounded-b Arps
  and exponential on synthetic fixtures.
- `tests/test_petbox_models.py` — optional tests that skip cleanly unless
  `petbox-dca` is installed.
- `docs/RESULTS_v3_petbox.md` — current synthetic-fixture result and caveats.

Core CI still runs without petbox. A separate optional-petbox check installs
`.[dev,petbox]` and runs the petbox tests when dependency resolution is
available.


## What v4 adds

v4 does **not** add a Bayesian forecaster yet. It adds the scoring layer needed
to test one fairly once it arrives. That matters because a probabilistic model's
main advantage is not just its P50 curve; it is whether its P10/P50/P90 bands
are honest.

Added pieces:

- `probabilistic.py` — empirical quantile coverage, calibration error,
  interval coverage, pinball loss, and a bundled probabilistic scorecard.
- `tests/test_probabilistic.py` — calibration/pinball tests, including the
  anti-gaming check that uselessly-wide bands do not win.
- `examples/demo_probabilistic_scoring.py` — shows two models with the same P50
  getting different probabilistic scores because one is overconfident.
- `docs/RESULTS_v4_probabilistic_scoring.md` — how a Bayesian or other
  probabilistic candidate should be evaluated.

The practical rule is unchanged: Bayesian is a candidate, not the answer. The
benchmark should score it on both point error and uncertainty quality.

## Ground rules

- **No benchmark, no model claim.** A forecasting idea does not get called
  better unless it wins on held-out data.
- **No holdout, no forecast brag.** Models are scored only on months they did
  not see.
- **No water data, no fake water forecast.** Missing water stays missing and is
  reported with phase availability. Do not zero-fill missing phases.
- **No synthetic-only victory laps.** Synthetic fixtures prove mechanisms and
  catch obvious regressions. Real validation still needs real wells.

## Quickstart

```bash
pip install -e ".[dev]"
pytest -q
```

Expected for the core install today:

```text
67 passed, 1 skipped
```

With optional petbox models installed:

```bash
pip install -e ".[dev,petbox]"
pytest -q
```

Expected with petbox available: `74 passed`.

Run the v2 secondary-phase demo:

```bash
python examples/demo_secondary_phase.py
```

Run the optional v3 petbox demo:

```bash
pip install -e ".[petbox]"
python examples/demo_petbox_vs_baseline.py
```

Or run the base benchmark programmatically:

```python
from forecast_benchmark.data import load_csv
from forecast_benchmark.arps import naive_last3, arps_hyperbolic_bounded_b
from forecast_benchmark.unified import unified_forecast
from forecast_benchmark.benchmark import run_benchmark
from forecast_benchmark.report import format_comparison

wells = load_csv("examples/tiny_wells.csv")

results = [
    run_benchmark(wells, naive_last3, model_name="naive_last3", cutoff_month=18, horizon=6),
    run_benchmark(wells, arps_hyperbolic_bounded_b, model_name="arps_bounded_b", cutoff_month=18, horizon=6),
    run_benchmark(wells, unified_forecast, model_name="unified", cutoff_month=18, horizon=6),
]
print(format_comparison(results))
```

## Layout

```text
src/forecast_benchmark/
    data.py                  WellSeries, CSV loader, honest NaN-vs-zero handling
    split.py                 point-in-time train/holdout split
    metrics.py               MAPE, sMAPE, MAE, bias, SPEE-style log-error score
    arps.py                  naive, exponential, bounded-b hyperbolic baselines
    unified.py               one-function/no-maturity-branching strawman
    benchmark.py             base data -> split -> model -> metrics runner
    report.py                plain-text comparison tables
    cohort.py                simple history-weighted cohort shrinkage
    ratios.py                self-contained ratio-shape fitting for secondary phases
    cohort_ratio.py          cohort ratio-shape prior for short-history wells
    secondary_benchmark.py   paired independent-vs-ratio benchmark
    petbox_models.py         optional petbox-dca THM/PLYield wrappers

examples/
    tiny_wells.csv           small synthetic base fixture
    ratio_wells.csv          synthetic ratio fixture
    bubblepoint_wells.csv    synthetic GOR-transition fixture
    demo_secondary_phase.py  reproducible v2 demo
    demo_petbox_vs_baseline.py optional v3 THM-vs-baseline demo

docs/
    METHODOLOGY.md
    OPEN_QUESTIONS.md
    PUBLIC_SCOPE.md
    RESEARCH_secondary_phases_and_short_history.md
    RESEARCH_open_source_landscape.md
    RESULTS_v2_secondary_phase.md
    RESULTS_v3_petbox.md
    RESULTS_v4_probabilistic_scoring.md
```

## Current status

The repo is a useful benchmark harness, not a production forecast engine.

What is working:

- Base holdout benchmark.
- SPEE-style log-error metric.
- Oil/gas/water phase-aware scoring.
- Secondary-phase ratio benchmark.
- Cheap cohort-prior ratio shape on synthetic short-history GOR fixture.
- petbox-dca-backed THM/PLYield models, added as separate benchmarked models
  (optional dependency; see `docs/RESULTS_v3_petbox.md`).
- Probabilistic scoring (calibration/coverage + pinball loss) so a model that
  emits P10/P50/P90 can be tested on its uncertainty bands, not just its P50
  (see `docs/RESULTS_v4_probabilistic_scoring.md`).

What is still open:

- Real-well validation.
- Real cohort definitions by basin/reservoir/vintage/completion context.
- Better water fixture and water-specific model shape.
- A probabilistic *forecaster* (e.g. Bayesian) to point at the new scoring
  layer — the metrics are ready, a candidate model is not yet wired in.

See `docs/OPEN_QUESTIONS.md` for the detailed list.
