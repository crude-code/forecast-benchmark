# v3: petbox-dca-backed models, added as separate benchmarked models

Reproducible: `python examples/demo_petbox_vs_baseline.py` (needs
`pip install petbox-dca`). All 61 tests pass; petbox tests skip cleanly if the
package isn't installed.

## What was added

- `src/forecast_benchmark/petbox_models.py` — two petbox-dca-backed models:
  - `thm_model(q)` — fits petbox's **Transient Hyperbolic Model** (Fulford &
    Blasingame, SPE-167242) to a monthly primary stream and returns a
    `ForecastFn` in the same shape as the `arps.py` baselines.
  - `thm_secondary_from_primary(...)` — a fully petbox-native secondary-phase
    forecast: THM primary + fitted `PLYield` ratio, gas/water read straight
    from petbox.
- `tests/test_petbox_models.py` — 7 tests, including the unit-bridge guard.
- `examples/demo_petbox_vs_baseline.py` — the head-to-head runner.
- `petbox-dca` added as an **optional** dependency (`.[petbox]` extra), never
  a hard requirement. Core benchmark and its tests run without it.

These are **separate models**, per the repo's own rule ("do not silently swap
model forms without benchmarking the change"). Nothing in `arps.py`,
`ratios.py`, or `cohort*.py` changed. THM competes; it did not replace.

## The unit bridge (the thing most likely to silently break this)

The benchmark speaks **monthly**: arrays indexed by month, values = volume
produced that month. petbox speaks **days** and **per-day rate**, with a
`monthly_vol(t_days)` helper. The wrapper maps month index `m` to the day at
the *end* of that month, `(m+1) * 30.4375`, and scores using `monthly_vol` —
never `rate()`. Getting this wrong would inflate/deflate everything ~30x. A
dedicated test (`test_thm_forecast_is_monthly_volume_not_daily_rate`) asserts
the cumulative THM forecast stays within 0.5x-2x of actuals so a units
regression can't slip through.

## The result — honest and nuanced

Median bias (closer to 0 is better), primary oil phase:

**bubblepoint_wells.csv** (realistic delayed GOR transition, harder decline):

| cutoff | THM | bounded-b Arps | exponential |
|---|---|---|---|
| 12 mo | **-14.6%** | **-4.0%** | -30.0% |
| 18 mo | -0.4% | +2.3% | -31.5% |
| 24 mo | -0.6% | +0.9% | -30.4% |

**ratio_wells.csv** (well-behaved hyperbolic decline):

| cutoff | THM | bounded-b Arps | exponential |
|---|---|---|---|
| 12 mo | **-1.8%** | +5.1% | -33.3% |
| 18 mo | **-1.1%** | +4.1% | -37.6% |
| 24 mo | +0.1% | +1.5% | -35.5% |

Three things this actually shows:

1. **Exponential is the dumb baseline, and it behaves like one** (~-30% biased
   everywhere). Confirms the harness discriminates rather than rating
   everything similar.
2. **THM wins once it has enough history to constrain its extra parameters.**
   On the well-behaved fixture it's near-zero-bias at every cutoff and clearly
   beats bounded-b Arps's persistent +4-5% over-forecast. The transient
   `bi -> bf` shape genuinely fits shale decline better than a single b.
3. **THM can LOSE on very short histories.** At the 12-month cutoff on the
   harder fixture it's -14.6% vs bounded-b's -4.0%: with only 12 months, THM's
   extra parameters (`bi`, `bf`, `telf`) are under-constrained and it overfits.
   The simpler, stiffer bounded-b model is more robust there.

That third point is the important one and the reason this went in as a
separate benchmarked model instead of a swap: **richer is not uniformly
better.** The right production behavior is probably to route by history
length — bounded-b (or cohort-shrunk) when young, THM when mature — which is
exactly the kind of thing the benchmark now lets us decide with numbers
instead of taste. That's a natural next experiment: a model that picks THM
vs bounded-b per well by available history, scored against both.

## Limitations (unchanged from v2, restated)

- Synthetic fixtures — these numbers prove the *mechanism* and the unit bridge,
  not calibrated basin accuracy. Real-well validation is still the biggest open
  item (`OPEN_QUESTIONS.md`).
- THM here fixes `bi=2.0` and uses no terminal segment (bterm/tterm=0) to keep
  the fit to 4 parameters. A terminal segment matters for long economic-limit
  forecasts, less so for these short holdouts; worth revisiting when EUR (not
  just near-term rate) is scored.
- The petbox-native secondary model is wired and tested but not yet in a
  head-to-head against the self-contained `ratios.py` version — that's the
  next comparison to run now that both exist.
