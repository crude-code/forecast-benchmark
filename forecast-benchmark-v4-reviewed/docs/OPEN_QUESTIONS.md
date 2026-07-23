# Open questions

Things this repo deliberately has not answered yet. Not a backlog — a list of
places where the honest answer right now is "we don't know yet."

> **v3 status (2026-07-22):** The repo now includes simple cohort shrinkage,
> ratio-based secondary-phase forecasting, a cohort-prior ratio model, and
> optional `petbox-dca` wrappers for THM/PLYield model forms. These are still
> benchmarked on synthetic fixtures only. The synthetic results prove mechanisms
> and catch failure modes; they are not calibrated evidence that any method wins
> on real basins.

## Real-well validation

This is the biggest gap.

The v2 fixtures are synthetic. They are shaped to mimic known failure modes
such as a delayed GOR transition, but the numbers are generated. Before any
model claim matters, this repo needs real held-out validation across wells with
known production history.

Required before claiming improvement:

- enough wells across oil, gas, and mixed-fluid cases
- basin/reservoir/vintage grouping
- thin-history and moderate-history slices
- per-phase metrics, not just aggregate BOE-style scoring
- error distributions, not just one average

## Water forecasting

Water is wired into the ratio framework, but it is not solved.

The current synthetic water examples are too simple. Real water often has
multiple regimes: flowback, cleanup, early formation water, later WOR rise,
and operating/proration artifacts. A single monotonic power-law ratio may be a
bad shape. The next useful step is a better synthetic water fixture that can
make a naive model lose for the right reason.

Do not fake water with an oil-shaped decline curve. If the data does not carry
water signal, report it as missing or out of scope.

## Secondary-phase / ratio forecasting

v2 proves the right warning: **forecasting a ratio is necessary but not
sufficient.**

A plain self-fit ratio model can lose. In the bubble-point fixture, the
power-law ratio fit extrapolates a mid-transition slope too far. The cohort
prior helps only because it borrows transition timing from comparable mature
wells.

Open items:

- test logistic vs. power-law ratio shapes on more fixtures
- add petbox-dca-backed `PLYield` wrappers as separate models
- compare against independent fits on real oil-well GOR and gas-well CGR cases
- define when ratio fitting should refuse and fall back

## Cohort definition

The current cohort prior uses "all other wells in the fixture." That is fine
for a small mechanism demo, not for real forecasting.

Real cohorts need filtering by at least:

- basin/play
- reservoir/formation
- operator or completion style where available
- vintage
- lateral length / completion intensity where available
- producing phase / fluid regime

Bad cohort selection can make the model confidently wrong.

## Does "unified" actually win anything?

Still unknown.

The original small fixture had every well with enough history that
`unified_forecast` and `arps_hyperbolic_bounded_b` followed the same effective
path and scored the same. v2 added cohort and secondary-phase tools, but it
still has not answered the core unified-vs-routing question on real messy wells.

To answer that, we need a benchmark slice with:

- no-history wells
- thin-history wells
- climbing wells
- peaked wells
- moderate/long-history wells

Until then, unified is a testable strawman, not a conclusion.

## petbox-dca integration

The research memo found `petbox-dca` 2.1.0 and flagged it as the right thing to
borrow for model forms: THM and PLYield-style secondary/water yields.

Current repo status: v3 adds optional petbox-backed wrappers in
`src/forecast_benchmark/petbox_models.py`. They are **separate benchmark models**,
not replacements for `arps.py` or `ratios.py`. The core install still works
without petbox; petbox tests skip cleanly unless `petbox-dca` is installed.

Still open:

- compare petbox-native PLYield secondary/water forecasting head-to-head against
  the self-contained `ratios.py` method
- decide a routing rule for THM vs bounded-b on short/moderate/long histories
- validate THM/PLYield on real held-out wells before making any model claim

Do not silently swap model forms without benchmarking the change.

## Per-foot lateral normalization

Not implemented.

Prior CrudeCode forecasting research flags that analog qi should be normalized
by lateral length, with the caveat that scaling is often sublinear. This repo
currently lacks the well header/completion metadata needed to do that honestly.
Add the data model before adding the formula.

## Full Bayesian posteriors

The forecasting side is not implemented; the *scoring* side now is.

The cheap cohort prior is an explicit approximation to hierarchical borrowing.
Full PyMC-backed posteriors with P10/P50/P90 remain a real project, not a patch.
Only do it with a product or competition driver.

However, as of v4 the benchmark CAN now score a probabilistic model fairly:
`probabilistic.py` adds calibration/coverage and pinball-loss metrics, so a
Bayesian candidate (e.g. an outside implementation offered for evaluation) can
be tested on its uncertainty bands, not just its P50. See
`RESULTS_v4_probabilistic_scoring.md` for how a candidate gets tested. The open
part is a probabilistic *forecaster* to point at these metrics — the metrics
themselves are ready.

## SPEE bake-off participation

Not decided.

This repo is the thing that would tell us whether entering makes sense. It is
not a commitment to enter.
