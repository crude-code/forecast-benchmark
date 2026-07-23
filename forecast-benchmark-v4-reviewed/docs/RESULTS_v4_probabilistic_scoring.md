# Probabilistic scoring + how a Bayesian model gets tested fairly

Reproducible: `python examples/demo_probabilistic_scoring.py`. Core install: 67 passed, 1 skipped when `petbox-dca` is absent. Optional petbox install: 74 passed.

## Why this exists

A team concern, correctly raised: single-well forecasting may be fighting a
losing battle because the *inputs* are contaminated — allocation error,
undocumented downtime, artificial-lift changes, reporting noise, short
histories, all mixed into the actuals before any model sees them. No fitter,
Bayesian or otherwise, can recover information that isn't in the data.

That concern is real, and it has a real answer: on a contaminated well, the
useful thing a model can produce is not a sharper point forecast — it's an
*honest* one, i.e. a wide uncertainty band that says "low confidence here,
don't trust this well." A Bayesian approach's main advantage is exactly this:
it outputs a distribution (P10/P50/P90), not a single line.

But — and this is the gap this module closes — **every metric in `metrics.py`
scores only a point forecast (the P50).** If a probabilistic model were dropped
into the benchmark as-is, it would be graded on its P50 alone and its entire
reason for existing (the uncertainty bands) would be invisible. We'd run an
unfair test that structurally can't see the method's advantage, then possibly
conclude "it didn't beat bounded-b." This module makes the test fair.

## What it measures (`probabilistic.py`)

Two questions, because either one alone is gameable:

1. **Calibration / coverage** — does a stated P90 actually contain the truth
   ~90% of the time? `assess_calibration` reports empirical coverage at each
   quantile and for central intervals (P10-P90 -> nominal 80%), plus a summary
   `calibration_error` (mean gap between nominal and empirical). Coverage alone
   is gameable: predict [-inf, +inf] and you have perfect coverage and zero
   usefulness. So we also need:
2. **Sharpness given calibration** — `pinball_loss` (the proper scoring rule
   for quantiles) and its average `mean_pinball`. It rewards being both
   well-centered and appropriately narrow, and provably *can't* be gamed by
   just widening bands (there's a test asserting exactly that). Lower is better.

A model is good only if it's calibrated AND sharp. Report both, always.

## The demonstration

`demo_probabilistic_scoring.py` builds a messy well (25% noise) and two models
with the **identical P50** but different bands:

| model | MAPE (P50) | bias (P50) | calib error | P10-P90 covers | pinball |
|---|---|---|---|---|---|
| A — overconfident (narrow) | 25.9% | +12.2% | 0.22 | **25%** (want 80%) | 69.8 |
| B — honest (calibrated) | 25.9% | +12.2% | 0.06 | **79%** (want 80%) | 57.5 |

A point-only benchmark rates A and B **identical** (same P50 -> same MAPE/bias).
The probabilistic layer correctly flags A as confidently wrong (its "80%" band
catches only 25% of outcomes) and B as honest, and ranks B better on pinball.
That difference is precisely the thing worth measuring on contaminated wells —
and it was invisible before this module.

## How a Bayesian candidate (e.g. Doug's) gets tested

When a probabilistic model arrives, it slots into the existing discipline —
same wells, same train/holdout split, no model blessed upfront — with one
addition: it's scored on BOTH layers.

1. **Point layer** (existing): run its P50 through `metrics.py` (bias, MAPE,
   SPEE log-error) head-to-head against naive / bounded-b Arps / cohort / THM.
   This answers "is its central forecast at least competitive?"
2. **Probabilistic layer** (this module): run its full P10/P50/P90 through
   `score_probabilistic`. This answers "are its uncertainty bands honest, and
   are they sharp?" — the question the point layer can't touch.

A Bayesian model earns its way into the production stack only if it wins on
something real: lower point error, OR calibrated-and-sharper bands, OR better
short-history behavior (test at low cutoffs), OR better secondary-phase
handling. If it only produces a smoother P50 at higher complexity, the
benchmark says so and we leave it out. "Bayesian is a candidate; the benchmark
decides" — this module is what lets the benchmark decide on the part that
actually matters for a probabilistic method.

### What to confirm about any candidate's output before wiring it in

- input format (monthly rate arrays? days? what units?)
- does it forecast oil / gas / water, or primary only?
- does it emit quantiles (which levels?) or just a mean/P50?
- does it need cohort/analog priors, or fit each well standalone?
- PyMC/MCMC (slow, heavy dep) or lighter scipy/analytic?
- can it run on synthetic/public data (so results are shareable in this repo)?

The last point matters for keeping this repo public-safe: candidate models are
tested here only on synthetic/public fixtures, never on private or licensed
well data.

## Honest limitations

- Coverage/calibration is only meaningful with enough wells — on a handful of
  holdout wells, empirical coverage is noisy. The synthetic tests use
  thousands of samples to verify the math; real evaluation needs a real,
  reasonably sized well set.
- This scores quantile forecasts. A model that emits a full posterior could be
  scored with richer rules (CRPS); pinball at a few quantiles is the practical,
  transparent version and is what most probabilistic DCA outputs support.
- Still synthetic. Like the rest of the repo, this proves the mechanism and the
  math; calibrated evidence on real basins remains the biggest open item.
