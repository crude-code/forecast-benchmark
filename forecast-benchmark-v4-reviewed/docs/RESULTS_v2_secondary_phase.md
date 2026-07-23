# v2 build results: secondary-phase forecasting + cohort priors

What got built, what it does, and — importantly — what the measurements
honestly show, including where the simple version does NOT win. Everything
here is reproducible: `python examples/demo_secondary_phase.py`.

## What was added

Four new modules, all wired into the existing harness, 54 tests passing
(22 original + 32 new), hygiene guard green:

- `cohort.py` — history-driven parameter shrinkage. Blends a well's own
  fitted qi/di/b toward its cohort's medians by a weight that grows with how
  much history the well has (`n / (n + 24)`). The smooth version of the
  production engine's hard `own_b` gate. Carries provenance (`b_source`,
  `weight_own`) so a forecast can explain itself.
- `ratios.py` — secondary-phase forecasting by RATIO, not independent fit.
  Fits GOR/CGR/WOR as a power-law or logistic curve, rides it on the primary
  forecast. Uses petbox-dca's `PLYield` math if installed, self-contained
  numpy otherwise (never *requires* the dependency).
- `cohort_ratio.py` — the piece that ties Gap 2 and Gap 3 together: borrows
  the cohort's normalized ratio-vs-month SHAPE (anchored to a young well's own
  early ratio level) so a well too young to have entered its bubble-point
  transition can still forecast the rise.
- `secondary_benchmark.py` — head-to-head: same wells, same holdout,
  independent-fit vs. ratio-method, scored on the existing metrics.

Two fixtures: `ratio_wells.csv` (clean rising-GOR / falling-CGR) and
`bubblepoint_wells.csv` (realistic four-stage GOR: flat, then a sharp rise at
month 12-20 — the shape that actually breaks self-contained fits).

## The honest findings

### 1. The plain ratio method is NOT a free win

On the bubble-point fixture cut at 18 months (well is mid-transition), the
naive power-law ratio method was *worse* than an independent fit
(bias -36% vs -4%). The power law, fit to a GOR that's partway up its S-curve,
extrapolates the steep slope too far. This is a real limitation, not a
rounding error, and it's exactly the "fitting a clean curve to messy data and
trusting it" trap the practitioner guides warn about. **Reported here rather
than buried** — a benchmark that only surfaced wins wouldn't be worth having.

Takeaway: ratio shape matters. The logistic/sigmoid shape is the right
structure for GOR (it plateaus; the power law doesn't), but it needs enough
history to locate the inflection, or it extrapolates wildly. Which leads to:

### 2. On short histories, NO self-contained method works — the cohort prior does

This is the headline, and it's the empirical justification for building the
cohort layer first. On a young well (BP-03) cut at month 10, *before* its GOR
rise begins:

| method | cumulative gas error |
|---|---|
| independent Arps fit on gas stream | -46% |
| self-ratio fit (flat, no rise seen) | -55% |
| **cohort-prior ratio (borrows transition timing)** | **-20%** |

The rise simply isn't in 10 months of data. The only way to forecast it is to
borrow the timing from analog wells that already went through it. The cohort's
median GOR multiplier (1.0x at mo 0 -> 1.7x at mo 12 -> 3.1x at mo 18) carries
that knowledge; anchoring it to the young well's own early GOR level applies
it. This is the cohort prior from Gong et al. / Lee & Mallick, done explicitly
in ratio space instead of via MCMC.

(The one-well number moves between -7% and -20% depending on whether the
cohort shape is blended with the well's own flat early data — `blend_with_self`.
The blend is more conservative and more honest for a general method; the pure
cohort shape fits this particular well better but would overfit in general.)

### 3. On moderate+ history, methods converge

At 18+ months on the clean fixture, independent and ratio methods are within a
few points of each other. The ratio method's value is concentrated exactly
where SPEE said the industry struggles: **short histories and the
secondary-phase transition** — not everywhere.

## What this means for the build plan

Confirmed, with measurements rather than assertion:

1. **The cohort layer was correctly prioritized.** It's not just foundational
   plumbing — it's the *only* thing that moved the needle on the hard
   short-history case. Both other gaps lean on it.
2. **"Forecast a ratio" is necessary but not sufficient.** The ratio has to be
   the right shape AND, on young wells, anchored to a cohort prior. A plain
   ratio fit alone can underperform.
3. **The measurement harness earned its keep.** It caught the ratio method
   losing at 18 months — which a "ship the clever idea" approach would have
   missed. That catch is the whole point of having built it first.

## Honest limitations (what this v2 is NOT)

- Fixtures are synthetic. The bubble-point shape is realistic in *form* but
  the numbers are generated, so the error magnitudes above are illustrative of
  the *mechanism*, not calibrated to real basins. Real validation needs real
  wells.
- WOR/water is wired in but barely exercised — the synthetic water is a simple
  rising trend, not the flowback-then-formation-water two-regime shape real
  water shows. Needs a better fixture and probably a two-segment model.
- No full Bayesian posteriors (P10/P50/P90) — still the "rebuild, only with a
  product driver" item. The cohort prior here is the cheap approximation.
- Cohort membership is "all other wells in the fixture." Real cohorts need
  basin/reservoir/vintage filtering, which the benchmark's data model doesn't
  carry yet.
