# Fixing the three gaps: water, secondary-phase ratios, and short history

Research memo, 2026-07-22. Maps the three big `OPEN_QUESTIONS.md` gaps to
concrete, literature-backed methods and what each would take to build here.

The framing question this answers: SPEE's 2024 bake-off found the whole
industry was weak on exactly the things we don't do yet. This is what the
published state of the art actually does about them, and which pieces are a
weekend vs. a rebuild.

Sources are cited inline by name/venue so they can be pulled and verified;
this memo is a synthesis, not a substitute for reading them.

---

## Gap 1 — Water (and the key insight: forecast the RATIO, not the rate)

### Why naive decline fails on water

Water in unconventional wells frequently does *not* decline like oil or gas.
Early life is dominated by frac-fluid flowback (very high, falling water),
then formation water often produces a *rising* water-oil ratio (WOR) over the
well's life. A single Arps hyperbolic fit on the water stream alone will fit
neither regime well — which is exactly why the SPEE deck (slide 26, the Oscar
DJ well) singled water out as "noisy, hard to forecast."

### The established method: WOR / WOR+1 vs. cumulative oil

This is the documented professional approach (IHS Harmony, Fekete/
petroleumsolutions, and the WOR-forecasting literature all describe the same
core method). The trick is that you don't forecast water directly — you
forecast the *ratio*, and let it ride on top of the oil forecast we already
produce:

1. Plot **WOR** (or **WOR+1**, which handles the low-water regime without
   blowing up on a log axis) against **cumulative oil**, not time.
2. In water-drive / boundary-dominated flow this tends toward a **straight
   line on a semi-log plot** — fit that line.
3. Extrapolate the line against the *oil forecast's* cumulative-oil
   projection to get future WOR, then back out water rate:
   `water_rate = WOR(cum_oil) × oil_rate`.

The reason this is better than an independent water curve: it couples water
to oil. The two forecasts constrain each other ("slope dependency" — when
total fluid is roughly constant, the oil-rate slope is the inverse of the
WOR+1 slope), so a water forecast that would imply a physically absurd total
fluid rate gets caught.

### What it would take here

Small-to-medium. We already produce the oil forecast and its cumulative. A
water model in this repo would be a new `ForecastFn`-shaped function that:

- takes the well's oil and water history,
- computes historical WOR+1 vs. cumulative oil,
- fits a log-linear trend (robust regression, not just least-squares —
  water data is noisy),
- and emits water rate by riding that trend on the oil forecast's cum-oil.

It reuses the metrics/split/benchmark harness unchanged. The one honest
caveat: WOR-vs-cum-oil is a *water-drive / high-water* method; on very
dry-gas wells it's the wrong tool (there, water is often a flat low rate or a
WGR play — see Gap 2). So the water model needs the same maturity/regime
awareness the oil path has, and should fall back to "flat trailing average"
rather than force a trend it can't see.

---

## Gap 2 — Secondary-phase ratios (GOR on oil wells, CGR on gas wells)

This is the gap SPEE flagged hardest, and the research explains *why* every
vendor got it wrong in the same direction.

### The physics that empirical DCA cannot see

From the GOR literature (URTeC 2020 "Observed GOR Trends in Liquids Rich
Shale," the Uinta Basin GOR study, the Bakken GOR characterization paper, and
a GOR-forecasting patent, US 11,767,750):

A single-phase empirical decline curve **structurally cannot** represent
solution-gas liberation. The real mechanism on an oil well:

1. Early life: bottomhole pressure is above bubble point → GOR is **flat** at
   the initial solution GOR.
2. BHP drops below bubble point → gas comes out of solution → after critical
   gas saturation is exceeded, free gas flows → **GOR rises**, often steeply,
   sometimes to several times the initial value.
3. Eventually plateaus.

So the true GOR profile is a rising S-curve. If you forecast gas on an oil
well by fitting an independent Arps curve to the gas history, you get a
*declining* gas forecast — but the physics says gas should *rise* relative to
oil as the well depletes. That's the systematic **under-forecast of GOR on
oil wells** the SPEE deck found. The mirror image (CGR over-forecast on gas
condensate wells) is the same mechanism running the other way: condensate
drops out as pressure declines, so CGR *falls*, and a naive independent fit
doesn't capture the drop.

### The established method: model the ratio with an S-curve, tie phases together

Two named approaches in the literature:

1. **Asymmetrical Sigmoid Model (ASM)** for GOR (SPE-LRBC 2019, "A New Way to
   Forecast Gas-Oil Ratios and Solution Gas Production"). Fits the rising-
   then-plateau GOR shape directly, then derives gas from
   `gas_rate = GOR(t) × oil_rate`. Validated against compositional simulation
   and field data with as little as 6 months of history.
2. **Four-stage bubble-point GOR trend** (the URTeC/Uinta/Bakken papers):
   constant → rising → plateau, keyed to the bubble-point transition.

Both share the same architectural move as the water fix: **forecast the
ratio, derive the secondary phase from the primary.** Never fit the secondary
phase independently.

This directly matches what our own recovered deep-research doc concluded
("Algorithms consistently under-forecast GOR on oil wells / over-forecast CGR
on gas wells — a more Bayesian approach could be helpful") and what the SPEE
deck's Kilo Eagle Ford example showed (rising CGRs even over 1,000 for wells
with entrance CGR <300).

### What it would take here

Medium, and it's the highest-value item because it's the thing SPEE grades
hardest. A GOR model would:

- classify the well (oil vs. gas-condensate) off early GOR — we already have
  the 3-month GOR bin logic conceptually from the SPEE data,
- fit a monotonic rising sigmoid to historical GOR (for oil wells) or a
  falling CGR trend (for gas wells),
- derive the secondary phase from the primary forecast × ratio.

The hard part isn't the curve — it's that a good GOR fit needs *enough
history to see the bubble-point transition start*. On a 6-month well the rise
often hasn't begun, so the model has to lean on an analog/basin prior for
where the transition happens (see Gap 3). This is why the literature keeps
landing on Bayesian priors for this.

---

## Gap 3 — Short history (the analog/prior problem underneath both fixes)

Both fixes above degrade on short-history wells for the same reason: not
enough data to see the regime. This is the gap our own shipped `own_b` fix
already addresses for the *primary* phase (long-history wells fit their own
b; short-history wells borrow a gated cohort b). The literature says the
principled general version is Bayesian.

### The established method: Bayesian / probabilistic DCA with priors

The most-cited line of work (Gong et al. SPE-147588-PA, "Bayesian
Probabilistic Decline-Curve Analysis"; the follow-on ABC/Gibbs comparison
study across 1,800 wells in 6 basins; Lee & Mallick's Bayesian *hierarchical*
model in the Eagle Ford):

- Treat qi, Di, b (and ratio-model parameters) as **random variables with
  prior distributions**, not point values to least-squares fit.
- Use MCMC (or approximate Bayesian computation) to get **posterior
  distributions** → P10/P50/P90 forecasts, not a single line.
- On short histories, the **prior** (from analog wells / the basin cohort)
  dominates; as history accrues, the **data** takes over. This is exactly the
  "borrow the cohort b when young, fit your own when mature" behavior we do
  crudely — Bayesian inference does it smoothly and gives uncertainty bands
  for free.

Hindcast evidence it works: Gong reports reliable P50 with as little as 6
months of history; the 1,800-well study found P50/P90 well-calibrated at ~2
years. The hierarchical version (Lee & Mallick) additionally borrows strength
across wells and can predict a *new* well's curve from completion + location
data.

### What it would take here

This is the "rebuild, not a patch" item our deep-research doc already flagged.
Two honest options:

1. **Cheap approximation (do this first):** we don't need full MCMC to get
   most of the benefit. A cohort/analog layer — fit the cohort, shrink each
   well's parameters toward the cohort mean by an amount that depends on how
   much history the well has — captures the "prior dominates when young"
   behavior without a sampler. This is also the piece the benchmark is
   currently missing that makes "unified vs. routing" not yet a fair fight
   (per OPEN_QUESTIONS.md), so it's worth building regardless.
2. **Full Bayesian (only with a product driver):** MCMC per well, real
   posteriors, uncertainty bands. Big lift, real payoff if we ever want to
   report P10/P50/P90 instead of a single number — which is where the whole
   profession is heading (Fulford's "artful management of assumptions and
   Bayesian priors," per Warren's write-up).

---

## Suggested build order (return-on-effort)

1. **Cohort/analog shrinkage layer** (Gap 3, cheap version). Unlocks a fair
   unified-vs-routing comparison *and* is the prior that Gaps 1 and 2 need for
   short-history wells. Foundational — do it first.
2. **GOR/CGR ratio model** (Gap 2). Highest external value — it's what SPEE
   grades hardest and what our own research already says we get wrong. Ride
   the secondary phase on the primary forecast × a fitted ratio S-curve.
3. **WOR ratio model** (Gap 1). Same architectural pattern as Gap 2 (ratio
   riding on the primary), so it's cheaper once Gap 2's plumbing exists.
4. **Full Bayesian posteriors** (Gap 3, expensive version). Only with a
   product driver — don't build on spec.

Everything in 1–3 reuses the existing split/metrics/benchmark harness
unchanged, and every one of them is measurable against held-out data the day
it's written — which is the whole point of having built the harness first.

## The honest bottom line for the bake-off question

The gaps are real work, but none of 1–3 is exotic — they're all the same move
(forecast a ratio, tie it to the primary, borrow from a cohort when history is
thin), and all are documented, hindcast-validated methods, not research bets.
The thing that makes this tractable is that we already built the measurement
harness, so each piece can prove it helps before it ships. That's the exact
discipline the SPEE committee spent 5 weeks and 7 engineers wishing the
vendors had.
