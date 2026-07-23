# What's already open-source: build vs. borrow for the three gaps

Research memo, 2026-07-22. Companion to
`RESEARCH_secondary_phases_and_short_history.md`. Before building anything,
this is what already exists in open source for each gap — searched GitHub,
PyPI, and practitioner sources.

Bottom line up front: **the single most important finding is petbox-dca
v2.1.0**, released 2026-07-09 (two weeks ago) by David Fulford — who sits on
the SPEE bake-off technical committee. It already implements the *model
forms* for all three of our gaps (GOR/CGR yield, water yield, transient
hyperbolic). It deliberately does **not** implement the fitting/automation
layer. So the split is clean: **borrow the curve math, build the harness and
the auto-fitting around it.** That is almost exactly the division of labor
our own recovered research doc predicted.

---

## petbox-dca (MIT license) — borrow the model forms

Repo: `github.com/petbox-dev/dca` · `pip install petbox-dca` · 60 stars,
actively maintained, v2.1.0 on 2026-07-09.

What it already has that maps to our gaps:

| Our gap | petbox-dca feature |
|---|---|
| Gap 2 — GOR on oil wells | `add_secondary(PLYield(...))` → `secondary.rate(t)`, `gor(t)` |
| Gap 2 — CGR on gas wells | same secondary interface → `cgr(t)` |
| Gap 1 — water | `add_water(PLYield(...))` → `water.rate(t)`, `wor(t)`, `wgr(t)` |
| b-factor / short history | `THM` (Transient Hyperbolic Model) — b transitions between flow regimes instead of a single fixed b |

The `PLYield` (power-law yield) model matches the "forecast the ratio, ride it
on the primary" architecture the research memo describes — it is attached to a
primary-phase model and derives the secondary/water phase from it. That is
the exact model-form pattern, already implemented and tested, for two of our
three gaps.

The `THM` model is Fulford & Blasingame's transient hyperbolic relation
(SPE-167242, URTeC-2018-2903036) — it's the physically-grounded answer to the
b-factor problem our `own_b` fix approaches empirically. Worth evaluating
`THM` as a model *in the benchmark* against our bounded-b approach.

### The critical caveat (this is why we still have work to do)

petbox-dca's README states plainly: **"No methods for regression are included
in this library."** It gives you `rate(t)` given parameters; it does **not**
fit parameters to data, classify wells, select which model to use, borrow
priors for short histories, or run any of it automatically. It even ships an
example using `scipy.optimize.least_squares` — the same optimizer we already
use — precisely because fitting is left to the user.

So petbox-dca replaces the part of our forecasting we'd least benefit from
rewriting (the curve equations, which are fiddly and easy to get subtly
wrong — the cumulative-volume `N(t)` functions especially), and leaves us
owning the part that's actually our value-add and the thing the bake-off
grades: **the automated fitting, well classification, ratio-model selection,
short-history prior-borrowing, and the measurement harness.**

### Recommendation

Adopt petbox-dca as a dependency for the *model forms* (both ratio yields and
THM), and build our benchmark's models as thin fitting wrappers around it,
rather than reimplementing Arps/THM/PLYield ourselves long-term. Two benefits
beyond saving work: (1) the curve math comes from an actively maintained,
petroleum-specific library authored by a SPEE technical volunteer, and (2)
it is less of our own code to test. The one cost is a dependency; given it's
MIT and small, that is likely cheap. We should keep our own `arps.py` as the
trivial baseline and add petbox-backed models alongside it, then let the
benchmark decide whether they improve results.

Current repo status: v3 adds `petbox-dca` as an optional dependency and wires
petbox-backed THM/PLYield wrappers as separate benchmark models. The
self-contained `ratios.py` implementation remains in place as a baseline and
comparison target. This is still a benchmark integration, not a production
forecasting decision.

---

## Primary-phase DCA: lots of repos, none worth adopting wholesale

The `decline-curve-analysis` GitHub topic has many repos. Almost all are
hobby/thesis-grade — single scripts, plain Arps least-squares, little or no
testing, no ratio forecasting, no short-history handling:

- `PrestonBlackburn/Decline-Curve-Analysis` — Arps vs. Prophet/RNN comparison, script-level.
- `dwb26/decline_curve_analysis` — Arps least-squares against an xlsx, script-level.
- `shirangi/DCA`, `Yous3ry/Python_Automated_DCA` — automated Arps with flow-regime/changepoint detection (the latter uses R via rpy2 for changepoints); closer to real, but oil-focused, no ratios, not packaged.
- `rashidwadani/Decline_Curve_Analysis_Tool`, `Jeffalltogether/well_decline_curve_analysis` — Arps fitting utilities, no secondary phase.

Two are more than scripts:

- **`dcapy`** (`scuervo91.github.io/dcapy`) — packaged, does *probabilistic*
  forecasts + well schedules + cashflow. Worth a closer look for the
  scheduling/economics side later, but its DCA core is Arps; it doesn't solve
  the ratio or short-history problems either.
- **`prodpy`** (`jshiriyev/data-driven-forecasting`) — packaged, vectorized
  Arps, plotting templates, multi-zone allocation. Nice engineering, but
  again primary-phase Arps at its core.

None of these change the build plan. petbox-dca is strictly better for the
model forms, and none of them do the ratio/short-history work.

---

## Bayesian / short-history (Gap 3): no turnkey package exists

This is the one gap with **no ready-made open-source solution**. The
published methods (Gong et al. SPE-147588, the ABC/Gibbs comparison study,
Lee & Mallick's hierarchical model) are papers, not libraries. What exists
open-source is the *general-purpose* Bayesian tooling:

- **PyMC** (`pymc-devs/pymc`) — the mature MCMC/variational-inference library.
  You'd write the DCA model (Arps or THM parameters as random variables with
  cohort priors) on top of it. This is real work, and it's the "rebuild, not
  a patch" item — but PyMC means we don't write a sampler, just the model.

So Gap 3 stays as the research memo framed it: the *cheap* version (cohort
shrinkage — fit the cohort, shrink each well toward it by an amount that
scales with how much history it has) is hand-rolled and small, needs no
Bayesian library at all, and should be done first. The *full* version
(PyMC-backed posteriors with P10/P50/P90) is a real project, only worth it
with a product driver, and even then PyMC does the heavy lifting.

---

## Revised build plan (updated from the research memo)

The ordering is unchanged, but the effort estimates drop for Gaps 1 and 2
because petbox-dca hands us the curve math:

1. **Cohort/analog shrinkage layer** (Gap 3, cheap). Still first — foundational,
   unlocks a fair unified-vs-routing test, and is the prior the other two
   gaps lean on for short-history wells. No open-source shortcut; hand-rolled
   and small. **No change.**
2. **GOR/CGR ratio model** (Gap 2). Now mostly *fitting* work: wrap
   petbox-dca's `PLYield` secondary-phase model, write the auto-fit + oil-vs-
   gas-condensate classification, ride it on our primary forecast. The curve
   itself is no longer ours to write or test. **Effort drops from "medium" to
   "small-medium."**
3. **WOR ratio model** (Gap 1). Same — wrap petbox-dca's `add_water`/`PLYield`
   water model, fit WOR+1 vs. cum-oil, fall back to flat on dry-gas wells.
   **Effort drops similarly.**
4. **Full Bayesian posteriors** (Gap 3, expensive). PyMC-backed. Only with a
   product driver. **No change.**

## What this means for the "is it a lot of work" question

Less than it looked in the first research memo. The parts that were most error-prone to build from scratch — the ratio-yield
curve math and the transient-hyperbolic equations — are already written,
MIT-licensed, and maintained in a petroleum-specific package. What's left for
us is the fitting/automation/measurement layer, which is (a) genuinely our
value-add, (b) the thing the bake-off actually grades, and (c) exactly what
our benchmark harness was built to support.
