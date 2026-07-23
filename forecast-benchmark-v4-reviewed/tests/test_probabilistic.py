"""Tests for probabilistic scoring. These lean on cases where the correct
answer is known analytically or by construction, because a scoring metric
that's subtly wrong is worse than none — it hands out confident wrong
verdicts."""
import numpy as np

from forecast_benchmark.probabilistic import (
    assess_calibration,
    empirical_coverage,
    mean_pinball,
    pinball_loss,
    score_probabilistic,
)


# ---- pinball loss ----

def test_pinball_zero_when_prediction_equals_actual():
    a = np.array([100.0, 200.0, 300.0])
    for q in (0.1, 0.5, 0.9):
        assert abs(pinball_loss(a, a, q)) < 1e-12


def test_pinball_median_is_half_abs_error():
    # at q=0.5 pinball loss = 0.5 * mean(|error|)
    a = np.array([100.0, 100.0])
    pred = np.array([80.0, 130.0])  # errors 20 and -30
    pl = pinball_loss(a, pred, 0.5)
    assert abs(pl - 0.5 * np.mean([20.0, 30.0])) < 1e-9


def test_pinball_penalizes_underprediction_more_at_high_q():
    # at q=0.9, being below actual (under-predict) should cost more than being
    # above by the same amount
    a = np.array([100.0])
    under = pinball_loss(a, np.array([90.0]), 0.9)   # pred below actual
    over = pinball_loss(a, np.array([110.0]), 0.9)   # pred above actual
    assert under > over


def test_pinball_rejects_bad_quantile():
    a = np.array([1.0, 2.0])
    for bad in (0.0, 1.0, -0.1, 1.5):
        try:
            pinball_loss(a, a, bad)
            assert False, "should have raised"
        except ValueError:
            pass


# ---- coverage ----

def test_coverage_perfect_when_all_below():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    high = np.array([10.0, 10.0, 10.0, 10.0])  # all actuals below pred
    assert empirical_coverage(a, high, 0.9) == 1.0


def test_coverage_zero_when_all_above():
    a = np.array([5.0, 6.0, 7.0])
    low = np.array([0.0, 0.0, 0.0])
    assert empirical_coverage(a, low, 0.1) == 0.0


def test_coverage_half():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    pred = np.array([2.5, 2.5, 2.5, 2.5])  # 2 of 4 at/below
    assert empirical_coverage(a, pred, 0.5) == 0.5


# ---- calibration ----

def _synthetic_calibrated_quantiles(n=2000, seed=0):
    """Build actuals and quantile predictions that are calibrated BY
    CONSTRUCTION: draw actuals from N(mu, sigma), set predictions to the true
    theoretical quantiles of that same normal. Empirical coverage must then
    match nominal within sampling noise."""
    from scipy.stats import norm
    rng = np.random.default_rng(seed)
    mu, sigma = 100.0, 15.0
    actual = rng.normal(mu, sigma, n)
    preds = {}
    for q in (0.1, 0.5, 0.9):
        preds[q] = np.full(n, mu + sigma * norm.ppf(q))
    return actual, preds


def test_well_calibrated_model_has_low_calibration_error():
    actual, preds = _synthetic_calibrated_quantiles()
    result = assess_calibration(actual, preds)
    assert result is not None
    # each empirical coverage should be within a few points of nominal
    for q, (cov, n) in result.by_quantile.items():
        assert abs(cov - q) < 0.04, f"q={q} coverage {cov}"
    assert result.calibration_error < 0.03


def test_overconfident_model_flagged():
    """Bands too narrow -> P90 contains truth much less than 90%."""
    from scipy.stats import norm
    rng = np.random.default_rng(1)
    mu, sigma = 100.0, 15.0
    actual = rng.normal(mu, sigma, 2000)
    # predictions use HALF the true sigma -> overconfident (too narrow)
    preds = {q: np.full(2000, mu + (sigma * 0.5) * norm.ppf(q)) for q in (0.1, 0.5, 0.9)}
    result = assess_calibration(actual, preds)
    # P90 should contain far less than 90%; P10 far more than 10%
    assert result.by_quantile[0.9][0] < 0.85
    assert result.by_quantile[0.1][0] > 0.15
    assert result.calibration_error > 0.05


def test_interval_coverage_reported():
    actual, preds = _synthetic_calibrated_quantiles()
    result = assess_calibration(actual, preds)
    # 0.1 & 0.9 present -> an 80% central interval should be reported
    assert "80%" in result.interval_coverage
    nominal, empirical = result.interval_coverage["80%"]
    assert nominal == 0.8
    assert abs(empirical - 0.8) < 0.04


# ---- the key anti-gaming property ----

def test_pinball_not_gamed_by_widening_bands():
    """Coverage alone is gameable: absurdly wide bands get perfect coverage.
    Pinball loss must PENALIZE that. A tight-but-calibrated model should beat a
    uselessly-wide one on mean pinball, even though both have decent coverage."""
    from scipy.stats import norm
    rng = np.random.default_rng(2)
    mu, sigma = 100.0, 15.0
    actual = rng.normal(mu, sigma, 3000)

    tight = {q: np.full(3000, mu + sigma * norm.ppf(q)) for q in (0.1, 0.5, 0.9)}
    absurd = {q: np.full(3000, mu + (sigma * 8) * norm.ppf(q)) for q in (0.1, 0.5, 0.9)}

    pin_tight = mean_pinball(actual, tight)
    pin_absurd = mean_pinball(actual, absurd)
    # the calibrated tight model must score better (lower) than the wide one
    assert pin_tight < pin_absurd


def test_score_probabilistic_bundle_shape():
    actual, preds = _synthetic_calibrated_quantiles(n=500)
    card = score_probabilistic(actual, preds)
    assert "mean_pinball" in card
    assert "calibration_error" in card
    assert "interval_coverage" in card
    assert card["mean_pinball"] is not None


def test_handles_nan_actuals():
    a = np.array([100.0, np.nan, 300.0])
    preds = {0.5: np.array([100.0, 999.0, 300.0])}
    # NaN month dropped; remaining two are exact -> pinball ~0
    assert abs(mean_pinball(a, preds)) < 1e-9
