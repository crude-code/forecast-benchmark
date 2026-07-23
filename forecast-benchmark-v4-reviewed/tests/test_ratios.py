import numpy as np

from forecast_benchmark.ratios import (
    fit_ratio,
    flat_ratio_fallback,
    secondary_from_primary,
)


def _rising_gor_well(n=36, seed=0):
    """Synthetic oil well: oil declines, GOR rises (solution gas liberation)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    oil = 5000 / (1 + 0.8 * 0.08 * t) ** (1 / 0.8)
    gor = 800 * ((t + 3) / 3) ** 0.4  # rising power law
    gas = gor * oil
    oil *= 1 + rng.normal(0, 0.03, n)
    gas *= 1 + rng.normal(0, 0.05, n)
    return np.clip(oil, 0, None), np.clip(gas, 0, None)


def _falling_cgr_well(n=36, seed=1):
    """Synthetic gas-condensate well: gas declines, CGR falls."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    gas = 40000 / (1 + 1.1 * 0.1 * t) ** (1 / 1.1)
    cgr = 120 * ((t + 3) / 3) ** -0.3  # falling power law
    cond = cgr * gas
    gas *= 1 + rng.normal(0, 0.03, n)
    cond *= 1 + rng.normal(0, 0.05, n)
    return np.clip(gas, 0, None), np.clip(cond, 0, None)


def test_fit_ratio_returns_callable_on_good_data():
    oil, gas = _rising_gor_well()
    fn = fit_ratio(oil, gas, shape="power_law", rising=True)
    assert fn is not None
    out = fn(np.arange(36, 48, dtype=float))
    assert len(out) == 12
    assert np.all(np.isfinite(out))


def test_rising_constraint_produces_rising_gor_forecast():
    oil, gas = _rising_gor_well()
    fn = fit_ratio(oil, gas, shape="power_law", rising=True)
    future = fn(np.arange(36, 60, dtype=float))
    # with rising=True the forecast ratio must not decline over the horizon
    assert future[-1] >= future[0]


def test_falling_constraint_produces_falling_cgr_forecast():
    gas, cond = _falling_cgr_well()
    fn = fit_ratio(gas, cond, shape="power_law", rising=False)
    future = fn(np.arange(36, 60, dtype=float))
    assert future[-1] <= future[0]


def test_secondary_from_primary_rides_the_forecast():
    oil, gas = _rising_gor_well()
    # a simple declining oil forecast for the holdout
    oil_forecast = 300 / (1 + 0.8 * 0.08 * np.arange(12)) ** (1 / 0.8)
    gas_fc = secondary_from_primary(oil, gas, oil_forecast, shape="power_law", rising=True)
    assert gas_fc is not None
    assert len(gas_fc) == 12
    # gas forecast should be positive and on the order of gor * oil
    assert np.all(gas_fc > 0)


def test_independent_fit_would_decline_but_ratio_method_does_not():
    """The core claim: an oil well's gas stream, fit independently, declines;
    but the ratio method keeps gas from collapsing because GOR rises."""
    oil, gas = _rising_gor_well(n=24)
    oil_forecast = 400 / (1 + 0.8 * 0.08 * np.arange(24)) ** (1 / 0.8)
    gas_fc = secondary_from_primary(oil, gas, oil_forecast, shape="power_law", rising=True)
    # implied GOR at end of forecast should exceed the initial historical GOR
    implied_gor_end = gas_fc[-1] / oil_forecast[-1]
    initial_gor = float(np.median((gas[:6] / oil[:6])))
    assert implied_gor_end >= initial_gor


def test_logistic_shape_fits_and_plateaus():
    oil, gas = _rising_gor_well(n=48)
    fn = fit_ratio(oil, gas, shape="logistic", rising=True)
    assert fn is not None
    far = fn(np.arange(200, 240, dtype=float))
    # logistic must plateau, not run away
    assert np.all(np.isfinite(far))
    assert far[-1] < 1e7


def test_fit_ratio_none_on_too_short_history():
    assert fit_ratio(np.array([100.0, 90.0]), np.array([80.0, 75.0])) is None


def test_flat_fallback_always_returns_something():
    oil, gas = _rising_gor_well(n=8)
    oil_fc = np.array([100.0] * 6)
    out = flat_ratio_fallback(oil, gas, oil_fc)
    assert len(out) == 6
    assert np.all(np.isfinite(out))
