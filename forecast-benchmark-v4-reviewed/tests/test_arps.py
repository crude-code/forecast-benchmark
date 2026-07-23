import numpy as np

from forecast_benchmark.arps import (
    arps_exponential,
    arps_hyperbolic_bounded_b,
    naive_last3,
)


def _synthetic_hyperbolic(qi=5000, di=0.08, b=0.8, n=36):
    t = np.arange(n, dtype=float)
    q = qi / np.power(1.0 + b * di * t, 1.0 / b)
    return q


def test_naive_last3_flat_lines_recent_average():
    q = np.array([100.0, 90.0, 80.0, 70.0, 60.0])
    fn = naive_last3(q)
    forecast = fn(4)
    expected = np.mean([80.0, 70.0, 60.0])
    assert np.allclose(forecast, expected)


def test_naive_last3_handles_nan_tail():
    q = np.array([100.0, 90.0, np.nan])
    fn = naive_last3(q)
    forecast = fn(2)
    assert np.all(np.isfinite(forecast))


def test_arps_exponential_recovers_declining_shape():
    q = _synthetic_hyperbolic(b=0.001, n=30)  # near-exponential source
    fn = arps_exponential(q)
    forecast = fn(12)
    assert len(forecast) == 12
    assert np.all(np.diff(forecast) <= 0)  # monotonically declining


def test_arps_hyperbolic_bounded_b_fits_reasonably():
    q = _synthetic_hyperbolic(qi=5000, di=0.08, b=0.8, n=36)
    fn = arps_hyperbolic_bounded_b(q)
    forecast = fn(12)
    # forecast should continue the decline, staying in a sane range
    assert forecast[0] < q[-1] * 1.2
    assert forecast[0] > q[-1] * 0.5


def test_arps_short_history_falls_back_to_naive():
    q = np.array([100.0, 90.0])  # too short to fit
    fn = arps_hyperbolic_bounded_b(q)
    forecast = fn(3)
    assert len(forecast) == 3
    assert np.all(np.isfinite(forecast))


def test_arps_handles_all_nan_gracefully():
    q = np.full(10, np.nan)
    fn = naive_last3(q)
    forecast = fn(3)
    # no history at all: naive_last3's mean-of-empty falls back to 0.0
    assert np.allclose(forecast, 0.0)
