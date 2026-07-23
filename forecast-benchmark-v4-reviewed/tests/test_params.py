import numpy as np
import pytest

from forecast_benchmark.arps import (
    arps_exponential,
    arps_hyperbolic_bounded_b,
    naive_last3,
)
from forecast_benchmark.data import load_csv
from forecast_benchmark.param_benchmark import (
    run_param_benchmark,
    summarize_by_model,
)
from forecast_benchmark.params import (
    DETERMINISTIC_PARAM_MODELS,
    HyperbolicParams,
    params_arps_bounded_b,
    params_arps_exponential,
    params_naive_last3,
    reanchor,
)

HORIZON = 12


def _decline_series(qi=1000.0, di=0.12, b=0.8, n=30, noise=0.0, seed=0):
    t = np.arange(n, dtype=float)
    if b < 1e-6:
        q = qi * np.exp(-di * t)
    else:
        q = qi / np.power(1.0 + b * di * t, 1.0 / b)
    if noise:
        rng = np.random.default_rng(seed)
        q = q * (1.0 + noise * rng.standard_normal(n))
    return q


# --- re-anchoring is a lossless time shift ----------------------------------

def test_reanchor_reproduces_shifted_curve():
    qi, di, b = 1200.0, 0.15, 0.9
    delta = 7
    shifted = reanchor(qi, di, b, delta)
    # q'(t) should equal q(delta + t) for the original curve.
    t = np.arange(1, 13, dtype=float)
    orig_at_shift = qi / np.power(1.0 + b * di * (delta + t), 1.0 / b)
    assert np.allclose(shifted.forecast(12), orig_at_shift)


def test_reanchor_exponential_branch():
    qi, di, delta = 900.0, 0.1, 5
    shifted = reanchor(qi, di, 0.0, delta)
    t = np.arange(1, 13, dtype=float)
    assert np.allclose(shifted.forecast(12), qi * np.exp(-di * (delta + t)))


def test_reanchor_zero_delta_is_identity():
    p = reanchor(500.0, 0.2, 0.7, 0.0)
    assert (p.qi, p.di, p.b) == (500.0, 0.2, 0.7)


# --- the consistency guarantee: params reproduce the base models' forecasts --

@pytest.mark.parametrize("noise,seed", [(0.0, 0), (0.05, 1), (0.15, 2)])
def test_bounded_b_params_match_base_forecast(noise, seed):
    q = _decline_series(noise=noise, seed=seed)
    base = arps_hyperbolic_bounded_b(q)(HORIZON)
    via_params = params_arps_bounded_b(q).forecast(HORIZON)
    assert np.allclose(base, via_params, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("noise,seed", [(0.0, 0), (0.08, 3)])
def test_exponential_params_match_base_forecast(noise, seed):
    q = _decline_series(b=0.0, noise=noise, seed=seed)
    base = arps_exponential(q)(HORIZON)
    via_params = params_arps_exponential(q).forecast(HORIZON)
    assert np.allclose(base, via_params, rtol=1e-6, atol=1e-6)


def test_naive_params_match_base_forecast():
    q = _decline_series(noise=0.1, seed=4)
    base = naive_last3(q)(HORIZON)
    via_params = params_naive_last3(q).forecast(HORIZON)
    assert np.allclose(base, via_params)
    p = params_naive_last3(q)
    assert p.di == 0.0  # flat line


def test_short_history_falls_back_to_naive():
    q = np.array([100.0, 90.0])  # < 3 clean points
    p = params_arps_bounded_b(q)
    assert isinstance(p, HyperbolicParams)
    assert p.di == 0.0  # naive fallback


# --- the benchmark runner produces a params + error table -------------------

def test_param_benchmark_table_and_leaderboard():
    wells = load_csv("examples/tiny_wells.csv")
    result = run_param_benchmark(
        wells, DETERMINISTIC_PARAM_MODELS, cutoff_month=18, horizon=6
    )
    assert result.n_wells_scored > 0
    rows = result.rows()
    assert rows, "expected at least one scored row"

    # Every scored oil row carries both parameters and error.
    scored_oil = [
        r for r in rows
        if r["phase"] == "oil" and r["phase_available"] and r["model"] == "arps_bounded_b"
    ]
    assert scored_oil
    for r in scored_oil:
        assert r["qi"] is not None and r["di"] is not None and r["b"] is not None
        assert r["spee_score"] is not None

    summary = summarize_by_model(result, phase="oil", metric="spee_score")
    assert set(summary) == set(DETERMINISTIC_PARAM_MODELS)
    for stats in summary.values():
        assert stats["n"] >= 1


def test_param_benchmark_records_unavailable_phase_without_forecasting_zero():
    # A well with no water: water rows must be phase_available=False, no params.
    wells = load_csv("examples/tiny_wells.csv")
    result = run_param_benchmark(
        wells, DETERMINISTIC_PARAM_MODELS, cutoff_month=18, horizon=6
    )
    water_rows = [r for r in result.per_well if r.phase == "water"]
    # If the fixture has no water, these are recorded but not forecast.
    for r in water_rows:
        if not r.phase_available:
            assert r.params is None
            assert r.scores == {}
