"""End-to-end smoke test: load the tiny fixture, run every model, make sure
nothing crashes and the summary shape is sane. Not a claim about which model
wins — that's what running against real data is for."""
import os

import pytest

from forecast_benchmark.arps import arps_exponential, arps_hyperbolic_bounded_b, naive_last3
from forecast_benchmark.benchmark import run_benchmark
from forecast_benchmark.data import load_csv
from forecast_benchmark.report import format_comparison, format_result
from forecast_benchmark.unified import unified_forecast

FIXTURE = os.path.join(os.path.dirname(__file__), "..", "examples", "tiny_wells.csv")


@pytest.fixture
def wells():
    return load_csv(FIXTURE)


def test_fixture_loads_and_has_wells(wells):
    assert len(wells) == 7
    ids = {w.well_id for w in wells}
    assert "EF-001" in ids


def test_fixture_water_availability_is_honest(wells):
    by_id = {w.well_id: w for w in wells}
    assert by_id["EF-001"].phase_available("water") is True
    assert by_id["WC-002"].phase_available("water") is False  # no water column data


def test_run_benchmark_naive(wells):
    result = run_benchmark(wells, naive_last3, model_name="naive_last3", cutoff_month=18, horizon=6)
    assert result.n_wells_scored > 0
    assert "oil" in result.summary
    assert "water" in result.summary
    # water should report unavailable wells honestly, not fake-score them
    assert result.summary["water"]["n_phase_unavailable"] >= 1


def test_run_benchmark_arps_exponential(wells):
    result = run_benchmark(wells, arps_exponential, model_name="arps_exp", cutoff_month=18, horizon=6)
    assert result.n_wells_scored > 0


def test_run_benchmark_arps_bounded_b(wells):
    result = run_benchmark(wells, arps_hyperbolic_bounded_b, model_name="arps_bounded_b", cutoff_month=18, horizon=6)
    assert result.n_wells_scored > 0


def test_run_benchmark_unified(wells):
    result = run_benchmark(wells, unified_forecast, model_name="unified", cutoff_month=18, horizon=6)
    assert result.n_wells_scored > 0


def test_report_format_result_runs(wells):
    result = run_benchmark(wells, naive_last3, model_name="naive_last3", cutoff_month=18, horizon=6)
    text = format_result(result)
    assert "naive_last3" in text
    assert "oil" in text


def test_report_format_comparison_runs(wells):
    results = [
        run_benchmark(wells, naive_last3, model_name="naive_last3", cutoff_month=18, horizon=6),
        run_benchmark(wells, arps_hyperbolic_bounded_b, model_name="arps_bounded_b", cutoff_month=18, horizon=6),
        run_benchmark(wells, unified_forecast, model_name="unified", cutoff_month=18, horizon=6),
    ]
    text = format_comparison(results)
    assert "naive_last3" in text
    assert "unified" in text


def test_skipped_wells_reported_not_silently_dropped(wells):
    # cutoff + horizon exceeds several wells' total history at this fixture size
    result = run_benchmark(wells, naive_last3, model_name="naive_last3", cutoff_month=30, horizon=12)
    assert result.n_wells_skipped > 0
    assert len(result.skipped_well_ids) == result.n_wells_skipped
