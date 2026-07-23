import numpy as np

from forecast_benchmark.arps import arps_hyperbolic_bounded_b, naive_last3
from forecast_benchmark.cum_benchmark import (
    CumResult,
    format_cum_report,
    run_cum_benchmark,
)
from forecast_benchmark.data import load_csv


def test_cum_result_errors():
    r = CumResult(well_id="W", model="m", pred_cum=110.0, actual_cum=100.0)
    assert r.error == 10.0
    assert abs(r.pct_error - 0.1) < 1e-9


def test_cum_pct_error_zero_actual_is_nan():
    r = CumResult(well_id="W", model="m", pred_cum=5.0, actual_cum=0.0)
    assert np.isnan(r.pct_error)


def test_run_cum_benchmark_predicts_and_compares_cum():
    wells = load_csv("examples/tiny_wells.csv")
    forecasters = {
        "naive_last3": naive_last3,
        "arps_bounded_b": arps_hyperbolic_bounded_b,
    }
    results, skipped = run_cum_benchmark(
        wells, forecasters, cutoff_month=18, horizon=12
    )
    # One result per (scored well, method).
    assert len(results) == 2 * len({r.well_id for r in results})
    for r in results:
        assert r.pred_cum >= 0.0
        assert r.actual_cum > 0.0

    # A perfect-ish decline fitter should beat a flat naive line on cum error.
    def med_abs(model):
        errs = [abs(r.pct_error) for r in results if r.model == model]
        return float(np.median(errs))

    assert med_abs("arps_bounded_b") < med_abs("naive_last3")

    report = format_cum_report(results)
    assert "arps_bounded_b" in report and "median_abs_%err" in report
