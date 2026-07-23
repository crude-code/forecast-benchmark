import numpy as np

from forecast_benchmark.cohort import (
    build_cohort_stats,
    shrink_params,
    shrinkage_weight,
)


def test_shrinkage_weight_monotonic_and_bounded():
    ws = [shrinkage_weight(n) for n in [0, 6, 12, 24, 48, 96, 240]]
    assert ws[0] == 0.0
    assert all(0.0 <= w <= 1.0 for w in ws)
    assert all(b >= a for a, b in zip(ws, ws[1:]))  # non-decreasing


def test_shrinkage_weight_half_trust_point():
    # at exactly half_trust_months, weight should be 0.5
    assert abs(shrinkage_weight(24, half_trust_months=24.0) - 0.5) < 1e-9


def test_build_cohort_stats_uses_medians():
    fits = [
        {"qi": 1000, "di": 0.1, "b": 0.8},
        {"qi": 1200, "di": 0.12, "b": 1.0},
        {"qi": 800, "di": 0.08, "b": 0.6},
    ]
    stats = build_cohort_stats(fits)
    assert stats is not None
    assert stats.b_median == 0.8
    assert stats.n_members == 3


def test_build_cohort_stats_none_when_no_usable_fits():
    fits = [{"qi": np.nan, "di": np.nan, "b": np.nan}, None]
    assert build_cohort_stats(fits) is None


def test_shrink_no_own_no_cohort_falls_back_to_default():
    out = shrink_params(None, None, n_months=6, default_b=0.8)
    assert out["b"] == 0.8
    assert out["b_source"] == "default_no_data"
    assert out["weight_own"] == 0.0


def test_shrink_no_own_uses_cohort_outright():
    cohort = build_cohort_stats([{"qi": 1000, "di": 0.1, "b": 0.9}])
    out = shrink_params(None, cohort, n_months=6)
    assert out["b"] == 0.9
    assert out["b_source"] == "cohort_only"


def test_shrink_own_no_cohort_clamps_b():
    own = {"qi": 1000, "di": 0.1, "b": 2.5}  # absurd b
    out = shrink_params(own, None, n_months=60)
    assert out["b"] <= 1.3  # clamped
    assert out["b_source"] == "own_only_clamped"


def test_shrink_young_well_leans_on_cohort():
    # young well fit says b=1.5, cohort says b=0.7
    own = {"qi": 1000, "di": 0.1, "b": 1.5}
    cohort = build_cohort_stats([{"qi": 900, "di": 0.09, "b": 0.7}])
    out_young = shrink_params(own, cohort, n_months=6)
    out_old = shrink_params(own, cohort, n_months=96)
    # young result should sit closer to the cohort's 0.7 than the old result does
    assert abs(out_young["b"] - 0.7) < abs(out_old["b"] - 0.7)
    # and both are clamped into range
    assert 0.3 <= out_young["b"] <= 1.3
    assert 0.3 <= out_old["b"] <= 1.3


def test_shrink_provenance_present():
    own = {"qi": 1000, "di": 0.1, "b": 0.9}
    cohort = build_cohort_stats([{"qi": 1000, "di": 0.1, "b": 0.8}])
    out = shrink_params(own, cohort, n_months=24)
    assert "weight_own" in out and "b_source" in out
    assert out["b_source"] == "shrunk"
