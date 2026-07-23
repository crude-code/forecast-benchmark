"""Plain-text summary table for a BenchmarkResult, or a comparison across
several. No plotting, no HTML — this is meant to paste into Slack or a
commit message, not to be a dashboard.
"""
from __future__ import annotations

from forecast_benchmark.benchmark import BenchmarkResult


def format_result(result: BenchmarkResult) -> str:
    lines = [
        f"{result.model_name} | cutoff={result.cutoff_month}mo | horizon={result.horizon}mo",
        f"wells scored: {result.n_wells_scored}  skipped (insufficient history): {result.n_wells_skipped}",
        "",
    ]
    header = f"{'phase':<8} {'n':>5} {'no_data':>8} {'bias_med':>10} {'mape_med':>10} {'spee_med':>10}"
    lines.append(header)
    lines.append("-" * len(header))
    for phase, s in result.summary.items():
        lines.append(
            f"{phase:<8} {s['n_scored']:>5} {s['n_phase_unavailable']:>8} "
            f"{_fmt(s.get('bias_median')):>10} {_fmt(s.get('mape_median')):>10} "
            f"{_fmt(s.get('spee_score_median'), pct=False):>10}"
        )
    return "\n".join(lines)


def format_comparison(results: list[BenchmarkResult]) -> str:
    """Side-by-side oil-phase comparison across models — the table you'd
    actually paste into Slack to answer 'did the new idea help.'"""
    lines = [
        f"{'model':<28} {'phase':<6} {'n':>5} {'bias_med':>10} {'mape_med':>10} {'spee_med':>10}",
    ]
    lines.append("-" * len(lines[0]))
    for result in results:
        for phase, s in result.summary.items():
            lines.append(
                f"{result.model_name:<28} {phase:<6} {s['n_scored']:>5} "
                f"{_fmt(s.get('bias_median')):>10} {_fmt(s.get('mape_median')):>10} "
                f"{_fmt(s.get('spee_score_median'), pct=False):>10}"
            )
    return "\n".join(lines)


def _fmt(v: float | None, *, pct: bool = True) -> str:
    if v is None:
        return "—"
    return f"{v:+.1f}%" if pct else f"{v:.3f}"
