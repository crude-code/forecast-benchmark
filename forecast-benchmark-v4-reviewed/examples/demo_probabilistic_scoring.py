"""Demo: why probabilistic scoring matters (and how it answers the
'is single-well forecasting even the right problem' concern).

    python examples/demo_probabilistic_scoring.py

Builds a deliberately messy well (25% noise, standing in for allocation /
downtime / lift / reporting contamination) and two models that share the
EXACT SAME P50 but differ in their uncertainty bands:

  A - overconfident: narrow bands that claim more certainty than the data supports
  B - honest:        wide bands calibrated to the actual noise level

A point-only benchmark (MAPE, bias) rates them identical. The probabilistic
layer tells them apart — which is the whole point: on a well whose actuals are
contaminated, the valuable model output isn't a sharper P50, it's an honest
'I don't know, here's how much.' This is how you measure that.
"""
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from scipy.stats import norm

from forecast_benchmark.metrics import bias, mape
from forecast_benchmark.probabilistic import score_probabilistic


def main():
    rng = np.random.default_rng(3)
    n = 24
    trend = 1000 * np.exp(-0.03 * np.arange(n))
    actual = np.clip(trend * (1 + rng.normal(0, 0.25, n)), 1, None)  # messy well

    p50 = trend

    def bands(sig):
        return {
            0.1: p50 * (1 + sig * norm.ppf(0.1)),
            0.5: p50,
            0.9: p50 * (1 + sig * norm.ppf(0.9)),
        }

    A = bands(0.05)  # overconfident: bands far narrower than the real noise
    B = bands(0.25)  # honest: bands matched to the real noise

    print("=" * 72)
    print("Probabilistic scoring demo — a messy well, two models, same P50")
    print("=" * 72)
    print("\nSame P50 => a point-only benchmark rates them IDENTICAL:")
    print(f"  Model A:  MAPE {mape(actual, p50):.1f}%   bias {bias(actual, p50):+.1f}%")
    print(f"  Model B:  MAPE {mape(actual, p50):.1f}%   bias {bias(actual, p50):+.1f}%")

    cardA = score_probabilistic(actual, A)
    cardB = score_probabilistic(actual, B)

    def row(name, card):
        p80 = card["interval_coverage"].get("80%", (0, 0))[1] * 100
        print(f"  {name:<34} calib_err {card['calibration_error']:.2f}   "
              f"P10-P90 covers {p80:.0f}% (want 80%)   pinball {card['mean_pinball']:.1f}")

    print("\nBut the probabilistic layer separates them:")
    row("A  overconfident (narrow bands)", cardA)
    row("B  honest (calibrated bands)", cardB)

    print("\nModel A claims an 80% band that actually catches far fewer outcomes —")
    print("it's confidently wrong. Model B admits the uncertainty and is calibrated.")
    print("On a contaminated well, B is the more useful (and more honest) model,")
    print("and only this layer can tell you that. Lower pinball = better overall.")
    print("=" * 72)


if __name__ == "__main__":
    main()
