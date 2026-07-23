"""Head-to-head: petbox-dca THM vs. self-contained baselines (primary phase).

    python examples/demo_petbox_vs_baseline.py

Requires petbox-dca (`pip install petbox-dca`). Prints median bias for THM,
bounded-b Arps, and exponential across cutoffs, on both synthetic fixtures.

The point is NOT "THM wins." The point is to MEASURE where a richer,
physically-grounded model helps and where it doesn't — which is the only
honest basis for adding it to the production stack. Spoiler from the synthetic
fixtures: THM tends to win once there's enough history to constrain its extra
parameters, and can lose to the simpler bounded-b model on very short
histories where those extra parameters overfit.
"""
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

try:
    import petbox.dca  # noqa: F401
except Exception:
    print("This demo requires petbox-dca: pip install petbox-dca")
    raise SystemExit(1)

from forecast_benchmark.arps import arps_exponential, arps_hyperbolic_bounded_b
from forecast_benchmark.data import load_csv
from forecast_benchmark.metrics import bias, mape
from forecast_benchmark.petbox_models import thm_model
from forecast_benchmark.split import make_splits

HERE = os.path.dirname(__file__)
MODELS = {
    "THM (petbox)": thm_model,
    "bounded-b Arps": arps_hyperbolic_bounded_b,
    "exponential": arps_exponential,
}


def _med(xs):
    return float(np.median(xs)) if xs else float("nan")


def run(fixture, cuts):
    wells = load_csv(os.path.join(HERE, fixture))
    print(f"\n===== {fixture} — primary OIL phase =====")
    print(f"  {'cutoff':>18}", end="")
    for name in MODELS:
        print(f"{name:>18}", end="")
    print()
    for cut, hor in cuts:
        splits, _ = make_splits(wells, cutoff_month=cut, horizon=hor)
        biases = {name: [] for name in MODELS}
        for s in splits:
            if not s.train.phase_available("oil"):
                continue
            actual = s.holdout_actuals["oil"]
            if not np.any(np.isfinite(actual)):
                continue
            train = np.asarray(s.train.oil, dtype=float)
            for name, fn in MODELS.items():
                b = bias(actual, fn(train)(hor))
                if b is not None:
                    biases[name].append(b)
        label = f"cut={cut} hor={hor} (n={len(biases['THM (petbox)'])})"
        print(f"  {label:>18}", end="")
        for name in MODELS:
            print(f"{_med(biases[name]):>+17.1f}%", end="")
        print()


def main():
    print("=" * 74)
    print("petbox THM vs. self-contained baselines — median bias (closer to 0 better)")
    print("=" * 74)
    run("bubblepoint_wells.csv", [(12, 18), (18, 18), (24, 12)])
    run("ratio_wells.csv", [(12, 18), (18, 18), (24, 12)])
    print("\nReading it: exponential is the dumb-baseline sanity check (should be")
    print("badly biased). THM vs bounded-b is the real comparison — note THM can")
    print("LOSE on the shortest histories where its extra params overfit.")
    print("=" * 74)


if __name__ == "__main__":
    main()
