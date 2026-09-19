"""Run Bayesian linear regression and measure prediction-interval coverage.

    python experiments.py --output-dir results/demo
"""

import argparse
import json
from math import floor
from pathlib import Path

import numpy as np

from uq import posterior_predictive, run_example


def save_figure(output_dir, report, data):
    """Show the fitted line and two different kinds of uncertainty."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "figure.dpi": 140, "savefig.dpi": 180,
        "svg.hashsalt": "uq-linear-regression",
    })
    x = np.linspace(-1, 1, 250)
    prediction = posterior_predictive(x, report["posterior"])
    fig, ax = plt.subplots(figsize=(8.5, 5.2), layout="constrained")
    ax.fill_between(x, prediction["lower"], prediction["upper"],
                    color="#187f9c", alpha=0.16, label="99% interval for a new observation")
    ax.fill_between(x, prediction["mean_lower"], prediction["mean_upper"],
                    color="#187f9c", alpha=0.4, label="99% interval for the mean at each x")
    ax.plot(x, prediction["mean"], color="#17627c", lw=1.8, label="Estimated mean")
    ax.scatter(data["train_x"], data["train_y"], s=30, color="#16324f",
               edgecolors="white", linewidths=0.5, label="20 training observations", zorder=3)
    ax.set(xlabel="Input x", ylabel="Output y", xlim=(-1, 1), ylim=(-3.0, 5.5))
    ax.set_title("A line, and a range for a new observation", pad=28)
    ax.text(0, 1.015, "The outer interval includes both uncertainty about the line and observation noise.",
            transform=ax.transAxes, fontsize=9, color="#45556a")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.12)
    for extension in ("png", "svg"):
        kwargs = {"metadata": {"Date": None}} if extension == "svg" else {}
        fig.savefig(output_dir / f"regression-interval.{extension}", **kwargs)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--test-size", type=int, default=10000,
                        help="Number of fresh test pairs, chosen in advance (default: 10000)")
    parser.add_argument("--output-dir", type=Path, default=Path("results/demo"))
    args = parser.parse_args(argv)
    try:
        report, data = run_example(seed=args.seed, test_size=args.test_size)
    except ValueError as error:
        parser.error(str(error))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_figure(args.output_dir, report, data)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    posterior, prediction, measured = report["posterior"], report["prediction"], report["validation"]
    print(f"Estimated line: y = {posterior['mean'][0]:.3f} + {posterior['mean'][1]:.3f} x")
    print(f"At x = {prediction['x']:.1f}, predicted mean: {prediction['mean']:.3f}")
    print(f"99% Bayesian prediction interval there: [{prediction['lower']:.3f}, "
          f"{prediction['upper']:.3f}]")
    print(f"Fresh observations inside their intervals: {measured['hits']:,} / {measured['size']:,} "
          f"({measured['estimate']:.2%})")
    print(f"Hoeffding allowance: {100 * measured['margin']:.3f} percentage points")
    minimum_percent = floor(1000 * measured["lower_bound"]) / 10
    print(f"At 95% confidence, the frozen prediction rule covers at least "
          f"{minimum_percent:.1f}% of fresh (X, Y) pairs from the same process.")
    print(f"Saved report.json and regression-interval.png/.svg in {args.output_dir}")
    return report


if __name__ == "__main__":
    main()
