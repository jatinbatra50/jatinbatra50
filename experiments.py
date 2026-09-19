"""Run the bottle example and save its measurements and one picture.

    python experiments.py --output-dir results/demo
"""

import argparse
import json
from math import floor
from pathlib import Path

import numpy as np

from uq import run_example


def save_figure(output_dir, report, data):
    """Show the fixed prediction range and the first 100 test bottles."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "figure.dpi": 140, "savefig.dpi": 180,
        "svg.hashsalt": "uq-bottles",
    })
    prediction, measured = report["prediction"], report["validation"]
    shown = min(100, measured["size"])
    bottles = np.arange(1, shown + 1)
    values, inside = data["test"][:shown], data["covered"][:shown]
    fig, ax = plt.subplots(figsize=(8.5, 4.8), layout="constrained")
    ax.axhspan(prediction["lower"], prediction["upper"], color="#187f9c", alpha=0.16,
               label="99% Bayesian prediction range")
    ax.axhline(prediction["posterior_mean"], color="#187f9c", lw=1.4,
               label="Estimated mean")
    ax.scatter(bottles[inside], values[inside], s=24, color="#16324f",
               label="Inside the range", zorder=3)
    ax.scatter(bottles[~inside], values[~inside], s=46, color="#bd552b", marker="x",
               linewidths=1.8, label="Outside the range", zorder=4)
    ax.set(xlabel="Test bottle number", ylabel="Amount in bottle (mL)",
           xlim=(0, shown + 1))
    ax.set_title("Predict a range, then check fresh bottles", pad=32)
    ax.text(0, 1.02,
            f"First {shown:,} of {measured['size']:,} test bottles shown; "
            f"all {measured['size']:,} used to measure coverage",
            transform=ax.transAxes, fontsize=9, color="#45556a")
    ax.legend(frameon=False, fontsize=9, loc="lower left", ncol=2)
    ax.grid(axis="y", alpha=0.12)
    for extension in ("png", "svg"):
        kwargs = {"metadata": {"Date": None}} if extension == "svg" else {}
        fig.savefig(output_dir / f"bottle-interval.{extension}", **kwargs)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--test-size", type=int, default=10000,
                        help="Number of fresh test bottles, chosen in advance (default: 10000)")
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
    prediction, measured = report["prediction"], report["validation"]
    print(f"Estimated mean: {prediction['posterior_mean']:.3f} mL")
    print(f"Uncertainty about that mean (posterior SD): {prediction['posterior_sd']:.3f} mL")
    print(f"99% Bayesian prediction range: [{prediction['lower']:.3f}, "
          f"{prediction['upper']:.3f}] mL")
    print(f"Fresh bottles inside the range: {measured['hits']:,} / {measured['size']:,} "
          f"({measured['estimate']:.2%})")
    print(f"Hoeffding allowance: {100 * measured['margin']:.3f} percentage points")
    minimum_percent = floor(1000 * measured["lower_bound"]) / 10
    print(f"At 95% confidence, this fixed range covers at least "
          f"{minimum_percent:.1f}% of same-source bottles.")
    print(f"Saved report.json and bottle-interval.png/.svg in {args.output_dir}")
    return report


if __name__ == "__main__":
    main()
