"""Propose a range at one fixed input, then measure its coverage there.

    python experiments.py --query 0.5 --output-dir results/demo
"""

import argparse
import json
from math import floor
from pathlib import Path

from uq import run_example


def save_figure(output_dir, report, data):
    """Plot only fresh responses and the frozen proposal at the chosen input."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "figure.dpi": 140, "savefig.dpi": 180,
        "svg.hashsalt": "uq-point-validation",
    })
    prediction, measured = report["prediction"], report["validation"]
    fig, ax = plt.subplots(figsize=(8.5, 5.2), layout="constrained")
    ax.hist(data["test_y"], bins=45, color="#7298ad", edgecolor="white",
            linewidth=0.5, label="Fresh responses at this input")
    ax.axvline(prediction["lower"], color="#16324f", linestyle="--", linewidth=2,
               label="Frozen proposal endpoints")
    ax.axvline(prediction["upper"], color="#16324f", linestyle="--", linewidth=2)
    ax.set(xlabel="Response y", ylabel="Number of responses")
    ax.set_title(f"Validate one interval at x = {prediction['x']:g}", pad=40)
    lower_percent = floor(1000 * measured["lower_bound"]) / 10
    ax.text(0, 1.025,
            f"Measured inside: {measured['estimate']:.2%}   |   "
            f"95% confidence lower bound: {lower_percent:.1f}%",
            transform=ax.transAxes, fontsize=10, color="#334155")
    ax.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=1,
              fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.12)
    for extension in ("png", "svg"):
        kwargs = {"metadata": {"Date": None}} if extension == "svg" else {}
        fig.savefig(output_dir / f"point-validation.{extension}", **kwargs)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--test-size", type=int, default=10000,
                        help="Fresh responses at the fixed query; choose in advance (default: 10000)")
    parser.add_argument("--query", type=float, default=0.5,
                        help="One input chosen before validation (default: 0.5)")
    parser.add_argument("--output-dir", type=Path, default=Path("results/demo"))
    args = parser.parse_args(argv)
    try:
        report, data = run_example(seed=args.seed, test_size=args.test_size, query=args.query)
    except ValueError as error:
        parser.error(str(error))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_figure(args.output_dir, report, data)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    prediction, measured = report["prediction"], report["validation"]
    print("The true mean is quadratic; the fitted Bayesian model assumes a line.")
    print(f"Query fixed before validation: x = {prediction['x']:g}")
    print(f"Frozen proposal: [{prediction['lower']:.3f}, {prediction['upper']:.3f}] "
          f"(the model's nominal {prediction['nominal_level']:.0%} interval)")
    print(f"Fresh responses inside at this input: {measured['hits']:,} / {measured['size']:,} "
          f"({measured['estimate']:.2%})")
    print(f"Hoeffding allowance: {100 * measured['margin']:.3f} percentage points")
    minimum_percent = floor(1000 * measured["lower_bound"]) / 10
    print(f"At 95% confidence, this interval covers at least {minimum_percent:.1f}% "
          f"of fresh responses at x = {prediction['x']:g} from this same process.")
    print("This confidence statement is for this one prespecified input.")
    print(f"Saved report.json and point-validation.png/.svg in {args.output_dir}")
    return report


if __name__ == "__main__":
    main()
