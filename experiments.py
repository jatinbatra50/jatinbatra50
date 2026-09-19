"""Propose prediction intervals, then measure their marginal coverage.

    python experiments.py --output-dir results/demo
"""

import argparse
import json
from math import floor
from pathlib import Path

from uq import run_example


def save_figure(output_dir, report, data):
    """Show held-out responses inside and outside their own proposed intervals."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "figure.dpi": 140, "savefig.dpi": 180,
        "svg.hashsalt": "uq-marginal-validation",
    })
    measured = report["validation"]
    fig, ax = plt.subplots(figsize=(8.5, 5.2), layout="constrained")
    counts = [measured["hits"], measured["size"] - measured["hits"]]
    bars = ax.bar(["Inside own proposed interval", "Outside"], counts,
                  color=["#527f95", "#bc7160"], width=0.6)
    ax.bar_label(bars, labels=[f"{count:,}" for count in counts], padding=5)
    ax.set(ylabel="Number of fresh test pairs", ylim=(0, 1.15 * measured["size"]))
    ax.set_title("Measure coverage across fresh (X, Y) pairs", pad=40)
    lower_percent = floor(1000 * measured["lower_bound"]) / 10
    ax.text(0, 1.025,
            f"Measured coverage: {measured['estimate']:.2%}   |   "
            f"95% confidence lower bound: {lower_percent:.1f}%",
            transform=ax.transAxes, fontsize=10, color="#334155")
    ax.grid(axis="y", alpha=0.12)
    ax.set_axisbelow(True)
    for extension in ("png", "svg"):
        kwargs = {"metadata": {"Date": None}} if extension == "svg" else {}
        fig.savefig(output_dir / f"marginal-validation.{extension}", **kwargs)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--test-size", type=int, default=10000,
                        help="Fresh input/response pairs; choose in advance (default: 10000)")
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
    measured = report["validation"]
    print("The true mean is quadratic; the fitted Bayesian model assumes a line.")
    print(f"Frozen proposal rule: the model's nominal {report['prediction']['nominal_level']:.0%} "
          "posterior predictive interval at each input.")
    print(f"Fresh responses inside their own intervals: {measured['hits']:,} / "
          f"{measured['size']:,} ({measured['estimate']:.2%})")
    print(f"Hoeffding allowance: {100 * measured['margin']:.3f} percentage points")
    minimum_percent = floor(1000 * measured["lower_bound"]) / 10
    print(f"At 95% confidence, the frozen rule covers at least {minimum_percent:.1f}% "
          "of fresh (X, Y) pairs from this same process.")
    print("This is marginal coverage, conditional on the trained rule; it is not a per-input guarantee.")
    print(f"Saved report.json and marginal-validation.png/.svg in {args.output_dir}")
    return report


if __name__ == "__main__":
    main()
