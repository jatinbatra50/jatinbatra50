"""Run the tutorial and save its numerical results and two figures.

    python experiments.py --output-dir results/demo
"""

import argparse
import json
from pathlib import Path

import numpy as np

from uq import hoeffding_radius, predict_line, run_demo


def save_figures(output_dir, report, data):
    """Draw a prediction interval and the cost of a smaller measurement error."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "figure.dpi": 140, "savefig.dpi": 180,
        "svg.hashsalt": "uq-tutorial",
    })
    navy, blue, amber = "#16324f", "#187f9c", "#bc6426"
    grid = np.linspace(-1, 1, 300)
    prediction = predict_line(data["coefs"], grid)
    q = data["half_width"]
    fig, ax = plt.subplots(figsize=(8, 4.7), layout="constrained")
    if np.isfinite(q):
        ax.fill_between(grid, prediction - q, prediction + q, color=blue, alpha=0.18,
                        label=f"{report['calibration']['nominal_coverage']:.1%} conformal interval")
    ax.scatter(data["train_x"], data["train_y"], s=15, color=navy, alpha=0.5,
               edgecolors="none", label="200 training readings")
    ax.plot(grid, prediction, color=blue, lw=2.3, label="Fitted mean")
    ax.plot(grid, 1 + 2 * grid, color=amber, ls="--", lw=1.4, label="True mean")
    ax.set(xlabel="Sensor input x", ylabel="Future reading y",
           title="A prediction comes with a range", xlim=(-1.02, 1.02))
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.12)
    for extension in ("png", "svg"):
        kwargs = {"metadata": {"Date": None}} if extension == "svg" else {}
        fig.savefig(output_dir / f"prediction-interval.{extension}", **kwargs)
    plt.close(fig)

    n = report["measurement"]["size"]
    delta = report["settings"]["delta"]
    counts = np.unique(np.geomspace(100, max(200000, 2 * n), 120).astype(int))
    errors = np.array([hoeffding_radius(int(count), delta) for count in counts])
    fig, ax = plt.subplots(figsize=(8, 4.7), layout="constrained")
    ax.loglog(counts, 100 * errors, color=blue, lw=2.3)
    chosen_error = 100 * report["measurement"]["radius"]
    ax.scatter([n], [chosen_error], color=amber, s=55, zorder=3)
    ax.axhline(chosen_error, color=amber, alpha=0.5, ls="--", lw=1)
    ax.annotate(f"{n:,} readings\n±{chosen_error:.2f} percentage points",
                xy=(n, chosen_error), xytext=(-12, 60), textcoords="offset points",
                ha="right", color=navy, fontsize=10)
    ax.set(xlabel="Independent test readings n",
           ylabel="Coverage measurement error (percentage points)",
           title=f"Choose measurement precision in advance ({1 - delta:.0%} confidence)")
    ax.grid(which="major", alpha=0.15)
    for extension in ("png", "svg"):
        kwargs = {"metadata": {"Date": None}} if extension == "svg" else {}
        fig.savefig(output_dir / f"measurement-budget.{extension}", **kwargs)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--epsilon", type=float, default=0.005,
                        help="Desired coverage measurement error (default: .005)")
    parser.add_argument("--delta", type=float, default=0.05,
                        help="Failure probability for the coverage measurement (default: .05)")
    parser.add_argument("--alpha", type=float, default=0.025,
                        help="Conformal miscoverage level (default: .025)")
    parser.add_argument("--output-dir", type=Path, default=Path("results/demo"))
    args = parser.parse_args(argv)
    try:
        report, data = run_demo(args.seed, args.epsilon, args.delta, args.alpha)
    except ValueError as error:
        parser.error(str(error))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_figures(args.output_dir, report, data)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    fit, calibration, measurement = report["fit"], report["calibration"], report["measurement"]
    print(f"Fitted mean: {fit['intercept']:.4f} + {fit['slope']:.4f} x")
    width_text = "unbounded" if calibration["half_width"] is None else f"±{calibration['half_width']:.4f}"
    print(f"{calibration['nominal_coverage']:.1%} prediction interval: fitted mean {width_text}")
    print(f"Independent test readings: {measurement['size']:,}")
    print(f"Observed coverage: {measurement['estimate']:.2%}")
    print(f"{measurement['confidence']:.0%} confidence interval for coverage: "
          f"[{measurement['lower']:.2%}, {measurement['upper']:.2%}]")
    print(f"Saved report.json and both figures (PNG + SVG) in {args.output_dir}")
    return report


if __name__ == "__main__":
    main()
