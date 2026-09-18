"""Run reproducible coverage audits and save an illustrated JSON report.

Example:
    python experiments.py --trials 1000 --mc-samples 1000 --seed 20260918 \
        --output-dir results/demo
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time

import numpy as np

from simulation import (
    audit_fixed_interval,
    audit_procedure,
    gaussian_interval_coverage,
    monte_carlo_samples,
)
from validation import coverage_certificate


def _parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trials", type=int, default=1000,
                        help="Independent outer trials per coverage claim (default: 1000)")
    parser.add_argument("--mc-samples", type=int, default=1000,
                        help="Inner draws for each interval construction (default: 1000)")
    parser.add_argument("--training-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--output-dir", type=Path, default=Path("results/demo"))
    parser.add_argument("--delta", type=float, default=0.05,
                        help="Overall failure budget, split equally between the two audits")
    parser.add_argument("--method", choices=("kl", "hoeffding"), default="kl")
    parser.add_argument("--nominal-levels", type=float, nargs="+",
                        default=[0.8, 0.9, 0.95, 0.99])
    parser.add_argument("--fixed-level", type=float, default=0.95,
                        help="Nominal level of the separately constructed frozen interval")
    parser.add_argument("--beta0", type=float, default=1.0)
    parser.add_argument("--beta1", type=float, default=2.0)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--x-star", type=float, default=1.0)
    return parser


def _sample_size_curves(delta, num_claims):
    # Multiples of 20 make the hypothetical success fraction exactly 19/20.
    counts = np.unique(np.rint(np.geomspace(1, 5000, 80)).astype(int)) * 20
    curves = {"trials": [int(n) for n in counts], "empirical_coverage": 0.95,
              "counts_are_hypothetical": True, "delta": delta, "num_claims": num_claims}
    for method in ("kl", "hoeffding"):
        curves[method + "_lower"] = [
            coverage_certificate(
                19 * (int(n) // 20), int(n), delta=delta, num_claims=num_claims,
                method=method,
            ).lower
            for n in counts
        ]
    return curves


def _save_figures(output_dir, levels, certificates, fixed, model, curves, joint_confidence):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
        "axes.spines.right": False, "axes.titleweight": "bold", "figure.dpi": 140,
        "savefig.dpi": 180,
    })
    navy, blue, amber = "#16324f", "#187f9c", "#bc6426"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), layout="constrained")
    ax = axes[0]
    observed = np.array([c.empirical_coverage for c in certificates])
    errors = np.array([[c.empirical_coverage-c.lower for c in certificates],
                       [c.upper-c.empirical_coverage for c in certificates]])
    ax.plot([0, 1], [0, 1], "--", color="#83929d", lw=1.3, label="Nominal target")
    ax.errorbar(levels, observed, yerr=errors, fmt="o", color=blue, capsize=5,
                markersize=6, lw=1.8, label="Empirical coverage + bounds")
    ax.set(xlabel="Nominal prediction-interval level", ylabel="Coverage probability",
           title="Rebuild the interval each trial")
    ax.set_xlim(max(0, min(levels)-0.04), min(1.01, max(levels)+0.03))
    ax.set_ylim(max(0, min(0.7, min(c.lower for c in certificates)-0.025)), 1.01)
    ax.grid(axis="y", alpha=0.18)
    ax.legend(loc="lower right", frameon=False, fontsize=9)

    ax = axes[1]
    mean = model["beta0"] + model["beta1"] * model["x_star"]
    sigma = model["sigma"]
    lower, upper = fixed["interval"]["lower"], fixed["interval"]["upper"]
    if sigma > 0:
        left, right = min(mean-4*sigma, lower), max(mean+4*sigma, upper)
        x = np.linspace(left, right, 700)
        density = np.exp(-0.5*((x-mean)/sigma)**2) / (sigma*np.sqrt(2*np.pi))
        ax.plot(x, density, color=navy, lw=1.5)
        ax.fill_between(x, 0, density, where=(lower <= x) & (x <= upper),
                        color=blue, alpha=0.24, label="Probability inside frozen interval")
        ax.set_ylabel("True future-observation density")
    else:
        ax.axvline(mean, color=navy, lw=3, label="Deterministic future observation")
        ax.set_xlim(mean-1, mean+1)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Point mass at the true mean")
    ax.axvline(lower, color=amber, ls="--", lw=1.2)
    ax.axvline(upper, color=amber, ls="--", lw=1.2)
    fixed_cert = fixed["certificate"]
    ax.set(xlabel="Future response Y", title="Freeze one interval, then audit it")
    ax.text(0.02, 0.98,
            f"Exact Gaussian coverage: {fixed['exact_gaussian_coverage']:.3f}\n"
            f"Audit: [{fixed_cert['lower']:.3f}, {fixed_cert['upper']:.3f}]\n"
            f"Decision: {fixed_cert['status']}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92})
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.15)
    fig.suptitle(f"Coverage validation | {joint_confidence:.0%} joint audit confidence",
                 color=navy, fontsize=14, weight="bold")
    for extension in ("png", "svg"):
        fig.savefig(output_dir / f"coverage.{extension}")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.6), layout="constrained")
    ax.semilogx(curves["trials"], curves["kl_lower"], color=blue, lw=2.2,
                label="Bernoulli Chernoff / KL")
    ax.semilogx(curves["trials"], curves["hoeffding_lower"], color=amber, lw=2.2,
                label="Hoeffding")
    ax.axhline(0.95, color="#83929d", ls="--", lw=1.1, label="Observed proportion 0.95")
    ax.set(xlabel="Number of independent outer trials N", ylabel="Lower coverage bound",
           title="How audit sample size controls precision", ylim=(0.4, 0.97))
    ax.text(0.02, 0.03, "Hypothetical counts k/N = 0.95; not experiment observations",
            transform=ax.transAxes, fontsize=9, color=navy)
    ax.grid(axis="y", alpha=0.18)
    ax.legend(loc="lower right", bbox_to_anchor=(1, 0.12), frameon=False)
    for extension in ("png", "svg"):
        fig.savefig(output_dir / f"sample-size.{extension}")
    plt.close(fig)
    return matplotlib.__version__


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    started = time.perf_counter()
    if not 0 < args.delta < 1:
        parser.error("--delta must be strictly between 0 and 1")
    if not 0 < args.fixed_level < 1:
        parser.error("--fixed-level must be strictly between 0 and 1")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    group_delta = args.delta / 2
    model = dict(beta0=args.beta0, beta1=args.beta1, sigma=args.sigma, x_star=args.x_star)
    child_seeds = np.random.SeedSequence(args.seed).spawn(3)
    rng_procedure, rng_construction, rng_fixed = [np.random.default_rng(s) for s in child_seeds]
    try:
        certificates = audit_procedure(
            rng_procedure, trials=args.trials, nominal_levels=args.nominal_levels,
            R=args.mc_samples, n=args.training_samples, delta=group_delta,
            method=args.method, **model,
        )
        cloud = monte_carlo_samples(
            rng_construction, R=args.mc_samples, n=args.training_samples, **model
        )
        tail = (1-args.fixed_level)/2
        lower, median, upper = [float(v) for v in np.quantile(cloud, [tail, 0.5, 1-tail])]
        fixed_certificate = audit_fixed_interval(
            rng_fixed, lower=lower, upper=upper, trials=args.trials,
            delta=group_delta, method=args.method, target_coverage=args.fixed_level,
            **model,
        )
    except (ValueError, ArithmeticError) as error:
        parser.error(str(error))
    exact = gaussian_interval_coverage(
        lower, upper, mean=args.beta0+args.beta1*args.x_star, sigma=args.sigma
    )
    fixed = {
        "scope": "Coverage of this one frozen interval, conditional on its construction.",
        "nominal_level": args.fixed_level,
        "interval": {"lower": lower, "median": median, "upper": upper},
        "exact_gaussian_coverage": exact,
        "certificate": fixed_certificate.to_dict(),
    }
    curves = _sample_size_curves(group_delta, len(certificates))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    matplotlib_version = _save_figures(
        args.output_dir, args.nominal_levels, certificates, fixed, model, curves,
        joint_confidence=1-args.delta,
    )
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter()-started,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "matplotlib": matplotlib_version},
        "randomness": {
            "seed": args.seed, "generator": type(rng_procedure.bit_generator).__name__,
            "streams": "SeedSequence.spawn(3): procedure, fixed construction, fixed audit",
            "shared_between_levels": True,
            "note": "Each outer repetition shares one MC cloud and one future response across levels. "
                    "Each level has independent repetitions. This preserves its marginal law from "
                    "the original notebook, but changes the seed protocol and correlates claims.",
        },
        "settings": {"trials": args.trials, "mc_samples": args.mc_samples,
                     "training_samples": args.training_samples, "method": args.method,
                     "model": model},
        "budgets": {"global_delta": args.delta, "joint_confidence": 1-args.delta,
                    "procedure_family_delta": group_delta,
                    "procedure_claims": len(certificates), "fixed_interval_delta": group_delta,
                    "fixed_interval_claims": 1},
        "distribution_scope": {
            "training": "Independent X ~ N(0,1); Y = beta0 + beta1*X + epsilon, epsilon ~ N(0,sigma^2).",
            "future": "Independent Y* ~ N(beta0 + beta1*x_star, sigma^2) at fixed x_star.",
            "limitation": "Oracle simulator parameters are supplied. These results do not certify "
                          "real-data coverage, arbitrary covariates, or a posterior distribution.",
        },
        "assumptions": ["Freeze settings, targets, and trial counts before seeing audit outcomes.",
                        "Independent repetitions within each coverage claim.",
                        "Fresh audit observations independent of the frozen interval construction.",
                        "Fixed-sample bounds: no data-dependent stopping or tuning on these outcomes.",
                        "Probability guarantees assume ideal independent random draws; numerical "
                        "experiments use a seeded pseudorandom generator."],
        "bounds": {"formula": "log_budget = log(2*num_claims/group_delta); "
                               "KL inverts N*kl(observed || p) <= log_budget; "
                               "Hoeffding uses sqrt(log_budget/(2*N)).",
                   "joint_statement": "With probability at least 1-global_delta, every procedure "
                                      "bound and the independent fixed-interval bound contain their "
                                      "respective true coverage probabilities."},
        "procedure_audit": {"scope": "Coverage averaged over interval construction and a fresh future response.",
                            "nominal_levels": args.nominal_levels,
                            "certificates": [c.to_dict() for c in certificates]},
        "fixed_interval_audit": fixed,
        "hypothetical_sample_size_plot": curves,
        "files": ["report.json", "coverage.png", "coverage.svg", "sample-size.png", "sample-size.svg"],
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8"
    )
    print(f"Joint audit confidence: {1-args.delta:.0%} (global delta={args.delta:g})")
    print(f"Procedure family delta={group_delta:g}; independent fixed-interval delta={group_delta:g}")
    print("Nominal   Covered        Empirical    Coverage bounds       Decision")
    for level, certificate in zip(args.nominal_levels, certificates):
        print(f"{level:7.3f}   {certificate.successes:5d}/{certificate.trials:<5d}    "
              f"{certificate.empirical_coverage:.4f}       "
              f"[{certificate.lower:.4f}, {certificate.upper:.4f}]    {certificate.status}")
    print(f"Frozen interval exact coverage={exact:.4f}; audit bounds="
          f"[{fixed_certificate.lower:.4f}, {fixed_certificate.upper:.4f}]; "
          f"{fixed_certificate.status}")
    print(f"Saved report.json, coverage.png, and sample-size.png (+ SVG) in {args.output_dir}")
    return report


if __name__ == "__main__":
    main()
