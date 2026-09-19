"""Small, reusable building blocks for prediction intervals and coverage measurement."""

from math import ceil, isfinite, log, sqrt
from numbers import Integral

import numpy as np


def _count(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}")
    return int(value)


def _probability(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be strictly between 0 and 1")
    value = float(value)
    if not isfinite(value) or not 0 < value < 1:
        raise ValueError(f"{name} must be strictly between 0 and 1")
    return value


def _vector(values, name):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be a nonempty, finite one-dimensional array")
    return values


def sample_sensor(rng, n):
    """Draw independent readings: X ~ Uniform[-1,1], Y = 1 + 2X + N(0,0.5²)."""
    n = _count(n, "n")
    x = rng.uniform(-1.0, 1.0, size=n)
    y = 1.0 + 2.0 * x + rng.normal(0.0, 0.5, size=n)
    return x, y


def fit_line(x, y):
    """Fit ordinary least squares and return [intercept, slope]."""
    x, y = _vector(x, "x"), _vector(y, "y")
    if x.shape != y.shape or x.size < 2 or np.ptp(x) == 0:
        raise ValueError("x and y must have equal length >= 2, with variation in x")
    design = np.column_stack((np.ones(x.size), x))
    return np.linalg.lstsq(design, y, rcond=None)[0]


def predict_line(coefs, x):
    """Evaluate a fitted line at new inputs."""
    coefs = _vector(coefs, "coefs")
    x = np.asarray(x, dtype=float)
    if coefs.shape != (2,) or not np.all(np.isfinite(x)):
        raise ValueError("coefs must contain intercept and slope, and x must be finite")
    return coefs[0] + coefs[1] * x


def conformal_radius(residuals, alpha):
    """Select the split-conformal absolute-residual order statistic (one-based rank)."""
    residuals = _vector(residuals, "residuals")
    alpha = _probability(alpha, "alpha")
    if np.any(residuals < 0):
        raise ValueError("residuals must be absolute residuals, hence nonnegative")
    k = ceil((residuals.size + 1) * (1 - alpha))
    if k > residuals.size:
        return float("inf")
    return float(np.partition(residuals, k - 1)[k - 1])


def hoeffding_radius(n, delta=0.05):
    """Two-sided error radius for a mean of n independent values in [0,1]."""
    n = _count(n, "n")
    delta = _probability(delta, "delta")
    return sqrt(log(2 / delta) / (2 * n))


def measurement_budget(epsilon=0.005, delta=0.05):
    """Choose the test-set size before looking at outcomes."""
    epsilon = _probability(epsilon, "epsilon")
    delta = _probability(delta, "delta")
    return ceil(log(2 / delta) / (2 * epsilon**2))


def coverage_interval(covered, delta=0.05):
    """Return observed coverage and a clipped two-sided Hoeffding interval."""
    covered = np.asarray(covered)
    if covered.ndim != 1 or covered.size == 0 or covered.dtype.kind != "b":
        raise ValueError("covered must be a nonempty, one-dimensional boolean array")
    radius = hoeffding_radius(covered.size, delta)
    estimate = float(np.mean(covered))
    return estimate, max(0.0, estimate - radius), min(1.0, estimate + radius)


def run_demo(seed=20260919, epsilon=0.005, delta=0.05, alpha=0.025):
    """Return (JSON-friendly report, NumPy datasets) for train/calibrate/measure.

    An unbounded conformal interval is encoded by half_width=None in the report;
    data["half_width"] retains the mathematical value infinity.
    """
    seed = _count(seed, "seed", minimum=0)
    epsilon = _probability(epsilon, "epsilon")
    delta = _probability(delta, "delta")
    alpha = _probability(alpha, "alpha")
    n_test = measurement_budget(epsilon, delta)
    streams = np.random.SeedSequence(seed).spawn(3)
    rng_train, rng_calibration, rng_test = [np.random.default_rng(s) for s in streams]
    train_x, train_y = sample_sensor(rng_train, 200)
    calibration_x, calibration_y = sample_sensor(rng_calibration, 4000)
    coefs = fit_line(train_x, train_y)
    residuals = np.abs(calibration_y - predict_line(coefs, calibration_x))
    half_width = conformal_radius(residuals, alpha)
    test_x, test_y = sample_sensor(rng_test, n_test)
    covered = np.abs(test_y - predict_line(coefs, test_x)) <= half_width
    estimate, lower, upper = coverage_interval(covered, delta)
    report = {
        "schema_version": 2,
        "model": "X ~ Uniform[-1,1]; Y = 1 + 2X + Normal(0, 0.5^2)",
        "settings": {"seed": seed, "training_size": 200, "epsilon": epsilon,
                     "delta": delta, "random_streams": "SeedSequence.spawn(3): train, calibrate, measure"},
        "fit": {"intercept": float(coefs[0]), "slope": float(coefs[1])},
        "calibration": {
            "size": 4000, "alpha": alpha, "nominal_coverage": 1 - alpha,
            "rank": ceil(4001 * (1 - alpha)),
            "half_width": half_width if isfinite(half_width) else None,
        },
        "measurement": {
            "size": n_test, "covered": int(covered.sum()), "estimate": estimate,
            "radius": hoeffding_radius(n_test, delta), "lower": lower, "upper": upper,
            "confidence": 1 - delta,
        },
    }
    data = {
        "train_x": train_x, "train_y": train_y,
        "calibration_x": calibration_x, "calibration_y": calibration_y,
        "test_x": test_x, "test_y": test_y, "covered": covered,
        "coefs": coefs, "half_width": half_width,
    }
    return report, data
