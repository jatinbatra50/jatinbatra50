"""Bayesian linear regression, followed by an independent coverage measurement."""

from math import isfinite, log, sqrt
from numbers import Integral, Real
from statistics import NormalDist

import numpy as np


def _count(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}")
    return int(value)


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a positive finite number")
    value = float(value)
    if not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return value


def _probability(value, name):
    value = _positive(value, name)
    if value >= 1:
        raise ValueError(f"{name} must be strictly between 0 and 1")
    return value


def fit_posterior(x, y, alpha=0.25, noise_sd=0.5):
    """Fit y = intercept + slope*x + Gaussian noise, with w ~ N(0, I/alpha).

    The noise standard deviation is fixed in advance. Return the Gaussian
    posterior's mean and covariance for w = [intercept, slope].
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if (x.ndim != 1 or y.ndim != 1 or x.size == 0 or x.shape != y.shape
            or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y))):
        raise ValueError("x and y must be nonempty, finite, equal-length vectors")
    alpha = _positive(alpha, "alpha")
    noise_sd = _positive(noise_sd, "noise_sd")
    phi = np.column_stack((np.ones(x.size), x))
    beta = 1 / noise_sd**2
    precision = alpha * np.eye(2) + beta * phi.T @ phi
    covariance = np.linalg.solve(precision, np.eye(2))
    mean = np.linalg.solve(precision, beta * phi.T @ y)
    return {"mean": mean.tolist(), "covariance": covariance.tolist(),
            "alpha": alpha, "noise_sd": noise_sd}


def posterior_predictive(x, posterior, probability=0.99):
    """At each x, return a central interval for a new observation.

    The predictive variance adds observation noise to uncertainty about the
    fitted line. mean_lower/mean_upper describe only the latter uncertainty.
    Scalar x returns scalar NumPy values; a vector x returns arrays.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim > 1 or x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError("x must be a finite scalar or nonempty vector")
    probability = _probability(probability, "probability")
    phi = np.stack((np.ones_like(x), x), axis=-1)
    mean = phi @ np.asarray(posterior["mean"])
    mean_variance = np.einsum("...i,ij,...j->...", phi, posterior["covariance"], phi)
    mean_sd = np.sqrt(mean_variance)
    predictive_sd = np.sqrt(posterior["noise_sd"]**2 + mean_variance)
    z = NormalDist().inv_cdf(0.5 + probability / 2)
    return {"mean": mean, "mean_sd": mean_sd, "predictive_sd": predictive_sd,
            "lower": mean - z * predictive_sd, "upper": mean + z * predictive_sd,
            "mean_lower": mean - z * mean_sd, "mean_upper": mean + z * mean_sd,
            "probability": probability}


def hoeffding_margin(test_size, delta=0.05):
    """One-sided error allowance for independent, same-source test pairs."""
    test_size = _count(test_size, "test_size")
    delta = _probability(delta, "delta")
    return sqrt(log(1 / delta) / (2 * test_size))


def hoeffding_lower(hits, test_size, delta=0.05):
    """A (1-delta) lower bound on the frozen interval rule's average coverage."""
    test_size = _count(test_size, "test_size")
    hits = _count(hits, "hits", minimum=0)
    if hits > test_size:
        raise ValueError("hits cannot exceed test_size")
    return max(0.0, hits / test_size - hoeffding_margin(test_size, delta))


def run_example(seed=20260919, test_size=10000):
    """Return (JSON-friendly report, arrays), without saving any files.

    Choose test_size in advance. Fit on 20 pairs; freeze the whole rule x ->
    interval. Draw independent test pairs from the same input/output process.
    """
    seed = _count(seed, "seed", minimum=0)
    test_size = _count(test_size, "test_size")
    train_stream, test_stream = np.random.SeedSequence(seed).spawn(2)
    train_rng = np.random.default_rng(train_stream)
    train_x = train_rng.uniform(-1.0, 1.0, size=20)
    train_y = 1.0 + 2.0 * train_x + train_rng.normal(0.0, 0.5, size=20)
    posterior = fit_posterior(train_x, train_y)
    prediction = {key: float(value) for key, value in
                  posterior_predictive(0.5, posterior).items()}
    prediction["x"] = 0.5
    test_rng = np.random.default_rng(test_stream)
    test_x = test_rng.uniform(-1.0, 1.0, size=test_size)
    test_y = 1.0 + 2.0 * test_x + test_rng.normal(0.0, 0.5, size=test_size)
    test_prediction = posterior_predictive(test_x, posterior)
    covered = (test_prediction["lower"] <= test_y) & (test_y <= test_prediction["upper"])
    hits = int(covered.sum())
    report = {
        "schema_version": 4,
        "model": {"actual_intercept": 1.0, "actual_slope": 2.0, "actual_noise_sd": 0.5,
                  "input_distribution": "Uniform(-1, 1)", "prior_mean": [0.0, 0.0],
                  "prior_covariance": [[4.0, 0.0], [0.0, 4.0]],
                  "alpha": 0.25, "assumed_noise_sd": 0.5, "beta": 4.0},
        "settings": {"seed": seed, "training_size": 20, "test_size": test_size,
                     "delta": 0.05, "random_streams": "SeedSequence.spawn(2): train, test"},
        "posterior": posterior,
        "prediction": prediction,
        "validation": {"size": test_size, "hits": hits, "estimate": hits / test_size,
                       "margin": hoeffding_margin(test_size),
                       "lower_bound": hoeffding_lower(hits, test_size),
                       "confidence": 0.95,
                       "target": "Average coverage over fresh (X, Y) pairs from the same process"},
    }
    return report, {"train_x": train_x, "train_y": train_y, "test_x": test_x,
                    "test_y": test_y, "covered": covered}
