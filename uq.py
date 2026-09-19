"""Propose intervals with a Bayesian model; measure their marginal coverage."""

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
    """Return the Bayesian model's central interval for a new observation.

    This is a proposal: its nominal probability is a model calculation, not a
    guarantee about the real process. Predictive variance includes new noise.
    Scalar x returns scalar NumPy values; a vector x returns arrays.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim > 1 or x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError("x must be a finite scalar or nonempty vector")
    probability = _probability(probability, "probability")
    phi = np.stack((np.ones_like(x), x), axis=-1)
    mean = phi @ np.asarray(posterior["mean"])
    mean_variance = np.einsum("...i,ij,...j->...", phi, posterior["covariance"], phi)
    predictive_sd = np.sqrt(posterior["noise_sd"]**2 + mean_variance)
    z = NormalDist().inv_cdf(0.5 + probability / 2)
    return {"mean": mean, "predictive_sd": predictive_sd,
            "lower": mean - z * predictive_sd, "upper": mean + z * predictive_sd,
            "nominal_level": probability}


def hoeffding_margin(test_size, delta=0.05):
    """One-sided error allowance for independent Bernoulli measurements."""
    test_size = _count(test_size, "test_size")
    delta = _probability(delta, "delta")
    return sqrt(log(1 / delta) / (2 * test_size))


def hoeffding_lower(hits, test_size, delta=0.05):
    """Lower confidence bound for a frozen rule's marginal coverage.

    The interval rule, sample size and delta are fixed before inspecting
    independent validation pairs from the future joint input/response law.
    Conditional on the trained rule, hits are independent Bernoulli trials.
    No fitted model or prior enters this calculation; it gives no per-input
    coverage guarantee.
    """
    test_size = _count(test_size, "test_size")
    hits = _count(hits, "hits", minimum=0)
    if hits > test_size:
        raise ValueError("hits cannot exceed test_size")
    return max(0.0, hits / test_size - hoeffding_margin(test_size, delta))


def run_example(seed=20260919, test_size=10000):
    """Return (JSON-friendly report, arrays), without saving any files.

    Choose test_size and the nominal interval level in advance. Fit a
    deliberately wrong linear model to 20 pairs from a quadratic process.
    Freeze its interval rule. Draw independent fresh input/response pairs,
    and check each response against the interval at its own input.
    """
    seed = _count(seed, "seed", minimum=0)
    test_size = _count(test_size, "test_size")
    train_stream, test_stream = np.random.SeedSequence(seed).spawn(2)
    train_rng = np.random.default_rng(train_stream)
    train_x = train_rng.uniform(-1.0, 1.0, size=20)
    train_y = (1.0 + 2.0 * train_x + 2.0 * train_x**2
               + train_rng.normal(0.0, 0.5, size=20))
    posterior = fit_posterior(train_x, train_y)
    nominal_level = 0.99
    test_rng = np.random.default_rng(test_stream)
    test_x = test_rng.uniform(-1.0, 1.0, size=test_size)
    test_y = (1.0 + 2.0 * test_x + 2.0 * test_x**2
              + test_rng.normal(0.0, 0.5, size=test_size))
    prediction = posterior_predictive(test_x, posterior, probability=nominal_level)
    covered = (prediction["lower"] <= test_y) & (test_y <= prediction["upper"])
    hits = int(covered.sum())
    report = {
        "schema_version": 6,
        "model": {"actual_mean_function": "1 + 2*x + 2*x**2",
                  "actual_noise_sd": 0.5,
                  "assumed_mean_function": "intercept + slope*x",
                  "training_input_distribution": "Uniform(-1, 1)",
                  "validation_and_future_input_distribution": "Uniform(-1, 1)",
                  "prior_mean": [0.0, 0.0],
                  "prior_covariance": [[4.0, 0.0], [0.0, 4.0]],
                  "alpha": 0.25, "assumed_noise_sd": 0.5, "beta": 4.0},
        "settings": {"seed": seed, "training_size": 20, "test_size": test_size,
                     "delta": 0.05,
                     "random_streams": "SeedSequence.spawn(2): train, test"},
        "posterior": posterior,
        "prediction": {"nominal_level": nominal_level,
                       "rule": "Frozen Bayesian linear posterior predictive interval at each input"},
        "validation": {"size": test_size, "hits": hits, "estimate": hits / test_size,
                       "margin": hoeffding_margin(test_size),
                       "lower_bound": hoeffding_lower(hits, test_size),
                       "confidence": 0.95,
                       "target": "Marginal coverage P(Y in I(X) | training data) for a fresh (X, Y) pair",
                       "confidence_scope": "One frozen prediction rule, conditional on its training data",
                       "sampling": "Independent fresh (X, Y) pairs from the same joint law as future pairs",
                       "per_input_guarantee": False},
    }
    return report, {"train_x": train_x, "train_y": train_y, "test_x": test_x,
                    "test_y": test_y, "covered": covered}
