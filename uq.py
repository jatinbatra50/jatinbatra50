"""One Bayesian prediction range, checked against fresh bottle measurements."""

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


def posterior_predictive(training, prior_mean=100.0, prior_sd=5.0,
                         noise_sd=2.0, probability=0.99):
    """Return a Bayesian range for the next bottle, with known bottle noise.

    The prior for the unknown mean is Normal(prior_mean, prior_sd**2).
    Given that mean, bottle measurements are independent Normal(mean, noise_sd**2).
    """
    training = np.asarray(training, dtype=float)
    if training.ndim != 1 or training.size == 0 or not np.all(np.isfinite(training)):
        raise ValueError("training must be a nonempty, finite one-dimensional array")
    if not isinstance(prior_mean, Real) or not isfinite(prior_mean):
        raise ValueError("prior_mean must be a finite number")
    prior_sd = _positive(prior_sd, "prior_sd")
    noise_sd = _positive(noise_sd, "noise_sd")
    probability = _probability(probability, "probability")

    variance = 1 / (1 / prior_sd**2 + training.size / noise_sd**2)
    mean = variance * (prior_mean / prior_sd**2 + float(training.sum()) / noise_sd**2)
    predictive_sd = sqrt(noise_sd**2 + variance)
    half_width = NormalDist().inv_cdf(0.5 + probability / 2) * predictive_sd
    return {
        "posterior_mean": mean, "posterior_sd": sqrt(variance),
        "predictive_sd": predictive_sd, "probability": probability,
        "lower": mean - half_width, "upper": mean + half_width,
    }


def hoeffding_margin(test_size, delta=0.05):
    """One-sided error allowance for independent, same-source test bottles."""
    test_size = _count(test_size, "test_size")
    delta = _probability(delta, "delta")
    return sqrt(log(1 / delta) / (2 * test_size))


def hoeffding_lower(hits, test_size, delta=0.05):
    """A (1-delta) lower confidence bound on this fixed range's coverage."""
    test_size = _count(test_size, "test_size")
    hits = _count(hits, "hits", minimum=0)
    if hits > test_size:
        raise ValueError("hits cannot exceed test_size")
    return max(0.0, hits / test_size - hoeffding_margin(test_size, delta))


def run_example(seed=20260919, test_size=10000):
    """Return (JSON-friendly report, arrays), without saving any files.

    Choose test_size before generating data. Freeze the range after 20 training
    bottles; use a separate random stream to measure its coverage.
    """
    seed = _count(seed, "seed", minimum=0)
    test_size = _count(test_size, "test_size")
    train_stream, test_stream = np.random.SeedSequence(seed).spawn(2)
    training = np.random.default_rng(train_stream).normal(100.5, 2.0, size=20)
    prediction = posterior_predictive(training)
    test = np.random.default_rng(test_stream).normal(100.5, 2.0, size=test_size)
    covered = (prediction["lower"] <= test) & (test <= prediction["upper"])
    hits = int(covered.sum())
    report = {
        "schema_version": 3,
        "model": {"actual_mean": 100.5, "actual_sd": 2.0, "prior_mean": 100.0,
                  "prior_sd": 5.0, "assumed_noise_sd": 2.0},
        "settings": {"seed": seed, "training_size": 20, "test_size": test_size,
                     "delta": 0.05, "random_streams": "SeedSequence.spawn(2): train, test"},
        "prediction": prediction,
        "validation": {"size": test_size, "hits": hits, "estimate": hits / test_size,
                       "margin": hoeffding_margin(test_size),
                       "lower_bound": hoeffding_lower(hits, test_size),
                       "confidence": 0.95},
    }
    return report, {"training": training, "test": test, "covered": covered}
