"""Reproducible simulations for finite-sample coverage audits.

The model and fitted-prediction law match ``uq1.ipynb``. Vectorization changes
the order in which random numbers are consumed, so equal seeds do not imply
equal draws to the original notebook. These routines use explicit Generator
objects and do not change NumPy's global random state.
"""

from numbers import Integral, Real
import math

import numpy as np

from validation import coverage_certificate


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _real(value, name, *, allow_infinite=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number")
    value = float(value)
    if math.isnan(value) or (not allow_infinite and not math.isfinite(value)):
        raise ValueError(f"{name} must be {'non-NaN' if allow_infinite else 'finite'}")
    return value


def _model(beta0, beta1, sigma, x_star):
    values = tuple(_real(v, key) for v, key in (
        (beta0, "beta0"), (beta1, "beta1"), (sigma, "sigma"), (x_star, "x_star")
    ))
    if values[2] < 0:
        raise ValueError("sigma must be nonnegative")
    if not math.isfinite(values[0] + values[1] * values[3]):
        raise ValueError("the future-observation mean must be finite")
    return values


def _generator(rng):
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a numpy.random.Generator")


def _levels(nominal_levels):
    try:
        levels = tuple(_real(v, "nominal level") for v in nominal_levels)
    except TypeError as error:
        raise ValueError("nominal_levels must be a nonempty sequence") from error
    if not levels or any(not 0 < level < 1 for level in levels):
        raise ValueError("nominal_levels must be nonempty and strictly between 0 and 1")
    if len(set(levels)) != len(levels):
        raise ValueError("nominal_levels must contain distinct levels")
    return levels


def _samples(rng, R, n, beta0, beta1, sigma, x_star):
    if sigma == 0:
        # OLS recovers an exactly linear noiseless model. Avoid introducing
        # roundoff noise into an otherwise deterministic prediction law.
        return np.full(R, beta0 + beta1 * x_star)
    x = rng.normal(0.0, 1.0, size=(R, n))
    y = beta0 + beta1 * x + rng.normal(0.0, sigma, size=(R, n))
    x_mean = x.mean(axis=1)
    y_mean = y.mean(axis=1)
    x_centered = x - x_mean[:, None]
    denominator = np.sum(x_centered * x_centered, axis=1)
    if np.any(denominator <= 0):
        raise ArithmeticError("a training design has zero variance; OLS is undefined")
    slope = np.sum(x_centered * (y - y_mean[:, None]), axis=1) / denominator
    intercept = y_mean - slope * x_mean
    return intercept + slope * x_star + rng.normal(0.0, sigma, size=R)


def monte_carlo_samples(
    rng, *, R=1000, n=50, beta0=1.0, beta1=2.0, sigma=1.0, x_star=1.0
):
    """Return R independent fitted-mean-plus-future-noise predictions.

    Each draw independently generates n Gaussian training covariates and
    response errors, fits an intercept and slope by OLS, and adds fresh
    Gaussian future noise at x_star. Memory is O(R*n).
    """
    _generator(rng)
    R, n = _integer(R, "R"), _integer(n, "n", minimum=2)
    beta0, beta1, sigma, x_star = _model(beta0, beta1, sigma, x_star)
    return _samples(rng, R, n, beta0, beta1, sigma, x_star)


def audit_procedure(
    rng, *, trials, nominal_levels=(0.8, 0.9, 0.95, 0.99), R=1000, n=50,
    beta0=1.0, beta1=2.0, sigma=1.0, x_star=1.0, delta=0.05, method="kl"
):
    """Certify procedure-average coverage simultaneously at planned levels.

    In each outer repetition, construct all intervals from one MC cloud and
    compare them with one independent true future response. Sharing these
    draws correlates different claims; the union bound needs no independence
    between claims. Repetitions remain independent for each fixed claim.

    Each certificate uses ``trials`` Bernoulli observations, not ``trials*R``.
    Its target is the corresponding nominal level. Total runtime is
    O(trials*R*n), with O(R*n) working memory. The true model parameters are
    supplied to this oracle simulation, as in the original notebook.
    """
    _generator(rng)
    trials = _integer(trials, "trials")
    R, n = _integer(R, "R"), _integer(n, "n", minimum=2)
    levels = _levels(nominal_levels)
    beta0, beta1, sigma, x_star = _model(beta0, beta1, sigma, x_star)
    # Validate certificate settings before doing potentially expensive work.
    coverage_certificate(0, trials, delta=delta, num_claims=len(levels), method=method)
    tails = (1 - np.asarray(levels)) / 2
    quantiles = np.stack((tails, 1 - tails))
    successes = np.zeros(len(levels), dtype=np.int64)
    mean = beta0 + beta1 * x_star
    for _ in range(trials):
        samples = _samples(rng, R, n, beta0, beta1, sigma, x_star)
        lower, upper = np.quantile(samples, quantiles)
        future = mean if sigma == 0 else rng.normal(mean, sigma)
        successes += (lower <= future) & (future <= upper)
    return [
        coverage_certificate(
            int(count), trials, delta=delta, num_claims=len(levels), method=method,
            target_coverage=level,
        )
        for level, count in zip(levels, successes)
    ]


def _interval(lower, upper):
    lower = _real(lower, "lower", allow_infinite=True)
    upper = _real(upper, "upper", allow_infinite=True)
    if lower > upper:
        raise ValueError("lower must not exceed upper")
    return lower, upper


def gaussian_interval_coverage(lower, upper, *, mean, sigma):
    """Exact Gaussian interval mass, up to floating-point CDF evaluation.

    Endpoints are inclusive. With sigma=0, the distribution is a point mass
    at mean, so the answer is exactly zero or one.
    """
    lower, upper = _interval(lower, upper)
    mean, sigma = _real(mean, "mean"), _real(sigma, "sigma")
    if sigma < 0:
        raise ValueError("sigma must be nonnegative")
    if sigma == 0:
        return float(lower <= mean <= upper)
    a = (lower - mean) / sigma / math.sqrt(2)
    b = (upper - mean) / sigma / math.sqrt(2)
    # Use survival functions in a single tail to avoid subtracting numbers
    # both rounded to one. erfc also handles infinite interval endpoints.
    if a >= 0:
        mass = (math.erfc(a) - math.erfc(b)) / 2
    elif b <= 0:
        mass = (math.erfc(-b) - math.erfc(-a)) / 2
    else:
        mass = (math.erf(b) - math.erf(a)) / 2
    return min(1.0, max(0.0, mass))


def audit_fixed_interval(
    rng, *, lower, upper, trials, beta0=1.0, beta1=2.0, sigma=1.0, x_star=1.0,
    delta=0.05, method="kl", target_coverage=None
):
    """Audit one frozen interval using independent true future responses.

    Choose the interval before observing these audit responses. The target
    is conditional on that particular interval, unlike audit_procedure.
    Draws are processed in bounded chunks, without storing all trials.
    """
    _generator(rng)
    trials = _integer(trials, "trials")
    lower, upper = _interval(lower, upper)
    beta0, beta1, sigma, x_star = _model(beta0, beta1, sigma, x_star)
    coverage_certificate(
        0, trials, delta=delta, method=method, target_coverage=target_coverage
    )
    mean = beta0 + beta1 * x_star
    if sigma == 0:
        successes = trials if lower <= mean <= upper else 0
    else:
        successes = 0
        remaining = trials
        while remaining:
            size = min(remaining, 4096)
            future = rng.normal(mean, sigma, size=size)
            successes += int(np.count_nonzero((lower <= future) & (future <= upper)))
            remaining -= size
    return coverage_certificate(
        successes, trials, delta=delta, method=method, target_coverage=target_coverage
    )
