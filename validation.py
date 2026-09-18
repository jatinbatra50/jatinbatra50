"""Finite-sample certificates for independent Bernoulli coverage trials.

Freeze the procedure before validation. One trial is one independently
constructed interval paired with one independent true future observation.
The resulting target averages over both sources of randomness. For a fixed
interval, instead validate fresh observations against that same interval.

No normal approximation or additional numerical dependencies are used.
See VALIDATION.md for the theorem, proof, and scope of the certificate.
"""

from dataclasses import asdict, dataclass
import math
from numbers import Integral, Real
from typing import Optional


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _probability(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or not 0 < value < 1
    ):
        raise ValueError(f"{name} must be finite and strictly between 0 and 1")
    return float(value)


def _log_budget(delta, num_claims):
    # Log separately to avoid overflow when delta is very small.
    return math.log(2) + math.log(num_claims) - math.log(delta)


def bernoulli_kl(p, q):
    """Binary relative entropy kl(p || q), with natural logs and endpoints."""
    for value in (p, q):
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError("p and q must be finite probabilities in [0, 1]")
    if p == q:
        return 0.0
    if q == 0 or q == 1:
        return math.inf
    if p == 0:
        return -math.log1p(-q)
    if p == 1:
        return -math.log(q)
    value = p * (math.log(p) - math.log(q))
    value += (1 - p) * (math.log1p(-p) - math.log1p(-q))
    return max(0.0, value)


def _kl_bounds(empirical, radius):
    if empirical == 0:
        return 0.0, min(1.0, math.nextafter(-math.expm1(-radius), math.inf))
    if empirical == 1:
        return max(0.0, math.nextafter(math.exp(-radius), -math.inf)), 1.0

    # Keep the endpoint outside the KL ball for conservative rounding.
    left, right = 0.0, empirical
    for _ in range(80):
        middle = (left + right) / 2
        if middle == left or middle == right:
            break
        if bernoulli_kl(empirical, middle) > radius:
            left = middle
        else:
            right = middle
    lower = left

    left, right = empirical, 1.0
    for _ in range(80):
        middle = (left + right) / 2
        if middle == left or middle == right:
            break
        if bernoulli_kl(empirical, middle) > radius:
            right = middle
        else:
            left = middle
    return max(0.0, math.nextafter(lower, -math.inf)), min(
        1.0, math.nextafter(right, math.inf)
    )


@dataclass(frozen=True)
class CoverageCertificate:
    """A two-sided confidence interval for the true coverage probability.

    With iid trials and a fixed protocol, all ``num_claims`` intervals cover
    their respective true probabilities with probability at least 1-delta.
    ``status`` concerns an optional minimum coverage target, not equality
    to a nominal confidence level or optimal interval width.
    """

    successes: int
    trials: int
    empirical_coverage: float
    lower: float
    upper: float
    delta: float
    num_claims: int
    method: str
    target_coverage: Optional[float]

    @property
    def status(self):
        if self.target_coverage is None:
            return "not_requested"
        if self.lower >= self.target_coverage:
            return "certified"
        if self.upper < self.target_coverage:
            return "below_target"
        return "inconclusive"

    def to_dict(self):
        """Return a JSON-serializable report including its decision."""
        return {**asdict(self), "status": self.status}


def coverage_certificate(
    successes,
    trials,
    *,
    delta=0.05,
    num_claims=1,
    method="kl",
    target_coverage=None,
):
    """Certify coverage using Hoeffding or binary-KL Chernoff inversion.

    ``successes`` must be the integer count of covered independent trials;
    ``trials`` is the outer validation sample size, never the inner Monte
    Carlo count R. ``num_claims`` counts all predeclared comparisons reported
    or selected using these data (e.g. all four nominal levels).

    Both methods use the two-sided budget log(2*num_claims/delta). No
    independence between different claims is required. Repeatedly checking
    and stopping when a bound passes is not covered by these fixed-n bounds.
    """
    trials = _positive_integer(trials, "trials")
    num_claims = _positive_integer(num_claims, "num_claims")
    delta = _probability(delta, "delta")
    if (
        isinstance(successes, bool)
        or not isinstance(successes, Integral)
        or not 0 <= successes <= trials
    ):
        raise ValueError("successes must be an integer between 0 and trials")
    successes = int(successes)
    if target_coverage is not None:
        target_coverage = _probability(target_coverage, "target_coverage")
    if method not in ("kl", "hoeffding"):
        raise ValueError("method must be 'kl' or 'hoeffding'")

    empirical = successes / trials
    radius = _log_budget(delta, num_claims) / trials
    if method == "hoeffding":
        epsilon = math.sqrt(radius / 2)
        lower = max(0.0, math.nextafter(empirical - epsilon, -math.inf))
        upper = min(1.0, math.nextafter(empirical + epsilon, math.inf))
    else:
        lower, upper = _kl_bounds(empirical, radius)
    return CoverageCertificate(
        successes, trials, empirical, lower, upper, delta, num_claims,
        method, target_coverage,
    )


def hoeffding_sample_size(epsilon, *, delta=0.05, num_claims=1):
    """Sufficient trials for simultaneous absolute error <= epsilon.

    This plans precision, not the probability that a minimum-coverage
    certificate will pass. Choose the sample size before seeing outcomes.
    """
    epsilon = _probability(epsilon, "epsilon")
    delta = _probability(delta, "delta")
    num_claims = _positive_integer(num_claims, "num_claims")
    denominator = 2 * epsilon**2
    if denominator == 0:
        raise ValueError("epsilon is too small for floating-point sample planning")
    required = _log_budget(delta, num_claims) / denominator
    if not math.isfinite(required):
        raise ValueError("requested sample size exceeds floating-point range")
    return math.ceil(required)
