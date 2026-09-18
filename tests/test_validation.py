"""Deterministic checks of the finite-sample coverage certificates.

Run from the repository root with ``python -m unittest discover -s tests``.
The coverage check sums binomial probabilities exactly up to floating-point
arithmetic; it does not rely on repeated random simulations.
"""

import json
import math
import unittest

from validation import bernoulli_kl, coverage_certificate, hoeffding_sample_size


class BinaryKLTests(unittest.TestCase):
    def test_endpoints_and_identity(self):
        for p in (0.0, 0.01, 0.25, 0.5, 0.99, 1.0):
            self.assertEqual(bernoulli_kl(p, p), 0.0)
        for q in (0.01, 0.25, 0.5, 0.99):
            self.assertAlmostEqual(bernoulli_kl(0, q), -math.log1p(-q))
            self.assertAlmostEqual(bernoulli_kl(1, q), -math.log(q))
        self.assertEqual(bernoulli_kl(0, 1), math.inf)
        self.assertEqual(bernoulli_kl(1, 0), math.inf)
        self.assertEqual(bernoulli_kl(0.5, 0), math.inf)
        self.assertEqual(bernoulli_kl(0.5, 1), math.inf)

    def test_known_value_reflection_and_monotonicity(self):
        expected = 0.25 * math.log(0.5) + 0.75 * math.log(1.5)
        self.assertAlmostEqual(bernoulli_kl(0.25, 0.5), expected)
        for p, q in ((0.1, 0.2), (0.4, 0.9), (0.75, 0.6)):
            self.assertAlmostEqual(bernoulli_kl(p, q), bernoulli_kl(1-p, 1-q))
        lower_side = [bernoulli_kl(0.4, q) for q in (0.01, 0.1, 0.2, 0.3, 0.4)]
        upper_side = [bernoulli_kl(0.4, q) for q in (0.4, 0.5, 0.6, 0.8, 0.99)]
        self.assertEqual(lower_side, sorted(lower_side, reverse=True))
        self.assertEqual(upper_side, sorted(upper_side))

    def test_invalid_probabilities(self):
        for value in (-0.01, 1.01, math.inf, -math.inf, math.nan, True, "0.5", None):
            for position in (0, 1):
                args = [0.5, 0.5]
                args[position] = value
                with self.subTest(value=value, position=position):
                    with self.assertRaises(ValueError):
                        bernoulli_kl(*args)


class CertificateNumericsTests(unittest.TestCase):
    def test_kl_all_successes_and_all_failures_exact_formulas(self):
        for n in (1, 2, 10, 1000, 1000000):
            for delta, claims in ((0.05, 1), (0.01, 4), (0.5, 100)):
                with self.subTest(n=n, delta=delta, claims=claims):
                    radius = math.log(2 * claims / delta) / n
                    failures = coverage_certificate(0, n, delta=delta, num_claims=claims)
                    successes = coverage_certificate(n, n, delta=delta, num_claims=claims)
                    self.assertEqual(failures.lower, 0.0)
                    self.assertEqual(successes.upper, 1.0)
                    self.assertAlmostEqual(failures.upper, -math.expm1(-radius), places=14)
                    self.assertAlmostEqual(successes.lower, math.exp(-radius), places=14)
                    self.assertAlmostEqual(failures.upper, 1-successes.lower, places=14)

    def test_hoeffding_formula_and_clipping(self):
        n, delta, claims = 1000, 0.05, 4
        epsilon = math.sqrt(math.log(2 * claims / delta) / (2*n))
        for k in (0, 1, 100, 500, 999, 1000):
            with self.subTest(successes=k):
                result = coverage_certificate(k, n, delta=delta, num_claims=claims,
                                              method="hoeffding")
                self.assertAlmostEqual(result.lower, max(0, k/n-epsilon), places=14)
                self.assertAlmostEqual(result.upper, min(1, k/n+epsilon), places=14)
        broad = coverage_certificate(0, 1, delta=0.001, method="hoeffding")
        self.assertEqual((broad.lower, broad.upper), (0.0, 1.0))

    def test_kl_inversion_solves_the_boundary(self):
        for k in (1, 50, 250, 500, 750, 950, 999):
            n, delta, claims = 1000, 0.05, 4
            with self.subTest(successes=k):
                result = coverage_certificate(k, n, delta=delta, num_claims=claims)
                radius = math.log(2 * claims / delta) / n
                self.assertLess(result.lower, k/n)
                self.assertGreater(result.upper, k/n)
                self.assertAlmostEqual(bernoulli_kl(k/n, result.lower), radius, places=12)
                self.assertAlmostEqual(bernoulli_kl(k/n, result.upper), radius, places=12)

    def test_kl_intervals_are_within_hoeffding_intervals(self):
        # Pinsker's inequality guarantees this containment analytically.
        for n in (1, 5, 25, 100):
            for k in range(n+1):
                for delta in (0.01, 0.2):
                    with self.subTest(n=n, k=k, delta=delta):
                        kl = coverage_certificate(k, n, delta=delta, num_claims=4)
                        hoeffding = coverage_certificate(k, n, delta=delta,
                                                         num_claims=4, method="hoeffding")
                        self.assertGreaterEqual(kl.lower + 1e-14, hoeffding.lower)
                        self.assertLessEqual(kl.upper, hoeffding.upper + 1e-14)
                        self.assertLessEqual(0.0, kl.lower)
                        self.assertLessEqual(kl.lower, k/n)
                        self.assertLessEqual(k/n, kl.upper)
                        self.assertLessEqual(kl.upper, 1.0)

    def test_confidence_and_family_size_monotonicity(self):
        for method in ("kl", "hoeffding"):
            for k in (0, 20, 50, 80, 100):
                with self.subTest(method=method, successes=k):
                    baseline = coverage_certificate(k, 100, method=method, delta=0.1)
                    more_confident = coverage_certificate(k, 100, method=method, delta=0.01)
                    more_claims = coverage_certificate(k, 100, method=method,
                                                      delta=0.1, num_claims=10)
                    self.assertLessEqual(more_confident.lower, baseline.lower)
                    self.assertGreaterEqual(more_confident.upper, baseline.upper)
                    self.assertLessEqual(more_claims.lower, baseline.lower)
                    self.assertGreaterEqual(more_claims.upper, baseline.upper)
                    self.assertAlmostEqual(more_confident.lower, more_claims.lower, places=14)
                    self.assertAlmostEqual(more_confident.upper, more_claims.upper, places=14)

    def test_more_samples_shrink_bounds_for_fixed_observed_proportion(self):
        for method in ("kl", "hoeffding"):
            for k in (0, 2, 5, 8, 10):
                small = coverage_certificate(k, 10, method=method)
                large = coverage_certificate(10*k, 100, method=method)
                self.assertGreaterEqual(large.lower, small.lower)
                self.assertLessEqual(large.upper, small.upper)

    def test_very_small_delta_stays_finite(self):
        for method in ("kl", "hoeffding"):
            for k in (0, 50, 100):
                result = coverage_certificate(k, 100, method=method,
                                              delta=1e-300, num_claims=1000000)
                self.assertTrue(math.isfinite(result.lower))
                self.assertTrue(math.isfinite(result.upper))
                self.assertLessEqual(result.lower, k/100)
                self.assertGreaterEqual(result.upper, k/100)


class ExactBinomialCoverageTests(unittest.TestCase):
    def test_noncoverage_probability_is_at_most_allocated_error(self):
        # For every possible observed count, compute its confidence interval;
        # sum the probability of all counts whose intervals exclude true p.
        # This verifies the probability statement itself, not just a formula.
        probabilities = (0.0, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5,
                         0.75, 0.9, 0.95, 0.99, 0.999, 1.0)
        for method in ("kl", "hoeffding"):
            for n in (1, 2, 5, 10, 25, 50, 100):
                for delta in (0.5, 0.1, 0.05, 0.001):
                    for claims in (1, 4, 100):
                        intervals = [coverage_certificate(k, n, method=method,
                                                           delta=delta, num_claims=claims)
                                     for k in range(n+1)]
                        for p in probabilities:
                            with self.subTest(method=method, n=n, p=p,
                                              delta=delta, claims=claims):
                                failure_probability = math.fsum(
                                    math.comb(n, k) * p**k * (1-p)**(n-k)
                                    for k, interval in enumerate(intervals)
                                    if p < interval.lower or p > interval.upper
                                )
                                self.assertLessEqual(failure_probability,
                                                     delta/claims + 2e-14)


class CertificateBehaviorTests(unittest.TestCase):
    def test_decisions_and_json_report(self):
        for method in ("kl", "hoeffding"):
            self.assertEqual(coverage_certificate(90, 100, method=method).status,
                             "not_requested")
            self.assertEqual(coverage_certificate(1000, 1000, method=method,
                                                 target_coverage=0.9).status,
                             "certified")
            self.assertEqual(coverage_certificate(0, 100, method=method,
                                                 target_coverage=0.9).status,
                             "below_target")
            result = coverage_certificate(90, 100, method=method, target_coverage=0.9)
            self.assertEqual(result.status, "inconclusive")
            report = json.loads(json.dumps(result.to_dict()))
            self.assertEqual(report["status"], result.status)
            self.assertEqual(report["successes"], 90)
            self.assertEqual(report["trials"], 100)
            self.assertEqual(report["empirical_coverage"], 0.9)
            self.assertEqual(report["method"], method)

    def test_decision_boundary_inclusion(self):
        for method in ("kl", "hoeffding"):
            result = coverage_certificate(60, 100, method=method)
            at_lower = coverage_certificate(60, 100, method=method,
                                             target_coverage=result.lower)
            at_upper = coverage_certificate(60, 100, method=method,
                                             target_coverage=result.upper)
            self.assertEqual(at_lower.status, "certified")
            self.assertEqual(at_upper.status, "inconclusive")

    def test_invalid_counts_and_claims(self):
        for invalid in (0, -1, True, 10.5, "10", None):
            with self.subTest(trials=invalid):
                with self.assertRaises(ValueError):
                    coverage_certificate(0, invalid)
            with self.subTest(num_claims=invalid):
                with self.assertRaises(ValueError):
                    coverage_certificate(0, 10, num_claims=invalid)
        for invalid in (-1, 11, True, 1.5, "1", None):
            with self.subTest(successes=invalid):
                with self.assertRaises(ValueError):
                    coverage_certificate(invalid, 10)

    def test_invalid_error_levels_targets_and_methods(self):
        for invalid in (0, 1, -0.1, 1.1, math.nan, math.inf, True, "0.1"):
            for name in ("delta", "target_coverage"):
                with self.subTest(parameter=name, value=invalid):
                    with self.assertRaises(ValueError):
                        coverage_certificate(5, 10, **{name: invalid})
        for method in (None, "KL", "chebyshev", 1):
            with self.subTest(method=method):
                with self.assertRaises(ValueError):
                    coverage_certificate(5, 10, method=method)


class SamplePlanningTests(unittest.TestCase):
    def test_size_is_sufficient_and_minimal_for_hoeffding_radius(self):
        for epsilon in (0.01, 0.05, 0.1, 0.3, 0.9):
            for delta in (0.01, 0.05, 0.5):
                for claims in (1, 4, 100):
                    with self.subTest(epsilon=epsilon, delta=delta, claims=claims):
                        n = hoeffding_sample_size(epsilon, delta=delta, num_claims=claims)
                        log_budget = math.log(2 * claims / delta)
                        self.assertIsInstance(n, int)
                        self.assertGreaterEqual(n, 1)
                        self.assertLessEqual(math.sqrt(log_budget/(2*n)), epsilon + 1e-14)
                        if n > 1:
                            self.assertGreater(math.sqrt(log_budget/(2*(n-1))), epsilon)

    def test_planning_monotonicity(self):
        baseline = hoeffding_sample_size(0.1)
        self.assertGreater(hoeffding_sample_size(0.05), baseline)
        self.assertGreater(hoeffding_sample_size(0.1, delta=0.01), baseline)
        self.assertGreater(hoeffding_sample_size(0.1, num_claims=4), baseline)

    def test_invalid_planning_inputs(self):
        for invalid in (0, 1, -0.1, 1.1, math.nan, math.inf, True, "0.1", None):
            for name in ("epsilon", "delta"):
                arguments = {"epsilon": 0.1, "delta": 0.05, name: invalid}
                with self.subTest(parameter=name, value=invalid):
                    with self.assertRaises(ValueError):
                        hoeffding_sample_size(**arguments)
        for invalid in (0, -1, True, 1.5, "1", None):
            with self.subTest(num_claims=invalid):
                with self.assertRaises(ValueError):
                    hoeffding_sample_size(0.1, num_claims=invalid)

    def test_unrepresentable_sample_plans_raise_clear_error(self):
        for epsilon in (1e-160, 1e-200):
            with self.subTest(epsilon=epsilon):
                with self.assertRaises(ValueError):
                    hoeffding_sample_size(epsilon)


if __name__ == "__main__":
    unittest.main()
