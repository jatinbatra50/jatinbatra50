"""Deterministic checks of the simulation and independent audit draw protocol."""

import math
import unittest

import numpy as np

from simulation import (
    audit_fixed_interval,
    audit_procedure,
    gaussian_interval_coverage,
    monte_carlo_samples,
)


def reference_samples(rng, *, R, n, beta0, beta1, sigma, x_star):
    """Independent row-wise least-squares calculation with the same draw order."""
    covariates = rng.normal(0.0, 1.0, size=(R, n))
    errors = rng.normal(0.0, sigma, size=(R, n))
    future_errors = rng.normal(0.0, sigma, size=R)
    predictions = []
    for x, error, future_error in zip(covariates, errors, future_errors):
        y = beta0 + beta1*x + error
        design = np.column_stack((np.ones(n), x))
        intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
        predictions.append(intercept + slope*x_star + future_error)
    return np.asarray(predictions)


class SimulationProtocolTests(unittest.TestCase):
    def test_vectorized_ols_matches_row_wise_least_squares(self):
        for n, R in ((2, 11), (7, 19), (50, 8)):
            with self.subTest(n=n, R=R):
                settings = dict(R=R, n=n, beta0=-1.5, beta1=0.7,
                                sigma=1.2, x_star=2.1)
                actual_rng = np.random.default_rng(827)
                reference_rng = np.random.default_rng(827)
                actual = monte_carlo_samples(actual_rng, **settings)
                expected = reference_samples(reference_rng, **settings)
                np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
                self.assertEqual(actual.shape, (R,))
                # Both computations consume every training and future-noise draw.
                self.assertEqual(actual_rng.random(), reference_rng.random())

    def test_procedure_reconstructs_fresh_cloud_and_future_in_each_trial(self):
        levels = (0.95, 0.6, 0.8)
        settings = dict(R=17, n=6, beta0=2.0, beta1=-0.5, sigma=0.8, x_star=1.4)
        trials, delta, seed = 23, 0.07, 2811
        rng = np.random.default_rng(seed)
        results = audit_procedure(rng, trials=trials, nominal_levels=levels,
                                  delta=delta, method="hoeffding", **settings)
        reference_rng = np.random.default_rng(seed)
        successes = [0] * len(levels)
        for _ in range(trials):
            cloud = reference_samples(reference_rng, **settings)
            future = reference_rng.normal(
                settings["beta0"] + settings["beta1"]*settings["x_star"],
                settings["sigma"],
            )
            for index, level in enumerate(levels):
                tail = (1-level)/2
                lower, upper = np.quantile(cloud, [tail, 1-tail])
                successes[index] += int(lower <= future <= upper)
        self.assertEqual(rng.random(), reference_rng.random())
        self.assertEqual([c.successes for c in results], successes)
        for level, certificate in zip(levels, results):
            self.assertIs(type(certificate.successes), int)
            self.assertEqual(certificate.trials, trials)
            self.assertEqual(certificate.num_claims, len(levels))
            self.assertEqual(certificate.delta, delta)
            self.assertEqual(certificate.method, "hoeffding")
            self.assertEqual(certificate.target_coverage, level)

    def test_fixed_interval_audit_matches_fresh_draws_across_chunks(self):
        seed, trials = 426, 8203  # More than two internal chunks of 4096.
        actual_rng = np.random.default_rng(seed)
        reference_rng = np.random.default_rng(seed)
        lower, upper, mean, sigma = -1.0, 1.5, 0.3, 1.2
        future = reference_rng.normal(mean, sigma, size=trials)
        successes = int(np.count_nonzero((lower <= future) & (future <= upper)))
        certificate = audit_fixed_interval(
            actual_rng, lower=lower, upper=upper, trials=trials, beta0=mean,
            beta1=0, sigma=sigma, delta=0.02, target_coverage=0.8,
        )
        self.assertEqual(certificate.successes, successes)
        self.assertIs(type(certificate.successes), int)
        self.assertEqual(certificate.trials, trials)
        self.assertEqual(certificate.num_claims, 1)
        self.assertEqual(certificate.delta, 0.02)
        self.assertEqual(certificate.target_coverage, 0.8)
        self.assertEqual(actual_rng.random(), reference_rng.random())

    def test_reproducibility_and_global_rng_is_untouched(self):
        global_before = np.random.get_state()
        first = audit_procedure(np.random.default_rng(101), trials=7, R=13, n=5)
        second = audit_procedure(np.random.default_rng(101), trials=7, R=13, n=5)
        self.assertEqual(first, second)
        a = monte_carlo_samples(np.random.default_rng(121), R=17, n=6)
        b = monte_carlo_samples(np.random.default_rng(121), R=17, n=6)
        np.testing.assert_array_equal(a, b)
        audit_fixed_interval(np.random.default_rng(77), lower=1, upper=5, trials=17)
        global_after = np.random.get_state()
        self.assertEqual(global_before[0], global_after[0])
        np.testing.assert_array_equal(global_before[1], global_after[1])
        self.assertEqual(global_before[2:], global_after[2:])

    def test_noiseless_model_is_exact_in_all_three_simulators(self):
        model = dict(beta0=0.1, beta1=0.2, sigma=0, x_star=0.3)
        mean = model["beta0"] + model["beta1"]*model["x_star"]
        samples = monte_carlo_samples(np.random.default_rng(5), R=11, n=3, **model)
        np.testing.assert_array_equal(samples, np.full(11, mean))
        certificates = audit_procedure(
            np.random.default_rng(5), trials=9, R=11, n=3, **model
        )
        self.assertEqual([c.successes for c in certificates], [9]*4)
        for lower, upper, expected in ((mean, mean, 9), (mean+1, mean+2, 0)):
            fixed = audit_fixed_interval(np.random.default_rng(5), lower=lower,
                                          upper=upper, trials=9, **model)
            self.assertEqual(fixed.successes, expected)


class GaussianCoverageTests(unittest.TestCase):
    def test_far_tail_mass_avoids_cdf_cancellation(self):
        # Independent Gaussian quadrature integrates the density directly.
        nodes, weights = np.polynomial.legendre.leggauss(32)
        x = 10.5 + 0.5*nodes
        expected = 0.5*np.sum(weights*np.exp(-0.5*x*x)/math.sqrt(2*math.pi))
        actual = gaussian_interval_coverage(10, 11, mean=0, sigma=1)
        reflected = gaussian_interval_coverage(-11, -10, mean=0, sigma=1)
        translated = gaussian_interval_coverage(37, 40, mean=7, sigma=3)
        self.assertGreater(actual, 0)
        self.assertAlmostEqual(actual/expected, 1.0, places=12)
        self.assertEqual(actual, reflected)
        self.assertEqual(actual, translated)

    def test_known_gaussian_masses_and_infinite_endpoints(self):
        self.assertEqual(gaussian_interval_coverage(-math.inf, math.inf, mean=2, sigma=3), 1)
        self.assertEqual(gaussian_interval_coverage(-math.inf, 2, mean=2, sigma=3), 0.5)
        self.assertEqual(gaussian_interval_coverage(2, math.inf, mean=2, sigma=3), 0.5)
        self.assertEqual(gaussian_interval_coverage(2, 2, mean=2, sigma=3), 0)
        for scale in (0.5, 1, 2, 3):
            with self.subTest(scale=scale):
                actual = gaussian_interval_coverage(2-3*scale, 2+3*scale, mean=2, sigma=3)
                self.assertAlmostEqual(actual, math.erf(scale/math.sqrt(2)), places=14)

    def test_point_mass_includes_endpoints(self):
        for lower, upper, expected in ((1, 1, 1), (0, 1, 1), (1, 2, 1),
                                        (-math.inf, math.inf, 1), (2, 3, 0), (-1, 0, 0)):
            with self.subTest(lower=lower, upper=upper):
                self.assertEqual(gaussian_interval_coverage(lower, upper, mean=1, sigma=0),
                                 expected)


class SimulationValidationTests(unittest.TestCase):
    def test_invalid_generators_and_integer_parameters(self):
        for invalid in (None, 42, np.random.RandomState(1)):
            with self.subTest(rng=invalid):
                with self.assertRaises(ValueError):
                    monte_carlo_samples(invalid, R=5, n=3)
                with self.assertRaises(ValueError):
                    audit_procedure(invalid, trials=5, R=5, n=3)
                with self.assertRaises(ValueError):
                    audit_fixed_interval(invalid, lower=0, upper=1, trials=5)
        for name, values in (("R", (0, -1, True, np.bool_(True), 2.5)),
                             ("n", (0, 1, True, 2.5))):
            for invalid in values:
                with self.subTest(parameter=name, value=invalid):
                    with self.assertRaises(ValueError):
                        monte_carlo_samples(np.random.default_rng(1),
                                            **{**dict(R=5, n=3), name: invalid})
        for invalid in (0, -1, True, 2.5):
            with self.subTest(trials=invalid):
                with self.assertRaises(ValueError):
                    audit_procedure(np.random.default_rng(1), trials=invalid, R=5, n=3)
                with self.assertRaises(ValueError):
                    audit_fixed_interval(np.random.default_rng(1), lower=0, upper=1,
                                         trials=invalid)

    def test_invalid_levels_models_and_certificate_settings(self):
        for invalid in ((), (0.8, 0.8), (0,), (1,), (math.nan,), (True,), 0.9, None):
            with self.subTest(levels=invalid):
                with self.assertRaises(ValueError):
                    audit_procedure(np.random.default_rng(1), trials=2, R=5, n=3,
                                    nominal_levels=invalid)
        for settings in (dict(sigma=-1), dict(sigma=math.nan), dict(beta0=math.inf),
                         dict(beta1=True), dict(x_star="1"),
                         dict(beta0=1e308, beta1=1e308, x_star=2)):
            with self.subTest(model=settings):
                with self.assertRaises(ValueError):
                    monte_carlo_samples(np.random.default_rng(1), R=5, n=3, **settings)
        for settings in (dict(delta=0), dict(delta=1), dict(method="normal")):
            rng, untouched = np.random.default_rng(1), np.random.default_rng(1)
            with self.subTest(settings=settings):
                with self.assertRaises(ValueError):
                    audit_procedure(rng, trials=2, R=5, n=3, **settings)
                # Invalid requests must fail before any expensive simulation.
                self.assertEqual(rng.random(), untouched.random())

    def test_invalid_intervals_and_gaussian_parameters(self):
        for lower, upper in ((2, 1), (math.nan, 1), (0, math.nan), (True, 1), (0, "1")):
            with self.subTest(lower=lower, upper=upper):
                with self.assertRaises(ValueError):
                    gaussian_interval_coverage(lower, upper, mean=0, sigma=1)
                with self.assertRaises(ValueError):
                    audit_fixed_interval(np.random.default_rng(1), lower=lower,
                                         upper=upper, trials=5)
        for settings in (dict(mean=math.inf, sigma=1), dict(mean=0, sigma=-1),
                         dict(mean=0, sigma=math.nan)):
            with self.subTest(settings=settings):
                with self.assertRaises(ValueError):
                    gaussian_interval_coverage(0, 1, **settings)


if __name__ == "__main__":
    unittest.main()
