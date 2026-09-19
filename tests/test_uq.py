"""Check the Bayesian regression calculation and independent coverage measurement."""

from contextlib import redirect_stdout
import io
import json
from math import comb, floor, log, sqrt
from pathlib import Path
from statistics import NormalDist
import tempfile
import unittest

import numpy as np

from experiments import main
from uq import fit_posterior, hoeffding_lower, hoeffding_margin, posterior_predictive, run_example


class TutorialTests(unittest.TestCase):
    def test_posterior_matches_a_hand_calculation_with_correlated_weights(self):
        # Phi = [[1,0],[1,1]]; precision = [[3,1],[1,2]].
        # Its inverse is [[2,-1],[-1,3]]/5 and posterior mean is [1,1].
        posterior = fit_posterior([0, 1], [1, 3], alpha=1, noise_sd=1)
        np.testing.assert_allclose(posterior["covariance"], [[0.4, -0.2], [-0.2, 0.6]])
        np.testing.assert_allclose(posterior["mean"], [1, 1])
        prediction = posterior_predictive(0.5, posterior)
        self.assertAlmostEqual(prediction["mean"], 1.5)
        self.assertAlmostEqual(prediction["mean_sd"]**2, 0.35)

    def test_predictive_interval_adds_observation_noise(self):
        posterior = fit_posterior([-1, 0, 1], [-1, 1, 3])
        prediction = posterior_predictive(np.array([-1, 0, 1]), posterior)
        np.testing.assert_allclose(prediction["predictive_sd"]**2,
                                   0.25 + prediction["mean_sd"]**2)
        half_width = (prediction["upper"] - prediction["lower"]) / 2
        np.testing.assert_allclose(half_width / prediction["predictive_sd"],
                                   NormalDist().inv_cdf(0.995))
        self.assertTrue(np.all(prediction["predictive_sd"] > prediction["mean_sd"]))
        self.assertTrue(np.all(prediction["lower"] < prediction["mean_lower"]))
        self.assertTrue(np.all(prediction["upper"] > prediction["mean_upper"]))

    def test_gaussian_prior_keeps_collinear_training_fit_well_defined(self):
        posterior = fit_posterior([0, 0, 0], [1, 1, 1])
        np.testing.assert_allclose(posterior["mean"], [12 / 12.25, 0])
        np.testing.assert_allclose(posterior["covariance"], [[1 / 12.25, 0], [0, 4]])

    def test_hoeffding_lower_bound_has_exact_binomial_error_at_most_delta(self):
        self.assertAlmostEqual(hoeffding_margin(10000), sqrt(log(20) / 20000))
        for n in (10, 25, 100):
            for p in (0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0):
                error = sum(comb(n, k) * p**k * (1 - p)**(n - k)
                            for k in range(n + 1) if hoeffding_lower(k, n) > p)
                self.assertLessEqual(error, 0.05 + 1e-14)

    def test_separate_streams_keep_fitted_rule_fixed_when_test_size_changes(self):
        first, data = run_example()
        repeated, repeated_data = run_example()
        small, small_data = run_example(test_size=100)
        self.assertEqual(first, repeated)
        self.assertEqual(first["posterior"], small["posterior"])
        self.assertEqual(first["prediction"], small["prediction"])
        for key in ("train_x", "train_y"):
            np.testing.assert_array_equal(data[key], small_data[key])
        for key in data:
            np.testing.assert_array_equal(data[key], repeated_data[key])
        train_stream, test_stream = np.random.SeedSequence(20260919).spawn(2)
        for stream, prefix, size in ((train_stream, "train", 20), (test_stream, "test", 10000)):
            rng = np.random.default_rng(stream)
            expected_x = rng.uniform(-1, 1, size)
            expected_y = 1 + 2 * expected_x + rng.normal(0, 0.5, size)
            np.testing.assert_array_equal(data[prefix + "_x"], expected_x)
            np.testing.assert_array_equal(data[prefix + "_y"], expected_y)

    def test_report_counts_each_input_specific_interval(self):
        report, data = run_example()
        prediction = posterior_predictive(data["test_x"], report["posterior"])
        expected = (prediction["lower"] <= data["test_y"]) & (data["test_y"] <= prediction["upper"])
        measured = report["validation"]
        np.testing.assert_array_equal(data["covered"], expected)
        self.assertEqual(measured["hits"], int(expected.sum()))
        self.assertEqual(measured["size"], 10000)
        self.assertEqual(measured["estimate"], measured["hits"] / 10000)
        self.assertAlmostEqual(measured["lower_bound"], measured["estimate"] - measured["margin"])
        self.assertEqual(measured["confidence"], 0.95)
        json.dumps(report, allow_nan=False)

    def test_invalid_inputs_and_zero_hits(self):
        self.assertEqual(hoeffding_lower(0, 100), 0)
        for n in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                run_example(test_size=n)
        for delta in (0, 1, -0.1, float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                hoeffding_margin(100, delta)
        for hits in (-1, 101, True, 2.5):
            with self.assertRaises(ValueError):
                hoeffding_lower(hits, 100)
        for x, y in (([], []), ([[1]], [1]), ([0, 1], [1]), ([float("nan")], [1])):
            with self.assertRaises(ValueError):
                fit_posterior(x, y)
        for kwargs in ({"alpha": 0}, {"noise_sd": 0}):
            with self.assertRaises(ValueError):
                fit_posterior([0], [1], **kwargs)
        with self.assertRaises(ValueError):
            posterior_predictive(float("inf"), fit_posterior([0], [1]))

    def test_cli_writes_the_report_and_one_figure(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()) as output:
            report = main(["--test-size", "250", "--output-dir", folder])
            self.assertEqual(json.loads((Path(folder) / "report.json").read_text()), report)
            self.assertEqual({p.name for p in Path(folder).iterdir()},
                             {"report.json", "regression-interval.png", "regression-interval.svg"})
            for extension in ("png", "svg"):
                self.assertGreater((Path(folder) / f"regression-interval.{extension}").stat().st_size, 1000)
            self.assertIn("At 95% confidence", output.getvalue())
            minimum_percent = floor(1000 * report["validation"]["lower_bound"]) / 10
            self.assertIn(f"at least {minimum_percent:.1f}%", output.getvalue())


if __name__ == "__main__":
    unittest.main()
