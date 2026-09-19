"""Checks of the Bayesian calculation and the independent coverage measurement."""

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
from uq import hoeffding_lower, hoeffding_margin, posterior_predictive, run_example


class TutorialTests(unittest.TestCase):
    def test_posterior_matches_a_hand_calculation(self):
        # Prior variance 4, noise variance 4, two observations summing to 10:
        # posterior variance = 1/(1/4+2/4) = 4/3; mean = (4/3)*(10/4).
        prediction = posterior_predictive([4, 6], prior_mean=0, prior_sd=2, noise_sd=2)
        self.assertAlmostEqual(prediction["posterior_mean"], 10 / 3)
        self.assertAlmostEqual(prediction["posterior_sd"]**2, 4 / 3)

    def test_predictive_range_includes_new_bottle_noise(self):
        prediction = posterior_predictive([100.5] * 20)
        self.assertAlmostEqual(prediction["predictive_sd"]**2,
                               4 + prediction["posterior_sd"]**2)
        half_width = (prediction["upper"] - prediction["lower"]) / 2
        self.assertAlmostEqual(half_width / prediction["predictive_sd"],
                               NormalDist().inv_cdf(0.995))
        self.assertGreater(prediction["predictive_sd"], prediction["posterior_sd"])

    def test_hoeffding_lower_bound_has_exact_binomial_error_at_most_delta(self):
        self.assertAlmostEqual(hoeffding_margin(10000), sqrt(log(20) / 20000))
        for n in (10, 25, 100):
            for p in (0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0):
                # Exactly sum the probability of reporting a lower bound above p.
                error = sum(comb(n, k) * p**k * (1 - p)**(n - k)
                            for k in range(n + 1) if hoeffding_lower(k, n) > p)
                self.assertLessEqual(error, 0.05 + 1e-14)

    def test_separate_streams_keep_prediction_fixed_when_test_size_changes(self):
        first, data = run_example()
        repeated, repeated_data = run_example()
        small, small_data = run_example(test_size=100)
        self.assertEqual(first, repeated)
        self.assertEqual(first["prediction"], small["prediction"])
        np.testing.assert_array_equal(data["training"], small_data["training"])
        np.testing.assert_array_equal(data["test"][:100], small_data["test"])
        np.testing.assert_array_equal(data["test"], repeated_data["test"])
        train_stream, test_stream = np.random.SeedSequence(20260919).spawn(2)
        np.testing.assert_array_equal(data["training"],
                                      np.random.default_rng(train_stream).normal(100.5, 2, 20))
        np.testing.assert_array_equal(data["test"],
                                      np.random.default_rng(test_stream).normal(100.5, 2, 10000))

    def test_report_counts_the_frozen_range_and_uses_all_test_bottles(self):
        report, data = run_example()
        prediction, measured = report["prediction"], report["validation"]
        expected = (prediction["lower"] <= data["test"]) & (data["test"] <= prediction["upper"])
        np.testing.assert_array_equal(data["covered"], expected)
        self.assertEqual(measured["hits"], int(expected.sum()))
        self.assertEqual(measured["size"], 10000)
        self.assertEqual(measured["estimate"], measured["hits"] / 10000)
        self.assertAlmostEqual(measured["lower_bound"],
                               measured["estimate"] - measured["margin"])
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
        for training in ([], [[1, 2]], [float("nan")]):
            with self.assertRaises(ValueError):
                posterior_predictive(training)
        with self.assertRaises(ValueError):
            posterior_predictive([100], noise_sd=0)

    def test_cli_writes_the_report_and_one_figure(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()) as output:
            report = main(["--test-size", "250", "--output-dir", folder])
            self.assertEqual(json.loads((Path(folder) / "report.json").read_text()), report)
            self.assertEqual({p.name for p in Path(folder).iterdir()},
                             {"report.json", "bottle-interval.png", "bottle-interval.svg"})
            for extension in ("png", "svg"):
                self.assertGreater((Path(folder) / f"bottle-interval.{extension}").stat().st_size, 1000)
            self.assertIn("At 95% confidence", output.getvalue())
            minimum_percent = floor(1000 * report["validation"]["lower_bound"]) / 10
            self.assertIn(f"at least {minimum_percent:.1f}%", output.getvalue())


if __name__ == "__main__":
    unittest.main()
