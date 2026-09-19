"""Check the model proposal and model-independent marginal coverage validation."""

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
    def test_posterior_matches_hand_calculation_and_handles_collinear_inputs(self):
        # Phi = [[1,0],[1,1]], precision = [[3,1],[1,2]], posterior mean = [1,1].
        posterior = fit_posterior([0, 1], [1, 3], alpha=1, noise_sd=1)
        np.testing.assert_allclose(posterior["covariance"], [[0.4, -0.2], [-0.2, 0.6]])
        np.testing.assert_allclose(posterior["mean"], [1, 1])
        self.assertAlmostEqual(posterior_predictive(0.5, posterior)["mean"], 1.5)
        collinear = fit_posterior([0, 0, 0], [1, 1, 1])
        np.testing.assert_allclose(collinear["mean"], [12 / 12.25, 0])
        np.testing.assert_allclose(collinear["covariance"], [[1 / 12.25, 0], [0, 4]])

    def test_predictive_interval_adds_new_observation_noise(self):
        posterior = fit_posterior([0, 1], [1, 3], alpha=1, noise_sd=1)
        prediction = posterior_predictive(0.5, posterior)
        # [1, .5] S [1, .5]^T = .35; observation variance = 1.
        self.assertAlmostEqual(prediction["predictive_sd"]**2, 1.35)
        half_width = (prediction["upper"] - prediction["lower"]) / 2
        self.assertAlmostEqual(half_width / prediction["predictive_sd"], NormalDist().inv_cdf(0.995))
        self.assertEqual(set(prediction), {"mean", "predictive_sd", "lower", "upper", "nominal_level"})

    def test_independent_validation_pairs_follow_the_future_joint_law(self):
        report, data = run_example()
        small, small_data = run_example(test_size=100)
        self.assertEqual(report["posterior"], small["posterior"])
        self.assertEqual(report["prediction"], small["prediction"])
        for key in ("train_x", "train_y"):
            np.testing.assert_array_equal(data[key], small_data[key])
        np.testing.assert_array_equal(data["test_x"][:100], small_data["test_x"])
        train_stream, test_stream = np.random.SeedSequence(20260919).spawn(2)
        train_rng, test_rng = np.random.default_rng(train_stream), np.random.default_rng(test_stream)
        train_x = train_rng.uniform(-1, 1, 20)
        train_y = 1 + 2 * train_x + 2 * train_x**2 + train_rng.normal(0, 0.5, 20)
        np.testing.assert_array_equal(data["train_x"], train_x)
        np.testing.assert_array_equal(data["train_y"], train_y)
        test_x = test_rng.uniform(-1, 1, 10000)
        test_y = 1 + 2 * test_x + 2 * test_x**2 + test_rng.normal(0, 0.5, 10000)
        np.testing.assert_array_equal(data["test_x"], test_x)
        np.testing.assert_array_equal(data["test_y"], test_y)

    def test_each_held_out_response_uses_the_interval_at_its_own_input(self):
        report, data = run_example()
        prediction = posterior_predictive(data["test_x"], report["posterior"])
        measured = report["validation"]
        covered = (prediction["lower"] <= data["test_y"]) & (data["test_y"] <= prediction["upper"])
        np.testing.assert_array_equal(data["covered"], covered)
        np.testing.assert_allclose(report["posterior"]["mean"], [1.7212065493873854, 2.26961763878656])
        self.assertEqual(measured["hits"], int(covered.sum()))
        self.assertEqual(measured["hits"], 9170)
        self.assertEqual(measured["size"], 10000)
        self.assertAlmostEqual(measured["lower_bound"], 0.904761265846596)
        self.assertEqual(measured["confidence"], 0.95)
        self.assertEqual(report["schema_version"], 6)
        self.assertIn("One frozen prediction rule", measured["confidence_scope"])
        self.assertIn("Marginal coverage", measured["target"])
        self.assertFalse(measured["per_input_guarantee"])
        self.assertEqual(report["prediction"]["nominal_level"], 0.99)
        json.dumps(report, allow_nan=False)

    def test_hoeffding_lower_bound_has_exact_binomial_error_at_most_delta(self):
        self.assertAlmostEqual(hoeffding_margin(10000), sqrt(log(20) / 20000))
        for n in (10, 25, 100):
            for p in (0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0):
                error = sum(comb(n, k) * p**k * (1 - p)**(n - k)
                            for k in range(n + 1) if hoeffding_lower(k, n) > p)
                self.assertLessEqual(error, 0.05 + 1e-14)

    def test_bound_uses_only_coverage_indicators_even_for_wrong_models(self):
        # Deliberately non-Gaussian responses; every proposal was fixed first.
        responses = np.array([-3, -1, 0, 0, 0, 0, 0, 1, 8, 100])
        for proposal in ((-1, 1), (-1000, -900), (-1000, 1000)):
            hits = int(((proposal[0] <= responses) & (responses <= proposal[1])).sum())
            expected = max(0, hits / len(responses) - sqrt(log(20) / (2 * len(responses))))
            self.assertEqual(hoeffding_lower(hits, len(responses)), expected)
        report, _ = run_example()
        self.assertNotEqual(report["model"]["actual_mean_function"],
                            report["model"]["assumed_mean_function"])

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

    def test_cli_writes_report_and_marginal_coverage_figure(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()) as output:
            report = main(["--test-size", "250", "--output-dir", folder])
            self.assertEqual(json.loads((Path(folder) / "report.json").read_text()), report)
            self.assertEqual({p.name for p in Path(folder).iterdir()},
                             {"report.json", "marginal-validation.png", "marginal-validation.svg"})
            for extension in ("png", "svg"):
                self.assertGreater((Path(folder) / f"marginal-validation.{extension}").stat().st_size, 1000)
            minimum_percent = floor(1000 * report["validation"]["lower_bound"]) / 10
            self.assertIn(f"at least {minimum_percent:.1f}%", output.getvalue())
            self.assertIn("not a per-input guarantee", output.getvalue())


if __name__ == "__main__":
    unittest.main()
