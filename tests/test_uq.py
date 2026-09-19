"""Numerical checks for the tutorial's learning steps and uncertainty statements."""

from contextlib import redirect_stdout
import io
import json
from math import comb
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments import main
from uq import (conformal_radius, coverage_interval, fit_line, hoeffding_radius,
                measurement_budget, predict_line, run_demo, sample_sensor)


class TutorialTests(unittest.TestCase):
    def test_sample_plan_meets_requested_precision(self):
        n = measurement_budget(0.005, 0.05)
        self.assertEqual(n, 73778)
        self.assertLessEqual(hoeffding_radius(n), 0.005)
        self.assertGreater(hoeffding_radius(n - 1), 0.005)

    def test_halving_error_needs_four_times_as_many_readings(self):
        coarse = measurement_budget(0.02)
        fine = measurement_budget(0.01)
        self.assertLessEqual(abs(fine - 4 * coarse), 3)
        self.assertGreater(measurement_budget(0.02, 0.01), coarse)

    def test_ols_recovers_a_known_line(self):
        x = np.array([-2, -1, 0, 1, 2], dtype=float)
        coefs = fit_line(x, 3 - 4 * x)
        np.testing.assert_allclose(coefs, [3, -4], atol=1e-12)
        np.testing.assert_allclose(predict_line(coefs, [0, 5]), [3, -17], atol=1e-12)

    def test_sensor_distribution_and_reproducibility(self):
        first = sample_sensor(np.random.default_rng(17), 20000)
        second = sample_sensor(np.random.default_rng(17), 20000)
        for x, y in zip(first, second):
            np.testing.assert_array_equal(x, y)
        x, y = first
        self.assertTrue(np.all((-1 <= x) & (x <= 1)))
        self.assertAlmostEqual(float(np.std(y - (1 + 2 * x))), 0.5, delta=0.015)

    def test_conformal_uses_finite_sample_rank(self):
        residuals = [0.4, 0.1, 0.5, 0.2, 0.3]
        self.assertEqual(conformal_radius(residuals, 0.25), 0.5)
        self.assertEqual(conformal_radius([0, 0, 1, 1], 0.4), 1)
        self.assertTrue(np.isinf(conformal_radius(residuals, 0.01)))

    def test_conformal_coverage_over_every_possible_new_rank(self):
        # A future score has uniform rank under exchangeability and no ties.
        for m in (4, 9, 19):
            scores = np.arange(m + 1, dtype=float)
            for alpha in (0.025, 0.1, 0.25, 0.5):
                covered = sum(
                    scores[i] <= conformal_radius(np.delete(scores, i), alpha)
                    for i in range(m + 1)
                )
                self.assertGreaterEqual(covered / (m + 1) + 1e-15, 1 - alpha)

    def test_coverage_interval_and_endpoint_clipping(self):
        flags = np.array([True] * 90 + [False] * 10)
        mean, lower, upper = coverage_interval(flags)
        self.assertAlmostEqual(mean, 0.9)
        self.assertAlmostEqual(lower, 0.9 - hoeffding_radius(100))
        self.assertEqual(upper, 1)
        self.assertEqual(coverage_interval(np.zeros(100, dtype=bool))[1], 0)

    def test_hoeffding_contains_binomial_parameter_at_claimed_rate(self):
        for n in (10, 25, 50):
            radius = hoeffding_radius(n, delta=0.05)
            for p in (0.01, 0.1, 0.5, 0.9, 0.99):
                failure = sum(comb(n, k) * p**k * (1 - p)**(n - k)
                              for k in range(n + 1) if abs(k / n - p) > radius)
                self.assertLessEqual(failure, 0.05 + 1e-14)

    def test_demo_is_reproducible_and_reports_measured_values(self):
        first, data = run_demo()
        second, _ = run_demo()
        self.assertEqual(first, second)
        self.assertEqual(first["measurement"]["size"], 73778)
        self.assertEqual(first["calibration"]["rank"], 3901)
        estimate, lower, upper = coverage_interval(data["covered"])
        self.assertEqual((estimate, lower, upper), tuple(
            first["measurement"][key] for key in ("estimate", "lower", "upper")))
        self.assertTrue(0.96 < estimate < 0.99)
        json.dumps(first, allow_nan=False)

    def test_test_precision_does_not_change_training_or_calibration(self):
        fine, data_fine = run_demo(epsilon=0.005)
        coarse, data_coarse = run_demo(epsilon=0.02)
        self.assertEqual(fine["fit"], coarse["fit"])
        self.assertEqual(fine["calibration"], coarse["calibration"])
        np.testing.assert_array_equal(data_fine["train_y"], data_coarse["train_y"])
        np.testing.assert_array_equal(data_fine["calibration_y"], data_coarse["calibration_y"])

    def test_invalid_inputs_are_rejected(self):
        for value in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                hoeffding_radius(value)
        for value in (0, 1, -0.2, float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                measurement_budget(value)
            with self.assertRaises(ValueError):
                hoeffding_radius(10, value)
        for flags in ([], [0, 1], [[True, False]]):
            with self.assertRaises(ValueError):
                coverage_interval(flags)
        with self.assertRaises(ValueError):
            conformal_radius([-1, 2], 0.1)
        with self.assertRaises(ValueError):
            fit_line([1, 1], [2, 3])

    def test_cli_writes_readable_report_and_figures(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            report = main(["--epsilon", ".02", "--output-dir", folder])
            self.assertEqual(json.loads((Path(folder) / "report.json").read_text()), report)
            for stem in ("prediction-interval", "measurement-budget"):
                for suffix in ("png", "svg"):
                    self.assertGreater((Path(folder) / f"{stem}.{suffix}").stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
