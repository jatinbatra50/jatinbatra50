"""Exercise notebook functions without running its expensive demo or sweep.

Only imports and function definitions are extracted from code cells. Tests
then run small real simulations and controlled coverage observations.
"""

import ast
import json
import os
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

from validation import CoverageCertificate


def load_notebook_functions():
    path = Path(__file__).resolve().parents[1] / "uq1.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace = {"__name__": "uq1_notebook_test"}
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]), filename=f"uq1.ipynb:{index}")
        tree.body = [
            node for node in tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))
        ]
        exec(compile(tree, filename=f"uq1.ipynb:{index}", mode="exec"), namespace)
    return namespace


class NotebookCoverageIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.namespace = load_notebook_functions()

    def setUp(self):
        self.random_state = np.random.get_state()
        self.coverage_test = self.namespace["coverage_test"]

    def tearDown(self):
        np.random.set_state(self.random_state)

    def test_default_float_and_certificate_use_identical_real_trials(self):
        arguments = dict(
            num_experiments=25,
            confidence=0.9,
            R=30,
            n=10,
            beta0=0.5,
            beta1=1.5,
            sigma=0.75,
            x_star=-0.5,
        )
        np.random.seed(8675309)
        empirical = self.coverage_test(**arguments)
        self.assertIsInstance(empirical, float)

        np.random.seed(8675309)
        certificate = self.coverage_test(
            **arguments,
            return_certificate=True,
            delta=0.02,
            num_claims=7,
            method="kl",
            target_coverage=0.85,
        )
        self.assertIsInstance(certificate, CoverageCertificate)
        self.assertEqual(certificate.empirical_coverage, empirical)
        self.assertEqual(certificate.successes / certificate.trials, empirical)
        self.assertIsInstance(certificate.successes, int)
        self.assertEqual(certificate.trials, arguments["num_experiments"])
        self.assertEqual(certificate.delta, 0.02)
        self.assertEqual(certificate.num_claims, 7)
        self.assertEqual(certificate.method, "kl")
        self.assertEqual(certificate.target_coverage, 0.85)

    def test_small_real_sweep_with_both_bound_methods(self):
        levels = [0.5, 0.8, 0.95]
        for method in ("kl", "hoeffding"):
            np.random.seed(20260918)
            for nominal in levels:
                with self.subTest(method=method, nominal=nominal):
                    result = self.coverage_test(
                        num_experiments=15,
                        confidence=nominal,
                        R=12,
                        n=8,
                        return_certificate=True,
                        delta=0.1,
                        num_claims=len(levels),
                        method=method,
                        target_coverage=nominal,
                    )
                    self.assertIsInstance(result, CoverageCertificate)
                    self.assertEqual(result.trials, 15)
                    self.assertEqual(result.num_claims, len(levels))
                    self.assertEqual(result.delta, 0.1)
                    self.assertEqual(result.method, method)
                    self.assertEqual(result.target_coverage, nominal)
                    self.assertGreaterEqual(result.lower, 0)
                    self.assertLessEqual(result.lower, result.empirical_coverage)
                    self.assertLessEqual(result.empirical_coverage, result.upper)
                    self.assertLessEqual(result.upper, 1)
                    self.assertIn(
                        result.status, {"certified", "below_target", "inconclusive"}
                    )

    def test_exact_counts_include_both_interval_endpoints(self):
        # There are four hits, including -1 and 1, among six independent
        # observations. R is deliberately unrelated to the trial count.
        for method in ("kl", "hoeffding"):
            with self.subTest(method=method):
                fixed_interval = Mock(
                    return_value=(-1.0, 0.0, 1.0, np.array([-1.0, 1.0]))
                )
                with patch.dict(
                    self.namespace,
                    {"monte_carlo_prediction_interval": fixed_interval},
                ), patch.object(
                    np.random, "normal", side_effect=[-1, 0, 0.5, 1, 2, -2]
                ) as future_draw:
                    result = self.coverage_test(
                        num_experiments=6,
                        R=137,
                        beta0=0,
                        beta1=0,
                        return_certificate=True,
                        method=method,
                    )
                self.assertIsInstance(result, CoverageCertificate)
                self.assertEqual(result.successes, 4)
                self.assertEqual(result.trials, 6)
                self.assertEqual(result.empirical_coverage, 4 / 6)
                self.assertEqual(result.status, "not_requested")
                self.assertEqual(future_draw.call_count, 6)
                self.assertEqual(fixed_interval.call_count, 6)
                self.assertEqual(fixed_interval.call_args.kwargs["R"], 137)

    def test_invalid_trial_counts_fail_before_sampling(self):
        invalid_counts = (0, -1, 1.5, np.float64(3), True, np.bool_(True), "3", None)
        for count in invalid_counts:
            with self.subTest(count=count):
                with patch.object(np.random, "normal") as draw:
                    with self.assertRaises(ValueError):
                        self.coverage_test(num_experiments=count, return_certificate=True)
                    draw.assert_not_called()

    def test_numpy_integer_trial_count_is_supported(self):
        np.random.seed(17)
        result = self.coverage_test(
            num_experiments=np.int64(3),
            R=10,
            n=8,
            return_certificate=True,
        )
        self.assertIsInstance(result, CoverageCertificate)
        self.assertEqual(result.trials, 3)


if __name__ == "__main__":
    unittest.main()
