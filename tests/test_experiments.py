"""Small end-to-end CLI runs, report accounting, and exported figure checks."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np

import experiments
from simulation import audit_fixed_interval, audit_procedure, monte_carlo_samples


REPOSITORY = Path(__file__).resolve().parents[1]


class ExperimentCLITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.output_dir = Path(cls.temporary.name) / "audit"
        cls.process = subprocess.run(
            [sys.executable, str(REPOSITORY / "experiments.py"),
             "--trials", "8", "--mc-samples", "15", "--training-samples", "6",
             "--seed", "912", "--delta", "0.08", "--output-dir", str(cls.output_dir)],
            cwd=REPOSITORY, capture_output=True, text=True, timeout=60,
        )
        if cls.process.returncode:
            raise AssertionError(f"CLI failed:\n{cls.process.stdout}\n{cls.process.stderr}")
        cls.report = json.loads((cls.output_dir / "report.json").read_text(encoding="utf-8"))

    def test_report_schema_and_joint_failure_budget(self):
        report = self.report
        self.assertEqual(report["schema_version"], 1)
        for key in ("created_at_utc", "elapsed_seconds", "versions", "randomness",
                    "settings", "budgets", "distribution_scope", "assumptions", "bounds",
                    "procedure_audit", "fixed_interval_audit", "hypothetical_sample_size_plot"):
            self.assertIn(key, report)
        self.assertEqual(report["settings"]["trials"], 8)
        self.assertEqual(report["settings"]["mc_samples"], 15)
        self.assertEqual(report["settings"]["training_samples"], 6)
        budgets = report["budgets"]
        self.assertEqual(budgets["global_delta"], 0.08)
        self.assertAlmostEqual(budgets["procedure_family_delta"] + budgets["fixed_interval_delta"],
                               budgets["global_delta"])
        self.assertEqual(budgets["joint_confidence"], 1-budgets["global_delta"])
        certificates = report["procedure_audit"]["certificates"]
        self.assertEqual(len(certificates), budgets["procedure_claims"])
        for level, certificate in zip(report["procedure_audit"]["nominal_levels"], certificates):
            self.assertEqual(certificate["delta"], budgets["procedure_family_delta"])
            self.assertEqual(certificate["num_claims"], len(certificates))
            self.assertEqual(certificate["trials"], 8)
            self.assertIs(type(certificate["successes"]), int)
            self.assertEqual(certificate["target_coverage"], level)
        fixed = report["fixed_interval_audit"]["certificate"]
        self.assertEqual(fixed["delta"], budgets["fixed_interval_delta"])
        self.assertEqual(fixed["num_claims"], budgets["fixed_interval_claims"])
        self.assertEqual(fixed["num_claims"], 1)
        allocated_errors = sum(c["delta"]/c["num_claims"] for c in certificates)
        allocated_errors += fixed["delta"]/fixed["num_claims"]
        self.assertAlmostEqual(allocated_errors, budgets["global_delta"])
        self.assertIn("Joint audit confidence: 92%", self.process.stdout)

    def test_report_reproduces_from_recorded_seed_and_three_distinct_streams(self):
        report = self.report
        settings = report["settings"]
        streams = np.random.SeedSequence(report["randomness"]["seed"]).spawn(3)
        procedure_rng, construction_rng, audit_rng = [np.random.default_rng(s) for s in streams]
        certificates = audit_procedure(
            procedure_rng, trials=settings["trials"], R=settings["mc_samples"],
            n=settings["training_samples"], **settings["model"],
            nominal_levels=report["procedure_audit"]["nominal_levels"],
            delta=report["budgets"]["procedure_family_delta"], method=settings["method"],
        )
        self.assertEqual([c.to_dict() for c in certificates],
                         report["procedure_audit"]["certificates"])
        cloud = monte_carlo_samples(construction_rng, R=settings["mc_samples"],
                                    n=settings["training_samples"], **settings["model"])
        fixed = report["fixed_interval_audit"]
        tail = (1-fixed["nominal_level"])/2
        lower, median, upper = [float(v) for v in np.quantile(cloud, [tail, 0.5, 1-tail])]
        self.assertEqual(fixed["interval"], dict(lower=lower, median=median, upper=upper))
        certificate = audit_fixed_interval(
            audit_rng, lower=lower, upper=upper, trials=settings["trials"],
            **settings["model"], delta=report["budgets"]["fixed_interval_delta"],
            method=settings["method"], target_coverage=fixed["nominal_level"],
        )
        self.assertEqual(certificate.to_dict(), fixed["certificate"])

    def test_exported_figures_are_nonempty_png_and_svg(self):
        expected = {"report.json", "coverage.png", "coverage.svg",
                    "sample-size.png", "sample-size.svg"}
        self.assertEqual(set(self.report["files"]), expected)
        for filename in expected:
            with self.subTest(filename=filename):
                path = self.output_dir / filename
                self.assertGreater(path.stat().st_size, 1000)
                if path.suffix == ".png":
                    self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
                elif path.suffix == ".svg":
                    self.assertEqual(ET.parse(path).getroot().tag,
                                     "{http://www.w3.org/2000/svg}svg")

    def test_planning_plot_is_explicitly_hypothetical(self):
        curves = self.report["hypothetical_sample_size_plot"]
        self.assertIs(curves["counts_are_hypothetical"], True)
        self.assertEqual(curves["empirical_coverage"], 0.95)
        self.assertEqual(curves["delta"], self.report["budgets"]["procedure_family_delta"])
        self.assertEqual(curves["num_claims"], self.report["budgets"]["procedure_claims"])
        self.assertEqual(curves["trials"], sorted(set(curves["trials"])))
        self.assertTrue(all(n % 20 == 0 for n in curves["trials"]))
        self.assertEqual(len(curves["trials"]), len(curves["kl_lower"]))
        self.assertEqual(len(curves["trials"]), len(curves["hoeffding_lower"]))


class ExperimentEdgeCaseTests(unittest.TestCase):
    def test_noiseless_hoeffding_run(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            report = experiments.main([
                "--trials", "8", "--mc-samples", "15", "--training-samples", "3",
                "--method", "hoeffding", "--sigma", "0", "--nominal-levels", "0.6", "0.9",
                "--output-dir", directory,
            ])
            for certificate in report["procedure_audit"]["certificates"]:
                self.assertEqual(certificate["successes"], 8)
                self.assertEqual(certificate["method"], "hoeffding")
                self.assertEqual(certificate["num_claims"], 2)
            fixed = report["fixed_interval_audit"]
            self.assertEqual(fixed["exact_gaussian_coverage"], 1)
            self.assertEqual(fixed["interval"], dict(lower=3.0, median=3.0, upper=3.0))
            self.assertEqual(fixed["certificate"]["successes"], 8)
            for filename in report["files"]:
                self.assertGreater((Path(directory) / filename).stat().st_size, 0)

    def test_invalid_cli_arguments_fail_before_creating_outputs(self):
        invalid_arguments = (
            ["--trials", "0"], ["--mc-samples", "0"], ["--training-samples", "1"],
            ["--sigma", "-1"], ["--delta", "0"], ["--delta", "1"],
            ["--fixed-level", "0"], ["--seed", "-1"], ["--nominal-levels", "0.8", "0.8"],
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "must-not-exist"
            for arguments in invalid_arguments:
                with self.subTest(arguments=arguments), redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as caught:
                        experiments.main(["--trials", "8", "--mc-samples", "15",
                                          "--output-dir", str(output), *arguments])
                    self.assertEqual(caught.exception.code, 2)
                    self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
