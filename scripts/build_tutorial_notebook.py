"""Build the standalone notebook from the repository's tutorial and source.

The builder may read local sources; the generated notebook never does.
Run this script from any directory. It writes an unexecuted nbformat-4 file.
"""

import ast
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "Provable_UQ_Tutorial.ipynb"
CELLS = []
COUNTERS = {}


def add(kind, source, key):
    source = source.strip() + "\n"
    COUNTERS[key] = COUNTERS.get(key, 0) + 1
    cell = {
        "cell_type": kind,
        "id": f"{key}-{COUNTERS[key]}",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }
    if kind == "code":
        compile(source, cell["id"], "exec")
        cell.update(execution_count=None, outputs=[])
    CELLS.append(cell)


def md(source, key):
    add("markdown", source, key)


def code(source, key):
    add("code", source, key)


CHAPTERS = {
    "01-probability-and-targets.md": "chapter-1",
    "02-concentration-and-certification.md": "chapter-2",
    "03-from-theorem-to-code.md": "chapter-3",
    "04-extensions-and-limits.md": "chapter-4",
    "05-exercises.md": "chapter-5",
}


def convert_link(match):
    label, destination = match.groups()
    if destination.startswith(("https://", "http://", "#")):
        return match.group(0)
    filename, _, fragment = destination.partition("#")
    filename = Path(filename).name
    if filename in CHAPTERS:
        target = CHAPTERS[filename]
        if fragment == "chapter-1-probability-and-targets":
            target = "solutions-chapter-1"
        elif fragment == "chapter-2-concentration-and-certification":
            target = "solutions-chapter-2"
    elif filename == "README.md":
        target = "contents"
    elif filename == "validation.py":
        target = "certificate-implementation"
    elif filename == "simulation.py":
        target = "simulation-implementation"
    elif filename == "VALIDATION.md":
        target = "references" if fragment == "references" else "chapter-2"
        if fragment.startswith("6-why"):
            target = "construction-distribution"
    elif filename == "uq1.ipynb":
        return f"[{label}](https://github.com/payal101/UQ_1/blob/eaa3830/uq1.ipynb)"
    else:
        raise ValueError(f"Unmapped relative link: {destination}")
    return f"[{label}](#{target})"


def adapt_markdown(text):
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", convert_link, text)
    # A single document title; chapters, sections, and subsections descend.
    text = re.sub(r"(?m)^(#{1,3}) ", lambda m: m[1] + "# ", text)
    text = text.replace("From the repository root:", "Run the following self-contained cell:")
    text = text.replace(
        "The notebook's recorded four-level run uses",
        "**Historical example: the original notebook's separate random-number protocol.**\n\n"
        "These counts are not outputs of this tutorial's main audit. That earlier run uses",
    )
    text = text.replace(
        "These are the counts from the original executed notebook.",
        "These are historical counts from the original executed notebook's separate protocol, "
        "not from the main experiment run below.",
    )
    text = text.replace(
        "The current [implementation](#certificate-implementation)",
        "The embedded [implementation](#certificate-implementation)",
    )
    return text


def chapter_content(text, key):
    """Make each Python example executable, leaving surrounding prose intact."""
    parts = re.split(r"(?ms)^```python\s*\n(.*?)^```\s*$", text)
    for index, part in enumerate(parts):
        if not part.strip():
            continue
        if index % 2:
            part = re.sub(r"(?m)^from validation import .*\n", "", part)
            code(part, key + "-example")
        else:
            md(adapt_markdown(part), key)


def definitions(filename, names):
    source = (ROOT / filename).read_text(encoding="utf-8")
    lines = source.splitlines(keepends=True)
    nodes = {node.name: node for node in ast.parse(source).body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    chunks = []
    for name in names:
        node = nodes[name]
        start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
        chunks.append("".join(lines[start:node.end_lineno]).rstrip())
    return "\n\n\n".join(chunks)


def certificate_definitions():
    md(r"""<a id="certificate-implementation"></a>
### Embedded certificate implementation

The following readable definitions are embedded from `validation.py`. They
depend only on the standard-library imports in the setup cell. The names
`successes`, `trials`, `num_claims`, and `delta` represent $S,N,K,\delta$ in
the proof. No repository file is imported or read when this notebook runs.
""", "certificate-source")
    code(definitions("validation.py", ["_positive_integer", "_probability", "_log_budget",
                                       "bernoulli_kl", "_kl_bounds"]), "certificate-numerics")
    code(definitions("validation.py", ["CoverageCertificate", "coverage_certificate",
                                       "hoeffding_sample_size"]), "certificate-api")


def simulation_definitions():
    md("""<a id="simulation-implementation"></a>
### Embedded simulator and independent audits

These definitions are embedded from `simulation.py`. Each row of a simulation
matrix is one independently generated training dataset. A `Generator` is passed
explicitly, so the examples do not reset or consume NumPy's global random stream.
The Gaussian benchmark uses `erf`/`erfc`, equivalent to the normal CDF, with a
stable subtraction in each tail. The noiseless case is handled as a point mass.
""", "simulation-source")
    code(definitions("simulation.py", ["_integer", "_real", "_model", "_generator",
                                       "_levels"]), "simulation-inputs")
    code(definitions("simulation.py", ["_samples", "monte_carlo_samples",
                                       "audit_procedure"]), "simulation-procedure")
    code(definitions("simulation.py", ["_interval", "gaussian_interval_coverage",
                                       "audit_fixed_interval"]), "simulation-frozen")


MAIN_AUDIT = '''# Predeclare all settings before generating any main-audit observations.
N = 1000                      # Independent outer trials per coverage claim.
R = 1000                      # Inner Monte Carlo draws per constructed interval.
training_n = 50
nominal_levels = [0.80, 0.90, 0.95, 0.99]
fixed_nominal_level = 0.95
global_delta = 0.05
procedure_delta = global_delta / 2
fixed_delta = global_delta / 2
method = "kl"                 # Change to "hoeffding" before a fresh audit.
seed = 20260918
model = dict(beta0=1.0, beta1=2.0, sigma=1.0, x_star=1.0)

# Independent streams for the procedure, one interval's construction, and its audit.
child_seeds = np.random.SeedSequence(seed).spawn(3)
rng_procedure, rng_construction, rng_fixed = [
    np.random.default_rng(child) for child in child_seeds
]
procedure_certificates = audit_procedure(
    rng_procedure, trials=N, nominal_levels=nominal_levels,
    R=R, n=training_n, delta=procedure_delta, method=method, **model,
)
fixed_cloud = monte_carlo_samples(rng_construction, R=R, n=training_n, **model)
tail = (1 - fixed_nominal_level) / 2
fixed_lower, fixed_median, fixed_upper = [
    float(value) for value in np.quantile(fixed_cloud, [tail, 0.5, 1-tail])
]
fixed_certificate = audit_fixed_interval(
    rng_fixed, lower=fixed_lower, upper=fixed_upper, trials=N,
    delta=fixed_delta, method=method, target_coverage=fixed_nominal_level, **model,
)
true_mean = model["beta0"] + model["beta1"] * model["x_star"]
fixed_exact = gaussian_interval_coverage(
    fixed_lower, fixed_upper, mean=true_mean, sigma=model["sigma"],
)

print(f"Joint audit confidence: {1-global_delta:.0%}; global delta = {global_delta}")
print(f"Procedure family delta = {procedure_delta}; fixed interval delta = {fixed_delta}")
print("Nominal  Successes/trials  Empirical   Coverage bounds        Decision")
for nominal, result in zip(nominal_levels, procedure_certificates):
    print(f"{nominal:6.2%}   {result.successes:4d}/{result.trials:<5d}       "
          f"{result.empirical_coverage:.4f}      "
          f"[{result.lower:.4f}, {result.upper:.4f}]   {result.status}")
print(f"\\nFrozen interval: [{fixed_lower:.4f}, {fixed_upper:.4f}]")
print(f"Exact Gaussian coverage: {fixed_exact:.6f}")
print(f"Audit coverage: {fixed_certificate.successes}/{fixed_certificate.trials}; "
      f"[{fixed_certificate.lower:.4f}, {fixed_certificate.upper:.4f}]; "
      f"{fixed_certificate.status}")

# A portable in-memory report. No files are opened or written.
main_report = {
    "seed": seed, "generator": type(rng_procedure.bit_generator).__name__,
    "versions": {"python": platform.python_version(), "numpy": np.__version__,
                 "matplotlib": matplotlib.__version__},
    "trials": N, "mc_samples": R, "training_samples": training_n, "model": model,
    "global_delta": global_delta, "procedure_delta": procedure_delta,
    "fixed_delta": fixed_delta, "shared_randomness_between_levels": True,
    "procedure_certificates": [c.to_dict() for c in procedure_certificates],
    "fixed_interval": [fixed_lower, fixed_upper],
    "fixed_exact_gaussian_coverage": fixed_exact,
    "fixed_certificate": fixed_certificate.to_dict(),
}
'''


MAIN_FIGURE = '''# Two targets: repeated construction on the left, one frozen interval on the right.
navy, blue, amber = "#16324f", "#187f9c", "#bc6426"
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
ax = axes[0]
observed = np.array([c.empirical_coverage for c in procedure_certificates])
errors = np.array([[c.empirical_coverage-c.lower for c in procedure_certificates],
                   [c.upper-c.empirical_coverage for c in procedure_certificates]])
ax.plot([0, 1], [0, 1], "--", color="#83929d", label="Nominal target")
ax.errorbar(nominal_levels, observed, yerr=errors, fmt="o", color=blue,
            capsize=5, markersize=6, label="Empirical coverage + bounds")
ax.set(xlabel="Nominal prediction-interval level", ylabel="Coverage probability",
       title="Rebuild the interval each trial")
ax.set_xlim(max(0, min(nominal_levels)-0.04), min(1.01, max(nominal_levels)+0.03))
ax.set_ylim(max(0, min(0.7, min(c.lower for c in procedure_certificates)-0.025)), 1.01)
ax.legend(loc="lower right", frameon=False, fontsize=9)
ax.grid(axis="y", alpha=0.2)

ax = axes[1]
sigma = model["sigma"]
if sigma > 0:
    x = np.linspace(min(true_mean-4*sigma, fixed_lower),
                    max(true_mean+4*sigma, fixed_upper), 700)
    density = np.exp(-0.5*((x-true_mean)/sigma)**2) / (sigma*np.sqrt(2*np.pi))
    ax.plot(x, density, color=navy)
    ax.fill_between(x, 0, density, where=(fixed_lower <= x) & (x <= fixed_upper),
                    color=blue, alpha=0.25)
    ax.set_ylabel("True future-response density")
else:
    ax.axvline(true_mean, color=navy, lw=3)
    ax.set_xlim(true_mean-1, true_mean+1)
    ax.set_ylabel("Point mass at the true mean")
ax.axvline(fixed_lower, color=amber, ls="--")
ax.axvline(fixed_upper, color=amber, ls="--")
ax.set(xlabel="Future response Y", title="Freeze one interval, then audit it")
ax.text(0.02, 0.98,
        f"Exact coverage: {fixed_exact:.3f}\\n"
        f"Audit: [{fixed_certificate.lower:.3f}, {fixed_certificate.upper:.3f}]\\n"
        f"Decision: {fixed_certificate.status}",
        transform=ax.transAxes, va="top", fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92})
ax.set_ylim(bottom=0)
ax.grid(axis="y", alpha=0.15)
fig.suptitle(f"Coverage validation | {1-global_delta:.0%} joint audit confidence",
             fontsize=14, color=navy)
plt.show()
plt.close(fig)
'''


PRECISION_FIGURE = '''# These counts are hypothetical; this cell performs no new statistical audit.
# N is a multiple of 20, so k/N = 19/20 = 0.95 exactly as a rational number.
sample_sizes = np.unique(np.rint(np.geomspace(1, 5000, 65)).astype(int)) * 20
lower_curves = {}
for bound_method in ("kl", "hoeffding"):
    lower_curves[bound_method] = [
        coverage_certificate(
            19*(int(size)//20), int(size), delta=procedure_delta,
            num_claims=len(nominal_levels), method=bound_method,
        ).lower
        for size in sample_sizes
    ]
fig, ax = plt.subplots(figsize=(7.4, 4.5), constrained_layout=True)
ax.semilogx(sample_sizes, lower_curves["kl"], color=blue, lw=2.2,
            label="Bernoulli Chernoff / KL")
ax.semilogx(sample_sizes, lower_curves["hoeffding"], color=amber, lw=2.2,
            label="Hoeffding")
ax.axhline(0.95, color="#83929d", ls="--", label="Stipulated proportion 0.95")
ax.set(xlabel="Number of independent outer trials N", ylabel="Lower coverage bound",
       title="How audit sample size controls precision", ylim=(0.4, 0.97))
ax.text(0.02, 0.03, "Hypothetical counts; not experiment observations or passing probabilities",
        transform=ax.transAxes, fontsize=8.5)
ax.legend(loc="lower right", bbox_to_anchor=(1, 0.11), frameon=False)
ax.grid(axis="y", alpha=0.18)
plt.show()
plt.close(fig)
'''


ASSERTIONS = '''# Deterministic numerical checks support, but do not replace, Chapter 2's proof.
for bound_method in ("kl", "hoeffding"):
    zero = coverage_certificate(0, 100, method=bound_method)
    one = coverage_certificate(100, 100, method=bound_method)
    assert zero.lower == 0.0 and one.upper == 1.0
    baseline = coverage_certificate(95, 100, delta=0.05, method=bound_method)
    multiple = coverage_certificate(95, 100, delta=0.05, num_claims=10,
                                    method=bound_method)
    assert multiple.lower <= baseline.lower <= baseline.upper <= multiple.upper

# Sum exact binomial probabilities instead of running a flaky random test.
checked = 0
for bound_method in ("kl", "hoeffding"):
    for trials in (1, 5, 12, 30):
        intervals = [coverage_certificate(s, trials, delta=0.05, num_claims=4,
                                           method=bound_method)
                     for s in range(trials+1)]
        for true_p in (0.0, 0.01, 0.2, 0.5, 0.9, 0.99, 1.0):
            missed_probability = math.fsum(
                math.comb(trials, s)*true_p**s*(1-true_p)**(trials-s)
                for s, interval in enumerate(intervals)
                if true_p < interval.lower or true_p > interval.upper
            )
            assert missed_probability <= 0.05/4 + 1e-13
            checked += 1

assert gaussian_interval_coverage(3, 3, mean=3, sigma=0) == 1.0
assert gaussian_interval_coverage(0, 2, mean=3, sigma=0) == 0.0
np.testing.assert_allclose(
    gaussian_interval_coverage(-1.96, 1.96, mean=0, sigma=1),
    0.950004209703559, rtol=0, atol=1e-14,
)
noiseless = audit_procedure(np.random.default_rng(37), trials=5, R=10, n=4,
                            sigma=0, nominal_levels=[0.8, 0.95])
assert all(c.successes == 5 and c.trials == 5 for c in noiseless)
try:
    coverage_certificate(3.5, 10)
except ValueError:
    pass
else:
    raise AssertionError("Fractional successes must not be accepted as binomial counts")
print(f"All checks passed, including {checked} deterministic binomial-coverage cases.")
'''


def build():
    CELLS.clear()
    COUNTERS.clear()
    md("""# Provable uncertainty quantification: a complete executable tutorial

**From probability and exponential Markov to reproducible coverage certificates.**

This is a standalone notebook: download this file, open it in Jupyter, and use
**Restart Kernel and Run All Cells**. Python **3.10 or newer**, NumPy, and
Matplotlib are the only requirements beyond your notebook environment. All
implementation code is included below. Execution does not download anything,
read companion files, invoke a shell, or require a repository checkout.

The default main experiment uses 1,000 independent outer trials and 1,000 inner
Monte Carlo draws per interval; it typically takes several seconds on a laptop.
Decrease those two values in Section 3.5 for a quicker smoke check, choosing
settings before seeing audit outcomes. Saved outputs can be read without running
the code. Mathematical statements are proved, and exercises include worked solutions.

<a id="contents"></a>
## Contents

1. [Probability, prediction, and the target](#chapter-1)
2. [Concentration and certification, with proofs](#chapter-2)
3. [The complete implementation and main experiment](#chapter-3)
4. [Images, selection, sequential audits, and limitations](#chapter-4)
5. [Exercises and worked solutions](#chapter-5)
6. [Provenance and original references](#references)

The main experiment allocates a joint 5% failure budget across its two audit
groups. Historical count examples, deterministic comparisons, and the separately
labeled sequential illustration are teaching exercises; they are not additional
claims included in that main experiment's joint 95% statement.
""", "introduction")
    code('''import sys
import math
import platform
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from typing import Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

assert sys.version_info >= (3, 10), "This tutorial requires Python 3.10 or newer."
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 110})
print(f"Python {platform.python_version()} | NumPy {np.__version__} | "
      f"Matplotlib {matplotlib.__version__}")
''', "setup")

    for filename, chapter_id in CHAPTERS.items():
        text = (ROOT / "docs" / filename).read_text(encoding="utf-8")
        sections = re.split(r"(?m)(?=^## )", text)
        for index, section in enumerate(sections):
            heading = section.splitlines()[0]
            key = f"{chapter_id}-section-{index}"
            if index == 0:
                section = f'<a id="{chapter_id}"></a>\n' + section
            if heading.startswith("## 1.3"):
                section = '<a id="construction-distribution"></a>\n' + section
            if heading == "## Chapter 1: Probability and targets":
                section = '<a id="solutions-chapter-1"></a>\n' + section
            if heading == "## Chapter 2: Concentration and certification":
                section = '<a id="solutions-chapter-2"></a>\n' + section
            if heading.startswith("## 3.5"):
                section = """## 3.5 Run the self-contained tutorial experiment

The next cell declares the whole main experiment before drawing any outcomes.
It allocates half the global error budget to the four procedure claims and half
to the frozen-interval claim. Section 3.6 proves their combined guarantee.
The three random streams match the companion command-line experiment exactly.

Keep this distinction in mind: the historical 953/1000 example above used the
original notebook's independent clouds for different levels and a separate
legacy random stream. This faster audit shares one cloud and one future response
across levels within each trial. It has the same marginal statistical experiment
for each level, but it is a different documented random-number protocol.

The code prints every decision, including `inconclusive`; observing empirical
coverage above the nominal level is not automatically a certificate. It also
keeps an in-memory `main_report` containing settings, versions, and results.
"""
            if heading.startswith("## 3.7"):
                section = re.sub(r"(?s)\nRun:\n.*?\n##", "\n##", section)
                section = section.split("\nRun:\n")[0]
                section += """\n\nRun the following deterministic checks directly in this notebook.
They enumerate finite binomial outcomes, verify endpoint behavior and multiple-claim
budgets, and check the exact Gaussian benchmark and the noiseless simulation.
No test framework, repository test directory, or shell command is needed.
"""
            chapter_content(section, key)

            if heading.startswith("## 1.4"):
                md("""### Small illustration: average coverage can hide bad realized intervals

Take $Y$ uniform on $[0,1]$. Independently return $[0,1]$ with probability
$0.95$ and $[2,3]$ otherwise. The code samples the interval construction only;
the conditional coverages are known exactly to be one and zero.
""", "targets-demo-intro")
                code('''illustration_rng = np.random.default_rng(73)
interval_is_good = illustration_rng.random(2000) < 0.95
conditional_coverages = interval_is_good.astype(float)
print("Exact procedure-average coverage: 0.95")
print("Possible fixed-interval coverages:", np.unique(conditional_coverages))
print("Illustrative average over constructed intervals:", conditional_coverages.mean())
''', "targets-demo")
            if heading.startswith("## 2.6"):
                code('''# Formula evaluation only: these are planned sample sizes, not new observations.
K, delta, precision = 4, 0.05, 0.01
budget = math.log(2*K/delta)
print("Outer trials for Hoeffding precision 0.01:", math.ceil(budget/(2*precision**2)))
print("Hoeffding margin at N=1000:", math.sqrt(budget/2000))
for target in (0.95, 0.99):
    print(f"If every trial succeeds, trials sufficient for target {target}: "
          f"{math.ceil(budget / -math.log(target))}")
''', "precision-planning")
            if heading.startswith("## 3.1"):
                certificate_definitions()
            if heading.startswith("## 3.4"):
                simulation_definitions()
            if heading.startswith("## 3.5"):
                code(MAIN_AUDIT, "main-audit")
                code(MAIN_FIGURE, "two-targets-figure")
                md("""### Precision at a stipulated observed proportion

The next figure holds the hypothetical ratio $S/N=0.95$ fixed and increases
$N$. It compares the two formulas using the procedure group's budget. It is
not based on additional audit observations and does not estimate the probability
of passing a test. The lower bounds approach $0.95$ from below.
""", "precision-figure-intro")
                code(PRECISION_FIGURE, "precision-figure")
            if heading.startswith("## 3.7"):
                code(ASSERTIONS, "self-contained-checks")
            if heading.startswith("## 4.2"):
                md("""### A million identical pixels do not make a million trials

This deterministic calculation compares the correct image-level radius with
the invalid radius obtained by multiplying the independent sample count by
the number of identical pixels. No simulated coverage claim is made here.
""", "pixels-demo-intro")
                code('''independent_images, pixels_per_image = 100, 1_000_000
correct_radius = math.sqrt(math.log(2/0.05)/(2*independent_images))
incorrect_radius = math.sqrt(
    math.log(2/0.05)/(2*independent_images*pixels_per_image)
)
print("Valid image-level Hoeffding radius:", correct_radius)
print("Invalid radius obtained by counting dependent pixels:", incorrect_radius)
assert math.isclose(correct_radius/incorrect_radius, math.sqrt(pixels_per_image))
''', "pixels-demo")
            if heading.startswith("## 4.4"):
                md("""### Separate sequential illustration

The next example has its own 5% time-uniform budget and a known simulation
probability $p=0.99$. It stops at the first certificate for the predeclared
target $0.95$, or at 3,000 trials. Stopping is permitted by the theorem just
proved. This demonstration is separate from the main experiment's joint budget.
There is no guarantee that it will certify before the cap.
""", "anytime-demo-intro")
                code('''sequential_rng = np.random.default_rng(812)
sequential_successes = 0
for t in range(1, 3001):
    sequential_successes += int(sequential_rng.random() < 0.99)
    sequential_result = anytime_certificate(sequential_successes, t, delta=0.05)
    if sequential_result.status == "certified":
        break
print(f"Stopped after {t} trials: {sequential_successes} successes; "
      f"[{sequential_result.lower:.4f}, {sequential_result.upper:.4f}]; "
      f"{sequential_result.status}")
print("This stopping rule used delta_t = 6*0.05/(pi**2*t**2), not repeated fixed-delta tests.")
''', "anytime-demo")

    validation_hash = hashlib.sha256((ROOT / "validation.py").read_bytes()).hexdigest()
    simulation_hash = hashlib.sha256((ROOT / "simulation.py").read_bytes()).hexdigest()
    md(f"""<a id="references"></a>
## Provenance and original references

The starting simulation is [payal101/UQ_1](https://github.com/payal101/UQ_1),
source snapshot `eaa3830` ("Revise README for clarity and structure"). Its
normal-design linear-regression model, OLS fit, and Monte Carlo interval
construction are retained in distribution. The embedded simulator vectorizes
the independent fits and uses a documented shared-randomness audit protocol.
The historical original-notebook counts are explicitly separated above.

The explanatory chapters and embedded implementation were assembled into this
standalone notebook by `scripts/build_tutorial_notebook.py`. The builder reads
repository sources at build time; this notebook needs no companion files at run
time. Embedded source fingerprints are:

- `validation.py` SHA-256: `{validation_hash}`
- `simulation.py` SHA-256: `{simulation_hash}`

These are classical concentration arguments and teaching demonstrations, not
claims of new theorems. The numerical examples use synthetic data only. No license
file was present in the original source snapshot; this notebook does not assign
a new license to that source material. Preserve its attribution.

Original mathematical references:

- W. Hoeffding (1963), [Probability Inequalities for Sums of Bounded Random
  Variables](https://doi.org/10.1080/01621459.1963.10500830), *Journal of the
  American Statistical Association* 58(301), 13–30.
- H. Chernoff (1952), [A Measure of Asymptotic Efficiency for Tests of a
  Hypothesis Based on the Sum of Observations](https://doi.org/10.1214/aoms/1177729330),
  *Annals of Mathematical Statistics* 23(4), 493–507.

To reproduce the tutorial, restart the kernel and run from the setup cell to
the end. To investigate a new model or target, decide the new protocol first
and allocate independent audit data and an appropriate confidence budget.
""", "provenance")

    # Validate self-containment statically; the parent workflow executes and
    # visually checks the full notebook before publishing its outputs.
    allowed_imports = {"sys", "math", "platform", "dataclasses", "numbers", "typing",
                       "numpy", "matplotlib"}
    for cell in CELLS:
        source = "".join(cell["source"])
        if cell["cell_type"] == "code":
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.ImportFrom):
                    assert node.module.split(".")[0] in allowed_imports
                elif isinstance(node, ast.Import):
                    assert all(alias.name.split(".")[0] in allowed_imports for alias in node.names)
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    assert node.func.id not in {"open", "exec", "eval", "__import__"}
        else:
            assert "```bash" not in source
            for destination in re.findall(r"\]\(([^)]+)\)", source):
                assert destination.startswith(("#", "https://", "http://")), destination
    notebook = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10", "file_extension": ".py",
                              "mimetype": "text/x-python", "pygments_lexer": "ipython3"},
            "title": "Provable UQ: a complete standalone tutorial",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False)+"\n", encoding="utf-8")
    print(f"Built {OUTPUT.name}: {len(CELLS)} cells, {OUTPUT.stat().st_size:,} bytes")
    print("All code cells compile; no local-file imports or runtime file access are present.")


if __name__ == "__main__":
    build()
