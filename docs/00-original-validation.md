# Monte Carlo prediction intervals with provable coverage validation

This notebook constructs intervals in a synthetic linear-regression experiment
and validates their **true coverage probability** using finite-sample Hoeffding
or binary-KL Chernoff bounds. The calibration plot includes simultaneous
confidence bounds, and each result reports whether a minimum coverage target
is certified, below target, or inconclusive.

The guarantee is a theorem under the independence and fixed-protocol assumptions
below. It does not assert that the intervals necessarily attain their nominal
coverage, or that the simulator represents real data.

## Run

Keep `uq1.ipynb` and `validation.py` in the same directory. Install NumPy,
Matplotlib, and a Jupyter notebook environment, then run the notebook from top
to bottom. Its original simulation loops are retained; the four-level audit
fits four million small regressions with the default settings.

The certificate module itself uses only the Python standard library. Run tests
from the repository root:

```bash
python -m unittest discover -s tests -v
```

## What is being validated?

The true model is

$$
Y=\beta_0+\beta_1x+\varepsilon,\qquad
\varepsilon\sim N(0,\sigma^2).
$$

Defaults are `beta0=1`, `beta1=2`, `sigma=1`, training size `n=50`, and
`x_star=1`. Each of the `R` Monte Carlo draws generates a new training dataset,
fits OLS, predicts at `x_star`, and adds fresh response noise. Quantiles of
these draws define an interval.

One **validation trial** independently constructs such an interval and checks
one fresh true response against it. `num_experiments=N` independent trials
estimate

$$
p=\Pr\{Y_{\rm future}\in I_{\rm MC}\},
$$

averaging over interval-construction randomness and future-response noise,
at the specified model parameters and `x_star`. This is not a guarantee for
every `x_star`, for a particular displayed interval, or for an unknown
real-world distribution. The inner count `R` is **not** the validation sample
size.

## Using the certificates

The notebook preserves `coverage_test(...)` returning a float by default.
Request a report as follows, after running its definition cells:

```python
report = coverage_test(
    num_experiments=1000,
    confidence=0.95,       # nominal prediction-interval level
    R=1000,
    return_certificate=True,
    delta=0.05,           # audit failure probability
    num_claims=1,         # use 4 for the predeclared four-level sweep
    method="kl",         # or "hoeffding"
    target_coverage=0.95,
)
print(report.to_dict())
```

For coverage counts from another independent validation dataset:

```python
from validation import coverage_certificate, hoeffding_sample_size

# Illustrative counts, not a new experimental result.
report = coverage_certificate(
    successes=956, trials=1000,
    delta=0.05, num_claims=4, method="kl", target_coverage=0.95,
)
print(report.lower, report.upper, report.status)
# Approximately 0.93220, 0.97366, "inconclusive".

print(hoeffding_sample_size(0.01, delta=0.05, num_claims=4))
# 25376 independent trials per claim for simultaneous 1-point precision.
```

For `K` predeclared claims, the returned intervals simultaneously contain all
true coverages with probability at least `1-delta`. The notebook uses `K=4`
for its 80%, 90%, 95%, and 99% levels. The claims need not be independent of
each other. `delta` and the nominal prediction level are different quantities.

| Status | Meaning for a requested minimum coverage `target_coverage` |
|---|---|
| `certified` | The lower confidence bound is at least the target. |
| `below_target` | The upper confidence bound is below the target. |
| `inconclusive` | The data do not establish either conclusion. |
| `not_requested` | Bounds were computed without a minimum target. |

A minimum-coverage certificate does not establish equality to the nominal
level. To certify calibration within a chosen tolerance, the entire confidence
interval must lie inside the nominal level plus/minus that tolerance.

## Executed example

The saved notebook was run with audit seed `20260918`, `N=1000`, `R=1000`,
the default model parameters, `method="kl"`, and one total `delta=0.05`
budget across all four levels:

| Nominal | Covered / trials | Empirical | Simultaneous coverage bounds | Verdict |
|---|---|---|---|---|
| 80% | 793 / 1000 | 79.3% | 75.03%–83.18% | inconclusive |
| 90% | 896 / 1000 | 89.6% | 86.26%–92.41% | inconclusive |
| 95% | 953 / 1000 | 95.3% | 92.86%–97.13% | inconclusive |
| 99% | 989 / 1000 | 98.9% | 97.50%–99.65% | inconclusive |

The audit does not certify minimum nominal coverage at any of these levels,
and it does not establish undercoverage either. The intervals quantify what
this finite validation sample can support; no tolerance was adjusted to
obtain a positive verdict.

## Assumptions that make the guarantee valid

- Freeze the interval procedure, settings, target, sample count, and set of
  comparisons before examining validation outcomes.
- Use independent validation trials with the same distribution. Any training
  or tuning on real data must be independent of the final audit data.
- Count all predeclared configurations selected or reported using the audit
  in `num_claims`. Additional non-nested bound methods need their own error
  allocation if selected or jointly reported. The implemented KL interval is
  contained in Hoeffding at identical inputs, so displaying this particular
  pair at the same budget needs no extra correction.
- Do not repeatedly inspect the results and stop when a certificate passes.
  These are fixed-sample bounds. Retuning requires a fresh audit or a valid
  preplanned simultaneous analysis.

## Important interpretation of the existing simulator

Construction samples `fitted_mean + fresh_noise`, while validation samples
`true_mean + fresh_noise`. Their distributions differ because the fitted mean
contains additional training-sample variation. Thus a nominal 95% Monte Carlo
quantile interval need not have exactly 95% coverage for the validation target.
In fact, its limiting central interval as `R` grows overcovers in this Gaussian
example. The finite-`R` audit measures what the implemented procedure achieves.

The code uses the known true generating parameters to simulate new datasets;
it is an oracle simulation study, not an interval fitted to one observed
dataset. Simply removing future noise does **not** turn these oracle quantiles
into a confidence interval for an unknown mean based on observed data. Fixing
the training design does not condition on the observed responses either.

For one observed Gaussian linear-regression dataset, the familiar formulas are

$$
\begin{aligned}
\text{mean CI}:&\quad \widehat y_\star\pm
 t_{n-2,1-\alpha/2}\,s\sqrt{h_\star},\\
\text{future-response PI}:&\quad \widehat y_\star\pm
 t_{n-2,1-\alpha/2}\,s\sqrt{1+h_\star},\\
h_\star=&\quad \frac1n+\frac{(x_\star-\bar x)^2}{S_{xx}}.
\end{aligned}
$$

Here `s` is the residual standard deviation from that observed fit. Their
classical coverage is conditional on the design, averaging over training
responses and, for prediction intervals, the independent future response.
The notebook's interval construction has been retained; the validation layer
does not replace it with these formulas.

See [VALIDATION.md](../VALIDATION.md) for the exact theorem, proof, sample-size
interpretation, and how to validate a single frozen interval.
