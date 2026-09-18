# 3. From theorem to code

[Previous: concentration and certification](02-concentration-and-certification.md) ·
[Contents](../README.md) · [Next: extensions and limits](04-extensions-and-limits.md)

The theorem is useful only when the program produces the random variables in
its hypotheses. This chapter treats a validation pipeline as a small program
whose interfaces encode its probability model.

## 3.1 Three pieces with different responsibilities

| Component | Input and output | Mathematical responsibility |
|---|---|---|
| Interval constructor | Simulation settings and randomness → interval | Defines the procedure being evaluated |
| Independent audit | Procedure and fresh validation randomness → integer successes $S$ and trials $N$ | Produces iid Bernoulli coverage indicators |
| Certificate | $(S,N,\delta,K)$ → bounds and verdict | Applies a theorem to those counts |

The [certificate module](../validation.py) has no NumPy, plotting, notebook,
network, or simulator dependency. It is a pure calculation from integer counts.
It cannot inspect whether the caller used independent observations. That is a
contract the experiment must satisfy, just as a numerical algorithm requires
its input to meet its mathematical hypotheses.

An important consequence is portability: the same module can audit intervals
from a neural network or a physics simulator, provided the validation indicators
really satisfy the assumptions. Changing the interval generator does not
require changing the concentration proof.

## 3.2 Start with a count

From the repository root:

```python
from validation import coverage_certificate

report = coverage_certificate(
    successes=953,
    trials=1000,
    delta=0.05,
    num_claims=4,
    method="kl",
    target_coverage=0.95,
)
print(report.to_dict())
```

These are the counts from the original executed notebook. The output includes
an empirical coverage of $0.953$, a lower bound approximately $0.9286$, an upper
bound approximately $0.9713$, and the status `inconclusive`.

`target_coverage` is a minimum acceptable value of the unknown coverage. The
`delta` argument is the probability budget for the audit theorem. The nominal
level used to construct an interval is a third quantity, which does not enter
this function at all unless the caller chooses it as the target.

The module deliberately accepts counts rather than a rounded reported
percentage. Rounding `95.3%` and then reconstructing $S$ is unnecessary and can
change the data. It also rejects zero trials and non-integer counts.

## 3.3 Inverting KL without a numerical optimization package

For fixed $a\in(0,1)$,

$$
\frac{\partial}{\partial q}\operatorname{kl}(a\Vert q)
=\frac{q-a}{q(1-q)}.
$$

Thus KL decreases from infinity to zero on $(0,a]$, and increases from zero
to infinity on $[a,1)$. Each equation

$$
\operatorname{kl}(a\Vert q)=\frac{\log(2K/\delta)}{N}
$$

has exactly one root on either side of $a$. Bisection is sufficient: no
initial guess, stochastic optimization, or external solver is needed.

On the lower side the implementation maintains an outside endpoint and an
inside endpoint. If the midpoint's KL exceeds the budget, replace the outside
endpoint by that midpoint; otherwise replace the inside endpoint. Return the
outside endpoint. Reverse the geometry on the upper side. A final outward
floating-point step avoids accidentally returning the inward endpoint.

The endpoint observations $a=0$ and $a=1$ have closed forms, so no logarithm
of zero needs to be evaluated. The implementation uses `log1p` and `expm1`
where cancellation would otherwise be avoidable. It computes
`log(2) + log(K) - log(delta)`, which avoids overflowing `2*K/delta` for a very
small error budget.

These choices make an ordinary floating-point implementation reliable in the
tested regimes. They are not a machine-checked real-arithmetic proof; formally
certified numerics would need interval arithmetic or a separately verified
rounding analysis.

## 3.4 A fast simulator with the same marginal experiment

The [original notebook](../uq1.ipynb) uses a Python loop for each fitted
regression. The [new simulator](../simulation.py) arranges the $R$ independent
training datasets as rows of an $R\times n$ matrix. NumPy computes all row
means, centered sums, slopes, and intercepts together.

Vectorization changes how the arithmetic is scheduled. It does not change the
experiment: every row is still a fresh normal design and response sample, every
row produces one OLS prediction, and a fresh future-noise draw is added to each
prediction. Broadcasting must preserve these axes. A column reused as noise
for every row would violate the intended experiment.

One outer trial does the following:

1. Generate a fresh cloud of $R$ simulated predictions.
2. Compute central quantiles for every predeclared nominal level.
3. Generate one fresh true future response, independent of that cloud.
4. Increment each level's success counter if its interval covers the response.

The same cloud and response are used across levels. This creates dependence
*between* the four indicators within a trial. It preserves independence
*across* outer trials, which is exactly what each individual concentration
bound requires. The union bound across levels never required independence
between levels.

This saves repeated interval construction. The runner does not reproduce the
original notebook's random stream: the notebook constructs separate clouds
for each level and uses NumPy's legacy random interface. Its recorded counts
and the new runner's counts are therefore separate experiments. Reproducibility
means repeating one documented protocol, not expecting different protocols to
yield identical random counts.

A conservative time bound is $O(N(Rn+R\log R+K))$ if quantiles are obtained by
sorting. The simulator stores one cloud's training arrays at a time, using
$O(Rn+K)$ working memory for the procedure audit instead of storing all $N$
clouds. The certificate calculation is tiny by comparison: a fixed number of
bisection iterations per reported interval.

## 3.5 Run the tutorial experiment

```bash
python -m pip install -r requirements.txt
python experiments.py --trials 1000 --mc-samples 1000 \
  --seed 20260918 --output-dir results/demo
```

For a quick installation check:

```bash
python experiments.py --trials 20 --mc-samples 30 \
  --output-dir /tmp/uq-smoke
```

The small run checks that the software works. It is not intended to establish
useful coverage precision.

The main command writes a JSON report and figures. Read the report before
interpreting the plots: it identifies the model settings, method, random seed,
trial counts, and failure-budget allocation. The first figure visualizes actual
coverage observations. The sample-size figure is a deterministic comparison
of certificate formulas at a stipulated empirical proportion; it is not another
simulation result or an estimate of the probability that certification passes.

## 3.6 Two targets, one declared total error budget

The runner demonstrates both an average over newly constructed intervals and
coverage of one frozen interval. With total budget $\delta$, it allocates
$\delta/2$ to the four-level procedure audit and $\delta/2$ to the frozen-interval
audit. Within the first group the certificate's $K=4$ correction is applied.

If $E_1$ and $E_2$ are the two group success events, then

$$
\Pr(E_1^c\cup E_2^c)
\le \delta/2+\delta/2=\delta.
$$

For the second group, construction of the displayed interval precedes its
independent validation. Its guarantee first holds conditional on that
constructed interval; averaging over the interval preserves the same failure
bound. This justifies the combined statement. Printing a separate 95% result
for each group and calling the whole report 95% would not justify it.

For the frozen interval, the Gaussian model supplies an exact CDF calculation
as an additional benchmark. The unknown-coverage theorem does not use that
benchmark. Seeing the theoretical coverage outside a nominal band on a rare
run is compatible with a confidence procedure; do not rerun until it lands
inside and then report the chosen run as if unselected.

## 3.7 What the tests establish

The test suite includes:

- endpoint and monotonicity checks for KL inversion;
- exact finite binomial sums of noncoverage probabilities on a fixed grid;
- tests that more claims or higher confidence widen the intervals;
- numerical agreement of fast OLS with explicit rowwise regression;
- notebook integration tests that catch missing certificate returns;
- reproducibility and smoke checks for the experiment report.

For a true Bernoulli probability $p$, the deterministic probability check sums

$$
\sum_{s=0}^{N}\mathbf1\{p\notin[L(s),U(s)]\}
{N\choose s}p^s(1-p)^{N-s}
$$

and verifies that it is at most the allocated budget. This is more informative
than a flaky test that repeatedly samples random audits. It still checks only
selected parameters and floating-point arithmetic. The all-parameter result
comes from the proof in Chapter 2.

Run:

```bash
python -m unittest discover -s tests -v
```

The GitHub Actions workflow also runs these tests and a small command-line
experiment on two Python versions. A successful local run establishes local
behavior; the remote CI result must be checked separately after publication.

## 3.8 Changing the experiment responsibly

Choose $N$, $R$, the candidate levels, the target, and the error budget before
inspecting the audit. Increase $R$ if you want a different, less noisy interval
constructor. Increase $N$ if you need a more precise estimate of the current
constructor's coverage. They solve different problems.

If an audit motivates changing the constructor, treat those observations as
development data. Freeze the new design and run a fresh audit, with a budget
appropriate to any combined claim you plan to make. To inspect accumulating
results and stop adaptively, use a bound valid simultaneously over time; the
next chapter derives a simple version.
