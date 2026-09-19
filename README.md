# Predict a range. Validate its coverage.

**Bayesian regression proposes intervals. Independent held-out data and
Hoeffding's bound tell us how often those intervals actually work.**

[Read the lesson](TUTORIAL.md) · [Run the notebook](Provable_UQ_Tutorial.ipynb)

## One ordinary test set

Fit a Gaussian Bayesian linear regression model to 20 observations.
Its posterior predictive distribution proposes a nominal 99% interval
at each new input.

The actual mean response is **quadratic**, so our straight-line model is
wrong. Its prior and predictive probabilities are working assumptions.

Freeze the fitted rule, then collect **10,000 independent input–response
pairs**. For each pair, check whether the response is inside the interval
proposed at its own input:

- **9,170** are inside: **91.70% measured marginal coverage**.
- One-sided Hoeffding subtracts **1.224 percentage points**.
- Rounding the lower bound downward gives:

> With 95% confidence, this frozen interval rule covers at least **90.4%**
> of future input–response pairs from the same population.

![Held-out outcomes inside and outside their own intervals](results/demo/marginal-validation.png)

**Marginal coverage** averages over the population of inputs. It does not
guarantee coverage at each particular input. There is no need to collect
repeated responses at exactly the same input.

The guarantee needs independent validation data from the future population,
a frozen prediction rule, and a sample size chosen in advance.
It does not need a correct linear model or Gaussian prior.

## Run

Python 3.10 or newer:

```bash
git clone --branch provable-uq-tutorial --single-branch https://github.com/jatinbatra50/jatinbatra50.git
cd jatinbatra50
python -m pip install -r requirements-notebook.txt
jupyter lab Provable_UQ_Tutorial.ipynb
```

The notebook is self-contained and includes executed outputs.
Regenerate the example with:

```bash
python experiments.py
```

All observations here are synthetic. Replace them with real held-out pairs
to validate a real application. The calculations are in [uq.py](uq.py);
[report.json](results/demo/report.json) records the experiment and coverage
bound. [References and attribution](ATTRIBUTION.md).
