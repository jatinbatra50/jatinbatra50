# Test a proposed interval at one input

**The model proposes an interval. Repeated measurements at the chosen input
supply the coverage guarantee.**

[Read the lesson](TUTORIAL.md) · [Run the notebook](Provable_UQ_Tutorial.ipynb)

## One input, one coverage statement

Fix **x = 0.5** before validation. A Gaussian Bayesian linear regression model
proposes **[1.516, 4.196]** for the next response at that input.

The actual sensor in this example is **nonlinear**. The linear model and its
Gaussian coefficient prior are only working assumptions; we do not trust
the model's nominal 99% probability.

Take **10,000 fresh, independent responses at exactly x = 0.5**:

- **9,745** land inside the proposed interval: **97.45% measured coverage**.
- A one-sided Hoeffding calculation subtracts **1.224 percentage points**.
- Rounded down, the supported statement is:

> At x = 0.5, with 95% confidence, this fixed interval covers at least
> **96.2% of future responses** from the same sensor at that input.

![Repeated measurements at the chosen input, with candidate interval endpoints](results/demo/point-validation.png)

There are no uncertainty bands across inputs. The endpoint lines mark the
candidate interval; the coverage statement comes entirely from validation.

## What is required

The chosen input, interval, and measurement count are fixed before checking
outcomes. Validation uses independent responses from the actual conditional
distribution at that input. The model and prior may be wrong.

This is a 95% confidence statement for **this input and interval**. It is not a
simultaneous guarantee over other inputs. To check another input, perform
a corresponding conditional validation experiment.

Randomly located test points generally cannot certify an exact new input
without additional assumptions. You need repeated measurements at that input
or access to its true conditional distribution. A simulator supplies such
a claim only for the process it faithfully represents.

## Run

Python 3.10 or newer:

```bash
git clone --branch provable-uq-tutorial --single-branch https://github.com/jatinbatra50/jatinbatra50.git
cd jatinbatra50
python -m pip install -r requirements-notebook.txt
jupyter lab Provable_UQ_Tutorial.ipynb
```

The notebook is self-contained. Regenerate its numerical example with:

```bash
python experiments.py
```

To run a separate check at a different, prechosen input:

```bash
python experiments.py --query 0.8 --output-dir results/query-08
```

Each run has its own pointwise confidence statement; several such statements
do not automatically give a joint 95% guarantee.

All observations here are synthetic. The reusable calculations are in
[uq.py](uq.py); [report.json](results/demo/report.json) records the input,
interval, counts, and confidence scope.
[References and attribution](ATTRIBUTION.md).
