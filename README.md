# Uncertainty quantification, one example at a time

**Predict a value. Put an interval around it. Measure how often that interval works.**

A short, runnable introduction to uncertainty quantification (UQ), using one
noisy sensor and ordinary Python. You will build prediction intervals with
split conformal calibration, then use Hoeffding's inequality to choose enough
fresh measurements for a precise coverage estimate.

**Start here: [the executed Jupyter notebook](Provable_UQ_Tutorial.ipynb)**
or [read the tutorial on GitHub](TUTORIAL.md).

## What you will learn

| Question | Tool | What you get |
|---|---|---|
| What reading should I predict? | Linear regression | A point prediction |
| How much can the next reading vary? | Conformal calibration | A prediction interval |
| How often does the interval contain the reading? | Fresh measurements | A coverage estimate |
| How accurate is that estimate? | Hoeffding's inequality | A finite-sample error bar |
| How many measurements do I need? | A sample-size formula | A budget chosen before sampling |

![A prediction interval around a fitted sensor response](results/demo/prediction-interval.png)

## A deliberately precise example

The example uses 200 training observations and 4,000 separate calibration
observations. It targets 97.5% prediction coverage and then takes **73,778 fresh
measurements**. That measurement count guarantees an error bar of at most
**±0.5 percentage points at 95% confidence** for the coverage of the fitted,
calibrated predictor, under independent sampling from the same distribution.

The three settings do different jobs:

- `alpha = 0.025`: the prediction interval's target miss rate.
- `epsilon = 0.005`: the desired coverage-measurement error.
- `delta = 0.05`: the error probability allowed for that measurement guarantee.

The notebook shows the actual estimate, interval width, and numerical error
bar. Change a setting, rerun from the top, and see what changes.

**Saved run:** measured coverage **97.39%**, with a **[96.89%, 97.89%]**
confidence interval. The prediction band is **2.236 sensor units** wide.

## Run it

Python 3.10 or newer, NumPy, and Matplotlib are enough for the scripts.
The notebook contains all its code and requires no data downloads or local
project imports.

```bash
git clone --branch provable-uq-tutorial --single-branch https://github.com/jatinbatra50/jatinbatra50.git
cd jatinbatra50
python -m pip install -r requirements-notebook.txt
jupyter lab Provable_UQ_Tutorial.ipynb
```

To regenerate the saved example without Jupyter:

```bash
python experiments.py
```

For twice the precision, take roughly four times as many measurements:

```bash
python experiments.py --epsilon 0.0025 --output-dir results/finer
```

## Files

| File | Purpose |
|---|---|
| [TUTORIAL.md](TUTORIAL.md) | The short lesson, with formulas, code, and figures |
| [Provable_UQ_Tutorial.ipynb](Provable_UQ_Tutorial.ipynb) | The same lesson, fully executable with saved outputs |
| [uq.py](uq.py) | Reusable UQ functions |
| [experiments.py](experiments.py) | Reproduce the example and figures |
| [results/demo/report.json](results/demo/report.json) | Settings and numerical results |
| [tests/test_uq.py](tests/test_uq.py) | Checks of calibration, measurement bounds, and the example |

All observations here are synthetic. See [attribution and references](ATTRIBUTION.md)
for the original inspiration and the statistical methods used.
