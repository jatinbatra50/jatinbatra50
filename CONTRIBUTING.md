# Keep the coverage claim precise

The model proposes an interval rule. Independent held-out input–response
pairs and a concentration bound supply its marginal coverage guarantee.
Keep the nonlinear true process separate from the assumed Gaussian linear
model.

`TUTORIAL.md` is the notebook source. Regenerate and check with:

```bash
python scripts/build_tutorial_notebook.py
python scripts/execute_tutorial_notebook.py
python experiments.py
python -m unittest discover -s tests -v
```

Freeze the fitted rule, nominal level, and sample size before validation.
Each response is checked against the interval at its own input. Validation
pairs are independent of training and each other, and come from the future
population. Do not substitute posterior draws for actual observations.

State that the bound averages over random inputs with the trained rule held
fixed. Do not claim coverage at each input. The confidence level concerns one
frozen rule and validation experiment. If the rule is revised using validation
results, check it on fresh data.

Round reported lower guarantees downward. Plot validation counts without
mean or predictive bands across inputs. Keep the notebook independent of
project imports and keep the prose short; no bound proofs are needed.
