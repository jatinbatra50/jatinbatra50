# Keep the explanation concrete

Explain one Bayesian regression example in ordinary language. Distinguish
uncertainty about the mean line from variation in a new observation. Keep the
model's predictive probability separate from confidence in measured coverage.

`TUTORIAL.md` is the notebook source. To regenerate and check the project:

```bash
python scripts/build_tutorial_notebook.py
python scripts/execute_tutorial_notebook.py
python experiments.py
python -m unittest discover -s tests -v
```

Use independent training and validation pairs. Freeze the fitted predictor
and test count before checking coverage. The coverage claim averages over
the stated input distribution; it is not a guarantee for each individual input.

Preserve full precision in calculations and round a reported guaranteed
minimum downward. The notebook must work without importing project files.
