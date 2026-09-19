# Keep the guarantee point-specific

The model generates a candidate interval. The only asserted real coverage
guarantee comes from fresh measurements at the chosen input and a
concentration bound. Keep the nonlinear true sensor separate from the
assumed Gaussian linear model.

`TUTORIAL.md` is the notebook source. Regenerate and check with:

```bash
python scripts/build_tutorial_notebook.py
python scripts/execute_tutorial_notebook.py
python experiments.py
python -m unittest discover -s tests -v
```

Freeze the query, proposed interval, and measurement count before validation.
Every validation response must be drawn at that query. Do not substitute
random-input coverage for conditional coverage at a point, or posterior
samples for measurements from the actual process.

State that confidence applies to one chosen query and interval. Round
reported lower guarantees downward. Plot only the conditional validation
experiment and candidate endpoints; no mean or prediction bands across inputs.
Keep the notebook independent of project imports.
