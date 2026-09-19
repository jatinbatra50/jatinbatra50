# Keep the explanation concrete

Explain the bottle example in ordinary language. Keep the model's prediction
separate from the evidence supplied by new measurements.

`TUTORIAL.md` is the notebook source. To regenerate and check the project:

```bash
python scripts/build_tutorial_notebook.py
python scripts/execute_tutorial_notebook.py
python experiments.py
python -m unittest discover -s tests -v
```

Keep the training and validation observations independent. Fix the interval
and test count before looking at validation outcomes. Preserve full precision
in calculations and round a reported guaranteed minimum downward.
The notebook must work without importing project files.
