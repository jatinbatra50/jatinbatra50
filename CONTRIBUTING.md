# Keep the lesson small and runnable

Use one example throughout. Explain what a number means before adding another
formula. Keep the notebook self-contained and retain the distinction between
prediction coverage and confidence in a measured coverage rate.

`TUTORIAL.md` is the notebook's source. After editing it, run:

```bash
python scripts/build_tutorial_notebook.py
python scripts/execute_tutorial_notebook.py
python experiments.py
python -m unittest discover -s tests -v
```

Commit the executed notebook and regenerated results together. Keep the
training, calibration, and measurement samples separate. Choose measurement
precision and sample size before drawing the measurement sample.

`requirements-reproduce.txt` records the versions used for the saved example.
CI also runs the tutorial and tests on Python 3.10 and 3.12.
