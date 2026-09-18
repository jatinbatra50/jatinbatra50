# Contributing

Every proposed statistical claim should identify the target probability, the
independent sampling unit, what was chosen before validation, and the scope of
the error budget. Include a derivation or a precise reference. Label numerical
evidence as such; passing simulation tests is not a proof.

For a code change:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python experiments.py --trials 20 --mc-samples 30 --output-dir /tmp/uq-smoke
```

For a new experiment, state the seed, parameters, trial count, Monte Carlo
count, candidate family, target, and failure budget before seeing outcomes.
Do not tune on the final audit and retain its original guarantee. Save the
machine-readable report with any figures; keep quick checks separate from
reported scientific results.

Use GitHub-supported `$...$` and `$$...$$` math in Markdown. Keep variable
names consistent across proofs, code, and figures. New bounds should have
non-random tests against known cases and, where practical, exact finite
binomial enumeration of their failure probabilities. Add integration tests
for behavior that crosses the simulation/certificate boundary.
