# Provenance and scope

This tutorial develops the simulation study in
[payal101/UQ_1](https://github.com/payal101/UQ_1), starting from commit
`eaa3830` ("Revise README for clarity and structure"). The source notebook's
linear-regression generator, OLS fit, Monte Carlo construction, and original
coverage experiment are retained in `uq1.ipynb`. The notebook now also contains
the finite-sample validation layer and its executed output.

The additions are the reusable certificate implementation, mathematical
tutorial chapters, fast vectorized simulator, experiment runner, documented
results, and automated tests. The vectorized runner reproduces each nominal
level's statistical experiment but uses a different random-number protocol
and shares randomness across levels; its counts need not match the notebook.

Hoeffding inequalities, Bernoulli Chernoff bounds, KL inversion, union bounds,
and the elementary repeated-inspection construction are established
mathematical tools. This repository is a teaching and reproducibility project;
it does not present those tools as new research theorems. The extensions
chapter distinguishes proved consequences from possible further projects.

No license file was present in the source snapshot. This tutorial does not
assign a new license to the source material. Preserve source attribution and
obtain the relevant authors' permission for uses that require a license.

The numerical examples contain synthetic data only. No real experimental
measurements or private user data are included.
