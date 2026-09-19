# Attribution and references

[Payal's UQ_1](https://github.com/payal101/UQ_1) was the starting inspiration
for this project. This edition is a standalone UQ lesson with a new synthetic
sensor example. It makes no assessment of that repository's results.

The methods are established statistical tools:

- **Split conformal prediction:** Angelopoulos and Bates,
  [A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification](https://arxiv.org/abs/2107.07511),
  especially Section 1.1. The tutorial uses absolute residuals and an exact
  order statistic for calibration.
- **Finite-sample measurement error:** Hoeffding (1963),
  [Probability Inequalities for Sums of Bounded Random Variables](https://www.cs.rpi.edu/academics/courses/spring06/random/hoefding.pdf).
  The tutorial applies the two-sided inequality to independent coverage
  indicators, which take values in {0, 1}.

The code, examples, and explanations in this edition are written for teaching.
The generated observations contain no private or real experimental data.
The original repository supplied no license in the source snapshot; this
project does not grant rights to material from that repository.
