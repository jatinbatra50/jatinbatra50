# References and attribution

This lesson uses two established calculations:

- Bayesian prediction with an unknown normal mean and known noise variance:
  Kevin Murphy, [Conjugate Bayesian analysis of the Gaussian distribution](https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf),
  Sections 2.3–2.4. The next observation's variance includes both observation
  noise and uncertainty about the mean.
- A lower confidence bound for a coverage rate:
  Wassily Hoeffding (1963), [Probability Inequalities for Sums of Bounded Random Variables](https://www.cs.rpi.edu/academics/courses/spring06/random/hoefding.pdf).
  Each new observation contributes 1 if it falls inside the fixed prediction
  interval and 0 otherwise.

[Payal's UQ_1](https://github.com/payal101/UQ_1) was the project's initial
inspiration. The current bottle example and lesson are standalone.
All example observations are synthetic.
