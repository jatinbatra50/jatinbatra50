# References and attribution

- **Bayesian linear regression:** Christopher M. Bishop,
  *Pattern Recognition and Machine Learning* (2006), Section 3.3,
  especially Sections 3.3.1 (parameter distribution) and 3.3.2
  (predictive distribution), Equations 3.52–3.59.
  [Author's book page](https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/).
  The author's [Chapter 3 slides](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/05/prml-slides-3.pdf)
  show the same calculations. This lesson uses the zero-mean isotropic Gaussian weight prior and fixed
  Gaussian observation noise. It holds the two precision parameters fixed.
- **Coverage validation:** Wassily Hoeffding (1963),
  [Probability Inequalities for Sums of Bounded Random Variables](https://www.cs.rpi.edu/academics/courses/spring06/random/hoefding.pdf).
  Each independent test pair contributes 1 when the observed output falls
  inside the frozen predictor's interval for its input, and 0 otherwise.

[Payal's UQ_1](https://github.com/payal101/UQ_1) was the initial inspiration.
The current regression example and lesson are standalone.
All example observations are synthetic.
