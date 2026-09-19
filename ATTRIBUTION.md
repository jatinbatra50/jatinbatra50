# References and attribution

- **A proposal mechanism:** Christopher M. Bishop,
  *Pattern Recognition and Machine Learning* (2006), Sections 3.3.1–3.3.2,
  Equations 3.52–3.59.
  [Author's book page](https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/)
  and [Chapter 3 slides](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/05/prml-slides-3.pdf).
  The Gaussian Bayesian linear regression calculation supplies a candidate
  interval. In this example the actual mean response is quadratic, outside
  the assumed linear family. No correctness of the working model or its
  Gaussian prior is used in validation.
- **The coverage guarantee:** Wassily Hoeffding (1963),
  [Probability Inequalities for Sums of Bounded Random Variables](https://www.cs.rpi.edu/academics/courses/spring06/random/hoefding.pdf).
  Each independent response at the fixed input contributes 1 if it lies
  inside the frozen candidate interval and 0 otherwise. The tutorial uses
  the one-sided exponential bound.

[Payal's UQ_1](https://github.com/payal101/UQ_1) was the initial inspiration.
The current example is standalone and uses synthetic observations.
