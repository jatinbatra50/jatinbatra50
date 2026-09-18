# Finite-sample coverage validation: theorem and proof

## 1. Define the probability before estimating it

Fix a nominal interval level `c` and all simulation settings. Let `A` denote
all randomness used to construct an interval `I(A)=[L(A),U(A)]`, including
all `R` simulated training datasets and future-noise draws. Independently draw

$$
Y=\mu+\varepsilon,\qquad
\mu=\beta_0+\beta_1x_\star,\quad
\varepsilon\sim N(0,\sigma^2).
$$

The target of `coverage_test` is

$$
p=\Pr_{A,Y}\{Y\in I(A)\}
=\mathbb E_A\big[\Pr_Y\{Y\in I(A)\mid A\}\big].
$$

For independent repetitions `(A_i,Y_i)`, define

$$
C_i=\mathbf1\{L(A_i)\le Y_i\le U(A_i)\},\qquad
S=\sum_{i=1}^{N}C_i,\qquad \widehat p=S/N.
$$

These are iid Bernoulli observations with mean `p`. The outer count `N`, not
`N*R`, determines the precision of the validation. Changing `R` changes the
randomized procedure and hence potentially `p`.

This reduction does not need Gaussian response noise: the same validation
theorem works for any distribution if the coverage indicators are iid and
the tested procedure is fixed independently of the audit data. The Gaussian
assumption describes this repository's chosen simulation target.

## 2. Simultaneous confidence intervals

Predeclare `K` configurations. Configuration `k` has true coverage `p_k`,
sample size `N_k`, and empirical coverage `p_hat_k`. Let `0<delta<1` and set

$$
b=\log(2K/\delta).
$$

**Theorem.** Each of the following constructions satisfies

$$
\Pr\{p_k\in[L_k,U_k]\text{ for every }k=1,\ldots,K\}\ge1-\delta.
$$

Each configuration requires iid trials; dependence between configurations
is allowed. The bounds are conditional on any independent training or tuning
data used to freeze the procedures.

**Hoeffding.** Set

$$
r_k=\sqrt{\frac b{2N_k}},\quad
L_k=\max(0,\widehat p_k-r_k),\quad
U_k=\min(1,\widehat p_k+r_k).
$$

**Binary-KL Chernoff.** Define, with the continuous endpoint conventions,

$$
\operatorname{kl}(a\Vert q)
=a\log\frac a q+(1-a)\log\frac{1-a}{1-q}.
$$

Let `[L_k,U_k]` be the sublevel interval

$$
\{q\in[0,1]:N_k\operatorname{kl}(\widehat p_k\Vert q)\le b\}.
$$

For interior empirical coverage, find its two endpoints by monotone
bisection, one on each side of `p_hat_k`. Endpoints have closed forms:

$$
\widehat p_k=0:\quad [L_k,U_k]=[0,1-e^{-b/N_k}],
\qquad
\widehat p_k=1:\quad [L_k,U_k]=[e^{-b/N_k},1].
$$

`validation.py` implements this inversion and uses outward bisection endpoints
with a final outward floating-point step. This is ordinary double-precision
numerical evaluation of the theorem, not formally verified interval arithmetic.
Binary KL obeys `kl(a||q) >= 2(a-q)^2`, so its sublevel interval is contained
in the corresponding Hoeffding interval (up to numerical rounding).

### Proof

For a Bernoulli variable of mean `p`,

$$
\mathbb E e^{t C}=1-p+pe^t.
$$

Independence and exponential Markov give, for `a>p`,

$$
\Pr\{\widehat p\ge a\}
\le\inf_{t>0}e^{-tNa}(1-p+pe^t)^N
=e^{-N\operatorname{kl}(a\Vert p)}.
$$

The optimizer for interior `a,p` satisfies
`exp(t)=a(1-p)/(p(1-a))`. Negative `t` yields the lower-tail bound for `a<p`;
endpoint cases follow by limits.

For a fixed true `p`, `kl(a||p)` increases as `a` moves away from `p` on
either side. The event `p<L` therefore lies in an upper tail of `p_hat`
with probability at most `exp(-b)`; the event `p>U` lies in a lower tail
with the same bound. This also follows by choosing the first attainable
binomial count in each tail. Thus the total failure probability for one
configuration is at most `2*exp(-b)=delta/K`.

Hoeffding's Bernoulli tail bounds

$$
\Pr\{\widehat p-p\ge r\}\le e^{-2Nr^2},\qquad
\Pr\{p-\widehat p\ge r\}\le e^{-2Nr^2}
$$

follow, for example, from the preceding Chernoff bounds and
`kl(a||p) >= 2(a-p)^2`. Substituting `r=sqrt(b/(2N))` gives the same
per-configuration failure budget. A union bound across the `K`
configurations proves both simultaneous statements. No independence across
configurations is used. QED.

## 3. What can the certificate establish?

For a prespecified minimum `p0`, report:

- `certified` if `L >= p0`;
- `below_target` if `U < p0`;
- `inconclusive` otherwise.

On the simultaneous confidence event, every affirmative decision of these
types is correct. The frequentist statement concerns the randomness of
validation data; it is not a posterior probability assigned to a fixed `p`.

There are two distinct notions of uncertainty: a nominal 95% prediction
interval, and a 95% confidence interval for that procedure's true coverage.
The latter needs independent validation. A point estimate of 95.6% does not
by itself establish that the true coverage is at least 95%.

To certify *calibration within tolerance* `tau`, require the full band to
lie in `[c-tau,c+tau]` (intersected with `[0,1]`). To certify minimum coverage
with tolerated shortfall `tau`, use `target_coverage=c-tau`. Choose `tau`
before examining outcomes. Very wide intervals can have excellent coverage;
this certificate says nothing by itself about informativeness or width.

## 4. Sample size and repeated use

To ensure simultaneous absolute estimation error at most `epsilon`,
Hoeffding suffices with

$$
N\ge\left\lceil\frac{\log(2K/\delta)}{2\epsilon^2}\right\rceil.
$$

At `N=1000`, `K=4`, `delta=.05`, the radius is about `0.0503745`.
One percentage-point precision requires `N=25376` per configuration by
this conservative sufficient bound. This is a precision calculation, not
a guarantee that the lower bound will exceed a given target. Certifying
coverage at the target requires favorable data, usually reflecting a true
margin above that target.

KL can be much sharper near coverage one. If all `N` trials succeed, its
lower bound is `exp(-b/N)`. Thus all-success data certify `p>=p0` whenever
`N >= ceil(b/(-log(p0)))`, ignoring a possible last-bit rounding effect at
exact equality. This is a best-case count, not a success-power guarantee.

All bounds here allocate `delta/(2K)` to each tail. If *only* lower bounds
are needed, a one-sided derivation can replace `2K` with `K`; the current API
intentionally returns two-sided bounds, also supporting undercoverage
detection and calibration plots.

Predeclare sample sizes and comparisons. Include all configurations selected
using the audit in `K`; this only protects a genuinely predeclared finite
family. Creating new configurations after looking at results is not repaired
by simply incrementing `K`. Repeated checking and optional stopping require
an anytime-valid method or a predetermined union-bound allocation across
inspection times. Retuning on an audit requires fresh validation data.

The two implemented methods each obey the theorem. At identical inputs the
KL interval is contained in Hoeffding, so their joint coverage follows from
the KL event without an extra correction. For additional non-nested methods,
allocate error across methods if they are jointly reported or selected using
validation results.

## 5. A particular displayed interval is a different target

Once a single interval `[lower,upper]` has been constructed, freeze it and
audit it against fresh true responses. Conditional on that interval, this
validates `p_fixed=Pr{lower <= Y <= upper}`. For example, after the notebook's
interval-construction cell, run this as a separately planned audit:

```python
import numpy as np
from validation import coverage_certificate

# Freeze lower/upper first; choose audit settings before seeing outcomes.
audit_rng = np.random.default_rng(271828)
N = 10000
mu, sigma = 1.0 + 2.0 * 1.0, 1.0  # must match the generating model
fresh_y = audit_rng.normal(mu, sigma, size=N)
successes = int(np.count_nonzero((lower <= fresh_y) & (fresh_y <= upper)))
fixed_report = coverage_certificate(
    successes, N, delta=0.05, target_coverage=0.95,
)
print(fixed_report.to_dict())
```

This example spends its own 5% failure budget. For a joint claim including
the notebook's four-level audit, allocate the total error budget across both
studies rather than claiming their separate 95% guarantees are jointly 95%.

In this oracle Gaussian example there is also an exact diagnostic for
`sigma>0`:

$$
p_{\rm fixed}=\Phi((U-\mu)/\sigma)-\Phi((L-\mu)/\sigma),
$$

where `Phi` is the standard normal cumulative distribution function. This
does not equal the average over newly constructed intervals in Section 1.
For `sigma=0`, the exact probability is simply the indicator that the
deterministic response `mu` lies inside the interval.

## 6. Why the construction distribution differs from the audit distribution

For one random training design `X`, write

$$
h_X=\frac1n+\frac{(x_\star-\bar x)^2}{S_{xx}}>0.
$$

Conditional on `X`, OLS prediction plus independent future noise has law

$$
\widetilde Y\mid X\sim N(\mu,\sigma^2(1+h_X)),
$$

whereas the audited future response has law `N(mu,sigma^2)`. For `sigma>0`
and any `a>0`,

$$
\Pr\{|\widetilde Y-\mu|\le a\}
=\mathbb E_X\left[2\Phi\!\left(\frac a{\sigma\sqrt{1+h_X}}\right)-1\right]
<2\Phi(a/\sigma)-1.
$$

The symmetric central interval from the limiting construction distribution
therefore covers a true future response with probability strictly greater
than its nominal level. This statement assumes a nondegenerate design
(almost surely for the notebook's normal design with `n>=2`). Finite `R`
adds random quantile error; the coverage certificate evaluates the actual
finite-`R` procedure without assuming exact nominal calibration.

## References

- W. Hoeffding (1963), [Probability Inequalities for Sums of Bounded Random
  Variables](https://doi.org/10.1080/01621459.1963.10500830), Journal of the
  American Statistical Association 58(301), 13–30.
- H. Chernoff (1952), [A Measure of Asymptotic Efficiency for Tests of a
  Hypothesis Based on the Sum of Observations](https://doi.org/10.1214/aoms/1177729330),
  Annals of Mathematical Statistics 23(4), 493–507.
