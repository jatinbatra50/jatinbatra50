# 5. Exercises, solutions, and small projects

[Contents](../README.md) · [Probability](01-probability-and-targets.md) ·
[Concentration](02-concentration-and-certification.md) ·
[Implementation](03-from-theorem-to-code.md) · [Extensions](04-extensions-and-limits.md)

Try each exercise before reading its solution. The first two sections answer
the exercises posed in the corresponding chapters; the last section contains
additional mathematical and programming tasks.

## Chapter 1: Probability and targets

### 1. One interval or many intervals?

**Question.** Write the probability estimated by testing one constructed
interval against $N$ fresh responses, then by independently rebuilding the
interval on each of $N$ trials.

**Solution.** If $A_0$ denotes the first interval's construction randomness,
the first experiment is conditional on $A_0$ and estimates

$$
q(A_0)=\Pr_Y\{Y\in I(A_0)\mid A_0\}.
$$

The second estimates

$$
p=\Pr_{A,Y}\{Y\in I(A)\}=\mathbb E_A q(A).
$$

The original notebook uses the second protocol. The new runner demonstrates
both, with distinct validation data and declared error budgets. An average
of conditional probabilities need not describe every conditional probability.

### 2. Marginal coverage need not hold at each query

**Question.** Define marginal and query-conditional coverage and construct a
counterexample to their equivalence.

**Solution.** For a fixed prediction procedure, let

$$
q(x)=\Pr\{Y\in I(X)\mid X=x\},\qquad
p=\Pr\{Y\in I(X)\}=\mathbb E q(X).
$$

Conditional probabilities are understood where a version is defined; claims
about every point require additional regularity or an explicit finite model.
Take $X\in\{0,1\}$ with $\Pr(X=0)=.99$ and deterministic $Y=0$. Use
$I(0)=[-1,1]$ and $I(1)=[1,2]$. Then $p=.99$ while $q(1)=0$.
A random-query audit estimates $p$; it does not establish $q(1)\ge.99$.

### 3. Average coverage can hide disastrous realized intervals

**Question.** Construct a randomized interval procedure with average coverage
$.95$ whose output sometimes has zero coverage.

**Solution.** Let $Y$ be uniform on $[0,1]$. Independently of $Y$, output
$[0,1]$ with probability $.95$ and $[2,3]$ otherwise. The conditional coverage
of the output is either $1$ or $0$, while its expectation is $.95$.
Repeatedly reconstructing intervals measures the expectation. Freezing one
output and auditing it measures that output's conditional coverage.

### 4. Derive the OLS prediction variance

**Question.** Prove that, conditional on the training design,
$\operatorname{Var}(\widehat\mu_\star\mid X)=\sigma^2h_X$.

**Solution.** Write $u_i=X_i-\bar X$ and $S_{xx}=\sum_i u_i^2$.
The slope error is

$$
\widehat\beta_1-\beta_1=\frac{\sum_i u_i\varepsilon_i}{S_{xx}}.
$$

Using $\widehat\mu_\star=\bar Y+\widehat\beta_1(x_\star-\bar X)$,

$$
\widehat\mu_\star-\mu_\star
=\sum_{i=1}^n w_i\varepsilon_i,
\qquad
w_i=\frac1n+\frac{(x_\star-\bar X)u_i}{S_{xx}}.
$$

Because $\sum_i u_i=0$,

$$
\sum_iw_i^2=\frac1n+\frac{(x_\star-\bar X)^2}{S_{xx}}=h_X.
$$

The errors are independent centered normal variables of variance $\sigma^2$
conditional on the design. A weighted sum is therefore normal with mean zero
and variance $\sigma^2\sum_i w_i^2$. Adding independent future noise gives
variance $\sigma^2(1+h_X)$, which explains the construction distribution.

## Chapter 2: Concentration and certification

### 1. The lower-tail optimizer

**Question.** Derive the Chernoff optimizer for a threshold $a<p$.

**Solution.** For $t<0$, the event $\sum_iC_i\le Na$ implies
$\exp(t\sum_iC_i)\ge e^{tNa}$. Exponential Markov gives the same objective
$e^{-tNa}(1-p+pe^t)^N$. Its stationary point obeys

$$
e^{t_*}=\frac{a(1-p)}{p(1-a)}<1.
$$

Thus $t_*<0$, as required. The second derivative of the log moment generating
function is nonnegative, so this is the minimizer. Substitution yields
$e^{-N\operatorname{kl}(a\Vert p)}$. Endpoint cases follow by limits or
by direct evaluation of all-zero/all-one events.

### 2. Why two bisections work

**Question.** Show that a KL sublevel set is an interval and determine its
endpoint cases. Which numerical bracket endpoints are conservative?

**Solution.** At a fixed empirical proportion $a\in(0,1)$,

$$
\partial_q\operatorname{kl}(a\Vert q)=\frac{q-a}{q(1-q)}.
$$

The function decreases to zero at $q=a$ and then increases, diverging at
both ends. A positive sublevel set is an interval whose endpoints are its
unique left and right roots. If $a=0$, the inequality is
$-N\log(1-q)\le b$, giving $[0,1-e^{-b/N}]$. If $a=1$, it is
$-N\log q\le b$, giving $[e^{-b/N},1]$.

For the lower root return the bracket endpoint farther toward zero; for the
upper root return the endpoint farther toward one. Those choices enlarge
the interval. They avoid a directional error in bracket selection, although
ordinary floating-point evaluation still has rounding error.

### 3. Interpret 953 successes

**Question.** Compute Hoeffding and KL intervals for $S=953$, $N=1000$,
$K=4$, and $\delta=.05$. What calibration tolerance around $.95$ would contain
the KL interval?

**Solution.** Run:

```python
from validation import coverage_certificate
for method in ("hoeffding", "kl"):
    c = coverage_certificate(953, 1000, delta=.05, num_claims=4,
                             method=method, target_coverage=.95)
    print(method, c.lower, c.upper, c.status)
```

Hoeffding gives approximately $[.90263,1]$; KL gives approximately
$[.92857,.97133]$. Neither certifies minimum coverage $.95$.
The smallest symmetric tolerance containing the KL interval is

$$
\tau_* = \max\{.95-L,U-.95\}\approx .02143.
$$

For example, a tolerance of $.022$ would contain the band. This is an
interpretation exercise, not a recommendation to choose the scientific
acceptance tolerance after seeing the audit. A prespecified $.022$ tolerance
would support a calibration-within-tolerance decision on this confidence
event. The confidence set itself remains valid without a threshold; reporting
its endpoints directly avoids pretending an acceptance rule was prespecified.

### 4. Power for all candidates

**Question.** Suppose every true coverage satisfies $p_k\ge p_{0,k}+\gamma$
for a common $\gamma>0$. Find a sufficient sample size for all Hoeffding
minimum-coverage certificates to pass with probability at least $1-\eta$.

**Solution.** The first union bound, over $2K$ confidence tails, sets
$r=\sqrt{\log(2K/\delta)/(2N)}$. A separate union bound on the probabilities
of unfavorable empirical fluctuations gives

$$
\Pr\left\{\forall k:\widehat p_k\ge
p_k-\sqrt{\frac{\log(K/\eta)}{2N}}\right\}\ge1-\eta.
$$

On that event, each lower confidence bound reaches its target whenever

$$
\sqrt{\frac{\log(2K/\delta)}{2N}}+
\sqrt{\frac{\log(K/\eta)}{2N}}\le\gamma.
$$

A sufficient integer sample size is therefore

$$
N\ge\left\lceil
\frac{(\sqrt{\log(2K/\delta)}+\sqrt{\log(K/\eta)})^2}{2\gamma^2}
\right\rceil.
$$

The parameter $\delta$ controls false claims; $\eta$ controls non-passing under
a positive-margin alternative. They have different roles. At $\gamma=0$,
this argument offers no high-probability passing guarantee.

## Further mathematical and programming exercises

### 1. One image with a million identical pixels

**Task.** Let all $d$ coverage indicators in image $i$ equal the same
Bernoulli variable $B_i$. Compare the image-average coverage estimator with
an incorrect calculation treating $Nd$ pixels as independent.

**Solution.** The empirical pixel average is exactly
$N^{-1}\sum_iB_i$. Its variance is $p(1-p)/N$, independent of $d$. Treating
pixels as independent would incorrectly divide this variance by $d$, or
shrink a Hoeffding radius by $\sqrt d$. Test code should preserve the image
axis and aggregate within each image before applying a bounded-mean theorem.

### 2. Rare failures and the cost of certification

**Task.** With no observed failures, derive the trial count needed to certify
coverage at least $1-\alpha$ and explain what this does not guarantee.

**Solution.** The two-sided KL lower bound is
$\exp[-\log(2K/\delta)/N]$. It is at least $1-\alpha$ if

$$
N\ge\frac{\log(2K/\delta)}{-\log(1-\alpha)}.
$$

As $\alpha\downarrow0$, $-\log(1-\alpha)\sim\alpha$, giving a count of
order $\log(K/\delta)/\alpha$. The calculation assumes all trials succeeded.
It does not promise that they will: if the true coverage is $p$, the
probability of observing all successes is $p^N$.

### 3. Implement a valid stopping rule

**Task.** Use the wrapper in Chapter 4, stop at the first time its lower
bound exceeds a fixed target, and explain the guarantee even though the
stopping time depends on the data.

**Solution.** Use $\delta_t=6\delta/(\pi^2t^2)$ for the certificate at
cumulative trial count $t$. With probability at least $1-\delta$, all
reported bands contain their target probabilities for every $t$. On that
single event any finite data-dependent selection of a time also has a valid
band. There is no promise that the stopping time is finite. Replacing
$\delta_t$ by the same fixed $\delta$ on every inspection loses this proof.

### 4. Add a bounded-mean API

**Task.** Design an interface for image-average coverage observations in
$[0,1]$ without converting them to binomial counts.

**Solution sketch.** Accept the number of independent images $N$ and their
empirical mean $\widehat m$, validate $N\ge1$ and $\widehat m\in[0,1]$,
and invert $N\operatorname{kl}(\widehat m\Vert q)\le\log(2K/\delta)$.
The proof is the bounded-observation moment inequality in Chapter 4. Keep
this API distinct from integer Bernoulli counts, and do not call an exact
binomial interval routine on fractional successes. Add tests for deterministic
fractional observations, endpoint means, and the pixel-dependence example.
This is a proposed extension, not an implemented public function here.

### 5. Build a comparison that remains valid after selection

**Task.** Predeclare several interval-width multipliers and select the
narrowest candidate whose coverage is certified. Explain which claim is
supported and what must be done if no candidate passes.

**Solution sketch.** Include every candidate in the simultaneous budget.
On the event that all bands are valid, whichever passing candidate is selected
has true coverage above the target. If none passes, report no certified
candidate; do not lower the target or invent a new multiplier using the same
audit and pretend it was one fixed candidate. If expected width is also to
be certified, define its statistical target and supply an appropriate bound.

### 6. Explain the Gaussian benchmark's limits

**Task.** Why can the runner compute the exact coverage of a frozen interval,
and why is that not already a certificate for an unknown physical system?

**Solution.** Here the response distribution and its true parameters are
explicitly supplied. For a fixed interval its probability is a Gaussian CDF
difference. In a physical system, those parameters and even the model family
may be uncertain. Simulator agreement does not establish their correctness.
Chapter 4's total-variation transfer statement shows exactly where an
additional justified model-discrepancy bound would enter.
