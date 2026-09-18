# 1. Probability, prediction, and the quantity being validated

The first mathematical task in uncertainty quantification is to specify an
event and the randomness over which its probability is taken. A computation
can estimate a probability accurately while answering the wrong question.
This chapter defines the questions answered by this repository.

The implementation is in [`validation.py`](../validation.py), and
[`VALIDATION.md`](../VALIDATION.md) collects the theorem and assumptions.
Continue to [Chapter 2](02-concentration-and-certification.md) for its proof.

## 1.1 A model, an observation, and an estimator

Fix real numbers $\beta_0,\beta_1$, a noise standard deviation $\sigma>0$,
and a query location $x_\star$. The regression mean at that location is the
fixed, unknown quantity

$$
\mu_\star=\beta_0+\beta_1x_\star.
$$

A future response is the random variable

$$
Y_\star=\mu_\star+\varepsilon_\star,
\qquad \varepsilon_\star\sim N(0,\sigma^2).
$$

A training dataset $D=((X_i,Y_i))_{i=1}^n$ is another random object. In this
notebook, the $X_i$ are independent $N(0,1)$ variables and
$Y_i=\beta_0+\beta_1X_i+\varepsilon_i$, with mutually independent Gaussian
noises independent of the design. Fitting ordinary least squares produces
the random estimator

$$
\widehat\mu_\star(D)=\widehat\beta_0(D)+\widehat\beta_1(D)x_\star.
$$

The mean $\mu_\star$, its estimator $\widehat\mu_\star$, and the future
observation $Y_\star$ are different objects. Uncertainty about estimating the
mean does not include the future observation's additional noise.

Explicitly, with sample means $\bar X=n^{-1}\sum_iX_i$ and
$\bar Y=n^{-1}\sum_iY_i$, the fitted coefficients are

$$
\widehat\beta_1=
\frac{\sum_i(X_i-\bar X)(Y_i-\bar Y)}{\sum_i(X_i-\bar X)^2},
\qquad
\widehat\beta_0=\bar Y-\widehat\beta_1\bar X.
$$

The denominator is positive almost surely for this normal design when
$n\ge2$.

## 1.2 Three intervals with different meanings

A **confidence interval for the mean** is a data-dependent set $J(D)$ whose
repeated-sampling guarantee concerns the event $\mu_\star\in J(D)$.
A **prediction interval** $I(D)$ concerns $Y_\star\in I(D)$. Later we will
construct a third interval: a **confidence interval for coverage**, meaning
an interval estimating the probability that a prediction procedure succeeds.

| Object | Quantity it contains | Randomness in its coverage statement |
|---|---|---|
| Mean confidence interval | Fixed $\mu_\star$ | Training data |
| Prediction interval | Future response $Y_\star$ | Training data and future response |
| Coverage confidence interval | Fixed coverage probability $p$ | Independent audit trials |

For example, under Gaussian linear regression with an intercept, fixed
nondegenerate design, and $n>2$, let

$$
S_{xx}=\sum_{i=1}^n(X_i-\bar X)^2,
\qquad
h_\star=\frac1n+\frac{(x_\star-\bar X)^2}{S_{xx}},
$$

and let $s^2=\sum_i(Y_i-\widehat\beta_0-\widehat\beta_1X_i)^2/(n-2)$.
Writing $t_{n-2,1-\alpha/2}$ for the indicated Student quantile, the familiar
intervals are

$$
\widehat\mu_\star\pm t_{n-2,1-\alpha/2}s\sqrt{h_\star}
\quad\text{and}\quad
\widehat\mu_\star\pm t_{n-2,1-\alpha/2}s\sqrt{1+h_\star}.
$$

The first estimates the mean; the second predicts a response. Their exact
coverage is conditional on the design $(X_i)$, averaging over training
responses and, for prediction, the independent future response. This is not
the assertion that every realized dataset produces an interval with
conditional future-response coverage exactly $1-\alpha$.

The notebook's oracle simulation does not construct these textbook intervals.

## 1.3 The notebook's actual construction

Fix a nominal level $c\in(0,1)$ and an inner simulation size $R$. For each
$r=1,\ldots,R$, the notebook independently generates an entire new training
dataset $D_r$ using the **known true generating parameters**, fits OLS, and
draws independent noise $\eta_r\sim N(0,\sigma^2)$. It forms

$$
\widetilde Y_r=\widehat\mu_\star(D_r)+\eta_r.
$$

The empirical quantiles at $(1-c)/2$ and $(1+c)/2$ become the endpoints
$L(A)$ and $U(A)$ of $I(A)$. Here $A$ denotes all randomness in this
construction, including every $D_r$ and $\eta_r$. The quantile convention
used by the code is also part of the fixed procedure.

These simulated values are not distributed like the audited response
$Y_\star$. Conditional on one training design $X$,

$$
\widehat\mu_\star(D)\mid X\sim N(\mu_\star,\sigma^2h_X),
\qquad
\widetilde Y\mid X\sim N(\mu_\star,\sigma^2(1+h_X)),
$$

where $h_X=1/n+(x_\star-\bar X)^2/S_{xx}>0$. By comparison,
$Y_\star\sim N(\mu_\star,\sigma^2)$. The construction has added variation
from a newly fitted training sample.

This distinction has an observable consequence. For every $a>0$, writing
$\Phi$ for the standard normal distribution function,

$$
\Pr\{|\widetilde Y-\mu_\star|\le a\}
=E_X\!\left[2\Phi\!\left(\frac{a}{\sigma\sqrt{1+h_X}}\right)-1\right]
<2\Phi(a/\sigma)-1.
$$

Consequently, the limiting central quantile interval as $R\to\infty$
strictly overcovers $Y_\star$ relative to its nominal level. Finite $R$
introduces random quantile error; its actual coverage still needs analysis
or validation. See [`VALIDATION.md`](../VALIDATION.md#6-why-the-construction-distribution-differs-from-the-audit-distribution)
for the same comparison in the theorem reference.

Removing $\eta_r$ would produce quantiles of the oracle sampling distribution
of an estimator. It would not, by itself, create a confidence interval based
on one observed dataset. Likewise, fixing the simulated design conditions
on that design, not on observed response values.

## 1.4 Two legitimate coverage targets

The existing `coverage_test` constructs a fresh random interval and tests a
fresh, independent response on every outer repetition. Its target is

$$
p_{\rm procedure}
=\Pr_{A,Y_\star}\{Y_\star\in I(A)\}
=E_A\left[\Pr\{Y_\star\in I(A)\mid A\}\right].
$$

This probability averages over Monte Carlo construction randomness and
future-response noise at the specified $x_\star$ and model parameters.
It is a property of the entire randomized procedure, including $R$.

Alternatively, construct one interval $I(A_0)=[L_0,U_0]$, freeze its
endpoints, and audit fresh responses against it. Conditional on this
construction, the target is

$$
p_{\rm fixed}(A_0)=\Pr\{L_0\le Y_\star\le U_0\mid A_0\}.
$$

Here the known Gaussian model gives an exact diagnostic:

$$
p_{\rm fixed}(A_0)
=\Phi\left(\frac{U_0-\mu_\star}{\sigma}\right)
-\Phi\left(\frac{L_0-\mu_\star}{\sigma}\right).
$$

For $\sigma=0$, this is instead the indicator that $\mu_\star$ lies between
the endpoints. A certificate for $p_{\rm procedure}$ does not automatically
certify $p_{\rm fixed}(A_0)$ for a selected interval. Conversely, auditing
one frozen interval does not evaluate repeated reconstruction.

Neither target covers all query locations. An audit at $x_\star=1$ concerns
that location. Sampling random query locations defines a different,
population-averaged target; finitely many predeclared locations can instead
receive simultaneous certificates.

## 1.5 Turning an experiment into independent trials

For the procedure target, independently generate pairs $(A_i,Y_i)$ for
$i=1,\ldots,N$, with $A_i$ independent of $Y_i$. Define

$$
C_i=\mathbf1\{Y_i\in I(A_i)\},\qquad
S=\sum_{i=1}^N C_i,\qquad \widehat p=S/N.
$$

The $C_i$ are independent, identically distributed Bernoulli variables with
mean $p_{\rm procedure}$. For the frozen-interval target, use the same
definition with $I(A_i)$ replaced by $I(A_0)$; the Bernoulli conclusion then
holds conditional on $A_0$.

Only the $N$ outer trials determine audit precision. The $R$ inner samples
help construct one interval; they are not $R$ additional successful or
unsuccessful validation trials. Increasing $R$ changes the procedure being
tested, while increasing $N$ measures that procedure more precisely.

With a learned procedure trained or tuned on external data $T$, the same
reasoning applies conditional on $T$, provided the audit data are independent
of $T$ and the tested procedure is then frozen. Integrating a conditional
$1-\delta$ guarantee over $T$ preserves that confidence level. Tuning on
audit outcomes violates this setup.

All guarantees concern the specified data-generating law. Independence is
an assumption about the experiment, not a fact that a favorable empirical
coverage establishes.

## Exercises

1. Write the probability measured when one interval is tested against
   $N$ responses. Then write the probability measured when $N$ independently
   reconstructed intervals are each tested once. Which construction does
   the notebook use?
2. Suppose the future query $X_\star$ is random. Define marginal coverage
   and coverage conditional on $X_\star=x$. Explain why the first can be high
   when the second is poor at some locations.
3. Construct a randomized procedure whose average coverage is $0.95$ but
   whose realized intervals sometimes have coverage zero.
4. Derive the conditional variance $\sigma^2h_X$ of the OLS fitted mean
   directly from the formulas for the slope and intercept.

[Solutions and discussion](05-exercises.md#chapter-1-probability-and-targets).
