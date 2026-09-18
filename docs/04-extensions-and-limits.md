# 4. Extensions, dependence, and the limits of a certificate

The preceding chapters establish a finite-sample guarantee for a fixed
validation protocol. This chapter asks how that guarantee changes when we
freeze one interval, predict an image, compare methods, or stop an experiment
early. Start with [the probability being estimated](01-probability-and-targets.md)
and [the concentration theorem](02-concentration-and-certification.md).

Throughout, logarithms are natural, $0<\delta<1$ is a failure budget, and
training and tuning are completed independently of the validation data unless
explicitly stated otherwise. Independence is a property of the experimental
design; the certificate cannot verify it from a success count.

## 4.1 A random procedure and one frozen interval answer different questions

Let $A$ contain the randomness used to construct an interval $I(A)$. Let $Y$
be an independent future response from the population of interest. Define

$$
q(a)=\Pr_Y\{Y\in I(a)\},
\qquad
p_{\mathrm{procedure}}=\Pr_{A,Y}\{Y\in I(A)\}=\mathbb E_A q(A).
$$

There are two valid experiments:

1. **Reconstruct on every trial.** Draw independent pairs $(A_i,Y_i)$ and
   record $C_i=\mathbf 1\{Y_i\in I(A_i)\}$. These are independent Bernoulli
   variables with mean $p_{\mathrm{procedure}}$.
2. **Freeze once.** Construct $A_\star$ independently of the audit, then draw
   fresh independent $Y_1,\ldots,Y_N$. Conditional on $A_\star=a$, the indicators
   $C_i=\mathbf 1\{Y_i\in I(a)\}$ are independent Bernoulli variables with mean
   $q(a)$.

For the second experiment, let $[L_N,U_N]$ be either confidence interval from
Chapter 2 with one claim. Conditional validity says, for almost every $a$,

$$
\Pr\{q(A_\star)\in[L_N,U_N]\mid A_\star=a\}\ge 1-\delta.
$$

Taking expectations gives the same unconditional confidence guarantee for
the **random target** $q(A_\star)$. It does not give a confidence interval for
$p_{\mathrm{procedure}}$.

For a sharp example, let $Y$ be uniform on $[0,1]$. A randomized procedure
returns $[0,1]$ with probability $0.95$ and $[2,3]$ otherwise. Its average
coverage is $0.95$, while every realized interval has coverage either one or
zero. A certificate for the average cannot tell you which interval was drawn.

Reusing one random fitted model across all trials is therefore compatible
with a **conditional certificate for that fitted model**. Averaging over
model fitting requires repeating that fitting step independently. Do not
silently switch targets halfway through an experiment.

## 4.2 Images: choose the independent unit before counting successes

Suppose a fixed procedure produces intervals $I_j(X)$ for the $d$ coordinates
of an image $Y=(Y_1,\ldots,Y_d)$ given an input $X$. Draw independent images
$(X_i,Y_i)$, $i=1,\ldots,N$, from the target distribution. Any fresh procedural
randomness must also be independent between images; a shared fitted model
can instead be conditioned on as above. Write

$$
C_{ij}=\mathbf 1\{Y_{ij}\in I_j(X_i)\}.
$$

Pixels within an image may be arbitrarily dependent. The following are three
different scientific questions.

### Expected fraction of covered pixels

Define one observation per image:

$$
Z_i=\frac1d\sum_{j=1}^d C_{ij},
\qquad m=\mathbb E Z_i,
\qquad \widehat m=\frac1N\sum_{i=1}^N Z_i.
$$

The $Z_i$ are independent, identically distributed, and lie in $[0,1]$.
Consequently, with probability at least $1-\delta$,

$$
|\widehat m-m|\le \sqrt{\frac{\log(2/\delta)}{2N}}.
$$

There is also a binary-KL Chernoff bound, even though $Z_i$ is not Bernoulli.
Convexity gives, for every $z\in[0,1]$ and every real $\lambda$,

$$
e^{\lambda z}\le (1-z)+ze^\lambda,
\qquad
\mathbb E e^{\lambda Z_i}\le 1-m+me^\lambda.
$$

Multiply these moment bounds using independence, apply exponential Markov,
and optimize in $\lambda$. This gives

$$
\Pr\{\widehat m\ge a\}\le e^{-N\operatorname{kl}(a\Vert m)}
\quad(a>m),
$$

and the corresponding lower-tail bound for $a<m$. The same inversion as in
Chapter 2 therefore yields the valid interval

$$
\{q\in[0,1]:N\operatorname{kl}(\widehat m\Vert q)\le\log(2/\delta)\}.
$$

This is a concentration argument for bounded observations, **not an exact
binomial model**. The current [implementation](../validation.py) accepts
integer Bernoulli counts. Do not represent this audit as $Nd$ independent
trials, or round $N\widehat m$ to fit that API. Implement a bounded-mean
interface if this target is needed.

### Coverage at every coordinate

For a fixed coordinate, $C_{1j},\ldots,C_{Nj}$ are independent Bernoulli
variables of mean $p_j$. A union bound over coordinates gives

$$
\Pr\left\{\forall j:\left|\frac1N\sum_i C_{ij}-p_j\right|
\le\sqrt{\frac{\log(2d/\delta)}{2N}}\right\}\ge1-\delta.
$$

Thus simultaneous absolute precision $\epsilon$ requires at most the
sufficient count $N=\lceil\log(2d/\delta)/(2\epsilon^2)\rceil$ independent
images. The image count grows logarithmically with $d$; evaluating all pixels
still costs $Nd$ indicator evaluations. For $K$ predetermined procedures,
replace $d$ by $Kd$. Dependence between coordinates does not invalidate this
union bound.

### Coverage of the entire image

Define instead

$$
W_i=\mathbf 1\{C_{ij}=1\text{ for every }j\},
\qquad p_{\mathrm{whole}}=\mathbb E W_i.
$$

These are $N$ Bernoulli trials, directly supported by the existing API.
Simultaneously knowing that every $p_j\ge0.95$ does **not** establish
$p_{\mathrm{whole}}\ge0.95$. The elementary implication is only

$$
p_{\mathrm{whole}}\ge
\max\left(0,1-\sum_{j=1}^d(1-p_j)\right).
$$

Conversely, if all pixel indicators in an image are identical, increasing
$d$ supplies no additional independent information at all. Pixel count and
independent sample count are different resources.

## 4.3 Selecting a method after validation

Suppose $K$ candidates are fixed before the audit. Let $E$ be the event that
all their confidence intervals contain their respective true coverages.
Chapter 2 gives $\Pr(E)\ge1-\delta$ when the budget includes all $K$ candidates.
For **any** selection rule $\widehat k$ based on those audit results,

$$
E\ \Longrightarrow\quad
p_{\widehat k}\in[L_{\widehat k},U_{\widehat k}].
$$

One may therefore choose the narrowest candidate among those certified to
meet a predetermined minimum coverage. This establishes the coverage claim
for the selected candidate; it does not separately certify its expected
width. Any additional statistical width claims need appropriate bounds and
error allocation.

All candidates may use the same validation images, response draws, or other
random inputs. This dependence often makes comparisons less noisy.
Independence is needed across trials **within each fixed candidate**, not
between candidates. The union bound never multiplies probabilities.

The premise fails when the audit is used to construct unrestricted new
candidates. Consider $X\sim\mathrm{Uniform}[0,1]$ and the deterministic response
$Y=0$. After seeing audit inputs $X_1,\ldots,X_N$, define

$$
I_D(x)=
\begin{cases}
[-1,1],&x\in\{X_1,\ldots,X_N\},\\
[1,2],&\text{otherwise}.
\end{cases}
$$

Its empirical coverage is one. Its coverage on a fresh input is zero because
a finite set has uniform probability zero. Calling this “one candidate”
and setting $K=1$ after constructing it does not produce a valid certificate.
The candidate depends on the very observations being treated as independent
validation data.

Increasing $K$ after inspection cannot generally repair this problem: the
fixed-candidate tail bound may no longer hold for a newly created candidate.
A union bound protects a finite family fixed independently of the audit,
even when only some members are eventually examined. Otherwise use fresh
independent validation data or prove a stronger theorem controlling the
entire allowed adaptive procedure.

## 4.4 A simple certificate valid at every sample size

Repeatedly checking a fixed-$N$ interval and stopping as soon as it passes
changes the experiment. Here is an elementary repair.

Fix $K$ candidates and, for each candidate $k$, an infinite sequence of
independent Bernoulli trials with constant mean $p_k$. Sequences may depend
on each other. At time $t\ge1$, each candidate has its first $t$ trials. Set

$$
\delta_t=\frac{6\delta}{\pi^2t^2},
\qquad
b_t=\log\frac{2K}{\delta_t}
=\log\frac{\pi^2Kt^2}{3\delta}.
$$

Apply either Chapter 2 construction at time $t$, replacing $\delta$ by
$\delta_t$. The Hoeffding radius is

$$
r_t=\sqrt{\frac{b_t}{2t}},
$$

and the KL interval for candidate $k$ is

$$
[L_{k,t},U_{k,t}]
=\{q\in[0,1]:t\operatorname{kl}(\widehat p_{k,t}\Vert q)\le b_t\}.
$$

**Theorem.** With probability at least $1-\delta$, every interval contains
its target, simultaneously for every candidate and every positive time:

$$
\Pr\{\forall k\le K,\ \forall t\ge1:
p_k\in[L_{k,t},U_{k,t}]\}\ge1-\delta.
$$

**Proof.** At a fixed time, allocate $\delta_t/(2K)$ to each candidate and
each tail. The probability of any failure at that time is at most $\delta_t$.
Countable subadditivity and $\sum_{t\ge1}t^{-2}=\pi^2/6$ give

$$
\Pr\{\text{any failure at any time}\}
\le\sum_{t\ge1}\delta_t=\delta.
$$

No independence between times or candidates is used. $\square$

Consequently, stopping at any data-dependent finite time and selecting any
candidate preserves coverage of its reported confidence interval. The
procedure itself and its target distribution must stay fixed. This theorem
does not authorize retraining candidates on the growing audit.

For Bernoulli data, the fixed-time implementation can be used as a building
block:

```python
import math
from validation import coverage_certificate

def anytime_certificate(successes, t, *, delta=0.05, num_claims=1):
    # t is this candidate's cumulative number of independent trials.
    delta_t = 6 * delta / (math.pi**2 * t**2)
    return coverage_certificate(
        successes, t, delta=delta_t, num_claims=num_claims,
        method="kl", target_coverage=0.95,
    )
```

The returned object's `delta` is the **local** budget $\delta_t$; the global
$\delta$ is justified by the theorem above. The same union argument also
allows different candidates to be inspected at different cumulative sample
counts, provided each candidate's underlying sequence satisfies the stated
assumptions. This is a standard union-bound construction, not a novel result.

## 4.5 What concentration cannot supply

A simulator certificate concerns the simulator's distribution. More simulated
trials reduce Monte Carlo error; they do not establish that the simulator
represents the physical experiment.

A possible transfer assumption makes the gap explicit. Let $P$ and $Q$ be
the complete trial distributions in simulation and reality, including any
randomness entering the coverage event $B$. If

$$
\operatorname{TV}(P,Q)=\sup_E|P(E)-Q(E)|\le\eta,
$$

then $Q(B)\ge P(B)-\eta$. A simulated lower confidence bound $L$ therefore
implies the real-world lower bound $\max(0,L-\eta)$ on the same confidence
event. The simulator audit does not estimate $\eta$; establishing that bound
is a separate scientific task.

Even exchangeability alone cannot replace trial independence. Draw one
latent $B\sim\mathrm{Bernoulli}(1/2)$ and set $C_i=B$ for every $i$. This
sequence is exchangeable, with every marginal mean equal to $1/2$, but

$$
\widehat p_N=B,
\qquad
\Pr\{|\widehat p_N-1/2|>1/4\}=1
$$

for every $N$. There is no concentration around the marginal mean as the
nominal sample size grows. A dependence model, independent experimental
replicates, or a different target is required.

## 4.6 Research directions left unresolved by this repository

These are extensions of this tutorial's implementation, not claims of new
theorems or assertions that the corresponding fields are unsolved:

- **Dependent physical observations:** specify a justified dependence model
  and derive a valid concentration bound with its parameters included.
- **Adaptive scientific iteration:** characterize permitted updates and
  control repeated reuse of validation data without silently treating new
  candidates as fixed.
- **Efficient sequential audits:** implement tighter confidence sequences
  and compare their required simulations with the elementary construction.
- **Simulator transfer:** obtain defensible discrepancy bounds for a stated
  physical experiment and propagate them to its coverage claim.
- **Useful high-dimensional intervals:** jointly assess width and the chosen
  coverage target, keeping average, coordinatewise, and whole-image coverage
  distinct.

Each direction begins by specifying the probability being certified and
identifying the independent information available to estimate it.
