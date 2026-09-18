# 2. From exponential Markov to a coverage certificate

This chapter proves the finite-sample guarantee implemented in
[`validation.py`](../validation.py). [Chapter 1](01-probability-and-targets.md)
identified the target and reduced validation to independent Bernoulli trials.
The compact reference is [`VALIDATION.md`](../VALIDATION.md).

Throughout, $C_1,\ldots,C_N$ are independent Bernoulli variables with the
same unknown mean $p\in[0,1]$, and $\widehat p=N^{-1}\sum_iC_i$. The
integer $N$ and the tested procedure are fixed before observing outcomes.
All logarithms are natural.

## 2.1 Exponential Markov: turning independence into multiplication

For a nonnegative random variable $W$ and $w>0$,
$E[W]\ge w\Pr\{W\ge w\}$, hence

$$
\Pr\{W\ge w\}\le E[W]/w.
$$

For $t>0$, apply this to $W=\exp(t\sum_iC_i)$ and $w=e^{tNa}$.
Independence is precisely what permits the product in

$$
\Pr\{\widehat p\ge a\}
\le e^{-tNa}\prod_{i=1}^N E[e^{tC_i}]
=\left[e^{-ta}(1-p+pe^t)\right]^N.
$$

This holds for every $t>0$, so choose the one giving the smallest bound.
For $0<p<a<1$, differentiating the logarithm of the bracket gives

$$
-a+\frac{pe^t}{1-p+pe^t}=0,
\qquad
e^{t_\star}=\frac{a(1-p)}{p(1-a)}>1.
$$

The logarithm is convex in $t$, so this stationary point is a global
minimum. Substitution produces the **Bernoulli Chernoff bound**

$$
\Pr\{\widehat p\ge a\}\le
\exp\{-N\operatorname{kl}(a\Vert p)\},
\qquad a>p,
$$

where the binary relative entropy is

$$
\operatorname{kl}(a\Vert q)
=a\log\frac aq+(1-a)\log\frac{1-a}{1-q}.
$$

For $a<p$, use $t<0$: the event $\sum_iC_i\le Na$ implies
$e^{t\sum_iC_i}\ge e^{tNa}$. The same calculation gives

$$
\Pr\{\widehat p\le a\}\le
\exp\{-N\operatorname{kl}(a\Vert p)\},
\qquad a<p.
$$

Endpoint thresholds follow by limits. If $p=0$ or $p=1$, all trials are
deterministic. The continuous conventions include
$\operatorname{kl}(0\Vert q)=-\log(1-q)$ and
$\operatorname{kl}(1\Vert q)=-\log q$. A positive probability divided by
zero in the entropy formula gives infinity.

## 2.2 Invert the tail bound

Suppose there are $K\ge1$ predeclared coverage claims and the desired total
audit failure probability is $\delta\in(0,1)$. Allocate $\delta/(2K)$ to
each of the two tails of each claim, and define

$$
b=\log(2K/\delta).
$$

After observing $\widehat p$, retain candidate probabilities $q$ satisfying

$$
N\operatorname{kl}(\widehat p\Vert q)\le b.
$$

This set is an interval $[L,U]$. Indeed, for $a\in(0,1)$,

$$
\frac{\partial}{\partial q}\operatorname{kl}(a\Vert q)
=\frac{q-a}{q(1-q)}.
$$

Thus entropy decreases up to $q=a$ and increases afterwards. Find $L$ by
monotone bisection on $[0,\widehat p]$ and $U$ on $[\widehat p,1]$.
For endpoint counts, no root-finding is needed:

$$
\widehat p=0:\quad [L,U]=[0,1-e^{-b/N}],
\qquad
\widehat p=1:\quad [L,U]=[e^{-b/N},1].
$$

Why is this a confidence interval? Fix the true $p$. If $p<L$, then
$\widehat p>p$ and $N\operatorname{kl}(\widehat p\Vert p)>b$.
For $a>p$, $\operatorname{kl}(a\Vert p)$ increases with $a$. Therefore
this failure event is an upper tail of the binomial count. Applying the
Chernoff bound at its first attainable count bounds its probability by
$e^{-b}$. The event $p>U$ is a lower tail with the same bound. Consequently,

$$
\Pr\{p\notin[L,U]\}\le2e^{-b}=\delta/K.
$$

The code keeps outward bisection brackets and takes a final outward
floating-point step. This is ordinary double-precision numerical evaluation,
not formally verified interval arithmetic.

## 2.3 Derive Hoeffding from binary relative entropy

There is a simpler, slightly wider interval with a closed-form radius.
For fixed $p\in(0,1)$, define

$$
g(a)=\operatorname{kl}(a\Vert p)-2(a-p)^2.
$$

Direct differentiation gives

$$
g(p)=g'(p)=0,
\qquad
g''(a)=\frac1{a(1-a)}-4\ge0,
$$

because $a(1-a)\le1/4$. Therefore $g$ is convex and minimized at $p$:

$$
\operatorname{kl}(a\Vert p)\ge2(a-p)^2.
$$

Continuity gives the remaining endpoint cases. Substitution into Chernoff
yields Hoeffding's two tails,

$$
\Pr\{\widehat p-p\ge r\}\le e^{-2Nr^2},
\qquad
\Pr\{p-\widehat p\ge r\}\le e^{-2Nr^2}.
$$

With $r=\sqrt{b/(2N)}$, the interval is

$$
[L_H,U_H]=[\max(0,\widehat p-r),\min(1,\widehat p+r)].
$$

The KL interval is contained in this Hoeffding interval at the same budget,
since $N\operatorname{kl}(\widehat p\Vert q)\le b$ implies
$|\widehat p-q|\le r$. This explains its improved precision, especially
near zero or one.

## 2.4 Simultaneous claims need a union bound

For claim $k$, let $F_k$ be the event that its confidence interval misses
its true coverage $p_k$. The preceding argument gives $\Pr(F_k)\le\delta/K$.
Since an indicator of a union is at most the sum of the indicators,

$$
\Pr\left(\bigcup_{k=1}^K F_k\right)
\le\sum_{k=1}^K\Pr(F_k)\le\delta.
$$

Thus all $K$ reported intervals are correct together with probability at
least $1-\delta$. Claims may share randomness; independence between them is
unnecessary. Each claim still needs independent, identically distributed
trials. Different predetermined sample sizes $N_k$ are allowed.

The family must be fixed before inspecting audit outcomes. Declaring more
claims afterwards does not repair arbitrary adaptive searches. Optional
stopping also needs a different guarantee or a preplanned allocation across
inspection times. These statements hold conditional on independent training
data used to freeze a procedure.

## 2.5 Certification is a decision based on a confidence set

For a minimum acceptable coverage $p_0\in(0,1)$:

| Returned status | Condition | Conclusion on the simultaneous confidence event |
|---|---|---|
| `certified` | $L\ge p_0$ | $p\ge p_0$ |
| `below_target` | $U<p_0$ | $p<p_0$ |
| `inconclusive` | Neither | Neither conclusion is established |

Failure to certify is not evidence of undercoverage. To certify approximate
calibration, $|p-c|\le\tau$, require the entire band to lie inside
$[c-\tau,c+\tau]$. Containing $c$ is insufficient.

The notebook's recorded four-level run uses $N=1000$, $K=4$,
$\delta=.05$, and KL bounds:

| Nominal level | Covered trials | Estimated coverage | Simultaneous coverage band |
|---|---:|---:|---:|
| 80% | 793 / 1000 | 79.3% | 75.03%–83.18% |
| 90% | 896 / 1000 | 89.6% | 86.26%–92.41% |
| 95% | 953 / 1000 | 95.3% | 92.86%–97.13% |
| 99% | 989 / 1000 | 98.9% | 97.50%–99.65% |

All four minimum-nominal-coverage decisions are inconclusive. In particular,
$953/1000>0.95$ does not establish $p\ge0.95$: its lower bound is about
$0.9286$. These are observed simulation results, not exact values of $p_k$.
The 95% audit confidence refers to repetitions of the entire audit; it is
not a posterior probability that a fixed $p$ lies in the realized band.

## 2.6 Precision and the probability of passing are different

To estimate all coverages with absolute error at most $\epsilon$, Hoeffding
gives the sufficient sample size

$$
N\ge\left\lceil\frac{\log(2K/\delta)}{2\epsilon^2}\right\rceil.
$$

For $K=4$, $\delta=.05$, and $\epsilon=.01$, this is $25{,}376$ trials
per claim. At $N=1000$ the radius is approximately $0.0503745$.

Precision alone does not ensure that certification passes. Suppose one
claim has true coverage $p\ge p_0+\gamma$ for a margin $\gamma>0$.
For a chosen non-passing probability $\eta\in(0,1)$, Hoeffding gives

$$
\widehat p\ge p-\sqrt{\frac{\log(1/\eta)}{2N}}
\quad\text{with probability at least }1-\eta.
$$

Hence the Hoeffding lower bound exceeds or equals $p_0$ on that event if

$$
N\ge
\frac{\left(\sqrt{\log(2K/\delta)}+
\sqrt{\log(1/\eta)}\right)^2}{2\gamma^2}.
$$

Round upwards for an integer sample size. This is a sufficient **power**
calculation for one claim: it controls the chance of obtaining a certificate
when a genuine positive margin exists. For all $K$ claims to pass together,
use a common minimum margin and replace $\log(1/\eta)$ by $\log(K/\eta)$.
At $p=p_0$ there is no positive margin and no corresponding high-probability
passing guarantee. Returning “inconclusive” near this boundary is legitimate.

Rare failures offer a different regime. If all $N$ trials succeed, the KL
lower bound is $e^{-b/N}$. Such data certify $p\ge p_0$ whenever

$$
N\ge\left\lceil\frac{b}{-\log p_0}\right\rceil.
$$

For $p_0=1-\alpha$ with small $\alpha>0$, this scales as $b/\alpha$,
rather than a generic quadratic precision bound. It is a best-case count:
all-success data occur with probability $p^N$, not with certainty.
For a two-sided audit with $K=4$ and $\delta=.05$, certifying $p\ge.95$
from all successes needs $99$ trials; certifying $p\ge.99$ needs $505$.
The API uses two-sided budgets throughout. An exclusively lower-bound
analysis could instead allocate $\delta/K$ per claim.

## Exercises

1. Derive the Chernoff optimizer for $a<p$ and explain why its sign reverses.
2. Prove that the KL sublevel set is an interval, including empirical
   coverage zero and one. Explain which bisection bracket endpoints are
   conservative.
3. For $953$ successes out of $1000$, compute both certificate types with
   $K=4$ and $\delta=.05$. What tolerance would suffice to certify
   calibration around $c=.95$ using the KL band?
4. Derive the sample-size bound for all $K$ positive-margin claims to pass
   with probability at least $1-\eta$. Identify the two separate union bounds.

[Solutions and discussion](05-exercises.md#chapter-2-concentration-and-certification).

These are classical Chernoff and Hoeffding arguments, not new concentration
results. Original references are linked in
[`VALIDATION.md`](../VALIDATION.md#references).
