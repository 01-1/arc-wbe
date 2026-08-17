# Competing with sampling {#competing-with-sampling .post__title .typgr8-alpha}

Posted by [Eric Neyman](/author/eric/), [Victor
Lecomte](/author/victor/) , [Wilson Wu](/author/wilson/) , [Michael
Winer](/author/michael-winer/) , [Jacob Hilton](/author/jacob/) &
[George Robinson](/author/george/) on November 18th, 2025

In 2025, ARC has been making conceptual and theoretical progress at the
fastest pace that I\'ve seen since I first interned in 2022. Most of
this progress has come about because of a re-orientation around a more
specific goal: **outperforming random sampling** when it comes to
understanding neural network outputs. Compared to our previous goals,
this goal has the advantage of being more concrete and more directly
tied to useful applications.

The purpose of this post is to:

1.  Explain and motivate our \"outperforming sampling\" agenda from the
    standpoint of preventing catastrophic AI misalignment.
2.  Introduce what we call the *matching sampling principle (MSP)* as a
    semi-formalization of the belief underpinning our research agenda,
    and discuss why we believe this principle.
3.  Discuss the progress we\'ve made toward matching sampling in some
    specific contexts, such as random MLPs and trained two-layer MLPs.

Also: we\'re hiring! If the research direction described in this post
excites you, you can [apply to ARC](https://www.alignment.org/hiring/)!



# Outperforming sampling as a step toward preventing AI misalignment

Consider the following simple scheme that attempts to align an AI model
$M$, which maps inputs $x$ to outputs $y$:

1.  Build a \"catastrophe detector\" $C$ that classifies model outputs
    $y$ as \"catastrophic\" (1) or \"non-catastrophic\"
    (0).^[\[1\]](#fn1){#fnref1}^ You can do this by, for example,
    scaffolding together a deliberative system of GPT-5\'s that
    carefully investigate whether the model is doing anything
    suspicious.
2.  Do adversarial training using the catastrophe detector. Concretely,
    this means:\
    a. Optimize a probability distribution $D$ over inputs $x$, so as to
    maximize $\mathbb{E} _ {x \sim D}[C(M(x))]$.\
    b. At the same time, optimize $M$ so as to minimize
    $\mathbb{E} _ {x \sim D}[C(M(x))]$.

I think that this is a fine starting point for alignment plan, but not a
complete plan in and of itself. It suffers from at least two issues:

1.  A catastrophe detector that\'s built in the way I described would be
    imperfect, even if you did a really good job with the engineering.
    If $M$ were \"smarter\" than $C$, then it could figure out ways to
    fool $C$ into assigning a low catastrophe probability to a
    catastrophic output.
2.  *Even if* the catastrophe detector were perfect, it is not clear how
    to compute $\mathbb{E} _ {x \sim D}[C(M(x))]$ efficiently enough to
    make this scheme work.^[\[2\]](#fn2){#fnref2}^ The most
    straightforward approach would involve estimating this expectation
    by taking random samples $x \sim D$, but this is extremely slow if
    catastrophes are rare on $D$.

(There are other technical issues as well,^[\[3\]](#fn3){#fnref3}^ but
these are the ones that seem hardest to surmount.)

We believe that ARC\'s technical research agenda is capable of
addressing both of these issues. However, issue #1 is mostly out of
scope for this post (though I\'ll very briefly describe our planned
approach in this footnote^[\[4\]](#fn4){#fnref4}^). The purpose of this
post is to explain in detail how we hope to address issue #2.



## Understanding structure helps outperform sampling

ARC\'s goal is to be able to estimate $\mathbb{E} _ {x \sim D}[C(M(x))]$
far better than one can just by drawing random samples from $D$. We
believe that this can be done by **understanding the structure** of $M$,
$C$, and $D$.

Here\'s a simple example that\'s meant to illustrate this point. Suppose
that, by understanding the internals of $C$, we are able to notice that
$C(y)$ is the conjunction of three predicates $P_1(y), P_2(y), P_3(y)$
\-- in other words, $C$ outputs 1 if and only if all of
$P_1, P_2, P_3$ are true of $M$\'s output.

And suppose that, furthermore, we understand the structure of $M$ and
$D$ to understand that $P_1(M(x)), P_2(M(x)), P_3(M(x))$ are independent
events for $x \sim D$.^[\[5\]](#fn5){#fnref5}^

Using this structural understanding, we can estimate that
$\mathbb{E} _ {x \sim D}[C(M(x))] = \prod _ {i = 1}^3 \mathbb{E} _ {x \sim D}[P _ i(M(x))]$.
If each $P_i$ is true on $D$ with probability one-in-a-million
($10^{-6}$), then we can estimate that
$\mathbb{E} _ {x \sim D}[C(M(x))] \approx 10^{-18}$. Obtaining this
estimate by sampling would have required roughly $10^{18}$ samples. By
contrast, our structural understanding lets us estimate this probability
with roughly $3 \cdot 10^6$ samples, and we can potentially do even
better than that if we have a structural understanding of the $P_i$\'s
themselves. Thus, **our structural understanding lets us estimate the
expected value far more efficiently than we could with sampling.**

This example is simplistic, of course: in practice, we will need to
understand structure that is far more sophisticated than \"the output is
a conjunction of three independent predicates.\" But the example
illustrates the point that having a detailed mechanistic understanding
of a neural net lets us estimate properties of its outputs far better
than black-box methods alone.



### A non-human understanding

When we speak of \"understanding the structure\" of $M$, $C$, and $D$,
we are not referring to human understanding. While a conjunctive
structure like the one above can be understood by a human, we believe
that in general, neural nets will be composed of mathematical structures
that are far too complex to allow for a full human understanding.

Instead, we are imagining that an explanation of the structure of a
neural net is written in some kind of formal language. The explanation
could be as large as the neural net itself, and may be as
incomprehensible to a human as the neural net. Thus, our goal is not to
have a human look at the structure and estimate the expectation of
$C(M(x))$. Instead, the goal is to invent an algorithm that takes as
input the explanation and estimates expectation of $C(M(x))$ based on
that explanation.

This contrasts to almost all other research on neural network
interpretability, which aims for a *partial, human understanding* of
neural nets. Our research is instead aimed at *full, algorithmic
understanding.*

In the next section, I will elaborate on what this means.



# The matching sampling principle

In this section, I will more formally describe what we hope to
accomplish by gaining structural understanding of neural networks. While
above I talked about *outperforming* sampling, in the fully general case
we can only hope to *match* the performance of sampling. In other words,
we expect that the performance of our algorithms in the practical
setting of trained neural nets will substantially exceed the worst-case
bounds that we will be able to state and prove. See
[below](#why-only-matching-sampling) for further discussion of this
point.

I will start with a first-pass attempt at stating the matching sampling
principle (MSP). As we will discuss, it does not quite make sense;
however, it carries across the key intuition.



## A first attempt at stating the MSP

In order to state the MSP, we will define a few pieces of notation:

- We will use the notation $M _ \theta$ to describe a neural network (or
  other function from a parameterized family). Here, $M$ denotes the
  *architecture* of the neural net, and $\theta$ denotes the parameters.
  - Concretely, $M _ \theta: \{0, 1\}^n \to \mathbb{R}$ is a function
    mapping $n$-bit inputs $x$ to real numbers. *(Note that this
    notation differs from the notation we used in the previous section;
    here, *$M _ \theta$* describes the composition *$C \circ M$* of the
    neural net and the catastrophe detector above.)*
  - We are interested in estimating
    $\mathbb{E} _ {x \sim \{0, 1\}^n}[M _ \theta(x)]$, where $x$ is
    sampled uniformly.^[\[6\]](#fn6){#fnref6}^
- We will use the notation $\pi$ to describe a *mechanistic explanation*
  of $M _ \theta$; $\pi$ is what provides us with a structural
  understanding of $M _ \theta$ and allows us to estimate its expected
  output. (This mirrors the way that ARC has
  [historically](https://arxiv.org/abs/2211.06738)
  [used](https://www.alignment.org/blog/a-computational-no-coincidence-principle/)
  the letter $\pi$.)
- We will use the notation $\mathbb{G} _ M$ to denote an estimator (the
  subscript $M$ is meant to emphasize that different architectures could
  have different estimators): $\mathbb{G} _ M$ takes as input the
  parameters $\theta$, a mechanistic explanation $\pi$, and a *tolerance
  parameter* $\varepsilon$. (As we discuss below, the smaller the
  tolerance parameter, the more accurate $\mathbb{G} _ M$\'s estimate,
  but the longer $\mathbb{G} _ M$ will be allowed to run.)

With this notation in place, we will make our first attempt to state the
MSP:

- For all architectures $M$ (with parameters $\theta$), there exists an
  estimator $\mathbb{G} _ M$ such that:
- For all parameters $\theta$, there exists a
  short^[\[7\]](#fn7){#fnref7}^ explanation $\pi$ (we require that
  $|\pi| \le O(|\theta|)$), such that:
- For all tolerance parameters $\varepsilon > 0$,
  $\mathbb{G} _ M(\theta, \pi, \varepsilon)$ satisfies the following
  three properties:
  1.  It runs in time
      $O \left( \frac{1}{\varepsilon^2} \text{Time}(M _ \theta) \right)$.
  2.  Its error is competitive with sampling:\
      $(\mathbb{G} _ M(\theta, \pi, \varepsilon) - \mathbb{E} _ {x \sim \{0, 1\}^n}[M _ \theta(x)])^2 \le \varepsilon^2 \text{Var} _ {x \sim \{0, 1\}^n}[M _ \theta(x)].$
  3.  It is **mechanistic.**

Let\'s parse these three requirements:

1.  $\mathbb{G} _ M$ needs to run in time
    $O \left( \frac{1}{\varepsilon^2} \text{Time}(M _ \theta) \right)$.
    This is the time that it takes to estimate
    $\mathbb{E} _ {x \sim \{0, 1\}^n}[M _ \theta(x)]$ by running
    $1/\varepsilon^2$ randomly sampled values of $x$ through
    $M _ \theta$.
2.  The squared error of $\mathbb{G} _ M$ must be small. Concretely, the
    right-hand side represents the expected squared error via taking the
    empirical average of $1/\varepsilon^2$ samples of $M _ \theta(x)$.
3.  $\mathbb{G} _ M$ is mechanistic. We have not defined
    \"mechanistic\", so this point requires elaboration.



### What makes an algorithm \"mechanistic\"?

We do not have a formal definition of \"mechanistic.\" But, loosely
speaking, we mean that $\mathbb{G} _ M$ estimates the expected
output of $M _ \theta$ **deductively, based on the structure of
**$M _ \theta$**.** This contrasts with *sampling-based* algorithms for
estimating the expected output of $M _ \theta$, which operate based on
*inductive* reasoning. Mechanistically estimating the expected output
involves *finding the reason* for the expected output being what it is;
meanwhile, sampling-based algorithms merely *infer the existence of a
reason* without learning anything about the reason.

To illustrate this difference, suppose that the explanation $\pi$ given
to $\mathbb{G} _ M$ is a simple heuristic argument (such as mean
propagation \-- see §D.2 [here](https://arxiv.org/pdf/2211.06738)),
which suggests that $\mathbb{E}[M _ \theta] = 0$ but is otherwise
uninformative about the structure of $M _ \theta$. Suppose further that
$\mathbb{G} _ M$ computes $M _ \theta(x)$ on a hundred inputs
$x \in \{0, 1\}^n$, and it finds that $M _ \theta(x) = 1$ on every one
of those hundred inputs. Then $\mathbb{G} _ M$ should return
$\frac{100}{2^n}$: that\'s because it knows that $M _ \theta(x) = 1$ on
the hundred inputs that it checks, but it has not seen any structural
evidence that would suggest that $M _ \theta$\'s behavior on those
hundred inputs has any bearing on how $M _ \theta$ behaves on the inputs
that it has not checked. By contrast, a *sampling-based* estimator that
checks the same hundred inputs would return $1$, implicitly assuming
that those inputs are representative.

(If indeed $M _ \theta$ always returns $1$, then we believe that there
*exists* a short explanation of this fact; but $\mathbb{G} _ M$ cannot
output $1$ unless it is given this explanation.)

In some of our [previous
work](https://www.alignment.org/blog/estimating-tail-risk-in-neural-networks/#method-1-gaussian-distribution),
we discussed *covariance propagation:* successively modeling each layer
of $M _ \theta$ as a multivariate normal
distribution.^[\[8\]](#fn8){#fnref8}^ Covariance propagation (and
[related methods](https://arxiv.org/abs/2211.06738), like mean
propagation and cumulant propagation) is mechanistic, because it deduces
an estimate based on the structure of
$M _ \theta$.^[\[9\]](#fn9){#fnref9}^ More generally,
[deduction-projection
estimators](https://dash.harvard.edu/items/4a7f19e8-c68d-4473-900f-a4ac3fd9ae17)
\-- estimators that successively model each layer of $M _ \theta$ by
finding the best-fit model from some parameterized class \-- are
mechanistic.

A simple, though not entirely correct, heuristic for whether an
estimation algorithm is deductive, is whether it avoids any random or
pseudorandom sampling. This heuristic should work for the purposes of
engaging with this post.

(See much more on mechanistic estimation in our earlier paper,
[\"Formalizing the presumption of
independence\"](https://arxiv.org/abs/2211.06738),^[\[10\]](#fn10){#fnref10}^
as well as in former ARC intern Gabe Wu\'s [senior
thesis](https://dash.harvard.edu/items/4a7f19e8-c68d-4473-900f-a4ac3fd9ae17)
on deduction-projection estimators.)



### Why do we require $\mathbb{G} _ M$ to be mechanistic? {#why-do-we-require-mathbbg-m-to-be-mechanistic}

There are multiple reasons for this; in a [previous blog
post](https://www.alignment.org/blog/mechanistic-anomaly-detection-and-elk/),
we discussed how mechanistic estimates can help us detect mechanistic
anomalies. But for the purposes of this post, the reason is pretty
straightforward: in cases where $M _ \theta$ has a lot of structure, we
think that $\mathbb{G} _ M$ can *substantially outperform* sampling, if
given an explanation $\pi$ that explains that structure (as motivated
[above](#understanding-structure-helps-outperform-sampling)).

Thus, loosely speaking, our hope is that if we find a $\mathbb{G} _ M$
that both (a) is mechanistic and (b) performs at least as well as
sampling for all $\theta$, then it will substantially outperform
sampling for parameters $\theta$ with a lot of structure, such as
trained neural nets.



### The intuition behind the MSP

Sampling is a really powerful tool, because *randomly drawn samples are
representative* (with high probability), and so a sampling-based
estimate can\'t be off by too much (with high probability). In light of
this, why do we think that a mechanistic estimation algorithm can
compete with sampling?

Suppose that $M _ \theta: \{0, 1\}^{100} \to \{0, 1\}$ is a boolean
circuit. Suppose, further, that a naive heuristic argument (like mean
propagation) suggests that $M _ \theta$\'s average output is $0.5$, but
that in fact its average output is roughly $0.49$ (far enough from $0.5$
that this discrepancy could not have [happened by
chance](https://www.alignment.org/blog/a-computational-no-coincidence-principle/)).
A sampling-based algorithm can pick up on this discrepancy given about
10,000 samples; but what can a mechanistic algorithm do?

Well, given that the discrepancy could not have happened by chance,
there must be structure that explains the discrepancy. For illustration,
let\'s consider two types of structure.

**First,** maybe the discrepancy is caused by different gates reusing
the same inputs, thereby inducing nontrivial correlations between
different parts of the circuit.^[\[11\]](#fn11){#fnref11}^ In that case,
$\pi$ should be able to point out this structure, causing
$\mathbb{G} _ M$ to understand the discrepancy (even without running any
inputs through $M _ \theta$).

**Second,** maybe only the first 10 input bits matter to the output of
the circuit (perhaps $M _ \theta$ ignores the last 90 input bits
entirely, or perhaps they end up not affecting the output for
complicated structural reasons). And then \-- just by chance \-- it so
happens that $M _ \theta$ outputs 1 on only 49% of the 1024 possible
10-bit inputs. In this case, $\pi$ points out that $M _ \theta$ depends
only on the first 10 input bits; it *does not* point out that
$M _ \theta$ outputs 1 on 49% of them, because that\'s part of the
unexplainable randomness of $M _ \theta$.^[\[12\]](#fn12){#fnref12}^
Instead, $\mathbb{G} _ M$ must determine this fact by using its allotted
time to check the value of $M _ \theta$ on those 1024 inputs.

(What if $\varepsilon$ is large enough that $\mathbb{G} _ M$ doesn\'t
have the necessary runtime to check $M _ \theta$ on all 1024 inputs? In
that case, it should check however many it can and estimate the rest as
being 50/50. This will still outperform sampling!)

More generally, the intuition is that knowing the structure of
$M _ \theta$ gives $\mathbb{G} _ M$ the knowledge it needs to do no
worse than random sampling. If $\mathbb{G} _ M$ still does worse than
random sampling after reading $\pi$, that can only be because $\pi$ did
not provide a full structural explanation of
$M _ \theta$.^[\[13\]](#fn13){#fnref13}^



### Why only *matching* sampling?

Given the [above
intuition](#understanding-structure-helps-outperform-sampling) that
understanding structure can outperform sampling, why are we only aiming
to *match* the performance of sampling?

Consider the above example, where the average output of $M _ \theta$
depends on 1024 effectively random computations, and suppose that
$1/\varepsilon^2 = 512$: enough time for $\mathbb{G} _ M$ to compute the
output of $M _ \theta$ on 512 of the 1024 inputs. In that case, we
expect both $\mathbb{G} _ M$ and sampling to have squared error on the
order of $\frac{1}{512}$: $\mathbb{G} _ M$\'s expected squared error
will be somewhat lower, but not dramatically so.

In general, we expect that there will often be a range of
$\varepsilon$-values for which the best mechanistic estimate is only
slightly better than sampling-based
estimation.^[\[14\]](#fn14){#fnref14}^ Thus, for some parameters
$\theta$ and tolerance parameters $\varepsilon$, we only expect to be
able to *match* (or perhaps slightly outperform) sampling, not to
strongly outperform sampling.

However, as discussed
[above](#why-do-we-require-mathbbg-m-to-be-mechanistic), we expect that
if our mechanistic estimator matches the performance of sampling in all
cases, then it will substantially outperform sampling in structured
cases such as trained neural nets, at least for non-tiny values of
$\varepsilon$. We expect that we can leverage this to help with the sort
of adversarial training process described [in the
introduction](#outperforming-sampling-as-a-step-toward-preventing-ai-misalignment).



### An issue: $\pi$ can just tell $\mathbb{G} _ M$ the answer {#an-issue-pi-can-just-tell-mathbbg-m-the-answer}

As mentioned earlier, out first attempt at stating the MSP doesn\'t
quite make sense. The idea of MSP is for $\pi$ to describe the structure
of $M _ \theta$. However, in order to satisfy the MSP statement
[above](#a-first-attempt-at-stating-the-msp), $\pi$ can just write down
the value of $\mathbb{E} _ {x \sim \{0, 1\}^n}[M _ \theta(x)]$. Then,
$\mathbb{G} _ M$ can output that value.

To fix this issue, we observe that if $\mathbb{G} _ M$ understands the
structure of $M _ \theta$, then it ought to be able to answer all sorts
of questions about $M _ \theta$ at least as well as sampling \-- not
just its expected value \-- so long as those questions are not
adversarially selected. To formalize this idea, we will modify the type
signature of $M _ \theta$ to take two inputs $(c, x)$ (here, $c$ stands
for \"context\"), and require that $\mathbb{G} _ M$ be able accurately
estimate $\mathbb{E} _ x[M _ \theta(c, x)]$ for a random choice of
$c$.^[\[15\]](#fn15){#fnref15}^ This change gives us an MSP statement
that we are willing to stand behind.



## Our actual MSP statement

Here is **ARC\'s mainline \"matching sampling principle\" (MSP):**

- *For all architectures* $M$ *(with parameters* $\theta$) *mapping
  pairs* $(c \in \{0, 1\}^{n_c}, x \in \{0, 1\}^{n_x})$ *to*
  $\mathbb{R}$ *, there exists an estimator* $\mathbb{G} _ M$ *mapping
  tuples* $(\theta, \pi, c, \varepsilon)$ *to* $\mathbb{R}$ *, such
  that:*
- *For all parameters *$\theta$*, there exists a short explanation
  *$\pi$* (*$|\pi| \le O(|\theta|)$*), such that:*
- *For all tolerance parameters *$\varepsilon > 0$*,
  *$\mathbb{G} _ M(\theta, \pi, c, \varepsilon)$* satisfies the
  following three properties:*
  1.  *It runs in time
      *$O \left( \frac{1}{\varepsilon^2} \text{Time}(M _ \theta) \right)$*.*
  2.  *Its error is competitive with sampling, on average over random
      *$c$*:\
      *$\mathbb{E} _ c[(\mathbb{G} _ M(\theta, \pi, c, \varepsilon) - \mathbb{E} _ x[M _ \theta(c, x)])^2] \le \varepsilon^2 \mathbb{E} _ c[\text{Var}_x[M _ \theta(c, x)]]$*,
      where *$c \sim \{0, 1\}^{n_c}$* and *$x \sim \{0, 1\}^{n_x}$*.*
  3.  *It is mechanistic.*

(Just as before, this statement isn\'t fully formal, because of the
informal \"mechanistic\" qualifier. But in practice, we have strong
enough opinions about what counts as \"mechanistic\" that this statement
is formal enough to guide our research.)

An interesting special case of the MSP is when $M _ \theta$ encodes a
universal Turing machine. See [the
appendix](#a-special-case-of-the-msp-universal-turing-machines) for
discussion.



### An important variant: Findable explanations

While the above MSP statement is the most theoretically clean one, on
its face the statement is not very useful. That\'s because it says
nothing about being able to *find* the explanation $\pi$; what use is it
to merely know that an adequate explanation *exists,* if we can\'t find
it?

This leads us to the following alternative statement, which we\'ve been
calling the **\"train and explain\" formulation of the MSP:**

- *For all architectures* $M$ *(with parameters* $\theta$) *mapping
  pairs* $(c \in \{0, 1\}^{n_c}, x \in \{0, 1\}^{n_x})$ *to*
  $\mathbb{R}$ *, there exists an estimator* $\mathbb{G} _ M$ *mapping
  tuples* $(\theta, \pi, c, \varepsilon)$ *to* $\mathbb{R}$ *, such
  that:*
- *For all \"training\" algorithms* $T$ *mapping random seeds*
  $s \in \{0, 1\}^r$ *to parameters* $\theta$ *, there exists an
  \"explaining\" algorithm* $E$ *mapping random seeds*
  $s \in \{0, 1\}^r$ *to explanations* $\pi$ *, with*
  $\text{Time}(E) \le O(\text{Time}(T))$ *, such that:*
- *For all tolerance parameters *$\varepsilon > 0$*,
  *$\mathbb{G} _ M(\theta, \pi, c, \varepsilon)$* satisfies the
  following three properties:*
  1.  *It runs in time
      *$O \left( \frac{1}{\varepsilon^2} \text{Time}(M _ \theta) \right)$*.*
  2.  *Its error is competitive with sampling, on average over random
      *$c$* and *$s$*:\
      *$\mathbb{E} _ {c, s}[(\mathbb{G} _ M(T(s), E(s), c, \varepsilon) - \mathbb{E} _ x[M _ {T(s)}(c, x)])^2] \le \varepsilon^2 \mathbb{E} _ {c, s}[\text{Var}_x[M _ {T(s)}(c, x)]]$*,
      where *$s \sim \{0, 1\}^r$*, *$c \sim \{0, 1\}^{n_c}$*, and
      *$x \sim \{0, 1\}^{n_x}$*.*
  3.  *It is mechanistic.*

In this statement, it is useful to think of $T$ as being the learning
algorithm used to find $\theta$ (e.g. SGD) and $s$ as being the random
bits used during training (the random initialization and random choices
of training data used at each training step). Then, $E$ is the algorithm
used to find the mechanistic explanation of $\theta$: intuitively, it
works \"in parallel\" with $T$, observing the training process and
\"building up\" the explanation $\pi$ in a way that mirrors the way that
$T$ \"builds up\" structure in $\theta$ by iteratively modifying
$\theta$ to get lower and lower loss.

Note that our mainline MSP statement is the special case of the \"train
and explain\" formulation where $T$ and $E$ are both computationally
unbounded (so that $T$ can select the *worst* parameters $\theta$ and
$E$ can select the *most helpful* advice $\pi$).

In general, we think that for any computational constraints placed on
$T$ (e.g. on time or memory), there is a corresponding $E$ with the same
computational constraints that can find an adequate explanation $\pi$.
If we are correct, then that potentially gives us a strategy for
efficiently computing properties of trained neural networks (such as
catastrophe probability), while paying a relatively small [alignment
tax](https://www.alignmentforum.org/w/alignment-tax). (If finding $\pi$
takes as much time as finding $\theta$, that\'s an alignment tax of
100%: a small price to pay for avoiding catastrophe.)



### Another modification: getting rid of $\varepsilon$

It turns out that the MSP can be stated without reference to a tolerance
parameter, by subsuming the number of samples into the architecture
instead. See [this appendix](#getting-rid-of-varepsilon) for details.



# Our progress so far

Over the course of 2025, ARC has made progress on the MSP in a few
different directions. Concretely:

- We have a mechanistic [algorithm](#intersection-of-random-half-spaces)
  that competes with sampling (in theory and in practice) for estimating
  the size of an intersection of randomly chosen
  halfspaces.^[\[16\]](#fn16){#fnref16}^ We have also generalized our
  algorithm to work for some other problems, such as estimating the
  satisfaction probability of a random
  [CNF](https://en.wikipedia.org/wiki/Conjunctive_normal_form), or the
  [permanent](https://en.wikipedia.org/wiki/Permanent_(mathematics)) of
  a random matrix.
- We believe we have a mechanistic [algorithm](#random-mlps) that
  competes with sampling for estimating the expected output of random
  MLPs on Gaussian inputs. (We have an empirical demonstration of
  competitiveness with sampling, and a proof sketch that we are working
  on expanding into a full proof.)
- We have made substantial progress toward a mechanistic
  [algorithm](#two-layer-mlps-with-a-trained-second-layer) that competes
  with sampling for estimating the expected output of a two-layer MLP on
  Gaussian inputs, where the second layer of the MLP is trained via
  gradient descent.

Our results are not yet ready for publication, but hope to get them
ready in the coming months. In this section, I will briefly summarize
these results and discuss the most interesting directions for future
work.



## Intersection of random half-spaces

The first problem we tackled in our \"matching sampling\" framework was
mechanistically estimating the volume of the intersection of random
half-spaces. Although it\'s a somewhat toy problem, it wasn\'t trivial
to solve, and we learned a lot from solving it.

### Problem statement

Find an algorithm $\mathbb{G}$ that takes as input unit vectors
$\mathbf{v} _ 1, \ldots, \mathbf{v} _ k \in \mathbb{R}^n$ and a
tolerance parameter $\varepsilon$, and mechanistically estimates the
probability that a randomly chosen unit vector $\mathbf{x}$ has a
nonnegative dot product with all of
$\mathbf{v} _ 1, \ldots, \mathbf{v} _ k$, such that:

- The expected squared error of $\mathbb{G}$ (over randomly chosen
  $\mathbf{v} _ 1, \ldots, \mathbf{v} _ k$) is less than the expected
  squared error of the naive sampling-based algorithm (i.e. sampling
  $1/\varepsilon^2$ random vectors $\mathbf{x}$ and outputting the
  empirical fraction that have a nonnegative dot product).
- The runtime of $\mathbb{G}$ is competitive with the runtime of the
  naive sampling-based algorithm, i.e.
  $O (kn/\varepsilon^2)$.^[\[17\]](#fn17){#fnref17}^

Note that this is the MSP in the particular case where $\theta$ is
the empty string; $c = (\mathbf{v} _ 1, \ldots, \mathbf{v} _ k)$; and
$M(c, \mathbf{x})$ returns 1 if $\mathbf{x} \cdot \mathbf{v} _ t \ge 0$
for all $t$. (The distribution of $\mathbf{x}$ and each vector in $c$ is
uniform over the unit sphere, rather than uniform over bit strings.)

Since $\theta$ is empty, there is no advice $\pi$ in this setting.
Despite that, we think that our solution to this MSP instance has helped
us progress toward solving the MSP in more generality.

(Alternatively, this problem can be viewed as an instance of the \"train
and explain\" version of the MSP, where
$\theta = (\mathbf{v} _ 1, \ldots, \mathbf{v} _ k)$, $c$ is empty, and
the training algorithm $T$ simply returns random vectors. In this
setting, the explaining algorithm $E$ does not have time to do any
interesting computation, so $\pi$ might as well be empty.)



### A sketch of our solution

In a sentence, our solution is to build up a polynomial approximation of
$M(c, \mathbf{x})$ by considering one vector $\mathbf{v} _ t$ at a time.

To elaborate on this, for $1 \le t \le k$, let $h_t(\mathbf{x})$ be the
function that outputs 1 if $\mathbf{v} _ t \cdot \mathbf{x} \ge 0$, and
0 otherwise. Let us define
$H_{\le t}(\mathbf{x}) := h_1(\mathbf{x}) \ldots h_t(\mathbf{x})$ (so
$M(c, \mathbf{x}) = H_{\le k}(\mathbf{x})$). We will:

- Define $\tilde{H}_{\le 0}$ to be the constant 1 function.
- For all $1 \le t \le k$, compute the best low-degree polynomial
  approximation $\tilde{H} _ {\le t}$ to
  $\tilde{H} _ {\le t - 1} h _ t$. In the end, $\tilde{H} _ {\le k}$
  will be a polynomial approximation to $H _ {\le k}$, and we will
  output the exact expectation $\mathbb{E}[\tilde{H} _ {\le k}]$ as our
  estimate of $\mathbb{E} _ x[M(c, \mathbf{x})]$.

To be more precise, \"low-degree polynomial\" here means degree $d$,
where $d$ is the largest integer such that $n^d \le 1/\varepsilon^2$ (it
turns out that that\'s the precision we need in order to compete with
sampling). And \"the best low-degree polynomial approximation\" means
the best approximation in terms of squared error, for $\mathbf{x}$ drawn
from the unit sphere.

In order for $\mathbb{G}$ to be efficient enough, it needs to be able to
compute this polynomial approximation in time $n/\varepsilon^2$. Getting
the dependence on $\varepsilon$ down to $1/\varepsilon^2$ turns out to
be pretty tricky.^[\[18\]](#fn18){#fnref18}^ However, we were able to
find a suitably efficient algorithm by working in the [Hermite
basis](https://en.wikipedia.org/wiki/Hermite_polynomials) of polynomials
instead of the standard monomial basis.

We have also generalized this approach to apply to a broader class of
problems than just \"intersection of randomly-chosen half-spaces.\"
Roughly speaking, we can apply our methods to estimate the expected
product of symmetric random functions, for a certain
representation-theoretic notion of symmetry. Concrete problems that we
have solved with this approach include:

- Estimating the satisfaction probability of
  $k$-[CNFs](https://en.wikipedia.org/wiki/Conjunctive_normal_form),
  where each literal is chosen uniformly at random.
- Estimating the
  [permanent](https://en.wikipedia.org/wiki/Permanent_(mathematics)) of
  a matrix where each entry is either $-1$ or $1$,
  independently with probability 50%.

We are aiming to publish the details of our algorithm and this
generalization in the coming months.



### Related questions

While we have solved this particular MSP instance, there are some
related settings for which we do not have a solution:

- **Non-isotropic half-spaces.** What if, instead of
  $\mathbf{v} _ 1, \ldots, \mathbf{v} _ k$ being uniformly randomly
  selected from the unit sphere (or, equivalently, the standard
  $n$-variate normal distribution), they are selected from some other
  $n$-variate normal distribution? Modulo some details, we believe that
  we have solved this problem if
  $\mathbf{v} _ 1, \ldots, \mathbf{v} _ k \sim \mathcal{N}(\mu, aI _ n + b\mu\mu^\top)$
  for scalars $a, b$ and $\mu \in \mathbb{R}^n$. But we don\'t have a
  general solution to
  $\mathbf{v} _ 1, \ldots, \mathbf{v} _ k \sim \mathcal{N}(\mu, \Sigma)$
  for arbitrary covariance matrices $\Sigma$.
- **Learned half-spaces.** What if some sort of learning algorithm is
  used to train $\mathbf{v} _ 1, \ldots, \mathbf{v} _ k$ to minimize
  some loss function? Is there an explanation string $\pi$ that we could
  learn in parallel with the learning algorithm, which would help us
  estimate the size of the intersection of the half-spaces? Or, perhaps,
  the setting is still simple enough that no explanation is necessary,
  even if the half-spaces are learned?



## Random MLPs

In the last couple of months, we have been tackling a more sophisticated
MSP instance: random
[MLPs](https://www.geeksforgeeks.org/deep-learning/multi-layer-perceptron-learning-in-tensorflow/).
We now believe that we have an algorithm and a proof of its correctness
and efficiency, though we are still verifying details. We also have an
empirical demonstration that our algorithm is competitive with sampling.

### Problem statement {#problem-statement}

Consider the following MLP architecture: the input size is $n$ (assume
that $n$ is very large^[\[19\]](#fn19){#fnref19}^); there are $k$ hidden
layers ($k$ is some fixed constant), each of width $n$; the output is a
scalar; and every hidden layer has some activation function (e.g. ReLU).

Find an algorithm $\mathbb{G}$ that takes as input the weights
$\mathbf{W}$ of an MLP with the above architecture and a tolerance
parameter $\varepsilon$, and mechanistically estimates the expected
output of the MLP on inputs drawn from $\mathcal{N}(\mathbf{0}, I_n)$,
such that:

- The expected squared error of $\mathbb{G}$ (over independent, normally
  distributed^[\[20\]](#fn20){#fnref20}^ weights) is less than the
  expected squared error of the naive sampling-based algorithm (i.e.
  sampling $1/\varepsilon^2$ random inputs and outputting the empirical
  average output).
- The runtime of $\mathbb{G}$ is competitive with the runtime of the
  naive sampling-based algorithm, i.e. $O(1/\varepsilon^2)$ forward
  passes.

Note that this is the MSP in the particular case where $\theta$ is
the empty string; $c = \mathbf{W}$; and $M(\mathbf{W}, \mathbf{x})$
returns the output of the MLP with weights $\mathbf{W}$ on input
$\mathbf{x}$. (The distribution of each component of $\mathbf{x}$ and
$\mathbf{W}$ is Gaussian.)

(Similarly to the case of random half spaces, this problem can also be
viewed as an instance of the \"train and explain\" version of the MSP,
where $\theta = \mathbf{W}$, $c$ is empty, the training algorithm $T$
simply returns random weights, and $\pi$ might as well be empty.)



### A sketch of our solution {#a-sketch-of-our-solution}

In a sentence, our solution to this problem is *cumulant propagation,* a
mechanistic estimation algorithm that we introduced in Appendix D of
[Formalizing the Presumption of
Independence](https://arxiv.org/abs/2211.06738).

[Cumulants](https://en.wikipedia.org/wiki/Cumulant) are a type of
summary statistic of a probability distribution. Loosely speaking, the
cumulant operator $\kappa$ takes a list of random variables and tells
you something like their \"multi-way correlation.\" For example,
$\kappa(X)$ is the mean of $X$; $\kappa(X, X)$ is the variance;
$\kappa(X_1, X_2)$ is the covariance of $X_1$ and $X_2$.

Cumulant propagation is a method that lets us make guesses about the
cumulants of layer $\ell$ of a neural net based on a partial list of
cumulants of layer $\ell - 1$. (The more complete the list of cumulants,
the more accurate the guesses become.) To a first approximation, then,
our algorithm is to:

- Start with a list of cumulants of the inputs (this is easy because the
  input is Gaussian: $\kappa(x_i, x_i) = 1$ for each input $i$,
  and all other cumulants are $0$).
- Use cumulant propagation to make guesses about the cumulants of the
  layer-1 activations,^[\[21\]](#fn21){#fnref21}^ going up to $d$-th
  order cumulants,^[\[22\]](#fn22){#fnref22}^ where $d$ is the largest
  integer such that $n^d < 1/\varepsilon^2$. Then do the same for the
  layer-2 activations, and so on.
- Output our guess about the mean (i.e. first cumulant) of the output.

(This description leaves out many details, but gets across the main
idea.)



### Related questions {#related-questions}

We would be really interested in finding a way to mechanistically
estimate the average output of **random recurrent neural networks
(RNNs).** We believe that this will be much more difficult than the MLP
setting, because of the weight sharing. (We think that interesting
structure can arise in random RNNs in a way that\'s far more improbable
for random MLPs; this is related to the fact that [RNNs are
Turing-complete](https://binds.cs.umass.edu/papers/1995_Siegelmann_JComSysSci.pdf).)
We think it\'s possible that finding an algorithm that solves the
\"random RNNs\" instance of the MSP would constitute major progress
toward finding an algorithm that solves the MSP in full generality (see
the [appendix](#compression-as-a-possible-msp-approach) for more
discussion).

A shorter-term project might be to adapt our solution to other
architectures. Can we solve the problem in the case of *narrow* MLPs (as
opposed to the infinite-width limit)? What about random CNNs? Random
transformers?



## Two-layer MLPs with a trained second layer

In parallel with tackling random MLPs, we have also been investigating
two-layer MLPs where the hidden layer is very wide, and where the second
layer of weights is trained. This is our first serious foray into
trained and/or worst-case instances \-- and while we haven\'t fully
solved it, we have made substantial progress.

### Problem statement {#problem-statement}

Consider the following MLP architecture: the input size is $n_0$; there
is one hidden layer of size $n_1$, where $n_1$ is very large; the output
is a scalar; and there is an activation function at the hidden layer and
the output layer.

**Problem 1:** Find an algorithm $\mathbb{G}$ that takes as input the
weights $(W, \mathbf{v})$ of an MLP with the above architecture
($W \in \mathbb{R}^{n_1 \times n_0}$ contains the first-layer weights;
$\mathbf{v} \in \mathbb{R}^{n_1}$ contains the second-layer weights), an
explanation $\pi$, and a tolerance parameter $\varepsilon$, and
mechanistically estimates the expected output of the MLP on inputs drawn
from $\mathcal{N}(\mathbf{0}, I_n)$, such that:

- **For all** $\mathbf{v}$, the expected squared error of $\mathbb{G}$
  (over independent, normally distributed weights in $W$) is less than
  the expected squared error of the naive sampling-based algorithm (i.e.
  sampling $1/\varepsilon^2$ random inputs and outputting the empirical
  average output).
- The runtime of $\mathbb{G}$ is competitive with the runtime of the
  naive sampling-based algorithm, i.e. $O(1/\varepsilon^2)$ forward
  passes.

Note that this is the MSP in the particular case where
$\theta = \mathbf{v}$; $c = W$; and
$M((W, \mathbf{v}), \mathbf{x})$ returns the output of the MLP with
weights $(W, \mathbf{v})$ on input $\mathbf{x}$.

**Problem 2:** Now, suppose that $\mathbf{v}$ is trained via SGD to make
the MLP match some target function. Extend the solution to Problem 1 by
finding a linear-time algorithm that takes as input the full
transcript of SGD and outputs $\pi$.

This is the \"train and explain\" version of the MSP in the particular
case where $\theta = \mathbf{v}$ and the training algorithm $T$ is SGD
with squared loss on an arbitrary target function.



### A look at our progress so far

Unlike in the case of random MLPs, we do not expect cumulant propagation
to work. That\'s because, for worst-case $\mathbf{v}$, the largest
cumulants will not necessarily be the low-order ones; thus, dropping
high-order might not produce a good approximation. So what can we do
instead?

Consider the function $f: \mathbb{R}^{n_0} \to \mathbb{R}$ that maps the
input to the final pre-activation (i.e. the output, but before the final
activation function is applied). If we could find the cumulants of
$f(\mathbf{x})$ (i.e. the mean, variance, etc. of the final
pre-activation on random inputs), then we would be able to find the mean
of the output of the MLP. So how can we estimate these cumulants?

The function $f$ can be well-approximated by a high-degree,
$n_0$-variable polynomial in the inputs. And as it turns out, there is a
neat way to express the cumulants of a multivariate polynomial as an
infinite sum in terms of the polynomial\'s coefficients in the [Hermite
basis](https://en.wikipedia.org/wiki/Hermite_polynomials). In
particular, for each $d$, the degree-$d$ coefficients can be written
down in a $d$-dimensional $n_0 \times \ldots \times n_0$-tensor. (Having
run out of Roman and Greek letters, we decided to call this tensor
ש$_ d$.^[\[23\]](#fn23){#fnref23}^) Then the $d$-th cumulant is the sum
of all tensor contractions across tensor networks consisting of copies
of ש$_ d$. This leaves us with two problems:

- Computing the ש-tensors.
- Approximating the infinite sum, given the ש-tensors.

Our solution to the first problem is to **receive the **ש**-tensors as
advice.** It turns out that, so long as $1/\varepsilon^2 < n_1$ (i.e. in
the infinite-width limit of the hidden layer), we have enough room in
$\pi$ to write down all of the ש-tensors we need. (And for the \"train
and explain\" version of the problem, we believe that we can learn the
ש-tensors in parallel with SGD.)

We are currently working on the second problem (summing up the tensor
networks), and we have made substantial partial progress. If the
hidden layer width is truly huge compared to the input size, then there
is enough time to approximate the sum by brute force. If the hidden
layer is large but not huge, then a more efficient algorithm is
necessary. We are working on finding efficient ways to contract
arbitrary tensor networks and being able to notice when a tensor network
can only contribute negligibly to the sum (so that we can drop it from
the sum).^[\[24\]](#fn24){#fnref24}^



### Related questions {#related-questions}

Once we have an on-paper solution, we will be interested to see how well
the solution works in practice. There are some reasons to believe that
it would be slower than sampling (the algorithm is likely to be quite
complex), but also some reasons to believe that it would be faster than
sampling (on paper, it would match the performance of sampling under
worst-case conditions; in typical conditions, it might outperform
sampling).

Additionally, even though we are excited about our progress on this
question, \"two-layer MLPs with a very large hidden layer, where only
the second layer is trained\" is ultimately a fairly narrow setting.
There are many directions in which we could try to generalize our
methods: deeper MLPs; hidden layers that are of similar size to the
input layer; training both layers; other distributions of input data;
and so on.



# Closing thoughts

I consider the MSP to be a significant step forward for ARC. Previously,
we were interested in [producing mechanistic
estimates](https://arxiv.org/abs/2211.06738) of mathematical quantities,
but had no particular benchmark by which to judge our progress or deem
our methods \"good enough.\" Now, we are holding ourselves to a standard
that is philosophically justified (we believe that it ought to be
possible for mechanistic estimates to compete with sampling), concrete
(we can check whether our methods compete with sampling using empirical
tests or formal proofs), and tied to a useful application (estimating
properties of neural nets, such as catastrophe probability).

Formulating the MSP has allowed us to ask more concrete questions (e.g.
\"How can we construct a mechanistic algorithm that competes with
sampling for estimating the average output of trained two-layer
MLPs?\"). We have solved some of these questions, made progress on
others, and are continuing to make progress.

We plan to continue attacking the MSP from a number of directions:

- Attempting to solve the MSP \"on paper\" (i.e. using mathematical
  tools) for specific instances (like the ones described in this post).
- Using our theoretical methods to create state-of-the-art algorithms
  for estimation problems.
- Approaching the problem from a more high-level or philosophical
  perspective in order to discern what sorts of mechanistic algorithms
  could compete with sampling in full generality.

If you\'re interested in working with us on any of these directions, you
can [apply here](https://www.alignment.org/hiring/)!



# Appendix

## A note on advice verifiability

In our various MSP statements, we do not ask for $\mathbb{G} _ M$ to be
able to \"verify\" that the explanation $\pi$ is \"accurate\" (i.e.
correctly describes the structure of $M _ \theta$, instead of making
false claims). Is that fine, or should we require $\pi$ to be verifiable
by $\mathbb{G} _ M$?

At least in the \"train and explain\" version of the MSP, we do not
believe that advice needs to be verifiable. This is for two reasons:

- Our eventual goal is to implement any MSP solution for actual neural
  nets, and to have strong accuracy guarantees (on average over the
  randomness of training). Solving the \"train and explain\" version of
  the MSP for a given neural architecture already comes with an accuracy
  guarantee, even without the ability to verify the explanation that the
  explaining algorithm $E$ produces.
- Suppose that the training algorithm $T$ finds $\theta$ by checking
  many random candidate values of $\theta$ until it finds a particularly
  unlucky one (e.g. one where $\mathbb{E}[M _ \theta]$ is much lager
  than a full structural understanding of $\theta$ would suggest, just
  by chance). By observing the computations done by $T$, $E$ can notice
  that $\theta$ is unlucky, but it cannot succinctly explain that fact
  in a verifiable way. All it can do is *assert* (in its explanation
  $\pi$) that $\theta$ is unlucky, and all $\mathbb{G} _ M$ can do is
  trust $E$\'s assertion.

This last point seems a little bit at odds with my [earlier
assertion](#the-intuition-behind-the-msp) that $\pi$ only makes claims
about the structure of $\theta$, not the randomness. A more refined
version of this assertion would be: $\pi$ should not assert any
randomness in $\theta$ that happened by accident; but if $\theta$ has
weird randomness due to some fact about the training process, then $\pi$
should reflect that fact.

What about our mainline MSP statement, where there is no explaining
algorithm to track optimization done during training? If $\pi$ \"comes
out of nowhere,\" are we comfortable with $\pi$ asserting facts about
$\theta$ without a possibility of verification?

In my opinion, it\'s fine for $\pi$ to be unverifiable, for essentially
the same reason. If it\'s fine for $\pi$ to claim that $\theta$ was
selected to be as adversarial as possible via a brute force search (in
the \"train and explain\" version of the MSP), then it seems fine for
$\pi$ to claim that $\theta$ was selected to be as adversarial as
possible by an omniscient oracle, if that\'s how $\theta$ was selected.

For example, imagine that we can model $M _ \theta$ as a [random
oracle](https://en.wikipedia.org/wiki/Random_oracle) \-- a completely
different random function for each $\theta$ \-- and the particular
$M _ \theta$ that\'s chosen happens to be the one whose average output
is furthest from 50/50. Then it seems fine for $\pi$ to assert that
$\theta$ is a random oracle whose average output just so happens to be
many standard deviations away from 50/50.

There might be natural versions of the MSP that require advice to be
verifiable. However, such statements would require giving
$\mathbb{G} _ M$ more time to run. Concretely, in the mainline MSP
statement, we would ask for $\mathbb{G} _ M$ to run in time
$O \left( \frac{|\theta|}{\varepsilon^2} \text{Time}(M _ \theta) \right)$,
where the extra factor of $|\theta|$ mitigates the selection pressure
put into choosing $\theta$. In the case where $M _ \theta$ is a random
oracle, this is exactly the amount of compute that $\mathbb{G} _ M$
needs to compete with sampling, if $\pi$ can convey that $M _ \theta$ is
a random oracle, but cannot assert anything about $M _ \theta$ being a
*particularly unlucky* random oracle. I like this version less, in part
because we don\'t expect that paying the extra $|\theta|$ factor will be
feasible in practice.



## A special case of the MSP: Universal Turing machines

One interesting case of the MSP is when the architecture $M$ is a
universal Turing machine $U$. In other words, $U_\theta(c, x)$
interprets $\theta$ as the encoding of a Turing machine, and runs
$\theta$ on the input $(c, x)$ \-- except that we will say that $\theta$
is forced to halt after one million steps (so that we don\'t need to
worry about runtime). Applying the MSP to this special case gives the
following assertion:

- There exists an estimator $\mathbb{G}(\theta, \pi, c, \varepsilon)$
  such that:
- For all Turing machines $\theta$, there is an explanation $\pi$
  ($|\pi| \le O(|\theta|)$), such that:
- For all tolerance parameters $\varepsilon > 0$, $\mathbb{G}$ satisfies
  the following three properties:
  - It runs in time $O(1/\varepsilon^2)$.
  - On average over random $c$, its error is competitive with sampling:\
    $\mathbb{E} _ c[(\mathbb{G}(\theta, \pi, c, \varepsilon) - \mathbb{E} _ {x \sim \{0, 1\}^n}[\theta(c, x)])^2] \le \varepsilon^2 \mathbb{E} _ c[\text{Var}_{x \sim \{0, 1\}^n}[\theta(c, x)]].$
  - It is mechanistic.

In other words, there is a single, universal $\mathbb{G}$ that is able
to mechanistically estimate the average output of any (time-bounded)
Turing machine, if it is given advice that explains the Turing
machine\'s structure.

Note also that a solution to this special case would yield a solution to
the full MSP: suppose that we had an estimator $\mathbb{G}$ for
universal Turing machines, and consider some other architecture $M$.
Then $M _ \theta = U_{\theta'}$, where $\theta'$ is a Turing machine
whose size is $|\theta|$ plus some constant that only depends on $M$.
Consider the estimator $\mathbb{G} _ M$ that, on input
$(\theta, \pi, c, \varepsilon)$, writes down the $\theta'$ such that
$M _ \theta = U_{\theta'}$ and returns
$\mathbb{G}(\theta', \pi, c, \varepsilon)$. If some $\pi$ causes
$\mathbb{G}$ to output accurate estimates for $U_{\theta'}$, then $\pi$
will also cause $\mathbb{G} _ M$ to output accurate estimates for
$M _ \theta$. Thus, this estimator $\mathbb{G} _ M$ solves our mainline
MSP for $M$.

The MSP statement can also be used to obtain a claim about
mechanistically estimating *random* Turing machines, but *without
advice.* Concretely, we will let $\theta$ be the empty string, and will
instead say that $M _ \theta(c, x) := M(c, x)$ interprets $c$ as the
encoding of a Turing machine, and runs $c$ on the input $x$. (As before,
we force $c$ to halt after a million steps.) Applying the MSP to this
special case gives the following assertion:

- There exists an estimator $\mathbb{G}(c, \varepsilon)$ such that:
- For all tolerance parameters $\varepsilon > 0$, $\mathbb{G}$ satisfies
  the following three properties:
  - It runs in time $O(1/\varepsilon^2)$.
  - Its error is competitive with sampling, on average over random $c$:\
    $\mathbb{E} _ c[(\mathbb{G}(c, \varepsilon) - \mathbb{E} _ x[c(x)])^2] \le \varepsilon^2 \mathbb{E} _ c[\text{Var}_x[c(x)]]$.
  - It is mechanistic.

This is an interesting and arguably bold statement: it says that as
$\mathbb{G}$ gets more time to run, it is able to get a more and more
accurate mechanistic estimate of the average output of the Turing
machine $c$. This is intuitive enough for Turing machines with no
interesting structure (as is the case for most random Turing machines).
However, in order to satisfy the accuracy guarantee above, $\mathbb{G}$
must converge to the right answer for *all* Turing machines (even if the
convergence is slower for Turing machines with more sophisticated
structure). Such a $\mathbb{G}$ would probably involve a systematic
search for structure: loosely speaking, since it isn\'t given an
explanation, it must find the explanation on its own.



## Getting rid of $\varepsilon$

Modulo a caveat (see below), it is possible to modify the MSP statement
to get rid of the tolerance parameter $\varepsilon$. Concretely, suppose
that the following statement \-- which specializes our mainline MSP
statement to the case of $\varepsilon = 1$ \-- is true:

- For all architectures $M$ (with parameters $\theta$) mapping pairs
  $(c \in \{0, 1\}^{n_c}, x \in \{0, 1\}^{n_x})$ to $\mathbb{R}$, there
  exists an estimator $\mathbb{G} _ M$ mapping tuples
  $(\theta, \pi, c)$ to $\mathbb{R}$, such that:
- For all parameters $\theta$, there exists a short explanation $\pi$
  ($|\pi| \le O(|\theta|)$), such that:
- $\mathbb{G} _ M(\theta, \pi, c)$ satisfies the following three
  properties:
  1.  It runs in time $O(\text{Time}(M _ \theta))$.
  2.  Its error is competitive with sampling, on average over random
      $c$:\
      $\mathbb{E} _ c[(\mathbb{G} _ M(\theta, \pi, c) - \mathbb{E} _ x[M _ \theta(c, x)])^2] \le \mathbb{E} _ c[\text{Var}_x[M _ \theta(c, x)]]$,
      where $c \sim \{0, 1\}^{n_c}$ and $x \sim \{0, 1\}^{n_x}$.
  3.  It is mechanistic.

We claim that our mainline MSP statement *almost* follows from this
$\varepsilon$-less version. To see this, consider an arbitrary
architecture $M$, and fix a positive integer $m$. We will define a
modified architecture $M'$ that has the same space of parameters as $M$.
Concretely, $M'_\theta$ works as follows: it takes as input a **list of
inputs** $x_1, \ldots, x_m$ to $M _ \theta$, runs $M _ \theta$ on all of
them, and outputs the **average value** of $M _ \theta(x_i)$ for
$i \in \{1, \ldots, m\}$. We claim if some estimator $\mathbb{G} _ {M'}$
solves the above MSP for all $M'$ regardless of the particular value of
$m$, then $\mathbb{G} _ {M'}$ also solves the mainline MSP for $M$.

To see this, suppose that we have an estimator $\mathbb{G} _ {M'}$ that
solves the above MSP for $M'$, in the case of $\varepsilon = 1$. This
means that for any $\theta$, there is an explanation $\pi$ such that
$\mathbb{G} _ {M'}$:

1.  Runs in time $O \left( \text{Time}(M'_\theta) \right)$.
2.  Has low error on average over random $c$:\
    $\mathbb{E} _ c[(\mathbb{G} _ {M'}(\theta, \pi, c) - \mathbb{E} _ {x = (x _ 1, \ldots, x _ m)}[M'_\theta(c, x)])^2] \le \mathbb{E} _ c[\text{Var} _ {x = (x _ 1, \ldots, x _ m)}[M' _ \theta(c, x)]]$.
3.  Is mechanistic.

Note that
$\mathbb{E} _ {(x_1, \ldots, x_m)}[M' _ \theta(c, (x_1, \ldots, x_m))] = \mathbb{E} _ x[M _ \theta(c, x)]$
and
$\text{Var}_{(x_1, \ldots, x_m)}[M' _ \theta(c, (x _ 1, \ldots, x _ m))] = \frac{1}{m} \text{Var} _ x[M _ \theta(c, x)]$.
Note also that $\mathbb{G} _ {M'}$ runs in time
$O(m \cdot \text{Time}(M _ \theta))$. This means that that
$\mathbb{G} _ {M'}$ also solves the mainline MSP for the architecture
$M$ if $\varepsilon^2 = \frac{1}{m}$.

Now, if there is a single $\mathbb{G} _ {M'}$ that solves the MSP
for $M'$ regardless of $m$, then $\mathbb{G} _ {M'}$ will solve the
mainline MSP for $M$ for all $\varepsilon$.

The fact that we need a uniform $\mathbb{G} _ {M'}$ regardless of $m$
means that we don\'t quite have a full reduction; however, the above
$\varepsilon$-less MSP statement is another interesting variant of MSP
that is *almost* the same. We decided to make our mainline MSP statement
contain a tolerance parameter $\varepsilon$ in order to make the
connection to the idea of matching sampling more intuitive.



## Compression as a possible MSP approach

In this section, I will outline one possible approach to solving the
MSP. For the sake of concreteness, I will consider the case where $M$ is
a universal Turing machine (which was discussed
[above](#a-special-case-of-the-msp-universal-turing-machines)). As a
reminder, this means that $M _ \theta(c, x)$ interprets $\theta$ as the
encoding of a Turing machine, and then runs $\theta$ on the input
$(c, x)$.

We will say that a Turing machine $\theta$ is *efficiently compressible*
if there is a significantly shorter Turing machine $\theta'$ that, on
any input $x$, constructs $\theta$ in time $O(|\theta|)$ and the runs
$\theta$ on input $x$. (We call $\theta'$ an *efficient compression* of
$\theta$.) One possible approach to solving the MSP looks something like
this:

1.  We solve the MSP for instances that are not efficiently
    compressible. Concretely, we find an estimation algorithm
    $\mathbb{G} _ M^{\text{inc}}(\theta, c, \varepsilon)$ that works for
    any $\theta$ that is not efficiently compressible, without any
    explanation $\pi$.
2.  We define the estimation algorithm
    $\mathbb{G} _ M(\theta, \pi, c, \varepsilon)$ to check that
    $\pi$ is an efficient compression of $\theta$, and then to
    return $\mathbb{G} _ M^{\text{inc}}(\pi, c, \varepsilon)$ if so.
    - If $\pi$ is not efficiently compressible, then $\mathbb{G} _ M$\'s
      output will be accurate (since $\mathbb{G} _ M^{\text{inc}}$
      solves the MSP for instances that are not efficiently
      compressible).

The hope for Step 1 is that *an estimation approach that works in the
average case (over random parameters) will work for all parameters that
are not efficiently compressible.*

The intuition underlying this hope is that *structure implies efficient
compression.* In other words, if $\theta$ has structure that would make
a mechanistic estimator mis-estimate its average output, then
understanding that structure would allow us to represent $\theta$ more
compactly (and in a way such that $\theta$ can be recovered quickly from
the representation).

What about the \"train and explain\" version of the MSP? In order to
adapt this approach to that setting, we also need to be able to *learn*
the efficient compression in parallel with learning $\theta$ itself. If
the training process has enough time to find a $\theta$ with special
structure, then is there an \"explaining process\" that would have
enough time to find the corresponding compression? That is unclear to
me, but I think this direction is promising enough to be worth
exploring.

If this approach is viable, then solving the MSP in the average case for
some Turing-complete architecture $M$ (such as RNNs) would be a major
step forward.



*Cross-posting for comments:
[LessWrong](https://www.lesswrong.com/posts/XdQd9gELHakd5pzJA/arc-progress-update-competing-with-sampling)*



------------------------------------------------------------------------

1.  
    We could instead imagine that $C$ outputs a probability of
    catastrophe, but we will keep the range of $C$ to $\{0, 1\}$ for
    simplicity of exposition. [↩︎](#fnref1){.footnote-backref}

2.  
    Running $C$ might take much longer than running $M$, which is why we
    can\'t just run $C$ on every input during deployment.
    [↩︎](#fnref2){.footnote-backref}

3.  
    For example, the parameterization of the distribution $D$ needs to
    be quite flexible, so as to allow distributions that are
    computationally intractable to sample from. For example, if $M$ acts
    catastrophically when it encounters a factorization of RSA-2048, we
    want to be able to train that behavior out of $M$ even if we can\'t
    factorize
    [RSA-2048](https://en.wikipedia.org/wiki/RSA_numbers#RSA-2048). (See
    [here](https://ai-alignment.com/training-robust-corrigibility-ce0e0a3b9b4d)
    for more discussion.) [↩︎](#fnref3){.footnote-backref}

4.  
    In brief, we hope to address issue #1 via [mechanistic anomaly
    detection](https://www.alignment.org/blog/mechanistic-anomaly-detection-and-elk/).
    A little more concretely, our plan is to:Build $C$ to be smart and
    robust enough that $M$ can only fool $C$ by thinking deliberately
    about how to fool $C$.In the [diamond-in-the-vault
    analogy](https://docs.google.com/document/d/1WwsnJQstPq91_Yh-Ch2XRL8H_EpsnjrC1dwZXR37PC8/edit?tab=t.0),
    this means that the cameras are robust enough that the SmartVault
    can only fool them deliberately.Build a mechanistic explanation for
    why $C(M(x))$ tends to be 0 on the training distribution. This
    would let us flag inputs $x^ *$ on which $C(M(x^ * ))=0$ for an
    unusual reason. Let $A(x)$ be this anomaly detector, i.e. $A(x)=1$
    if $C(M(x))=0$ for an unusual reason.Instead of training $M$ to
    minimize $\mathbb{E} _ {x \sim D}[C(M(x))]$, train $M$ to minimize
    $\mathbb{E} _ {x \sim D}[\max\{C(M(x)), A(x)\}]$ (i.e. to maximize
    the probability that $C(M(x))=0$ for the usual reason).And similarly
    to above, train $D$ to maximize this expectation.This leaves many
    details unexplained, but that\'s the basic concept.
    [↩︎](#fnref4){.footnote-backref}

5.  
    Or rather, independent up to small random variation that is
    unpredictable just from understanding the structure of $M$ and $D$.
    [↩︎](#fnref5){.footnote-backref}

6.  
    In practice, we will be interested in the behavior of neural nets on
    structured (rather than uniformly random) inputs. However, note that
    it is possible to create structured inputs out of random inputs via
    a generative model. For example, if we are interested in the
    behavior of a classification model on pictures of animals, we could
    let $M _ \theta$ consist of two parts: first, a generative model
    that creates an image of an animal from random noise, and second, a
    classifier that takes the animal image as input.
    [↩︎](#fnref6){.footnote-backref}

7.  
    Why do we require $\pi$ to be short? The basic reason is that, as we
    discuss later, we will be interested in learning $\pi$ in parallel
    with learning the parameters $\theta$, and so we will want to be
    able to do a backward pass through $\pi$ as quickly as doing a
    backward pass through $\theta$. We also have some amount of
    philosophical justification for believing an explanation the size of
    $\theta$ is sufficient. Essentially, we think that any object\'s
    structure can be described compactly, because if the amount of
    (non-redundant) structure in an object is much larger than the size
    of the object itself, that would constitute an \"[outrageous
    coincidence](https://mxphi.com/wp-content/uploads/2023/04/MxPhi-Gowers2023.pdf)\".
    [↩︎](#fnref7){.footnote-backref}

8.  
    This works by taking our (Gaussian) model of layer $k - 1$ and then
    modeling layer $k$ by finding the normal distribution that would
    minimize the KL divergence from the pushforward of our model of
    layer $k - 1$. [↩︎](#fnref8){.footnote-backref}

9.  
    Covariance propagation does not require an explanation $\pi$.
    However, some modifications of covariance propagation could require
    advice. For example, if $\varepsilon$ is too large to allow for
    $\mathbb{G} _ M$ to compute all of the covariances, then $\pi$ could
    advise $\mathbb{G} _ M$ to only keep track of some particular
    covariances. Or, $\pi$ could tell $\mathbb{G} _ M$ about some
    important third-order correlations to keep track of.
    [↩︎](#fnref9){.footnote-backref}

10. 
    Note that we use the word \"heuristic\" in place of \"mechanistic\"
    in that paper. I think that the word \"mechanistic\" conveys our
    goal slightly better. [↩︎](#fnref10){.footnote-backref}

11. 
    As a very simple example, consider the circuit
    $(x_1 \wedge x_2) \wedge (x_2 \wedge x_3)$. Mean propagation
    estimates this circuit\'s average output as $1/16$ rather than $1/8$
    because it fails to notice the correlation induced by the presence
    of $x_2$ in the two conjunctive clauses.
    [↩︎](#fnref11){.footnote-backref}

12. 
    Though, see the [appendix on advice
    verifiability](#a-note-on-advice-verifiability) for some nuance on
    this point. [↩︎](#fnref12){.footnote-backref}

13. 
    Eliezer Yudkowsky\'s [Worse Than
    Random](https://www.lesswrong.com/posts/GYuKqAL95eaWTDje5/worse-than-random)
    makes a similar point:As a general principle, on any problem for
    which you know that a particular unrandomized algorithm is unusually
    stupid - so that a randomized algorithm seems wiser - you should be
    able to use the same knowledge to produce a superior derandomized
    algorithm. [↩︎](#fnref13){.footnote-backref}

14. 
    Roughly speaking, this is the range where $1/\varepsilon^2$ is a
    substantial fraction of the number of times that one needs to run
    $M _ \theta$ to fully estimate its unstructured randomness.
    [↩︎](#fnref14){.footnote-backref}

15. 
    We cannot require $\mathbb{G} _ M$ to be accurate for all $c$. For
    example, suppose that $M _ \theta(c, x)$ interprets $c$ as a Turing
    machine and runs $x$ on the Turing machine $c$. Requiring
    $\mathbb{G} _ M$ to be accurate for all $c$ would mean expecting
    $\mathbb{G} _ M$ to be able to mechanistically estimate the output
    of a worst-case Turing machine, without any structural advice at
    all. (After all, $\pi$ cannot depend on $c$.) This is too much to
    ask for. [↩︎](#fnref15){.footnote-backref}

16. 
    Note that this problem is equivalent to estimating the probability
    that a one-layer ReLU network outputs all zeros on a random input.
    Concretely, if the network is $\text{ReLU}(W\mathbf{x})$, then the
    output is all zeros if and only if
    $\mathbf{w} _ i \cdot \mathbf{x} \le 0$ for every row
    $\mathbf{w} _ i$ of $W$. [↩︎](#fnref16){.footnote-backref}

17. 
    Our algorithm\'s runtime is
    $O \left( \frac{ k(\log 1/\varepsilon)^2 } { \varepsilon^2 } \right)$,
    which is technically too slow in the case where
    $1/\varepsilon > 2^{\sqrt{n}}$. However, we are most interested in
    the regime where $\varepsilon = \text{poly}(n)$.
    [↩︎](#fnref17){.footnote-backref}

18. 
    The naive approach is to treat this as a linear regression problem,
    where the covariance (inner product) between two polynomials $p_1$
    and $p_2$ is defined as the expectation of
    $p_1(\mathbf{x})p_2(\mathbf{x})$ for $\mathbf{x}$ drawn from the
    unit sphere. However, doing this involves multiplying a
    $\frac{1}{\varepsilon^2} \times \frac{1}{\varepsilon^2}$ matrix by a
    $\frac{1}{\varepsilon^2}$-vector, so the dependence of this
    algorithm on $\varepsilon$ looks like $1/\varepsilon^4$: not fast
    enough. [↩︎](#fnref18){.footnote-backref}

19. 
    We believe that our proof of correctness and efficiency works in the
    limit as $n \to \infty$, where the MLP depth is constant and
    $1/\varepsilon = \text{poly}(n)$. [↩︎](#fnref19){.footnote-backref}

20. 
    Mean zero; standard deviation is chosen so that all activations have
    the same variance. [↩︎](#fnref20){.footnote-backref}

21. 
    One complication to this picture is that, although we\'ve
    [defined](https://arxiv.org/abs/2211.06738) cumulant propagation for
    sums and products of random variables, it\'s not clear what it means
    to apply cumulant propagation to an activation function like ReLU:
    given the cumulants of $X$, how does one estimate the cumulants of
    $\text{ReLU}(X)$? Our strategy is to find a polynomial approximation
    to the ReLU function (see the next paragraph of this footnote for
    details). Once we\'ve done that, we can apply cumulant propagation
    as we\'ve already defined it for sums and products.What is the
    appropriate notion of polynomial approximation? It turns out that we
    can take the polynomial that minimizes mean squared error if $X$ is
    assumed to be normally distributed with mean equal to our estimate
    of $\kappa(X)$ and covariance equal to our estimate of
    $\kappa(X, X)$. This is equivalent to taking the first several terms
    of the [Hermite
    expansion](https://en.wikipedia.org/wiki/Hermite_polynomials#Hermite_polynomial_expansion)
    of ReLU (appropriately centered and scaled).
    [↩︎](#fnref21){.footnote-backref}

22. 
    Actually, it is more important to keep track of cumulants in which
    the same activation appears multiple times, so we need to keep track
    of some cumulants of order higher than $d$ that involve repeated
    indices. [↩︎](#fnref22){.footnote-backref}

23. 
    That\'s the Hebrew letter shin. [↩︎](#fnref23){.footnote-backref}

24. 
    Concretely, a tensor network can only contribute substantially to
    the sum if every tensor in the network has a large operator norm.
    Thus, if in the process of contracting the tensor network, we find a
    tensor that has a small operator norm, we can cut off the
    computation and move onto the next tensor network.
    [↩︎](#fnref24){.footnote-backref}


\
