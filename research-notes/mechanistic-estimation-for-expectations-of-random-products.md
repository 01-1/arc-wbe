# Mechanistic estimation for expectations of random products {#mechanistic-estimation-for-expectations-of-random-products .post__title .typgr8-alpha}

Posted by [Jacob Hilton](/author/jacob/), [George
Robinson](/author/george/) , [Eric Neyman](/author/eric/) , [Paul
Christiano](/author/paul/) , [Michael Winer](/author/michael-winer/) ,
[Victor Lecomte](/author/victor/) , [Wilson Wu](/author/wilson/) &
[Gabriel Wu](/author/gabriel/) on May 15th, 2026

We have developed some relatively general methods for [mechanistic
estimation competitive with
sampling](https://www.alignment.org/blog/competing-with-sampling/) by
studying problems that are expressible as *expectations of random
products*. This includes several different estimation problems, such as
random halfspace intersections, random #3-SAT and random permanents. In
this post, we will give a high-level introduction to these methods
before sharing some more [detailed notes](#detailed-notes). This is
intended as an interim technical update and will be relatively light on
motivation: for a broader discussion of this line of research, see our
[prior post](https://www.alignment.org/blog/competing-with-sampling/).

## Random instances of the matching sampling principle

All of the problems discussed in this post can be thought of particular
choices of \"architecture\" $M$ in our [matching sampling
principle](https://www.alignment.org/blog/competing-with-sampling/#our-actual-msp-statement).
In fact, they are all choices in which $M$ has *no learned or worst-case
parameters* $\theta$. They still have *random* parameters, which are
captured in the \"context\" variable $c$, making them similar to
[randomly-initialized
networks](https://www.alignment.org/blog/mechanistic-estimation-for-wide-random-mlps/)
rather than trained networks. Note that when $\theta$ is missing from
$M$, no \"explanation\" $\pi$ is required by the estimation algorithm
$\mathbb G_M$, which reflects the fact that there is no \"structure\" in
a randomly-initialized network that needs to be pointed out.

Why study random instances of the matching sampling principle? One
simple reason is their difficulty: they are often easy enough to be
tractable, but hard enough to pose a challenge that expands our
inventory of methods. Even more than this, it is essential for us to be
able to handle randomly-initialized networks, since they serve as the
\"base case\" upon which our study of trained networks must be built.
Somewhat more speculatively, we also have some hope that we will be able
to view learned networks as random instances with a more complex
architecture, as discussed
[here](https://www.alignment.org/blog/mechanistic-estimation-for-wide-random-mlps/#extending-to-trained-networks).
For these reasons, random instances continue to occupy a significant
amount of our attention.

## Expectations of random products

The particular choices of architecture $M$ for the problems in this post
have an even more specific form, beyond having no parameters $\theta$.
The \"context\" variable is a tuple $c=\left(c_1,\ldots,c_k\right)$ for
some integer $k$, which is used by $M$ to compute a random product based
on the simple function $F$:\
$$M\left(c,x\right)=F\left(c_1,x\right)\cdots F\left(c_k,x\right).$$

Recall that the job of the mechanistic estimator
$\mathbb G_M\left(c,\varepsilon\right)$ is to approximate
$\mathbb E_x\left[M\left(c,x\right)\right]$ with low mean squared error
on average over $c$. By varying the function $F$, as well as the
distributions from which $c_i$ and $x$ are drawn, we can arrange for
this expectation to instantiate a variety of computational problems. For
example (up to constant multipliers):

- By drawing $c_i$ and $x$ from $\mathcal N\left(0,\mathbf I_n\right)$
  and taking $F\left(c_i,x\right)=1_{c_i\cdot x\geq 0}$, we obtain the
  spherical volume of the intersection of $k$ random halfspaces passing
  through the origin, which can also be thought of as the probability
  that all $k$ output neurons of a 1-layer ReLU MLP are active.
- By taking $c_i$ to be a random 3-CNF clause, drawing $x$ from
  $\left\{-1,1\right\}^n$ and taking
  $F\left(c_i,x\right)=c_i\left(x\right)$, we obtain random #3-SAT, the
  number of satisfying assignments to a random 3-CNF formula with $k$
  clauses.^[\[1\]](#fn1){#fnref1}^
- By drawing $c_i$ from $\left\{0,1\right\}^n$, drawing $x$ from the
  symmetric group $S_n$ and taking
  $F\left(c_i,x\right)=\left(c_i\right)_{x\left(i\right)}$, we obtain a
  random $\left\{0,1\right\}$-permanent in the case $k=n$.

## Deduction--projection estimators {#deduction%E2%80%93projection-estimators}

In order to produce mechanistic estimates for these expectations of
random products, we use [*deduction--projection
estimators*](https://gabrieldwu.com/assets/thesis.pdf). The general idea
of a deduction--projection estimator is to split up an expensive
computation into multiple steps, and then to alternate between
\"deduction\" steps, which perform one step of the computation exactly,
and \"projection\" steps, which simplify the computational state,
preventing an exponential blowup in complexity. For example, our
[cumulant propagation algorithm for wide random
MLPs](https://www.alignment.org/blog/mechanistic-estimation-for-wide-random-mlps/)
has this form: in the \"deduction\" step, we use the activation
cumulants at one layer to deduce the activation cumulants at the next
layer exactly, and in the \"projection\" step, we drop terms that are
insignificant relative to their computational
expense.^[\[2\]](#fn2){#fnref2}^

In the case of expectations of random products, given $c$, we build up
an approximate representation of the function
$x\mapsto M\left(c,x\right)$ by multiplying in each of the factors one
step at a time. That is, for each $i=0,\ldots,k$, we represent the
\"head\" of the product\
$$g^{\left(i\right)}\left(x\right):=F\left(c_1,x\right)\cdots F\left(c_i,x\right).$$\
In the deduction step, we multiply our representation of
$g^{\left(i\right)}\left(x\right)$ by $F\left(c_{i+1},x\right)$ to
obtain a representation of $g^{\left(i+1\right)}\left(x\right)$, and in
the projection step, we simplify that representation to control its
complexity. At the end of this process, we obtain a simplified
representation of $g^{\left(k\right)}\left(x\right)=M\left(c,x\right)$,
which we can use to estimate
$\mathbb E_x\left[M\left(c,x\right)\right]$.

Importantly, we can be judicious in how we simplify the representation
of $g^{\left(i\right)}\left(x\right)$. Not all information about this
function is equally useful: the most useful information is the
information that most improves our estimate of
$\mathbb E_x\left[M\left(c,x\right)\right]=\mathbb E_x\left[f ^ {\left(i\right)}\left(x\right)g ^ {\left(i\right)}\left(x\right)\right]$,
where $f^{\left(i\right)}\left(x\right)$ is the \"tail\" of the product\
$$f^{\left(i\right)}\left(x\right):=F\left(c_{i+1},x\right)\cdots F\left(c_k,x\right).$$\
Fortunately, we know a lot about what $f^{\left(i\right)}\left(x\right)$
might look like, because we know the function $F$ and the distribution
from which the $c_i$ are drawn. This brings us to problem of mechanistic
$L^2$ sketching.

## Mechanistic $L^2$ sketching

In designing the projection step of our deduction--projection estimator,
we are left with the following abstract problem: given a function
$g\left(x\right)$, can we produce a simplified \"sketch\" function
$\tilde g\left(x\right)$ such that
$\mathbb E_x\left[f\left(x\right)\tilde g\left(x\right)\right]\approx\mathbb E_x\left[f\left(x\right)g\left(x\right)\right]$,
where $f\left(x\right)$ is a random function drawn from a known
distribution? (We have dropped the $^{\left(i\right)}$ superscripts for
clarity.) We refer to this problem as *mechanistic $L^2$ sketching*,
because we want the representation of $\tilde g\left(x\right)$ to be
\"mechanistic\" (rather than consisting of the value of the function on
random points, say), and because we need the mean squared error\
$$\mathbb E_f\left[\left(\mathbb E_x\left[f\left(x\right)\tilde g\left(x\right)\right]-\mathbb E_x\left[f\left(x\right)g\left(x\right)\right]\right)^2\right]$$\
to be small relative to the $L^2$ norms
$\mathbb E_f\left[\mathbb E_x\left[f\left(x\right)^2\right]\right]$ and
$\mathbb E_x\left[g\left(x\right)^2\right]$.

Once the problem is isolated in this way, the solution ends up being a
straightforward application of linear algebra. Viewing $f\left(x\right)$
and $g\left(x\right)$ as a vectors $\mathbf f$ and $\mathbf g$ in the
space of $L^2$ functions from the domain containing $x$ to $\mathbb R$,
we can diagonalize the kernel\
$$\mathbb E_{\mathbf f}\left[\mathbf f\,\mathbf f^{\mathsf T}\right]$$\
and take $\tilde{\mathbf g}$ to be the orthogonal projection of
$\mathbf g$ onto the space spanned by the eigenfunctions of this kernel
with the largest eigenvalues. The maximum mean squared error is then
determined by the largest unused eigenvalue, which we can control using
the $L^2$ constraint on $\mathbf f$.

## Detailed notes

The basic approach we have described, of using a deduction--projection
estimator with mechanistic $L^2$ sketching for the projection step, is
shared between all of the estimation problems we study here. However, in
each particular case, some work is required to diagonalize the kernel,
and to check that the orthogonal projection onto its leading
eigenfunctions can be performed quickly enough. Generally speaking, we
have been able to do this in cases where the function $F$ and the
distributions from which $c_i$ and $x$ are drawn have sufficient
symmetry, rather than being highly complex.

We hope to describe more of the details of this approach in a future
paper, but we may not get around to this for some time. We are therefore
sharing some notes that we have been using internally, so that these
details are available to those who are interested:

- **[Deductive estimation for random halfspace
  intersections](https://www.alignment.org/content/files/2026/05/deductive_estimation_for_random_halfspace_intersections.pdf)
  (written up by Jacob Hilton)**: this is focused on the case of random
  halfspace intersections, but also includes introductory content on
  deduction--projection estimators and mechanistic $L^2$ sketching. Our
  method is the same as the one that we previously summarized
  [here](https://www.alignment.org/blog/competing-with-sampling/#intersection-of-random-half-spaces).
- **[Products of symmetric random
  functions](https://www.alignment.org/content/files/2026/05/rand_sym_dedproj.pdf)
  (written up by Wilson Wu)**: this provides a general framework based
  on the theory of Gelfand pairs into which our algorithm for random
  halfspace intersections fits. It then derives a mechanistic estimation
  algorithm for #3-SAT as a special case of this framework.
- **[Deduction--projection for the
  permanent](https://www.alignment.org/content/files/2026/05/perm_dedproj.pdf)
  (written up by Wilson Wu)**: this discusses the case of random
  $\left\{0,1\right\}$-permanents, which require a kind of symmetry that
  does not quite fit into the theory of Gelfand pairs. The focus is on
  $\left\{0,1\right\}$-permanents rather than
  $\left\{-1,1\right\}$-permanents because it turns out to be trivial to
  match sampling mechanistically for $\left\{-1,1\right\}$-permanents,
  since all of the terms are pairwise uncorrelated.
- **[Worst-case-up-to-signs mechanistic estimations for products of
  functions](https://www.alignment.org/content/files/2026/05/products_fourier_simple_v5.pdf)
  (written up by Mike Winer)**: this provides a general approach based
  on random walk simulations for Boolean functions with a small number
  of non-zero Fourier coefficients. We obtain an alternative algorithm
  for #3-SAT as a special case. More generally, the approach can be
  applied to both products of linear functions and $q$-local functions
  on the Boolean cube, and even allows the non-zero Fourier coefficients
  to have arbitrary magnitudes, as long as their signs are chosen
  randomly.

In each case, we roughly match or outperform random sampling up to
logarithmic factors, except for the third case, where we are further
behind.

## Conclusion

We have been studying random instances of the matching sampling
principle in order to discover general methods for mechanistic
estimation that we can apply more broadly. Mechanistic $L^2$ sketching
is one such method that has fallen out of our study of expectations of
random products.

*Cross-postings for comments:
[LessWrong](https://www.lesswrong.com/posts/7RyAefESvb6BQ3tMz/mechanistic-estimation-for-expectations-of-random-products),
[Alignment
Forum](https://www.alignmentforum.org/posts/7RyAefESvb6BQ3tMz/mechanistic-estimation-for-expectations-of-random-products#Detailed_notes)*

------------------------------------------------------------------------

1.  
    A 3-CNF clause is a Boolean formula of the form
    $l_1\lor l_2\lor l_3$ where each of $l_1$, $l_2$ and $l_3$ is either
    $x_i$ or $\lnot x_i$ for some $i$, and a 3-CNF formula is an AND of
    3-CNF clauses. [↩︎](#fnref1){.footnote-backref}

2.  
    In practice, the deduction and projection steps are not so cleanly
    separated, and instead we perform an approximate deduction step that
    combines both. [↩︎](#fnref2){.footnote-backref}


\
