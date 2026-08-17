# Packed gate-clustered two-sided pruning gate (pre-registration)

Status: **PRE-REGISTERED BEFORE FLY**. This document fixes the measurement,
cost model, and pass/fail thresholds before any result is observed.

## Question

Can label-free row clustering turn exact input-union sparsity plus rigorously
certified output-dead coordinates into a small number of packed batched kernels
that make 27--28 current-route Hadamard blocks fit at the score floor?

The existing exact union-prune mode saves about 10.5% raw FLOPs, while
contiguous 512-row groups saved only another ~3% raw and paid a large residual
penalty. This gate therefore does not reopen arbitrary small fragmented
matmuls. It measures whether sorting rows by their own activation geometry and
packing equal padded shapes changes the economics enough to justify one exact
mode-gated implementation.

## Legitimacy and data boundary

- Run only through `make fly-payload` on all 100 fresh truth-bank MLP seeds.
- Each Machine rebuilds its MLP from the bank seed and verifies the stored
  weight checksum.
- Candidate logic uses only the rebuilt MLP and the real current estimator's
  own 16-block activations. Truth-bank activation means are not read.
- No local estimator run or scoring, no grader/public/private labels or seeds,
  no reference outputs, no network input to candidate logic, and no flopscope
  or budget bypass.
- This is research measurement only. `estimator.py` and
  `ESTIMATOR_HISTORY.md` remain untouched in this lane.

## Current-route reproduction

Machine-side generation follows the live 16-block depth-32 route:

1. randomized antithetic Walsh-Hadamard halves from the MLP seed;
2. fp32 L3 Strassen first-layer apply;
3. exact zero-mean Gaussian ReLU mean/covariance target and current Cholesky
   recolor, with the same fp32 propagation/fp64 moment split;
4. the live `1.5x` first-successor post-ReLU variance transform;
5. fp32 L3 Strassen propagation through all remaining layers.

Full dense preactivations are retained only as research ground truth for
whether a candidate output is truly dead and for certificate-violation checks.
They do not enter the certificate.

## Grouping strategies

At each propagation input after the first-successor transform, independently
measure fixed group sizes `32`, `64`, and `128` under:

- `contiguous`: unchanged row order (negative control);
- `activation_norm`: stable sort by row squared norm;
- `pc1`: stable sort by a deterministic, label-free global PC1-like score;
- `gatekey32`: stable sort by the 32 most variable activation-support bits;
- `support_lex`: lexicographic sort by the complete 256-bit activation support,
  ordered from most to least variable coordinate.

Sorting may permute rows but never changes their values or weights. Candidate
gate keys are derived only from the current activation matrix.

## Exact quantities

For every group and propagation layer:

- `live_input`: fraction of input coordinates nonzero in at least one group
  row;
- `true_output_dead`: fraction of output coordinates whose full dense fp32 L3
  preactivation is `<= 0` for every group row;
- certificate recall: certified-dead / true-dead;
- certificate violations: certified dead but full preactivation positive;
- `live_input * uncertified_output`, where uncertified output includes every
  output not rigorously certified dead.

The decision layer set is weight layers `3..31` inclusive. Layer 2 is still
included in projected total FLOPs because a real candidate must pay it.

## Rigorous certificates

For group coordinate bounds `l_i <= X_ri <= u_i`, the box upper bound is

`U_j = sum_i [u_i (w_ij)_+ + l_i (w_ij)_-]`.

Certification requires `U_j + eps_j <= 0`, with

`eps_j >= 64 * gamma_k * sum_i max(|l_i|, |u_i|) |w_ij|`,

`gamma_k = k*u/(1-k*u)` and fp32 unit roundoff `u=2^-24`. The factor 64 is a
deliberately conservative enlargement for the live L3 Strassen arithmetic;
the projected packed kernel is plain/batched fp32 and has a smaller dot-product
rounding envelope.

Also measure a deterministic rank-2 residual certificate. With group mean
`mu`, an orthonormal label-free global activation basis `Q`,
`Y=(X-mu)Q`, `E=X-mu-YQ^T`, and exact row residual radii `R_r=||E_r||_2`,

`U_j = mu^T w_j + max_r [Y_r^T(Q^T w_j) + R_r ||(I-QQ^T)w_j||_2]`.

The same conservative fp32 envelope is added. Box-only and union(box, rank-2)
results are reported separately. **Any violation kills the gate**, even if the
aggregate economics pass.

## Packed-kernel projection

For each group, gather its exact live inputs and uncertified outputs, round
both dimensions up to `16`, `32`, or `64`, and bucket groups by the padded
`(input_width, output_width)` pair. A bucket is one batched rectangular
matmul/einsum; certified outputs are reconstructed as exact zeros after ReLU.

Report for box-only and union(box, rank-2):

- packed plain-matmul FLOPs including padding;
- certificate screening FLOPs, sorting-score arithmetic, and conservative
  reconstruction/reduction arithmetic;
- mean/max buckets per layer, groups per bucket, packed bytes, and gather/
  scatter element volume as wall/memory proxies;
- projected 28-block raw FLOPs.

Projection anchor: the clean current-route raw count is `25,353,276,460` at
16 blocks. The exact L3 arithmetic cost of each full `8192 x 256 @ 256 x 256`
matmul is `748,176,384` FLOPs. The projection keeps the non-replaced route
cost conservatively linear in rows, replaces layers `2..31` with the measured
packed cost, and scales row-dependent work by `28/16`. The rank-2 projection
must include basis, projection, residual-bound, and screening arithmetic; it
is not credited as free.

## Frozen decision gate

An implementation may be designed, but not written in this lane, only if one
strategy/group/padding/certificate plan satisfies all of:

1. over layers `3..31`, mean
   `live_input * uncertified_output <= 0.31`; `<=0.29` is the strong pass;
2. per-MLP q90 projected 28-block raw FLOPs `<= 2.4e10`;
3. zero certificate violations across every group, layer, and all 100 MLPs;
4. a credible packing plan with few large kernels -- preregistered as mean
   `<= 8` and max `<= 16` nonempty buckets per layer at padding 32 or 64,
   median groups per nonempty bucket `>= 4`, peak packed working memory
   `<= 512 MiB`, and no more than `2.5x` the dense activation element volume
   in gather/scatter traffic -- plausibly keeping residual compute `<=3e9`.

Failure of (1), (2), or (3) closes two-sided pruning. A borderline arithmetic
pass that fails (4) is reported as an implementation-economics failure, not a
score candidate. If multiple plans pass, select the lowest q90 projected raw
FLOPs subject to the few-kernel constraint; ties prefer the larger group and
coarser padding.

## Required report

The aggregate must include all 100 MLPs or explicitly fail, per-MLP spread,
unsorted controls, layer curves, exact certificate violation counts, the
selected or rejected packing plan, and an exact estimator-mode design only if
the frozen gate passes.
