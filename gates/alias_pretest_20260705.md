# Alias-Correlation Pre-Test for High-Order-Even Cubature

Run date: 2026-07-05. Offline only; local self-generated MLPs and MC truth.
MLP seeds: [11, 22]; R per MLP: 80; truth samples: 200,000.

## Pooled CV R^2

| Feature family | n features | per-coordinate variance-weighted CV R^2 | scalar mean-error CV R^2 | mean feature std |
|---|---:|---:|---:|---:|
| q4_unweighted | 12 | -0.0757 | -0.0239 | 1.002 |
| q4_weighted | 4 | -0.0550 | -0.0665 | 1.015 |
| q6_unweighted | 4 | -0.0558 | -0.0807 | 1.029 |
| q2_control | 4 | -0.0372 | -0.0335 | 0.959 |
| all_q4 | 16 | -0.1262 | -0.0959 | 1.005 |
| all_alias | 20 | -0.1723 | -0.1532 | 1.010 |
| all_with_controls | 24 | -0.1988 | -0.1756 | 1.001 |

## Per MLP

### MLP seed 11

| Feature family | per-coordinate CV R^2 | scalar mean-error CV R^2 |
|---|---:|---:|
| q4_unweighted | -0.2633 | -0.3407 |
| q4_weighted | -0.1312 | -0.1695 |
| q6_unweighted | -0.1397 | -0.2267 |
| q2_control | -0.0804 | -0.1107 |
| all_q4 | -0.3951 | -0.4490 |
| all_alias | -0.6213 | -0.7222 |
| all_with_controls | -0.7295 | -0.9255 |

### MLP seed 22

| Feature family | per-coordinate CV R^2 | scalar mean-error CV R^2 |
|---|---:|---:|
| q4_unweighted | -0.1245 | 0.0019 |
| q4_weighted | -0.0921 | -0.1367 |
| q6_unweighted | -0.1133 | -0.1579 |
| q2_control | -0.0980 | -0.1350 |
| all_q4 | -0.2288 | -0.1184 |
| all_alias | -0.3840 | -0.2957 |
| all_with_controls | -0.4992 | -0.5257 |

## Power and Coverage

Positive control target R^2: 0.50; recovered pooled CV R^2: 0.4867.
Quadruple sketches used 20,000 tuples each across 16 requested degree-4 sketches, covering about 46.32% of the distinct XOR-closed quadruple count before duplicate overlap.
Degree-6 sketches used 8,000 sextuples each; this is sparse relative to the rough XOR-closed sextuple count.

## Gate Verdict

Pre-registered pooled all-alias per-coordinate CV R^2: -0.1723. Candidate 3 is **DEAD** under the 0.35 gate.
The tested alias sketches do not resolve the fingerprint; alias-targeted sign designs should not be implemented from this evidence, and external information remains the remaining lever.
