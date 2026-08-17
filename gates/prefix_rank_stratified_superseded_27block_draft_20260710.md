# Superseded 27-block draft (never launched)

This note preserves the prelaunch draft that was stopped before any Fly
submission. It proposed a 27-block pool, K including 16, and terminal counts
that treated all terminal blocks as additional main blocks. The coordinator
identified the hard full-pool variance ceiling (about `1.7e-6`) and required
the corrected 32-block scalar-GREG v2 instead. The old draft produced no
results and must not be launched.

The live v2 launch used the corrected constants: 32 pool blocks, 8192 pairs,
K `{2,4,8,12}`, terminal budgets `{13,11,8,3}`, and selected-main counts
`{2816,2304,1536,256}`.
