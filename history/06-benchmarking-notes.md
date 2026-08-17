# Benchmarking Notes

Use current scorer-path comparisons, not stale flops-only proxies. For
deterministic Hadamard knob comparisons on the fixed Fly dataset, prefer
full per-MLP JSON and matched pairs over repeated summary-only run means:
override `FLY_RUN_FLAGS` on the `make` command line to omit `--summary-only`,
request all 100 results, and compare per-MLP final-layer MSE deltas against
the same baseline rows. Summary-only Fly means are still useful for quick
smoke tests, but they discard the pairing and can recreate the old returned-set
`bounce` ambiguity. For estimator changes, follow [`AGENTS.md`](../repo/AGENTS.md):
compile `estimator.py` and use the Fly fast runner by default unless the owner
asks for a different proof. For docs-only changes, a link/search check and
Markdown sanity are sufficient.
