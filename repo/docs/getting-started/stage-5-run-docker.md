# Stage 5: Docker Runner

> [← Tutorial](README.md)

> Ladder: [1](stage-1-standalone.md) · [2](stage-2-validate.md) · [3](stage-3-run-local.md) · [4](stage-4-run-subprocess.md) · **5** · [6](stage-6-package.md)

> **Coming soon.** The Docker runner gives you the exact image the grader uses — same Python version, same OS packages, same wheel cache. Until it ships in `whestbench`, treat Stage 4 as your closest grader-parity check.

When it's ready, the command will be:

```bash
uv run whest run --estimator estimator.py --runner docker
```

Track progress in the [whestbench changelog](https://github.com/AIcrowd/whestbench/blob/main/CHANGELOG.md).

## In the meantime

Move on to [Stage 6: package your submission](stage-6-package.md).
