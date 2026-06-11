"""ESTIMATOR SOURCE OVERWRITTEN BEFORE PUBLICATION.

This version of `estimator.py` fitted final-ReLU mean coefficients, expanded
shortcut features, and a seed-keyed residual overlay against cached public
evaluation labels. That is not a legitimate estimator improvement under the
challenge rules.

It was caught and reverted within the hour, on 2026-06-11, during the warmup
round -- see commits `11f5f95` ("Remove invalid public mini residual overlay")
and `1c0ceb9` ("Remove public label final ReLU fits"). The standing prohibition
is recorded in the history under "Public-label calibration".

The incident is disclosed deliberately; the working code is not. The original
source was replaced with this notice on 2026-08-17 so that the history records
what happened without shipping a runnable public-label fitter. The unmodified
history is retained privately.

Original blob: fad1bda4d980118906440d355ec2611ef9ac8bf0
"""

raise RuntimeError(
    "This historical estimator version was removed before publication. "
    "See the module docstring."
)
