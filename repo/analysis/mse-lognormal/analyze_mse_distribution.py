#!/usr/bin/env python3
"""Fit simple positive distributions to the 50 per-MLP MSE values."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import scipy.stats as st


DATA = Path(__file__).with_name("mlp_mse_values.csv")


def aic(log_likelihood: float, n_params: int) -> float:
    return 2 * n_params - 2 * log_likelihood


def main() -> None:
    with DATA.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    x = np.array([float(row["final_layer_mse"]) for row in rows])
    log_x = np.log(x)

    mu = float(log_x.mean())
    sigma = float(log_x.std(ddof=0))
    mean = float(x.mean())
    sd = float(x.std(ddof=0))

    lognorm_ll = float(st.lognorm.logpdf(x, s=sigma, loc=0, scale=math.exp(mu)).sum())
    normal_ll = float(st.norm.logpdf(x, loc=mean, scale=sd).sum())
    gamma_shape, _, gamma_scale = st.gamma.fit(x, floc=0)
    gamma_ll = float(st.gamma.logpdf(x, gamma_shape, loc=0, scale=gamma_scale).sum())
    weibull_shape, _, weibull_scale = st.weibull_min.fit(x, floc=0)
    weibull_ll = float(st.weibull_min.logpdf(x, weibull_shape, loc=0, scale=weibull_scale).sum())

    lognorm = st.lognorm(s=sigma, loc=0, scale=math.exp(mu))
    cv2 = math.exp(sigma * sigma) - 1.0
    mean50_sigma = math.sqrt(math.log1p(cv2 / len(x)))
    mean50_mu = math.log(float(x.mean())) - 0.5 * mean50_sigma * mean50_sigma
    mean50 = st.lognorm(s=mean50_sigma, loc=0, scale=math.exp(mean50_mu))

    print(f"n: {len(x)}")
    print(f"arithmetic mean: {x.mean():.6g}")
    print(f"median: {np.median(x):.6g}")
    print(f"sample sd: {x.std(ddof=1):.6g}")
    print(f"cv: {x.std(ddof=1) / x.mean():.6g}")
    print(f"min/max: {x.min():.6g} / {x.max():.6g} ({x.max() / x.min():.3g}x)")
    print()
    print("fitted log-normal:")
    print(f"  mu: {mu:.9g}")
    print(f"  sigma: {sigma:.9g}")
    print(f"  geometric mean: {math.exp(mu):.6g}")
    print(f"  individual 95% spread: {lognorm.ppf(0.025):.6g} to {lognorm.ppf(0.975):.6g}")
    print(f"  50-MLP mean 95% spread: {mean50.ppf(0.025):.6g} to {mean50.ppf(0.975):.6g}")
    print()
    print("AIC:")
    scores = {
        "log-normal": aic(lognorm_ll, 2),
        "gamma": aic(gamma_ll, 2),
        "weibull": aic(weibull_ll, 2),
        "raw-normal": aic(normal_ll, 2),
    }
    best = min(scores.values())
    for name, score in sorted(scores.items(), key=lambda item: item[1]):
        print(f"  {name}: {score:.3f} (delta {score - best:.3f})")
    print()
    print("goodness checks:")
    print(f"  KS fitted log-normal p: {st.kstest(x, 'lognorm', args=(sigma, 0, math.exp(mu))).pvalue:.6g}")
    print(f"  KS raw-normal p: {st.kstest(x, 'norm', args=(mean, sd)).pvalue:.6g}")
    print(f"  Shapiro log(MSE) p: {st.shapiro(log_x).pvalue:.6g}")
    ad = st.anderson(log_x, "norm")
    print(f"  Anderson log(MSE) statistic: {ad.statistic:.6g}")


if __name__ == "__main__":
    main()
