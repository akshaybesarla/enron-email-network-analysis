"""Network structure analysis on top of the pipeline.

Three questions, each from the original brief:

1. Concentration. Do the top 20% of nodes account for ~80% of edge weight?
2. Growth. How does the maximum degree grow as the network accumulates nodes?
3. Degree distribution. Is it plausibly a power law, and with what exponent?

On question 3 this module reports two estimators side by side, which is the main
analytical change from the original submission. See ``fit_powerlaw_ols`` and
``fit_powerlaw_mle``.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone

import numpy as np

from .pipeline import (
    convert_to_weighted_network,
    get_in_degree_dist,
    get_in_degrees,
    get_out_degree_dist,
    get_out_degrees,
)


# --------------------------------------------------------------------------
# 1. Concentration
# --------------------------------------------------------------------------

@dataclass
class Concentration:
    direction: str
    total_nodes: int
    total_degree: int
    top_20pct_nodes: int
    top_20pct_degree: int
    share: float

    @property
    def follows_80_20(self) -> bool:
        return self.share >= 0.80


def concentration(degree_rdd, direction: str) -> Concentration:
    """Share of total degree held by the highest-degree 20% of nodes.

    ``degree_rdd`` is (degree, node) as produced by get_out_degrees /
    get_in_degrees, already sorted descending.
    """
    degrees = [d for d, _ in degree_rdd.collect()]
    total_nodes = len(degrees)
    total_degree = int(sum(degrees))

    cutoff = max(1, int(round(total_nodes * 0.20)))
    top_degree = int(sum(degrees[:cutoff]))

    return Concentration(
        direction=direction,
        total_nodes=total_nodes,
        total_degree=total_degree,
        top_20pct_nodes=cutoff,
        top_20pct_degree=top_degree,
        share=(top_degree / total_degree) if total_degree else 0.0,
    )


# --------------------------------------------------------------------------
# 2. Growth of the maximum degree
# --------------------------------------------------------------------------

@dataclass
class GrowthPoint:
    months: int
    nodes: int
    k_max_in: int
    k_max_out: int


def growth_curve(edge_rdd, start: datetime, months: int = 12) -> list[GrowthPoint]:
    """Cumulative slices: for the first n months, record node count and max degree.

    Each slice includes everything from ``start`` up to the end of month n, so
    the network only grows. ``edge_rdd`` should be cached by the caller; it is
    scanned once per slice.
    """
    points: list[GrowthPoint] = []

    for n in range(1, months + 1):
        end_year = start.year + (start.month - 1 + n) // 12
        end_month = (start.month - 1 + n) % 12 + 1
        end = datetime(end_year, end_month, 1, tzinfo=timezone.utc)

        weighted = convert_to_weighted_network(edge_rdd, (start, end)).cache()
        out_degrees = get_out_degrees(weighted).collect()
        in_degrees = get_in_degrees(weighted).collect()
        weighted.unpersist()

        if not out_degrees:
            continue

        points.append(
            GrowthPoint(
                months=n,
                nodes=len(out_degrees),
                k_max_in=max(d for d, _ in in_degrees),
                k_max_out=max(d for d, _ in out_degrees),
            )
        )

    return points


def growth_exponent(points: list[GrowthPoint], attribute: str) -> float:
    """Slope of log(k_max) against log(n).

    Interpretation: ~1 means the maximum degree grows in step with the network
    (linear), >1 means hubs accumulate connections faster than the network adds
    nodes (superlinear, the "rich get richer" pattern).
    """
    nodes = np.array([p.nodes for p in points], dtype=float)
    k_max = np.array([getattr(p, attribute) for p in points], dtype=float)
    mask = (nodes > 0) & (k_max > 0)
    if mask.sum() < 2:
        return float("nan")
    slope, _ = np.polyfit(np.log(nodes[mask]), np.log(k_max[mask]), 1)
    return float(slope)


# --------------------------------------------------------------------------
# 3. Degree distribution and power-law fitting
# --------------------------------------------------------------------------

@dataclass
class PowerLawFit:
    method: str
    alpha: float
    x_min: int
    n_tail: int
    r_squared: float | None = None
    ks_distance: float | None = None

    @property
    def valid_power_law(self) -> bool:
        """A power law p(k) ~ k^-alpha is only normalisable for alpha > 1."""
        return self.alpha > 1.0


def fit_powerlaw_ols(dist: list[tuple[int, int]], x_min: int = 1) -> PowerLawFit:
    """Least-squares fit of a straight line to the log-log degree histogram.

    This is the estimator used in the original submission and it is reported
    here for comparison, not because it is the right one. Fitting OLS to
    log-binned counts gives every point equal weight, so the sparse, noisy tail
    pulls the slope as hard as the dense head. It is known to be biased, and it
    carries no constraint keeping alpha above 1, so it can return a value that
    does not describe a normalisable distribution at all.

    Reference: Clauset, Shalizi & Newman (2009), "Power-law distributions in
    empirical data", SIAM Review 51(4).
    """
    pairs = [(k, c) for k, c in dist if k >= max(x_min, 1) and c > 0]
    if len(pairs) < 2:
        return PowerLawFit(method="ols-loglog", alpha=float("nan"), x_min=x_min, n_tail=0)

    log_k = np.log10([k for k, _ in pairs])
    log_c = np.log10([c for _, c in pairs])

    slope, intercept = np.polyfit(log_k, log_c, 1)
    predicted = slope * log_k + intercept
    ss_res = float(np.sum((log_c - predicted) ** 2))
    ss_tot = float(np.sum((log_c - np.mean(log_c)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot else float("nan")

    return PowerLawFit(
        method="ols-loglog",
        alpha=float(-slope),
        x_min=max(x_min, 1),
        n_tail=int(sum(c for _, c in pairs)),
        r_squared=float(r_squared),
    )


def _mle_alpha(observations: np.ndarray, x_min: int) -> float:
    """Discrete MLE for alpha with the continuous approximation and -0.5 correction."""
    tail = observations[observations >= x_min]
    if tail.size < 2:
        return float("nan")
    return 1.0 + tail.size / float(np.sum(np.log(tail / (x_min - 0.5))))


def _ks_distance(observations: np.ndarray, alpha: float, x_min: int) -> float:
    """Kolmogorov-Smirnov distance between the empirical and fitted CDFs."""
    tail = np.sort(observations[observations >= x_min])
    if tail.size < 2 or not np.isfinite(alpha):
        return float("inf")

    empirical = np.arange(1, tail.size + 1) / tail.size
    # Continuous power-law CDF, adequate for the discrete case at these scales.
    fitted = 1.0 - (tail / x_min) ** (1.0 - alpha)
    return float(np.max(np.abs(empirical - fitted)))


def fit_powerlaw_mle(dist: list[tuple[int, int]]) -> PowerLawFit:
    """Maximum-likelihood fit with the lower cutoff chosen by KS minimisation.

    This is the method Clauset, Shalizi & Newman recommend: for each candidate
    x_min, estimate alpha by maximum likelihood over the tail above it, and keep
    the x_min that minimises the KS distance between the empirical and fitted
    distributions.

    ``dist`` is the (degree, node_count) histogram; it is expanded back into the
    per-node observations the estimator needs.
    """
    observations = np.repeat(
        [k for k, _ in dist], [c for _, c in dist]
    ).astype(float)
    observations = observations[observations > 0]
    if observations.size < 10:
        return PowerLawFit(method="mle-ks", alpha=float("nan"), x_min=1, n_tail=0)

    candidates = np.unique(observations)
    candidates = candidates[candidates >= 1]
    # Keep enough tail for the estimate to mean anything.
    candidates = candidates[: max(1, len(candidates) - 5)]

    best = PowerLawFit(method="mle-ks", alpha=float("nan"), x_min=1, n_tail=0,
                       ks_distance=float("inf"))

    for x_min in candidates:
        x_min_int = int(x_min)
        if x_min_int < 1:
            continue
        alpha = _mle_alpha(observations, x_min_int)
        if not np.isfinite(alpha):
            continue
        distance = _ks_distance(observations, alpha, x_min_int)
        if distance < (best.ks_distance if best.ks_distance is not None else float("inf")):
            best = PowerLawFit(
                method="mle-ks",
                alpha=float(alpha),
                x_min=x_min_int,
                n_tail=int((observations >= x_min_int).sum()),
                ks_distance=distance,
            )

    return best


def analyse_slice(edge_rdd, start: datetime, end: datetime) -> dict:
    """Run all three analyses over one time slice and return plain dictionaries."""
    weighted = convert_to_weighted_network(edge_rdd, (start, end)).cache()

    out_degrees = get_out_degrees(weighted).cache()
    in_degrees = get_in_degrees(weighted).cache()

    out_dist = get_out_degree_dist(weighted).collect()
    in_dist = get_in_degree_dist(weighted).collect()

    result = {
        "slice": {"start": start.isoformat(), "end": end.isoformat()},
        "concentration": {
            "out": asdict(concentration(out_degrees, "out")),
            "in": asdict(concentration(in_degrees, "in")),
        },
        "degree_distribution": {
            "out": [list(p) for p in out_dist],
            "in": [list(p) for p in in_dist],
        },
        "power_law": {
            "out": {
                "ols": asdict(fit_powerlaw_ols(out_dist)),
                "mle": asdict(fit_powerlaw_mle(out_dist)),
            },
            "in": {
                "ols": asdict(fit_powerlaw_ols(in_dist)),
                "mle": asdict(fit_powerlaw_mle(in_dist)),
            },
        },
    }

    out_degrees.unpersist()
    in_degrees.unpersist()
    weighted.unpersist()
    return result
