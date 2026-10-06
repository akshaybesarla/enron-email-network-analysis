#!/usr/bin/env python3
"""End-to-end entry point: build the network, analyse it, write results and charts.

    python run_analysis.py --input data/raw --start 2000-01-01 --months 12

Outputs land in ``--outdir`` (default ``output/``):
    results.json          every number the README quotes
    degree_distribution.png
    growth.png
    concentration.png
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from pyspark import SparkConf, SparkContext  # noqa: E402

from src import sources  # noqa: E402
from src.analysis import analyse_slice, growth_curve, growth_exponent  # noqa: E402
from src.pipeline import extract_email_network, get_monthly_contacts  # noqa: E402

PALETTE = {
    "out": "#2563eb",
    "in": "#ea580c",
    "fit": "#64748b",
    "grid": "#e2e8f0",
}


def build_context(app_name: str = "EnronNetwork") -> SparkContext:
    conf = SparkConf().setAppName(app_name)
    if not conf.contains("spark.master"):
        conf.setMaster("local[*]")
    sc = SparkContext.getOrCreate(conf=conf)
    sc.setLogLevel("WARN")
    return sc


def plot_degree_distribution(results: dict, path: Path) -> None:
    """Complementary CDF on log-log axes, with the fitted tail overlaid.

    The CCDF is plotted rather than the raw histogram. A histogram of a
    heavy-tailed variable has a long stretch of degree values observed exactly
    once, which turns the tail into a flat line of 1s and makes any fit look
    wrong. The CCDF needs no binning and is the form Clauset, Shalizi & Newman
    recommend for judging a power-law fit by eye.
    """
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for direction in ("out", "in"):
        dist = [(k, c) for k, c in results["degree_distribution"][direction] if k > 0]
        if not dist:
            continue

        observations = np.repeat([k for k, _ in dist], [c for _, c in dist]).astype(float)
        observations.sort()
        total = observations.size
        # P(X >= k) evaluated at each observed value.
        ccdf = 1.0 - np.arange(total) / total

        ax.scatter(observations, ccdf, s=10, alpha=0.5, color=PALETTE[direction],
                   label=f"{direction}-degree", edgecolors="none")

        mle = results["power_law"][direction]["mle"]
        alpha, x_min = mle["alpha"], mle["x_min"]
        if np.isfinite(alpha) and alpha > 1 and x_min >= 1:
            tail = observations[observations >= x_min]
            if tail.size:
                # For p(k) ~ k^-alpha, P(X >= k) ~ k^-(alpha-1), scaled to meet
                # the empirical CCDF at x_min.
                anchor = tail.size / total
                ax.plot(tail, anchor * (tail / x_min) ** (1.0 - alpha),
                        color=PALETTE[direction], linestyle="--", linewidth=1.6,
                        label=f"{direction} MLE fit: a={alpha:.2f}, k>={x_min}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Weighted degree k")
    ax.set_ylabel("P(degree >= k)")
    ax.set_title("Degree distributions (CCDF, log-log) with maximum-likelihood fits")
    ax.grid(True, which="both", color=PALETTE["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_growth(points, path: Path) -> None:
    if not points:
        return
    nodes = [p.nodes for p in points]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(nodes, [p.k_max_out for p in points], marker="s", color=PALETTE["out"],
            linewidth=1.8, markersize=5, label="max out-degree")
    ax.plot(nodes, [p.k_max_in for p in points], marker="o", color=PALETTE["in"],
            linewidth=1.8, markersize=5, label="max in-degree")

    ax.set_xlabel("Number of nodes in the cumulative slice")
    ax.set_ylabel("Maximum weighted degree")
    ax.set_title("Maximum degree against network size")
    ax.grid(True, color=PALETTE["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_concentration(results: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 3.4))

    directions = ("out", "in")
    labels = [f"{d}-degree" for d in directions]
    top = [results["concentration"][d]["share"] * 100 for d in directions]
    rest = [100 - value for value in top]

    y = np.arange(len(labels))
    ax.barh(y, top, color=[PALETTE[d] for d in directions], height=0.55)
    ax.barh(y, rest, left=top, color=PALETTE["grid"], height=0.55)

    ax.axvline(80, color="#0f172a", linestyle=":", linewidth=1.2, zorder=3)
    ax.annotate("80% reference", xy=(80, len(labels) - 0.45), fontsize=8,
                color="#0f172a", ha="center", va="bottom")

    for i, value in enumerate(top):
        inside = value > 18
        ax.text(value - 1.5 if inside else value + 1.5, i, f"{value:.1f}%",
                va="center", ha="right" if inside else "left",
                color="white" if inside else "#0f172a",
                fontsize=11, fontweight="bold")

    ax.set_yticks(y, labels)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, len(labels) - 0.1)
    ax.set_xlabel("Share of total edge weight held by the top 20% of nodes (%)")
    ax.set_title("Degree concentration", loc="left", fontsize=12)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/raw",
                        help="directory of text messages, or an HDFS sequence-file path")
    parser.add_argument("--format", choices=["auto", "text", "sequence"], default="auto")
    parser.add_argument("--start", default="2000-01-01", help="slice start, YYYY-MM-DD")
    parser.add_argument("--months", type=int, default=12, help="slice length in months")
    parser.add_argument("--outdir", type=Path, default=Path("output"))
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_year = start.year + (start.month - 1 + args.months) // 12
    end_month = (start.month - 1 + args.months) % 12 + 1
    end = datetime(end_year, end_month, 1, tzinfo=timezone.utc)

    args.outdir.mkdir(parents=True, exist_ok=True)
    sc = build_context()

    try:
        raw = sources.load(sc, args.input, args.format)
        edges = extract_email_network(raw).cache()

        edge_count = edges.count()
        print(f"Extracted {edge_count:,} distinct (sender, recipient, timestamp) edges")
        if edge_count == 0:
            print("No edges extracted. Check the input path and message format.")
            return

        results = analyse_slice(edges, start, end)
        results["edges_total"] = edge_count

        points = growth_curve(edges, start, months=args.months)
        results["growth"] = {
            "points": [asdict(p) for p in points],
            "exponent_out": growth_exponent(points, "k_max_out"),
            "exponent_in": growth_exponent(points, "k_max_in"),
        }

        top_contacts = get_monthly_contacts(edges).take(10)
        results["top_monthly_contacts"] = [list(t) for t in top_contacts]

        (args.outdir / "results.json").write_text(json.dumps(results, indent=2))
        plot_degree_distribution(results, args.outdir / "degree_distribution.png")
        plot_growth(points, args.outdir / "growth.png")
        plot_concentration(results, args.outdir / "concentration.png")

        out_c = results["concentration"]["out"]
        in_c = results["concentration"]["in"]
        out_mle = results["power_law"]["out"]["mle"]
        in_mle = results["power_law"]["in"]["mle"]

        print(f"\nSlice {args.start} +{args.months}m")
        print(f"  nodes                   {out_c['total_nodes']:,}")
        print(f"  top 20% share (out)     {out_c['share']:.1%}")
        print(f"  top 20% share (in)      {in_c['share']:.1%}")
        print(f"  MLE alpha (out)         {out_mle['alpha']:.2f}  (k >= {out_mle['x_min']})")
        print(f"  MLE alpha (in)          {in_mle['alpha']:.2f}  (k >= {in_mle['x_min']})")
        print(f"  k_max growth exponent   out {results['growth']['exponent_out']:.2f}, "
              f"in {results['growth']['exponent_in']:.2f}")
        print(f"\nWrote results and charts to {args.outdir}/")
    finally:
        sc.stop()


if __name__ == "__main__":
    main()
