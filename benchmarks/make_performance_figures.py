"""Draw the performance figures from a committed benchmark run.

Reads ``benchmarks/results/latest.json`` -- produced by ``run_benchmarks.py`` -- and
writes SVGs into ``docs/figures/``. The output is committed, so building the
documentation needs neither ``matplotlib`` nor a GPU, in the same spirit as
``docs/make_accuracy_figures.py``. Re-run it only when the numbers change::

    python benchmarks/make_performance_figures.py

Rows the harness flagged as noisy are drawn hollow rather than dropped. On a
power-capped laptop GPU a short kernel can measure almost anything, and hiding that
behind a solid bar would be the wrong kind of tidy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

#: One colour per *series*, not per backend: the two GPU precisions are the comparison
#: the figure exists to make, so they must not share a colour.
SERIES = [
    ("scipy", "float64", "#7f7f7f", "scipy (fallback)"),
    ("numba", "float64", "#1f77b4", "numba (default)"),
    ("jax-cpu", "float32", "#9467bd", "cosmotile.jax, CPU f32"),
    ("jax-gpu", "float64", "#ff7f0e", "cosmotile.jax, GPU f64"),
    ("jax-gpu", "float32", "#d62728", "cosmotile.jax, GPU f32"),
]
GPU_COLOUR = "#d62728"


def _select(rows: list[dict[str, Any]], **where: Any) -> list[dict[str, Any]]:
    return [
        r
        for r in rows
        if "error" not in r and all(r.get(key) == value for key, value in where.items())
    ]


def throughput_by_order(rows: list[dict[str, Any]], box_size: int, nside: int, out: Path) -> None:
    """Throughput against interpolation order, one group of bars per order."""
    orders = sorted({r["order"] for r in rows})
    fig, ax = plt.subplots(figsize=(7.4, 4.4))

    width = 0.16
    for index, (backend, dtype, colour, label) in enumerate(SERIES):
        heights, hollow, positions = [], [], []
        for slot, order in enumerate(orders):
            found = _select(
                rows, backend=backend, dtype=dtype, order=order, box_size=box_size, nside=nside
            )
            if not found:
                continue
            heights.append(found[0]["mpix_per_second"])
            hollow.append(found[0].get("noisy", False))
            positions.append(slot + (index - len(SERIES) / 2 + 0.5) * width)
        ax.bar(
            positions,
            heights,
            width,
            label=label,
            color=["none" if h else colour for h in hollow],
            edgecolor=colour,
            linewidth=1.2,
        )

    ax.set_yscale("log")
    ax.set_xticks(range(len(orders)))
    ax.set_xticklabels([str(o) for o in orders])
    ax.set_xlabel("interpolation order")
    ax.set_ylabel("throughput / Mpix s$^{-1}$")
    ax.set_title(f"Tiling a ${box_size}^3$ box onto an nside={nside} shell")
    ax.grid(axis="y", which="both", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    # Headroom so the legend never sits on top of the tallest bar.
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom, top * 8)
    ax.legend(fontsize=8, ncol=2, framealpha=0.95, loc="upper center")
    fig.text(
        0.01,
        0.01,
        "hollow bars: the harness flagged the measurement as noisy",
        fontsize=7,
        color="#555555",
    )
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)


def speedup_over_default(rows: list[dict[str, Any]], nside: int, out: Path) -> None:
    """How far each backend is ahead of the path a user is actually on."""
    boxes = sorted({r["box_size"] for r in rows})
    orders = sorted({r["order"] for r in rows})
    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    for box_size, marker in zip(boxes, ("o", "s", "^"), strict=False):
        xs, ys = [], []
        for order in orders:
            default = _select(
                rows, backend="numba", dtype="float64", order=order, box_size=box_size, nside=nside
            )
            gpu = _select(
                rows,
                backend="jax-gpu",
                dtype="float32",
                order=order,
                box_size=box_size,
                nside=nside,
            )
            if not default or not gpu:
                continue
            xs.append(order)
            ys.append(gpu[0]["mpix_per_second"] / default[0]["mpix_per_second"])
        ax.plot(xs, ys, marker=marker, label=f"${box_size}^3$", color=GPU_COLOUR, alpha=0.8)

    ax.axhline(1.0, color="#333333", linewidth=1.0, linestyle="--")
    ax.text(0.05, 1.05, "parity with the default NumPy path", fontsize=8, color="#333333")
    ax.set_yscale("log")
    ax.set_xlabel("interpolation order")
    ax.set_ylabel("GPU float32 speedup over numba")
    ax.set_title(f"What the GPU actually buys, nside={nside}")
    ax.grid(axis="y", which="both", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, title="box")
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)


def main() -> None:
    """Read the committed results and write the figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("benchmarks/results/latest.json"))
    parser.add_argument("--out", type=Path, default=Path("docs/figures"))
    parser.add_argument("--box-size", type=int, default=256)
    parser.add_argument("--nside", type=int, default=256)
    args = parser.parse_args()

    rows = json.loads(args.results.read_text())["rows"]
    args.out.mkdir(parents=True, exist_ok=True)

    by_order = args.out / "throughput_by_order.svg"
    throughput_by_order(rows, args.box_size, args.nside, by_order)
    print(f"wrote {by_order}")

    speedup = args.out / "gpu_speedup.svg"
    speedup_over_default(rows, args.nside, speedup)
    print(f"wrote {speedup}")


if __name__ == "__main__":
    main()
