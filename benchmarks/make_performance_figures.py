"""Draw the performance figures from a committed benchmark run.

Reads ``benchmarks/results/latest.json`` -- produced by ``run_benchmarks.py`` -- and
writes SVGs (or PDFs, with ``--format pdf``) into ``docs/figures/``. The output is
committed, so building the documentation needs neither ``matplotlib`` nor a GPU, in the
same spirit as ``docs/make_accuracy_figures.py``. Re-run it only when the numbers
change::

    python benchmarks/make_performance_figures.py

Measurements flagged as noisy (e.g. measured on a power-capped laptop GPU) are drawn as
unfilled markers.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

#: One colour per series, so the two GPU precisions can be told apart.
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
    fig.savefig(out)
    plt.close(fig)


def throughput_scaling(
    rows: list[dict[str, Any]], box_size: int, nside: int, order: int, out: Path
) -> None:
    """Throughput against interpolation order, box size and nside, one line per backend.

    Each panel varies one parameter and holds the other two at ``box_size``, ``nside``
    and ``order``. Only HEALPix nested-ordering rows are used.
    """
    rows = _select(rows, healpix_order="nested")
    if not rows:
        raise SystemExit("no nested-ordering rows in the results")

    panels = [
        ("order", "interpolation order", {"box_size": box_size, "nside": nside}),
        ("box_size", "box side $N$ / cells", {"nside": nside, "order": order}),
        ("nside", r"$N_{\rm side}$", {"box_size": box_size, "order": order}),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), sharey=True, layout="constrained")

    for ax, (key, xlabel, fixed) in zip(axes, panels, strict=True):
        for backend, dtype, colour, label in SERIES:
            found = sorted(
                _select(rows, backend=backend, dtype=dtype, **fixed), key=lambda r: r[key]
            )
            if not found:
                continue
            xs = [r[key] for r in found]
            ys = [r["mpix_per_second"] for r in found]
            ax.plot(xs, ys, "o-", ms=4, lw=1.3, color=colour, label=label)
            # Noisy measurements are drawn hollow, as in the bar chart.
            noisy = [(x, y) for x, y, r in zip(xs, ys, found, strict=True) if r.get("noisy")]
            if noisy:
                ax.plot(*zip(*noisy, strict=True), "o", ms=4, mfc="white", color=colour)
        ax.set_yscale("log")
        if key != "order":
            ax.set_xscale("log", base=2)
        xticks = sorted({r[key] for r in rows if all(r.get(k) == v for k, v in fixed.items())})
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(x) for x in xticks])
        ax.minorticks_off()
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.set_title(
            ", ".join(
                f"{name} {value}" if name == "order" else f"{name}={value}"
                for name, value in {"box_size": box_size, "nside": nside, "order": order}.items()
                if name in fixed
            ).replace("box_size", "$N$"),
            fontsize=9,
        )

    axes[0].set_ylabel("throughput / Mpix s$^{-1}$")
    axes[0].legend(fontsize=7.5, frameon=False, loc="upper right")
    fig.savefig(out)
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
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    """Read the committed results and write the figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("benchmarks/results/latest.json"))
    parser.add_argument("--out", type=Path, default=Path("docs/figures"))
    parser.add_argument("--box-size", type=int, default=256)
    parser.add_argument("--nside", type=int, default=256)
    parser.add_argument("--order", type=int, default=3, help="order held fixed in the scaling plot")
    parser.add_argument("--format", choices=["svg", "pdf"], default="svg")
    args = parser.parse_args()

    rows = json.loads(args.results.read_text())["rows"]
    args.out.mkdir(parents=True, exist_ok=True)

    by_order = args.out / f"throughput_by_order.{args.format}"
    throughput_by_order(rows, args.box_size, args.nside, by_order)
    print(f"wrote {by_order}")

    scaling = args.out / f"throughput_scaling.{args.format}"
    throughput_scaling(rows, args.box_size, args.nside, args.order, scaling)
    print(f"wrote {scaling}")

    speedup = args.out / f"gpu_speedup.{args.format}"
    speedup_over_default(rows, args.nside, speedup)
    print(f"wrote {speedup}")


if __name__ == "__main__":
    main()
