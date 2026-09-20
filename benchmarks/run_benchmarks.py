"""Measure how fast ``cosmotile`` tiles, across backends, orders and precisions.

Run manually and commit the result; nothing builds this at documentation time, in the
same spirit as ``docs/make_accuracy_figures.py``::

    python benchmarks/run_benchmarks.py --out benchmarks/results/latest.json

Two things about the method matter enough to state here, because getting either wrong
changes the answer by nearly an order of magnitude.

**The coordinates must be a real shell.** This kernel is bound by memory latency, so
what it costs depends on how its samples are laid out in the box, not just how many
there are. Random coordinates -- the obvious thing to benchmark with -- understate real
throughput by up to eight times, and would make every backend look artificially close
together. So the sweep tiles actual HEALPix shells.

**HEALPix ring order is slower than nested order**, by about a factor of 1.7 on the same
pixels, purely because neighbouring pixels in the nested scheme land nearer each other
in the box. That is a real result about how to use the library, so it is measured rather
than assumed: pass ``--healpix-order ring`` to see it.

The ``scipy`` row is the *fallback*, not the default: with ``numba`` installed the
NumPy backend gathers through :mod:`cosmotile._gather` instead, which is the ``numba``
row. Both are measured, because ``scipy`` is the reference everything else is calibrated
against.

The JAX rows are skipped if ``jax`` is not installed, and the GPU rows if no GPU is
visible, so this runs anywhere and reports what it could measure.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import platform
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from astropy_healpix import HEALPix
from scipy.ndimage import map_coordinates, spline_filter

from cosmotile import _gather

ORDERS = (0, 1, 3, 5)
BOX_SIZES = (128, 256, 384)
NSIDES = (128, 256)


def shell_coordinates(
    nside: int, radius: float, box_size: int, healpix_order: str = "nested"
) -> np.ndarray:
    """Pixel coordinates of a real HEALPix shell, centred in the box."""
    healpix = HEALPix(nside=nside, order=healpix_order)
    lon, lat = healpix.healpix_to_lonlat(np.arange(healpix.npix))
    lon = lon.to_value("radian")
    lat = lat.to_value("radian")
    polar = np.pi / 2 - lat
    sin_polar = np.sin(polar)
    direction = np.array([sin_polar * np.cos(lon), sin_polar * np.sin(lon), np.cos(polar)])
    return radius * direction + box_size / 2


def time_it(
    call: Callable[[], Any],
    repeats: int = 9,
    warmups: int = 3,
    min_seconds: float = 0.5,
) -> dict[str, float]:
    """Time ``call``, reporting both the best and the median per-call time.

    Warming up is not a formality on a laptop GPU. The card here idles at a tenth of its
    maximum clock and ramps under load, then gets pulled back by its power cap -- it had
    accumulated half a minute of power capping by the time these numbers were taken. A
    fixed three warm-up calls is enough for a long kernel and nowhere near enough for a
    short one, so warming continues until ``min_seconds`` of wall clock have gone by as
    well. Without that, the same kernel measured minutes apart differed by a factor of
    seven, which is larger than most of the effects being measured.

    Both statistics are reported, deliberately. The best run is the machine's
    capability; the median is what you will actually get. When they disagree badly the
    measurement is not to be trusted, and writing both down makes that visible in the
    data rather than hiding it behind one authoritative-looking number.
    """
    start = time.perf_counter()
    count = 0
    while count < warmups or time.perf_counter() - start < min_seconds:
        call()
        count += 1

    times = []
    for _ in range(repeats):
        tick = time.perf_counter()
        call()
        times.append(time.perf_counter() - tick)
    return {"best": min(times), "median": float(np.median(times))}


def _scipy_rows(box: np.ndarray, coords: np.ndarray, order: int) -> dict[str, Any]:
    coefficients = (
        spline_filter(box, order=order, mode="grid-wrap", output=np.float64) if order > 1 else box
    )
    prefilter_seconds = (
        time_it(
            lambda: spline_filter(box, order=order, mode="grid-wrap", output=np.float64),
            repeats=3,
            warmups=1,
            min_seconds=0.0,
        )["best"]
        if order > 1
        else 0.0
    )
    timing = time_it(
        lambda: map_coordinates(
            coefficients, coords, order=order, mode="grid-wrap", prefilter=False
        )
    )
    return {
        "gather_seconds": timing["best"],
        "gather_seconds_median": timing["median"],
        "prefilter_seconds": prefilter_seconds,
    }


def _numba_rows(box: np.ndarray, coords: np.ndarray, order: int) -> dict[str, Any]:
    """Time the default NumPy path, which is this whenever ``numba`` is installed."""
    coefficients = (
        spline_filter(box, order=order, mode="grid-wrap", output=np.float64) if order > 1 else box
    )
    _gather.gather(coefficients, coords, order)  # compile before timing
    timing = time_it(lambda: _gather.gather(coefficients, coords, order))
    return {
        "gather_seconds": timing["best"],
        "gather_seconds_median": timing["median"],
        # Same pre-filter as the scipy row: only the gather is replaced.
        "prefilter_seconds": 0.0,
    }


def _jax_rows(
    box: np.ndarray, coords: np.ndarray, order: int, device: Any, precision: str
) -> dict[str, Any]:
    import contextlib as _contextlib

    import jax
    import jax.numpy as jnp

    from cosmotile import jax as cjax

    # Double precision is opt-in and scoped: without this, jnp silently truncates to
    # float32 and the "float64" row would be a float32 timing under another name.
    x64 = jax.enable_x64() if precision == "float64" else _contextlib.nullcontext()

    with x64, jax.default_device(device):
        dtype = jnp.float64 if precision == "float64" else jnp.float32
        array = jnp.asarray(box, dtype=dtype)
        prefilter_seconds = time_it(
            lambda: jax.block_until_ready(cjax.prefilter_coeval(array, order).coefficients),
            repeats=3,
            warmups=2,
            min_seconds=0.3,
        )["best"]
        coefficients = cjax.prefilter_coeval(array, order)
        coordinates = jnp.asarray(coords, dtype=dtype)
        gather = jax.jit(lambda c, x: cjax.shell_from_coordinates(c, x, order=order))
        result = gather(coefficients, coordinates)
        if result.dtype != dtype:
            raise RuntimeError(f"expected {dtype}, got {result.dtype}")
        timing = time_it(lambda: jax.block_until_ready(gather(coefficients, coordinates)))

    return {
        "gather_seconds": timing["best"],
        "gather_seconds_median": timing["median"],
        "prefilter_seconds": prefilter_seconds,
    }


def run(
    box_sizes: tuple[int, ...],
    nsides: tuple[int, ...],
    orders: tuple[int, ...],
    healpix_order: str,
    radius_fraction: float,
) -> list[dict[str, Any]]:
    """Sweep every available backend over the requested grid."""
    try:
        import jax

        devices = [("jax-cpu", jax.devices("cpu")[0])]
        with contextlib.suppress(RuntimeError):  # no GPU on this machine
            devices.append(("jax-gpu", jax.devices("gpu")[0]))
    except ImportError:  # pragma: no cover - depends on the environment
        devices = []

    rng = np.random.default_rng(20260919)
    rows: list[dict[str, Any]] = []

    for box_size in box_sizes:
        box = rng.standard_normal((box_size,) * 3)
        radius = radius_fraction * box_size
        for nside in nsides:
            coords = shell_coordinates(nside, radius, box_size, healpix_order)
            npix = coords.shape[1]
            for order in orders:
                common = {
                    "box_size": box_size,
                    "nside": nside,
                    "npix": npix,
                    "order": order,
                    "radius": radius,
                    "healpix_order": healpix_order,
                }
                rows.append(
                    {**common, "backend": "scipy", "dtype": "float64"}
                    | _scipy_rows(box, coords, order)
                )
                if _gather.NUMBA:
                    rows.append(
                        {**common, "backend": "numba", "dtype": "float64"}
                        | _numba_rows(box, coords, order)
                    )
                for name, device in devices:
                    for precision in ("float32", "float64"):
                        try:
                            rows.append(
                                {**common, "backend": name, "dtype": precision}
                                | _jax_rows(box, coords, order, device, precision)
                            )
                        except Exception as exc:  # a backend that runs out of memory
                            rows.append(
                                {
                                    **common,
                                    "backend": name,
                                    "dtype": precision,
                                    "error": type(exc).__name__,
                                }
                            )
                print(f"  {box_size}^3 nside={nside} order={order}: {len(rows)} rows so far")

    for row in rows:
        if "gather_seconds" in row:
            row["mpix_per_second"] = row["npix"] / row["gather_seconds"] / 1e6
            row["mpix_per_second_median"] = row["npix"] / row["gather_seconds_median"] / 1e6
            # A run whose best and median differ a lot was measuring the machine, not
            # the kernel. Flag it rather than quietly averaging it away.
            row["noisy"] = row["gather_seconds_median"] > 1.5 * row["gather_seconds"]
    return rows


def main() -> None:
    """Parse arguments, run the sweep, write the JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("benchmarks/results/latest.json"))
    parser.add_argument("--box-sizes", type=int, nargs="+", default=list(BOX_SIZES))
    parser.add_argument("--nsides", type=int, nargs="+", default=list(NSIDES))
    parser.add_argument("--orders", type=int, nargs="+", default=list(ORDERS))
    parser.add_argument("--healpix-order", choices=("nested", "ring"), default="nested")
    parser.add_argument(
        "--radius-fraction",
        type=float,
        default=0.39,
        help="Shell radius as a fraction of the box side; the default inscribes the shell.",
    )
    args = parser.parse_args()

    rows = run(
        tuple(args.box_sizes),
        tuple(args.nsides),
        tuple(args.orders),
        args.healpix_order,
        args.radius_fraction,
    )

    environment: dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    try:
        import jax

        environment["jax"] = jax.__version__
        environment["jax_devices"] = [str(d) for d in jax.devices()]
    except ImportError:  # pragma: no cover - depends on the environment
        pass

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"environment": environment, "rows": rows}, indent=1))
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
