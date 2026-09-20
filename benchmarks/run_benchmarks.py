"""Measure how fast ``cosmotile`` tiles, across backends, orders and precisions.

Run manually and commit the result::

    python benchmarks/run_benchmarks.py --out benchmarks/results/latest.json

Every row times a public ``cosmotile`` entry point -- the interpolator returned by
:func:`cosmotile.make_lightcone_slice_interpolator`, or :func:`cosmotile.jax.shell` --
rather than the library it happens to call underneath, so the numbers stay true if the
implementation changes.

The shells are real HEALPix shells, not random coordinates. Tiling is bound by memory
latency, so what it costs depends on where the samples land and not just how many there
are; random coordinates understate throughput several-fold and make every backend look
alike. Pass ``--healpix-order ring`` to measure the cost of the other pixel ordering.

The ``scipy`` rows are the *fallback* path, reached by turning the parallel gather off.
They are measured because ``scipy`` is the reference the test suite is calibrated
against. JAX rows are skipped if ``jax`` is not installed, and GPU rows if no GPU is
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

import cosmotile as cmt
from cosmotile import _gather

ORDERS = (0, 1, 3, 5)
BOX_SIZES = (128, 256, 384)
NSIDES = (128, 256)


def shell_angles(nside: int, healpix_order: str = "nested") -> tuple[np.ndarray, np.ndarray]:
    """Latitude and longitude of every pixel of a HEALPix shell, in radians."""
    healpix = HEALPix(nside=nside, order=healpix_order)
    lon, lat = healpix.healpix_to_lonlat(np.arange(healpix.npix))
    return lat.to_value("radian"), lon.to_value("radian")


def time_it(
    call: Callable[[], Any],
    repeats: int = 9,
    warmups: int = 3,
    min_seconds: float = 0.5,
) -> dict[str, float]:
    """Time ``call``, reporting both the best and the median per-call time.

    Warming continues until ``min_seconds`` of wall clock have passed as well as
    ``warmups`` calls, because a GPU that has been idle is also a GPU at its idle clock.
    Without that, the same kernel measured minutes apart differed by a factor of seven on
    the laptop card these numbers came from.

    Both statistics are reported: the best run is the machine's capability, the median is
    what you will actually get, and when they disagree badly the measurement is not to be
    trusted.
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


def _numpy_rows(
    box: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    radius: float,
    order: int,
    use_scipy: bool,
) -> dict[str, Any]:
    """Time the NumPy backend, through the interpolator a user would build."""
    _gather.use_scipy_gather(use_scipy)
    try:
        prefilter = time_it(
            lambda: cmt.prefilter_coeval(box, order), repeats=3, warmups=1, min_seconds=0.0
        )["best"]
        coefficients = cmt.prefilter_coeval(box, order)
        interpolate = cmt.make_lightcone_slice_interpolator(
            latitude=latitude,
            longitude=longitude,
            distance_to_shell=radius,
            interpolation_order=order,
        )
        timing = time_it(lambda: interpolate(coefficients))
    finally:
        _gather.use_scipy_gather(False)
    return {
        "gather_seconds": timing["best"],
        "gather_seconds_median": timing["median"],
        "prefilter_seconds": prefilter,
    }


def _jax_rows(
    box: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    radius: float,
    order: int,
    device: Any,
    precision: str,
) -> dict[str, Any]:
    """Time the JAX backend, through :func:`cosmotile.jax.shell`."""
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
        sampling = cjax.make_shell_sampling(latitude=latitude, longitude=longitude, order=order)

        prefilter = time_it(
            lambda: jax.block_until_ready(cjax.prefilter_coeval(array, order).coefficients),
            repeats=3,
            warmups=2,
            min_seconds=0.3,
        )["best"]
        coefficients = cjax.prefilter_coeval(array, order)

        shell = jax.jit(lambda c: cjax.shell(c, sampling, radius))
        result = shell(coefficients)
        if result.dtype != dtype:
            raise RuntimeError(f"expected {dtype}, got {result.dtype}")
        timing = time_it(lambda: jax.block_until_ready(shell(coefficients)))

    return {
        "gather_seconds": timing["best"],
        "gather_seconds_median": timing["median"],
        "prefilter_seconds": prefilter,
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
            latitude, longitude = shell_angles(nside, healpix_order)
            npix = latitude.size
            for order in orders:
                common = {
                    "box_size": box_size,
                    "nside": nside,
                    "npix": npix,
                    "order": order,
                    "radius": radius,
                    "healpix_order": healpix_order,
                }
                for backend, use_scipy in (("scipy", True), ("numba", False)):
                    if backend == "numba" and not _gather.NUMBA:
                        continue
                    rows.append(
                        {**common, "backend": backend, "dtype": "float64"}
                        | _numpy_rows(box, latitude, longitude, radius, order, use_scipy)
                    )
                for name, device in devices:
                    for precision in ("float32", "float64"):
                        try:
                            rows.append(
                                {**common, "backend": name, "dtype": precision}
                                | _jax_rows(
                                    box, latitude, longitude, radius, order, device, precision
                                )
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
