r"""Measure how fast ``cosmotile`` tiles, across backends, orders and precisions.

Run manually and commit the result::

    python benchmarks/run_benchmarks.py --out benchmarks/results/latest.json

The shells are real HEALPix shells, not random coordinates. Tiling is bound by memory
latency, so what it costs depends on where the samples land and not just how many there
are; random coordinates understate throughput several-fold and make every backend look
alike. Pass ``--scaling`` to sweep box size, nside and order one at a time about a reference
point, instead of the full grid, e.g. for the three-panel scaling figure::

    python benchmarks/run_benchmarks.py --scaling --box-sizes 64 128 256 384 512 \\
        --nsides 32 64 128 256 512 1024

To add to an earlier run rather than redo it, e.g. the GPU rows once CUDA is working::

    python benchmarks/run_benchmarks.py --scaling <same axes> --append --backends jax-gpu

Pass ``--healpix-order ring`` to measure the cost of the other pixel ordering.

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

BACKENDS = ("scipy", "numba", "jax-cpu", "jax-gpu")
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


def configurations(
    box_sizes: tuple[int, ...],
    nsides: tuple[int, ...],
    orders: tuple[int, ...],
    scaling: tuple[int, int, int] | None = None,
) -> list[tuple[int, int, int]]:
    """Return the ``(box_size, nside, order)`` triples to measure.

    By default this is the full grid. With ``scaling=(box_size, nside, order)`` it is
    instead three one-at-a-time sweeps through that reference point -- every box size at
    the reference nside and order, every nside at the reference box size and order, and
    every order at the reference box size and nside -- which grows linearly, not
    multiplicatively, with the number of values on each axis.
    """
    if scaling is None:
        triples = [(b, n, o) for b in box_sizes for n in nsides for o in orders]
    else:
        box, nside, order = scaling
        triples = (
            [(b, nside, order) for b in box_sizes]
            + [(box, n, order) for n in nsides]
            + [(box, nside, o) for o in orders]
        )
    return sorted(set(triples))


def run(
    configs: list[tuple[int, int, int]],
    healpix_order: str,
    radius_fraction: float,
    backends: tuple[str, ...] = BACKENDS,
) -> list[dict[str, Any]]:
    """Sweep the requested, available backends over ``(box, nside, order)`` triples."""
    try:
        import jax

        devices = [("jax-cpu", jax.devices("cpu")[0])]
        with contextlib.suppress(RuntimeError):  # no GPU on this machine
            devices.append(("jax-gpu", jax.devices("gpu")[0]))
    except ImportError:  # pragma: no cover - depends on the environment
        devices = []
    devices = [(name, device) for name, device in devices if name in backends]
    missing = {b for b in backends if b.startswith("jax") and b not in dict(devices)}
    if missing:
        print(f"warning: no device for {sorted(missing)}; those backends will be skipped")

    rng = np.random.default_rng(20260919)
    rows: list[dict[str, Any]] = []

    # One box per size and one shell per nside, however many triples reuse them.
    boxes = {size: rng.standard_normal((size,) * 3) for size in sorted({c[0] for c in configs})}
    shells = {nside: shell_angles(nside, healpix_order) for nside in {c[1] for c in configs}}

    for box_size, nside, order in configs:
        box = boxes[box_size]
        radius = radius_fraction * box_size
        latitude, longitude = shells[nside]
        common = {
            "box_size": box_size,
            "nside": nside,
            "npix": latitude.size,
            "order": order,
            "radius": radius,
            "healpix_order": healpix_order,
        }
        for backend, use_scipy in (("scipy", True), ("numba", False)):
            if backend not in backends or (backend == "numba" and not _gather.NUMBA):
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
                        | _jax_rows(box, latitude, longitude, radius, order, device, precision)
                    )
                except Exception as exc:  # a backend that runs out of memory
                    rows.append(
                        {**common, "backend": name, "dtype": precision, "error": type(exc).__name__}
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


def _key(row: dict[str, Any]) -> tuple[Any, ...]:
    """Get what identifies a measurement, so a re-measurement can replace the old one."""
    names = ("backend", "dtype", "box_size", "nside", "order", "healpix_order")
    return tuple(row.get(name) for name in names)


def merge_results(
    old: dict[str, Any], rows: list[dict[str, Any]], environment: dict[str, Any]
) -> dict[str, Any]:
    """Add ``rows`` to an earlier results file, replacing any row measured again."""
    fresh = {_key(row) for row in rows}
    kept = [row for row in old["rows"] if _key(row) not in fresh]
    merged_env = {**old.get("environment", {}), **environment}
    devices = old.get("environment", {}).get("jax_devices", []) + environment.get("jax_devices", [])
    if devices:
        merged_env["jax_devices"] = list(dict.fromkeys(devices))
    return {"environment": merged_env, "rows": kept + rows}


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
    parser.add_argument(
        "--scaling",
        action="store_true",
        help="Instead of the full grid, sweep each of --box-sizes, --nsides and --orders "
        "on its own, holding the other two at the --ref-* values.",
    )
    parser.add_argument("--ref-box-size", type=int, default=256)
    parser.add_argument("--ref-nside", type=int, default=256)
    parser.add_argument("--ref-order", type=int, default=3)
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=BACKENDS,
        default=list(BACKENDS),
        help="Measure only these backends.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Add to the existing --out file instead of overwriting it; rows measured "
        "again (same backend, dtype, box size, nside and order) replace the old ones. "
        "E.g. `--append --backends jax-gpu` after fixing the GPU install.",
    )
    args = parser.parse_args()

    configs = configurations(
        tuple(args.box_sizes),
        tuple(args.nsides),
        tuple(args.orders),
        (args.ref_box_size, args.ref_nside, args.ref_order) if args.scaling else None,
    )
    rows = run(configs, args.healpix_order, args.radius_fraction, tuple(args.backends))

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
    results = {"environment": environment, "rows": rows}
    if args.append and args.out.exists():
        results = merge_results(json.loads(args.out.read_text()), rows, environment)
    args.out.write_text(json.dumps(results, indent=1))
    print(f"wrote {len(rows)} new rows ({len(results['rows'])} in total) to {args.out}")


if __name__ == "__main__":
    main()
