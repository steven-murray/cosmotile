"""A parallel B-spline gather for the NumPy backend.

:func:`scipy.ndimage.map_coordinates` is single-threaded and carries a fair amount of
per-point overhead, which for this workload -- a few hundred thousand independent
gathers into a periodic box -- leaves most of the machine idle. The kernel here is the
same mathematics (:mod:`cosmotile._spline`), written so that ``numba`` can run it across
every core: about eight times faster at order 3 on sixteen cores, for results that agree
with ``scipy`` to roundoff.

``numba`` is optional. Without it, this module reports itself unavailable and the NumPy
backend falls straight back to ``scipy``, exactly as it did before.

The fallback is also reachable deliberately, via :func:`use_scipy_gather`.
"""

import math
from typing import Any

import numpy as np

from ._spline import MAX_ORDER

# `numba` is untyped, so both names are `Any` either way -- which is what lets the
# kernel below call `prange` under mypy's `disallow_untyped_calls`.
njit: Any
prange: Any

try:
    from numba import njit, prange

    NUMBA = True
except ImportError:  # pragma: no cover - exercised by the tests-nojit session
    NUMBA = False
    prange = range


#: Set by :func:`use_scipy_gather`. Module state rather than an argument, because the
#: point is to flip the whole library over at once while bisecting a moved number.
_DISABLED = False


def use_scipy_gather(disabled: bool = True) -> None:
    """Route the NumPy backend back through :func:`scipy.ndimage.map_coordinates`.

    Parameters
    ----------
    disabled
        ``True`` to use ``scipy``, ``False`` to go back to the parallel kernel.
    """
    global _DISABLED
    _DISABLED = bool(disabled)


def available() -> bool:
    """Whether the parallel kernel should be used for this call."""
    return NUMBA and not _DISABLED


def _gather_impl(coefficients: Any, coordinates: Any, order: int, out: Any) -> Any:
    """Evaluate the spline at every coordinate, one output point per thread.

    The weights are the same uniform-knot de Boor recursion as
    :func:`cosmotile._spline.spline_weights`, unrolled into scratch arrays because
    ``numba`` cannot call the array-namespace-generic version. The recursion is done in
    place: ``term`` reads the old value before it is overwritten, and ``saved`` carries
    the part that belongs to the next tap, so no second buffer is needed.
    """
    n0, n1, n2 = coefficients.shape
    sizes = (n0, n1, n2)
    ntap = order + 1

    for point in prange(coordinates.shape[1]):
        weights = np.empty((3, MAX_ORDER + 1), np.float64)
        indices = np.empty((3, MAX_ORDER + 1), np.int64)

        for axis in range(3):
            shifted = coordinates[axis, point] - order / 2.0 + 0.5
            start = math.floor(shifted)
            offset = shifted - start
            first = int(start) % sizes[axis]

            # Wrap every tap here rather than in the triple loop below: that is
            # 3 * (order + 1) modulos per point instead of (order + 1) ** 3, and it has
            # to be a full modulo because an axis may be shorter than the stencil -- a
            # 2-cell axis tiled at order 5 wraps around it three times.
            for tap in range(ntap):
                indices[axis, tap] = (first + tap) % sizes[axis]

            weights[axis, 0] = 1.0
            for depth in range(1, ntap):
                saved = 0.0
                for tap in range(depth):
                    term = weights[axis, tap] / depth
                    weights[axis, tap] = saved + (tap + 1 - offset) * term
                    saved = (offset + depth - tap - 1) * term
                weights[axis, depth] = saved

        total = 0.0
        for a in range(ntap):
            ia = indices[0, a]
            for b in range(ntap):
                ib = indices[1, b]
                plane = weights[0, a] * weights[1, b]
                for c in range(ntap):
                    total += plane * weights[2, c] * coefficients[ia, ib, indices[2, c]]
        out[point] = total
    return out


#: The compiled kernel when numba is present, and the plain Python loop when it is not
#: -- which is far too slow to use, but keeps the module importable and testable.
_gather = njit(cache=True, parallel=True)(_gather_impl) if NUMBA else _gather_impl


def gather(coefficients: np.ndarray, coordinates: np.ndarray, order: int) -> np.ndarray:
    """Interpolate a periodic 3D box at the given pixel coordinates.

    Equivalent to ``scipy.ndimage.map_coordinates(..., mode="grid-wrap",
    prefilter=False)``, to roundoff.

    Parameters
    ----------
    coefficients
        The 3D B-spline coefficients, already pre-filtered for ``order`` if it exceeds 1.
    coordinates
        ``(3, nsample)`` coordinates in cells. Need not lie inside the box.
    order
        Spline order, in the range 0-5.

    Returns
    -------
    values
        ``(nsample,)`` interpolated values.
    """
    coefficients = np.ascontiguousarray(coefficients, dtype=np.float64)
    coordinates = np.ascontiguousarray(coordinates, dtype=np.float64)
    out = np.empty(coordinates.shape[1], dtype=np.float64)
    return _gather(coefficients, coordinates, order, out)
