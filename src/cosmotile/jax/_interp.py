"""The separable B-spline gather, orders 0-5."""

from __future__ import annotations

import functools
from typing import Any

import jax
import jax.numpy as jnp

from .._geometry import shell_coordinates, wrap_index
from .._plan import PrefilteredCoeval, ShellSampling
from .._spline import base_index, spline_weights


def _average_subsamples(values: Any, weights: Any, npix: int) -> Any:
    """Collapse sub-sample-major values onto one value per output pixel."""
    return (weights[:, None] * values.reshape(-1, npix)).sum(axis=0)


def _wrap(index: Any, size: int, order: int) -> Any:
    """Wrap a stencil index into ``[0, size)``, given a ``base`` already inside the grid.

    ``size`` and ``order`` are both known when the function is traced, so the choice
    between the two branches costs nothing at run time -- and it is worth making. The
    index is at most ``size + order - 1``, so one conditional subtract is enough as soon
    as the axis is longer than the stencil, which it essentially always is. A general
    modulo has to handle negative operands too, and on a GPU that extra work is not
    free: measured on an RTX A2000, it took the order-3 gather from 443 Mpix/s to 61.

    The modulo is still needed for the pathological case, though. An axis may be shorter
    than the stencil -- a 2-cell axis tiled at order 5 wraps around it three times -- and
    a single subtract would leave the index off the end of the array.
    """
    if size > order:
        return jnp.where(index >= size, index - size, index)
    return index % size


@functools.partial(jax.jit, static_argnames=("order",))
def _gather(coefficients: Any, base: Any, weights: Any, order: int) -> Any:
    """Evaluate the spline at every sample, given a pre-wrapped stencil.

    Unrolls the outer two stencil axes and vectorises the innermost. Unrolling all three
    would emit ``(order + 1) ** 3`` graph nodes -- 216 at order 5 -- and compile slowly;
    vectorising all three would materialise ``(order + 1) ** 3`` index arrays at once,
    which is 768 MB of ``int32`` alone for three million samples at order 3. Unrolling
    two and vectorising one is 16 nodes and 48 MB for the same case.

    Parameters
    ----------
    coefficients
        The 3D B-spline coefficients.
    base
        ``(3, nsample)`` first stencil index per axis, already wrapped into the grid.
    weights
        ``(3, order + 1, nsample)`` separable stencil weights.
    order
        The spline order. Static.
    """
    nx, ny, nz = coefficients.shape
    flat = coefficients.reshape(-1)

    # Only the vectorised axis is stacked. Hoisting the other two into arrays as well
    # keeps every tap of every axis live across the whole loop -- tens of megabytes at
    # full resolution -- and XLA fuses the kernel markedly worse for it. The unrolled
    # axes are cheaper recomputed in place.
    last = jnp.stack([_wrap(base[2] + c, nz, order) for c in range(order + 1)])

    total = jnp.zeros(base.shape[1], coefficients.dtype)
    for a in range(order + 1):
        row = _wrap(base[0] + a, nx, order) * ny
        for b in range(order + 1):
            # One flat index, so XLA emits a single gather. The wrap has to be per-axis:
            # wrapping a flat index would run off the end of a row into the next one.
            linear = (row + _wrap(base[1] + b, ny, order))[None, :] * nz + last
            values = jnp.take(flat, linear)
            total = total + (weights[0, a] * weights[1, b]) * jnp.einsum(
                "cn,cn->n", weights[2], values
            )
    return total


def _coefficients(coeval: Any, order: int) -> Any:
    """Unwrap a pre-filtered box, checking it was filtered for this order."""
    if isinstance(coeval, PrefilteredCoeval):
        if coeval.order != order:
            raise ValueError(
                f"coeval was pre-filtered for order {coeval.order}, but is being "
                f"interpolated at order {order}. Pre-filter at the order you will tile with."
            )
        return coeval.coefficients
    if order > 1:
        raise ValueError(
            f"interpolating at order {order} needs B-spline coefficients: pass the box "
            "through cosmotile.jax.prefilter_coeval(box, order) first. Unlike the NumPy "
            "backend this is not done for you, because filtering inside a jit-ed shell "
            "would redo it for every shell of the lightcone."
        )
    return coeval


def shell_from_coordinates(
    coeval: Any,
    coordinates: Any,
    *,
    order: int,
    weights: Any = None,
    npix: int | None = None,
) -> Any:
    """Interpolate a box at arbitrary pixel coordinates.

    The primitive the rest of the backend is built on. Pure, so it may be wrapped in
    :func:`jax.jit`, :func:`jax.vmap` or :func:`jax.grad` directly; it is deliberately
    not jit-ed itself, so that jit-ing the caller does not compile it twice.

    Parameters
    ----------
    coeval
        A :class:`~cosmotile.PrefilteredCoeval` for ``order >= 2``, or a raw 3D array
        for orders 0 and 1.
    coordinates
        ``(3, nsample)`` coordinates in cells, sub-sample-major. Need not lie inside
        the box: the grid is periodic.
    order
        Interpolation order, in the range 0-5. Static under ``jit``.
    weights
        Sub-sample weights, or ``None`` to return one value per sample.
    npix
        Number of output pixels. Required when ``weights`` is given.

    Returns
    -------
    values
        ``(npix,)`` if ``weights`` is given, else ``(nsample,)``.
    """
    coefficients = _coefficients(coeval, order)
    base, offset = base_index(coordinates, order, xp=jnp)
    base = jnp.stack(
        [wrap_index(base[axis], coefficients.shape[axis], xp=jnp) for axis in range(3)]
    )
    stencil = jnp.swapaxes(spline_weights(offset, order, xp=jnp), 0, 1)
    values = _gather(coefficients, base, stencil.astype(coefficients.dtype), order)
    if weights is None:
        return values
    if npix is None:
        raise ValueError("npix is required when weights are given")
    return _average_subsamples(values, weights.astype(values.dtype), npix)


def shell(
    coeval: Any,
    sampling: ShellSampling,
    radius: Any,
    *,
    rotation: Any = None,
    origin: Any = None,
) -> Any:
    """Interpolate a box onto one spherical shell.

    Pure and differentiable in ``coeval``, ``radius`` and ``origin``. The coordinates
    are built inside the computation rather than passed in, which is what lets a whole
    lightcone be folded with :func:`lightcone_scan` while carrying only one scalar per
    shell.

    Parameters
    ----------
    coeval
        A :class:`~cosmotile.PrefilteredCoeval` for ``order >= 2``, else a 3D array.
    sampling
        The radius-independent geometry, from
        :func:`~cosmotile._plan.make_shell_sampling`.
    radius
        Shell radius in cells. May be a tracer.
    rotation
        A ``(3, 3)`` rotation matrix, or ``None``. Call
        :meth:`scipy.spatial.transform.Rotation.as_matrix` on the host to get one.
    origin
        A ``(3,)`` shell centre in cells, or ``None``.

    Returns
    -------
    values
        ``(sampling.npix,)`` interpolated values.
    """
    coordinates, weights = shell_coordinates(
        jnp.asarray(sampling.directions),
        radius,
        jnp.asarray(sampling.nodes),
        jnp.asarray(sampling.node_weights),
        half_width=sampling.half_width,
        rotation=rotation,
        origin=origin,
        n_angular=sampling.n_angular,
        xp=jnp,
    )
    if sampling.n_angular == 1 and weights.size == 1:
        weights = None
    return shell_from_coordinates(
        coeval,
        coordinates,
        order=sampling.order,
        weights=weights,
        npix=sampling.npix,
    )
