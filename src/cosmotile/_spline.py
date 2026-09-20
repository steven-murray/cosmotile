"""Cardinal B-spline mathematics, shared by every backend.

Everything here is written in operations that :mod:`numpy` and :mod:`jax.numpy` spell
identically, and is selected with the private ``xp`` argument. Nothing in this module
imports ``jax``, ``scipy`` or ``astropy``: it is pure arithmetic on an array namespace.

Three things live here, and they are the same three facts about the cardinal B-spline
seen from different sides:

:func:`base_index` and :func:`spline_weights`
    The interpolation stencil -- which samples an evaluation point draws on, and with
    what weights. Together they reproduce :func:`scipy.ndimage.map_coordinates` for
    every order in the range 0-5.

:func:`cardinal_bspline_at_integers`
    The kernel sampled on the grid it is defined on. This is what the spline pre-filter
    has to invert, and it is the special case of :func:`spline_weights` at the offset
    that lands the stencil on integer arguments.

:func:`bspline_dtft`
    The Fourier response of those samples. Dividing by it *is* the pre-filter (see
    :mod:`cosmotile.jax`), and it is also the denominator of the interpolation window
    (see :func:`cosmotile.theory.interpolation_window`).
"""

from __future__ import annotations

from typing import Any

import numpy as np

MAX_ORDER = 5


def spline_weights(offset: Any, order: int, xp: Any = np) -> Any:
    r"""Evaluate the ``order + 1`` non-zero B-spline weights at a fractional offset.

    Uses the Cox-de Boor recursion on uniform integer knots, where every denominator
    collapses to the recursion depth. One loop therefore covers every order, with no
    per-order special case -- which is what lets the same code serve orders 0-5 under
    :func:`jax.jit`, since ``order`` appears only as a Python ``range`` bound.

    Parameters
    ----------
    offset
        The fractional offset into the stencil, as returned by :func:`base_index`. Lies
        in ``[0, 1)``. Any shape.
    order
        The spline order (degree), in the range 0-5.
    xp
        The array namespace to compute in. :mod:`numpy` by default; pass
        :mod:`jax.numpy` to build a traceable computation.

    Returns
    -------
    weights
        Shape ``(order + 1, *offset.shape)``. Weight ``r`` multiplies the sample at
        ``base + r``, and the weights sum to one.

    Examples
    --------
    >>> import numpy as np
    >>> from cosmotile._spline import spline_weights
    >>> np.asarray(spline_weights(np.array(0.0), order=1))
    array([1., 0.])
    """
    weights = [xp.ones_like(offset)]
    for depth in range(1, order + 1):
        saved = xp.zeros_like(offset)
        updated = []
        for r in range(depth):
            term = weights[r] / depth
            updated.append(saved + (r + 1 - offset) * term)
            saved = (offset + depth - r - 1) * term
        updated.append(saved)
        weights = updated
    return xp.stack(weights)


def base_index(coordinate: Any, order: int, xp: Any = np) -> tuple[Any, Any]:
    r"""Split a coordinate into the first stencil sample and the offset into it.

    The stencil for an order-``p`` spline is centred on the evaluation point, so it
    starts at :math:`\lfloor x - p/2 + 1/2 \rfloor`. That single expression reproduces
    ``scipy``'s convention for every order: ``floor(x)`` for odd orders, and
    ``floor(x + 1/2) - p//2`` for even ones -- including order 0, whose stencil is the
    round-half-up nearest neighbour.

    Splitting here rather than carrying a raw coordinate is also what keeps the kernel
    accurate in single precision. ``offset`` lies in ``[0, 1)``, so it costs about
    ``6e-8`` absolute in ``float32``, whereas a shell radius of a thousand cells carried
    as a ``float32`` coordinate would already be uncertain by ``1e-4`` cells.

    Parameters
    ----------
    coordinate
        Evaluation points in pixel units. Need not lie inside the grid.
    order
        The spline order (degree), in the range 0-5.
    xp
        The array namespace to compute in.

    Returns
    -------
    base
        Index of the first sample of the stencil, as ``int32``. Not yet wrapped
        into the grid -- see :func:`cosmotile._geometry.wrap_index`.
    offset
        The fractional offset into the stencil, in ``[0, 1)``. Pass to
        :func:`spline_weights`.
    """
    shifted = coordinate - order / 2 + 0.5
    base = xp.floor(shifted)
    return base.astype(np.int32), shifted - base


def cardinal_bspline_at_integers(order: int) -> np.ndarray:
    r"""Sample the centred cardinal B-spline of the given order at integer offsets.

    Returns :math:`\beta(0), \beta(1), \ldots, \beta(p // 2)`; the kernel is symmetric,
    and it vanishes at every larger integer.

    These are the coefficients the spline pre-filter must deconvolve. For orders 0 and 1
    the kernel is interpolating, so this is ``[1.0]`` and no pre-filter is needed.

    Parameters
    ----------
    order
        The spline order (degree), in the range 0-5.

    Returns
    -------
    samples
        The non-zero, non-negative-offset integer samples of the kernel.

    Notes
    -----
    This is :func:`spline_weights` evaluated at the one offset that puts the stencil on
    integer arguments -- zero for odd orders, a half for even ones -- so it needs no
    separate implementation and cannot drift out of step with the interpolation.

    Examples
    --------
    >>> from cosmotile._spline import cardinal_bspline_at_integers
    >>> cardinal_bspline_at_integers(3)
    array([0.66666667, 0.16666667])
    """
    half = order // 2
    weights = spline_weights(np.array(0.0 if order % 2 else 0.5), order)
    return np.asarray(weights[half : 2 * half + 1], dtype=float)


def bspline_dtft(k: Any, order: int, xp: Any = np) -> Any:
    r"""Fourier response of the B-spline sampled on the grid.

    .. math:: b_p(k) = \beta(0) + 2 \sum_{m \ge 1} \beta(m) \cos(m k),

    the discrete-time transform of :func:`cardinal_bspline_at_integers`. It is real,
    even and strictly positive for every order in range, so dividing by it is stable.

    It appears in two places, and they are the same operation seen forwards and
    backwards: dividing a box's spectrum by it *is* the periodic spline pre-filter, and
    it is the denominator of :func:`cosmotile.theory.interpolation_window`, where it
    divides out the pre-filter the tiling applied.

    Parameters
    ----------
    k
        Angular wavenumbers, in inverse cell units. Any shape.
    order
        The spline order (degree), in the range 0-5.
    xp
        The array namespace to compute in.

    Returns
    -------
    response
        The response, of the same shape as ``k``. Identically one for orders 0 and 1.
    """
    samples = cardinal_bspline_at_integers(order)
    response = samples[0] * xp.ones_like(k)
    for offset, value in enumerate(samples[1:], start=1):
        response = response + 2 * float(value) * xp.cos(offset * k)
    return response
