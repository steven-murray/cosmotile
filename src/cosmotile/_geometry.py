"""Shell geometry: where on the coeval grid each output pixel is sampled.

Split in two, because the two halves have opposite requirements.

**Host geometry** -- :func:`unit_vectors` and :func:`radial_quadrature` -- runs once per
shell *geometry*, not once per field, and is free to use ``astropy`` and ``scipy``. It is
``O(npix)``, never ``O(N^3)``, and nobody wants to differentiate with respect to a
HEALPix pixel centre. It always runs in NumPy at double precision.

**Traced geometry** -- :func:`shell_coordinates` and :func:`wrap_index` -- has to run
inside a :func:`jax.jit` trace, because a lightcone's per-shell coordinates are far too
large to precompute and hold (nineteen gigabytes for a thousand shells at ``nside=256``),
and because a gradient with respect to the shell radius needs them to be traced. These
are written only in operations that :mod:`numpy` and :mod:`jax.numpy` spell identically,
selected by the private ``xp`` argument, so one implementation serves both backends.

:class:`scipy.spatial.transform.Rotation` never crosses that boundary: call
:meth:`~scipy.spatial.transform.Rotation.as_matrix` on the host and pass the ``(3, 3)``
array in.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def residual_radial_width(target_width: float, cell_size: float = 1.0) -> float:
    r"""Return the extra radial top-hat needed to reach a given total window.

    A cell-averaged box already carries a top-hat of one cell along the line of sight,
    so averaging over the full ``target_width`` on top of it double-counts: the
    delivered window would be the product ``sinc(k w / 2) sinc(k D / 2)``, not
    ``sinc(k w / 2)`` alone. Both expand as ``1 - k^2 x^2 / 24``, so their widths add in
    quadrature to leading order and the extra width to apply is

    .. math:: w = \sqrt{\max(\Delta r^2 - \Delta^2,\; 0)}.

    That reproduces the wanted window to better than 2.5% out to its own Nyquist for any
    ratio of the two widths, against up to 36% for applying ``target_width`` directly.

    :func:`make_lightcone_slice_interpolator` does this for you -- its ``radial_width``
    is the total window you want, and it subtracts ``coeval_cell_width`` itself. This
    function is the arithmetic behind that, exposed for anyone reasoning about windows
    on their own.

    Parameters
    ----------
    target_width
        Width of the radial top-hat you want the output to carry, in cells.
    cell_size
        Width of the cell top-hat already present in the box, in cells. One by default;
        pass zero if your box holds point samples rather than cell averages.

    Returns
    -------
    width
        The extra top-hat to apply. Zero when the box already supplies enough.
    """
    if target_width < 0 or cell_size < 0:
        raise ValueError("target_width and cell_size must be non-negative")

    return float(np.sqrt(max(target_width**2 - cell_size**2, 0.0)))


def unit_vectors(latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    """Cartesian unit vectors for a set of angular coordinates.

    These do not depend on the shell radius, so they are computed once for a geometry
    and scaled per shell. That is what makes a thousand-shell lightcone tractable: the
    unit vectors are shared, and each shell carries only its radius.

    Parameters
    ----------
    latitude
        Latitudes in radians, from ``-pi/2`` to ``pi/2``. Any shape; flattened.
    longitude
        Longitudes in radians, from 0 to ``2 pi``. Same shape as ``latitude``.

    Returns
    -------
    directions
        Shape ``(3, latitude.size)``, unit norm along the first axis.
    """
    latitude = np.asarray(latitude, dtype=float).ravel()
    longitude = np.asarray(longitude, dtype=float).ravel()
    polar = np.pi / 2 - latitude
    sin_polar = np.sin(polar)
    return np.array(
        [
            sin_polar * np.cos(longitude),
            sin_polar * np.sin(longitude),
            np.cos(polar),
        ]
    )


def radial_quadrature(width: float, n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes and weights for averaging across a shell's radial extent.

    Returned on the reference interval ``[-1, 1]``, so that a shell can scale them to its
    own radius inside a trace. The ``r^2`` volume element is *not* applied here, since it
    depends on the radius; see :func:`shell_coordinates`.

    Parameters
    ----------
    width
        The extra radial top-hat to apply, in cells -- what
        :func:`~cosmotile.residual_radial_width` returns. Zero means point sampling.
    n_nodes
        Number of quadrature nodes. One means point sampling.

    Returns
    -------
    nodes
        Node positions on ``[-1, 1]``, scaled by ``width / 2`` at use.
    weights
        The matching quadrature weights, before the volume element.
    """
    if n_nodes == 1 or width == 0:
        return np.zeros(1), np.ones(1)
    nodes, weights = np.polynomial.legendre.leggauss(n_nodes)
    return nodes, weights


def shell_coordinates(
    directions: Any,
    radius: Any,
    nodes: Any,
    node_weights: Any,
    *,
    half_width: Any = 0.0,
    rotation: Any = None,
    origin: Any = None,
    n_angular: int = 1,
    xp: Any = np,
) -> tuple[Any, Any]:
    r"""Pixel coordinates and sub-sample weights for one shell.

    Sub-samples are laid out radial-major, then angular, then pixel, so that the weights
    are the outer product of the two sets -- the same layout the NumPy backend uses, so
    the two agree element by element.

    The ``r^2`` volume element is applied to the radial weights here rather than on the
    host, because it depends on the radius and the radius may be a tracer. It is always
    applied: it is what makes the result the mean over a shell of finite thickness rather
    than an unweighted average of radii.

    Parameters
    ----------
    directions
        ``(3, n_angular * npix)`` unit vectors, from :func:`unit_vectors`.
    radius
        The shell radius in cells. May be a scalar tracer.
    nodes, node_weights
        Gauss-Legendre rule on ``[-1, 1]``, from :func:`radial_quadrature`.
    half_width
        Half the extra radial top-hat, in cells. Zero for point sampling.
    rotation
        A ``(3, 3)`` rotation matrix, or ``None``. Applied before ``origin``.
    origin
        A ``(3,)`` shell centre in cells, or ``None``.
    n_angular
        Number of angular sub-samples per output pixel, used to normalise the weights.
    xp
        The array namespace to compute in.

    Returns
    -------
    coordinates
        ``(3, n_radial * n_angular * npix)`` pixel coordinates.
    weights
        ``(n_radial * n_angular,)`` sub-sample weights, summing to one.
    """
    radii = radius + half_width * nodes
    radial_weights = node_weights * radii**2
    radial_weights = radial_weights / radial_weights.sum()

    if rotation is not None:
        directions = rotation @ directions

    # (n_radial, 3, M) -> (3, n_radial, M) -> (3, n_radial * M), i.e. radial-major.
    scaled = radii[:, None, None] * directions[None, :, :]
    coordinates = xp.swapaxes(scaled, 0, 1).reshape(3, -1)

    if origin is not None:
        coordinates = coordinates + origin[:, None]

    weights = xp.repeat(radial_weights, n_angular) / n_angular
    return coordinates, weights


def wrap_index(base: Any, size: int, xp: Any = np) -> Any:
    """Wrap a stencil's first index into ``[0, size)``, periodically.

    This is ``mode="grid-wrap"``: the grid has period exactly ``size``, with no
    reflection or repeated edge sample.
    """
    return xp.remainder(base, size)
