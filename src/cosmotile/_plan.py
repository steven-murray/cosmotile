"""Backend-agnostic containers for work that does not depend on field values.

These are plain frozen dataclasses. They hold no arrays of their own beyond what they
are given, import neither ``jax`` nor ``scipy``, and are registered as JAX pytrees by
:mod:`cosmotile.jax` when that backend is imported -- which is why the fields that must
stay static under :func:`jax.jit` (orders, counts) are kept separate from the fields
that hold data.
"""

import dataclasses
import warnings
from typing import Any

import numpy as np


@dataclasses.dataclass(frozen=True)
class PrefilteredCoeval:
    """A coeval box converted to B-spline coefficients for one interpolation order.

    Produced by :func:`~cosmotile.prefilter_coeval` (or its JAX counterpart). The order
    is carried explicitly rather than hidden on the array, so the same object works for
    a NumPy array, a JAX array or anything else, and so a static type checker can see
    it.

    The tag is what tells :func:`~cosmotile.make_lightcone_slice` to skip the filter it
    would otherwise apply, and what lets it reject a box filtered for the wrong order.

    Deliberately not an array: it supports neither arithmetic nor indexing, because the
    pre-filter of a slice is not the slice of the pre-filter and the pre-filter of twice
    a box is not twice the pre-filter. Both would be silently wrong, so both are a loud
    :exc:`TypeError`. Use :func:`numpy.asarray` to get the coefficients out, and
    re-filter the result if you derive a new box from them.
    """

    #: The B-spline coefficients. For orders 0 and 1 these are the box values unchanged.
    coefficients: Any

    #: The interpolation order the box was filtered for.
    order: int

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the underlying coefficients."""
        return tuple(self.coefficients.shape)

    @property
    def ndim(self) -> int:
        """Number of dimensions of the underlying coefficients."""
        return int(self.coefficients.ndim)

    @property
    def dtype(self) -> Any:
        """Dtype of the underlying coefficients."""
        return self.coefficients.dtype

    @property
    def spline_order(self) -> int:
        """Deprecated alias for :attr:`order`.

        .. deprecated:: 2.0
            Use :attr:`order`.
        """
        warnings.warn(
            "PrefilteredCoeval.spline_order is deprecated; use .order instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.order

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        """Return the coefficients as a plain :class:`numpy.ndarray`."""
        return np.asarray(self.coefficients, dtype=dtype)


@dataclasses.dataclass(frozen=True)
class ShellSampling:
    """Everything about a shell's geometry that does not depend on its radius.

    Built by :func:`make_shell_sampling`. One of these is shared by every shell of a
    lightcone: the unit vectors are the expensive part and they are radius-independent,
    so a thousand-shell lightcone carries one of these plus a thousand scalars, rather
    than a thousand coordinate arrays.

    The integer fields are kept separate from the array fields because
    :mod:`cosmotile.jax` registers this as a pytree with the integers as static
    metadata -- ``order`` in particular cannot be a tracer, since the gather unrolls
    over it.
    """

    #: ``(3, n_angular * npix)`` unit vectors, angular-sub-sample-major.
    directions: Any

    #: Gauss-Legendre nodes on ``[-1, 1]`` across the shell's radial extent.
    nodes: Any

    #: The matching quadrature weights, before the ``r^2`` volume element.
    node_weights: Any

    #: Half the extra radial top-hat, in cells. Zero for point sampling.
    half_width: float

    #: Number of output pixels.
    npix: int

    #: Number of angular sub-samples per output pixel.
    n_angular: int

    #: The interpolation order. Static: the gather unrolls over it.
    order: int


def make_shell_sampling(
    *,
    latitude: np.ndarray,
    longitude: np.ndarray,
    order: int = 1,
    radial_width: float = 1.0,
    coeval_cell_width: float = 1.0,
    n_radial_samples: int = 1,
) -> ShellSampling:
    """Build the radius-independent half of a shell's geometry.

    The arguments mirror :func:`~cosmotile.make_lightcone_slice_interpolator`, minus
    everything that varies from shell to shell (radius, rotation, origin), so that one
    of these serves a whole lightcone.

    Parameters
    ----------
    latitude, longitude
        Angular coordinates in radians. Either 1D, one per output pixel, or 2D with
        shape ``(n_angular, npix)`` to average over sub-samples of each pixel -- see
        :func:`~cosmotile.healpix_subpixel_lonlat`.
    order
        Interpolation order, in the range 0-5.
    radial_width
        The **total** radial top-hat the output should carry, in cells. The box already
        supplies ``coeval_cell_width``, so only the remainder is applied; see
        :func:`~cosmotile.residual_radial_width`.
    coeval_cell_width
        The radial top-hat the box already carries, in cells.
    n_radial_samples
        Number of Gauss-Legendre nodes across the residual width.

    Returns
    -------
    sampling
        Pass to :func:`cosmotile.jax.shell`, with a radius.
    """
    from ._geometry import radial_quadrature, residual_radial_width, unit_vectors
    from ._spline import MAX_ORDER

    if not isinstance(order, int):
        raise TypeError("order must be an integer")
    if order < 0 or order > MAX_ORDER:
        raise ValueError(f"order must be in the range 0-{MAX_ORDER}")
    if latitude.shape != longitude.shape:
        raise ValueError("latitude and longitude must have the same shape")
    if latitude.ndim > 2:
        raise ValueError("latitude and longitude must be 1D or 2D arrays")
    if n_radial_samples < 1:
        raise ValueError("n_radial_samples must be at least 1")
    if coeval_cell_width < 0:
        raise ValueError("coeval_cell_width must be non-negative")
    if radial_width < coeval_cell_width:
        raise ValueError(
            "radial_width must be at least coeval_cell_width: averaging cannot sharpen, "
            "so the output cannot carry a narrower window than the box it came from"
        )

    n_angular = latitude.shape[0] if latitude.ndim == 2 else 1
    npix = latitude.size // n_angular
    extra_width = residual_radial_width(radial_width, coeval_cell_width)
    nodes, node_weights = radial_quadrature(extra_width, n_radial_samples)

    return ShellSampling(
        directions=unit_vectors(latitude, longitude),
        nodes=nodes,
        node_weights=node_weights,
        half_width=0.5 * extra_width,
        npix=npix,
        n_angular=n_angular,
        order=order,
    )
