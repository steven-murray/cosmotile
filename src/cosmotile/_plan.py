"""Backend-agnostic containers for work that does not depend on field values.

These are plain frozen dataclasses. They hold no arrays of their own beyond what they
are given, import neither ``jax`` nor ``scipy``, and are registered as JAX pytrees by
:mod:`cosmotile.jax` when that backend is imported -- which is why the fields that must
stay static under :func:`jax.jit` (orders, counts) are kept separate from the fields
that hold data.
"""

from __future__ import annotations

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
