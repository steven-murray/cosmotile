"""A JAX backend for ``cosmotile``.

Requires ``jax``, an optional dependency: ``pip install cosmotile[jax]``. The NumPy API
in :mod:`cosmotile` never imports it.

The functions here take arrays and return arrays, so you can wrap them in
:func:`jax.jit`, :func:`jax.vmap` or :func:`jax.grad`. They compute the same thing as the
NumPy API, which returns lazy iterators and carries ``astropy`` units -- neither of which
survives being traced -- so this is a smaller, flatter surface rather than a mirror of it.

:func:`prefilter_coeval`
    Convert a box to B-spline coefficients. Needed once, before tiling at order 2 or above.
:func:`shell`
    Interpolate a box onto one spherical shell.
:func:`shell_from_coordinates`
    The same, when you already have the pixel coordinates.
:func:`lightcone_scan`
    Build many shells one at a time, accumulating a result instead of keeping them all.
:func:`apply_rsds`
    Apply redshift-space distortions.

See :doc:`the backend guide </jax-backend>` for what it is for, what you can
differentiate, and how to keep a large lightcone in memory.
"""

# Needed despite Python 3.11: the documentation build mocks `jax` (see docs/conf.py),
# so `jax.Array` is a Mock there and an eagerly-evaluated `Array | ...` would fail.
from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

import jax
from jax import Array

from .._plan import PrefilteredCoeval, ShellSampling, make_shell_sampling
from .._rsd import RsdPlan, make_rsd_plan
from ._interp import shell, shell_from_coordinates
from ._prefilter import prefilter_coeval
from ._rsd import apply_rsds

#: Whatever :func:`lightcone_scan` is accumulating -- a scalar, an array, or a pytree.
Carry = TypeVar("Carry")

__all__ = [
    "PrefilteredCoeval",
    "RsdPlan",
    "ShellSampling",
    "apply_rsds",
    "lightcone_scan",
    "make_rsd_plan",
    "make_shell_sampling",
    "prefilter_coeval",
    "shell",
    "shell_from_coordinates",
]


jax.tree_util.register_dataclass(
    PrefilteredCoeval, data_fields=["coefficients"], meta_fields=["order"]
)
jax.tree_util.register_dataclass(
    RsdPlan,
    data_fields=[
        "fine_widths",
        "refine_index",
        "source_mask",
        "interp_index",
        "interp_weight",
        "rebin_index",
        "rebin_frac",
        "out_widths",
        "fine_cumulative",
    ],
    meta_fields=["nslice", "n_near", "n_far", "n_subcells", "outside"],
)
jax.tree_util.register_dataclass(
    ShellSampling,
    data_fields=["directions", "nodes", "node_weights"],
    meta_fields=["half_width", "npix", "n_angular", "order"],
)


def lightcone_scan(
    coeval: Array | PrefilteredCoeval,
    sampling: ShellSampling,
    radii: Array,
    body: Callable[[Carry, Array], Carry],
    init: Carry,
    *,
    remat: bool = False,
    **kwargs: Any,
) -> Carry:
    """Build the shells of a lightcone one at a time, accumulating a result.

    A whole lightcone rarely fits in GPU memory -- a thousand shells at ``nside=256`` is
    3.1 GB, and building all their coordinates at once needs a further 19 GB. This makes
    one shell at a time and hands it to ``body``, which combines it with a running
    result and returns the updated one. Only a single shell is ever in memory.

    Use it when what you want out is a *summary* of the lightcone rather than the
    lightcone itself -- a likelihood, a power spectrum, a sum of squares, the maximum
    brightness -- which is the usual case when you are differentiating through it. If
    you want the shells themselves and they fit, just call :func:`shell` in a loop.

    Parameters
    ----------
    coeval
        A :class:`~cosmotile.PrefilteredCoeval` for ``order >= 2``, else a 3D array.
    sampling
        The shared, radius-independent geometry.
    radii
        ``(nshell,)`` shell radii in cells.
    body
        Called as ``body(result, shell)`` for each shell, and must return the updated
        result. Both must have the same shape and dtype every time.
    init
        The starting value of the result, before any shell has been seen.
    remat
        Recompute each shell during the backward pass rather than storing it. Only worth
        it when ``body`` is expensive; the interpolation itself stores almost nothing.
    **kwargs
        Passed to :func:`shell` -- ``rotation`` and ``origin``.

    Returns
    -------
    result
        What ``body`` returned after the last shell.

    Examples
    --------
    The mean squared brightness of every shell in a lightcone, differentiated with
    respect to the box it came from::

        import jax, jax.numpy as jnp
        from cosmotile import jax as cjax

        radii = jnp.linspace(100.0, 400.0, 1000)

        def total_power(box):
            coeff = cjax.prefilter_coeval(box, order=3)
            return cjax.lightcone_scan(
                coeff, sampling, radii, lambda acc, shell: acc + jnp.mean(shell**2), 0.0
            )

        gradient = jax.grad(total_power)(box)
    """

    def step(carry: Carry, radius: Array) -> tuple[Carry, None]:
        return body(carry, shell(coeval, sampling, radius, **kwargs)), None

    stepper = jax.checkpoint(step) if remat else step
    carry, _ = jax.lax.scan(stepper, init, radii)
    return carry
