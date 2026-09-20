r"""A JAX backend for ``cosmotile``.

Import this only if you want it: ``import cosmotile.jax`` requires ``jax``, which is an
optional dependency (``pip install cosmotile[jax]``). The NumPy API in
:mod:`cosmotile` never imports it.

This is deliberately **not** a mirror of the NumPy API. The functions here are pure and
array-in, array-out, so that you can wrap them in :func:`jax.jit`, :func:`jax.vmap` or
:func:`jax.grad` yourself. The NumPy API's lazy iterators, ``astropy`` units and
value-dependent validation are all useful there and impossible here, so rather than
reproduce shapes that cannot be traced, this module offers one function per operation:

:func:`prefilter_coeval`
    Box to B-spline coefficients, as an FFT deconvolution.
:func:`shell`
    One spherical shell, from a radius and a :class:`~cosmotile._plan.ShellSampling`.
:func:`shell_from_coordinates`
    The primitive, when you already have coordinates.
:func:`lightcone_scan`
    Fold a function over many shells without materialising the lightcone.
:func:`apply_rsds`
    Redshift-space distortions, on a plan built by :func:`make_rsd_plan`.

Why it is worth it
------------------

**Gradients.** The interpolation is linear in the box values, so the derivative of a
shell with respect to the coeval box is exactly the transpose gather -- a scatter-add.
JAX derives it from the forward pass; nothing here hand-writes a VJP. That is what lets
``cosmotile`` sit inside a differentiable forward model.

**Speed.** The gather is memory-latency bound, which is the workload a GPU is best at.
Measured on a 256-cubed box at ``nside=256``, order 3: 3.5 Mpix/s through ``scipy``,
30 Mpix/s through the NumPy backend's own parallel kernel (which is the default whenever
``numba`` is installed, and is what this has to beat), and 161 Mpix/s in double precision
or 544 Mpix/s in single through this one.

What to watch out for
---------------------

*Nothing here is decorated with* :func:`jax.jit` *at the top level.* Wrap the call site
yourself; pre-jitting a library primitive compiles it twice when you jit the caller.

*Use* :func:`lightcone_scan`\\ *, not* :func:`jax.vmap`\\ *, over shells.* A thousand
shells at ``nside=256`` is 3.1 GB of output and 19 GB of coordinates if they are all
built at once; scanning builds one shell at a time from shared unit vectors.

*JAX defaults to single precision.* The gather is fine there -- the coordinate is split
into an integer base and a fraction in ``[0, 1)`` on the host, so it never carries a
large radius through a ``float32`` -- but expect around ``2e-6`` relative rather than
``6e-15``. Use :func:`jax.enable_x64` if you need double. This module never
changes that setting for you: it is global process state, and flipping it would change
the numerics of every other JAX library you have loaded.

*Gradients with respect to geometry need order 3 or more.* Order 0 has zero gradient
everywhere and order 1 is only piecewise linear. Gradients with respect to the *box* are
exact at every order.
"""

from __future__ import annotations

from typing import Any

import jax

from .._plan import PrefilteredCoeval, ShellSampling, make_shell_sampling
from .._rsd import RsdPlan, make_rsd_plan
from ._interp import shell, shell_from_coordinates
from ._prefilter import prefilter_coeval
from ._rsd import apply_rsds

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
    coeval: Any,
    sampling: ShellSampling,
    radii: Any,
    body: Any,
    init: Any,
    *,
    remat: bool = False,
    **kwargs: Any,
) -> Any:
    """Fold a function over the shells of a lightcone, one shell at a time.

    ``body(carry, shell) -> carry`` is applied to each shell in turn under
    :func:`jax.lax.scan`, so only one shell is ever live. That is the difference between
    a lightcone that fits on a GPU and one that does not: a thousand shells at
    ``nside=256`` is 3.1 GB of output, and building their coordinates with
    :func:`jax.vmap` instead would need a further 19 GB.

    Carrying the result rather than collecting the shells is the point. If ``body``
    returns the shells themselves, reverse-mode needs every shell's cotangent live and
    the saving is gone -- fold them into whatever statistic you actually want.

    Parameters
    ----------
    coeval
        A :class:`~cosmotile.PrefilteredCoeval` for ``order >= 2``, else a 3D array.
    sampling
        The shared, radius-independent geometry.
    radii
        ``(nshell,)`` shell radii in cells.
    body
        ``(carry, shell) -> carry``.
    init
        The initial carry.
    remat
        Re-materialise each step under :func:`jax.checkpoint` instead of saving its
        residuals. Rarely worth it for the interpolation itself, whose residuals are
        nearly free because the map is linear with constant indices -- turn it on when
        ``body`` is the expensive part.
    **kwargs
        Passed to :func:`shell` (``rotation``, ``origin``).

    Returns
    -------
    carry
        The final carry.
    """

    def step(carry: Any, radius: Any) -> tuple[Any, None]:
        return body(carry, shell(coeval, sampling, radius, **kwargs)), None

    stepper = jax.checkpoint(step) if remat else step
    carry, _ = jax.lax.scan(stepper, init, radii)
    return carry
