"""The periodic spline pre-filter, as an FFT-domain deconvolution."""

import functools

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from .._plan import PrefilteredCoeval
from .._spline import MAX_ORDER, bspline_dtft


@functools.partial(jax.jit, static_argnames=("order",))
def _deconvolve(coeval: Array, order: int) -> Array:
    """Divide the spectrum by ``b_p`` along every axis."""
    transformed = jnp.fft.rfftn(coeval)
    for axis, size in enumerate(coeval.shape):
        last = axis == coeval.ndim - 1
        freq = jnp.fft.rfftfreq(size) if last else jnp.fft.fftfreq(size)
        broadcast = [1] * coeval.ndim
        broadcast[axis] = -1
        response = bspline_dtft(2 * jnp.pi * freq, order, xp=jnp)
        transformed = transformed / response.reshape(broadcast)
    return jnp.fft.irfftn(transformed, s=coeval.shape, axes=tuple(range(coeval.ndim)))


def prefilter_coeval(coeval: ArrayLike, order: int) -> PrefilteredCoeval:
    r"""Convert a coeval box to B-spline coefficients, once, for re-use across shells.

    Interpolating at ``order >= 2`` requires the box to be converted to B-spline
    coefficients first, so that the reconstruction passes through the grid values rather
    than merely being attracted to them.

    ``scipy`` does this with a recursive IIR filter, which is inherently sequential and
    has no JAX equivalent. Under the periodic boundary condition ``cosmotile`` uses,
    though, the filter is a *circular* deconvolution, so it is exactly a division by the
    Fourier response of the sampled kernel,

    .. math:: c = \mathcal{F}^{-1}\left[ \frac{\mathcal{F}[g]}{\prod_i b_p(k_i)} \right],

    with :math:`b_p` from :func:`~cosmotile._spline.bspline_dtft`. This is not an
    approximation to ``scipy``'s filter: it agrees with it to roundoff (about
    ``7e-15`` relative at ``384^3`` in double precision), and being an FFT it is very
    much faster on a GPU -- around fifteen to thirty times ``scipy``'s filter on a CPU.

    Unlike :func:`cosmotile.prefilter_coeval`, this is not optional: the JAX backend
    will not filter a box for you inside :func:`~cosmotile.jax.shell`, because that
    would redo the filter for every shell of a lightcone.

    Parameters
    ----------
    coeval
        The box to filter. Any array JAX accepts.
    order
        The interpolation order the box is being prepared for, in the range 0-5. Orders
        0 and 1 use interpolating kernels and need no filter, so for them this only
        tags the box.

    Returns
    -------
    prefiltered
        The coefficients, tagged with ``order``. Pass to :func:`~cosmotile.jax.shell`.

    Notes
    -----
    In single precision the deconvolution is a *global* operation, so its error is not
    the local error of the gather: expect a relative error around ``3e-6`` rather than
    the ``1e-15`` of double precision. See the precision discussion in the
    documentation.
    """
    if not isinstance(order, int):
        raise TypeError("order must be an integer")
    if order < 0 or order > MAX_ORDER:
        raise ValueError(f"order must be in the range 0-{MAX_ORDER}")

    array = jnp.asarray(coeval)
    if order > 1:
        array = _deconvolve(array, order)
    return PrefilteredCoeval(array, order)
