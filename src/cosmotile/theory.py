r"""Closed-form predictions for the statistics of a tiled shell.

``cosmotile`` cuts a spherical shell out of a periodically-tiled coeval box. Given the
three-dimensional power spectrum of the coeval box, the angular power spectrum of the
shell is fixed, and this module evaluates it.

Three functions are provided.

:func:`continuum_angular_power`
    What the shell *would* have if the box were infinite,

    .. math::

        C_\ell = \frac{2}{\pi} \int \mathrm{d}k\, k^2 P(k) j_\ell^2(kr).

:func:`discrete_angular_power`
    What a periodic box of finite size actually gives, since it contains only the
    discrete modes :math:`\mathbf{k} = 2\pi\mathbf{j}/L`,

    .. math::

        C_\ell = \frac{4\pi}{V} \sum_{\mathbf{k}} P(k) j_\ell^2(kr).

:func:`interpolation_window`
    The Fourier-space response :math:`W(\mathbf{k})` of the spline interpolation that
    ``cosmotile`` uses to reconstruct the field between grid points, which multiplies
    the box modes before they are summed.

The difference between the first two is the power a finite box is missing; the third is
the power the interpolation suppresses at the small-scale end. Together they bracket the
window of validity documented in :doc:`accuracy`.

This is also how to check that ``cosmotile`` is working. Measure the angular power
spectrum of a shell you have actually tiled -- one realisation, or better an average over
several -- and compare it against :func:`discrete_angular_power` with
:func:`interpolation_window` squared folded into the weights. That is the prediction the
tiling is supposed to reproduce, and inside the window of validity it does so to a few
percent. Comparing instead against :func:`continuum_angular_power` mixes in the finite
box's missing power, and comparing against ``P(l / r) / r**2`` is the Limber
approximation, which a geometrically thin shell does not have.

Notes
-----
Everything here works in *cell units*: the coeval box is ``n`` cells on a side, the cell
size is unity, so the box length is ``L = n``, the volume is ``V = n**3``, wavenumbers
are ``k = 2 pi j / n`` and distances to shells are in cells.

The Fourier convention is

.. math::

    \delta(\mathbf{x}) = \sum_{\mathbf{k}} \delta_{\mathbf{k}}
                          e^{i \mathbf{k}\cdot\mathbf{x}},
    \qquad
    \langle |\delta_{\mathbf{k}}|^2 \rangle = P(k) / V,

which is the convention in which the angular power spectrum of a thin shell takes the
familiar form above. It is also ``powerbox``'s default (``a = b = 1``,
``vol_normalised_power=True``) once ``boxlength`` is given in cells.

These are exact statements about a geometrically thin shell; none of them is the Limber
approximation, which has no thin-shell limit because it needs a radial kernel of finite
width to integrate over.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from scipy.integrate import trapezoid
from scipy.interpolate import BSpline
from scipy.special import spherical_jn

__all__ = [
    "continuum_angular_power",
    "discrete_angular_power",
    "interpolation_window",
]

_MAX_ORDER = 5

# Bins per oscillation of j_ell^2, and a floor for shells so close that the criterion
# asks for almost none. See the ``nbin`` discussion in discrete_angular_power.
_BINS_PER_OSCILLATION = 16
_MIN_BINS = 100
_MIN_RADIAL_NODES = 8


def _cardinal_bspline_at_integers(order: int) -> np.ndarray:
    """Sample the centred cardinal B-spline of the given order at integer offsets.

    Returns ``beta(0), beta(1), ..., beta(order // 2)``; the kernel is symmetric, and
    it vanishes at every larger integer.

    Parameters
    ----------
    order
        The spline order (degree).

    Returns
    -------
    samples
        The non-zero, non-negative-offset integer samples of the kernel.
    """
    knots = np.arange(order + 2) - (order + 1) / 2
    kernel = BSpline.basis_element(knots, extrapolate=False)
    return np.asarray(kernel(np.arange(order // 2 + 1, dtype=float)), dtype=float)


def _interpolation_window_1d(k: np.ndarray, order: int) -> np.ndarray:
    """Fourier response of one dimension of order-``order`` spline interpolation.

    Parameters
    ----------
    k
        Angular wavenumbers, in inverse cell units.
    order
        The spline order.

    Returns
    -------
    window
        The response, of the same shape as ``k``.
    """
    # The continuous kernel is the order-``order`` B-spline, whose transform is
    # sinc^(order + 1)(k / 2). np.sinc(y) = sin(pi y) / (pi y), so sin(k/2) / (k/2) is
    # np.sinc(k / (2 pi)).
    response = np.sinc(k / (2 * np.pi)) ** (order + 1)

    # Orders above 1 are not interpolating on their own: cosmotile pre-filters the box
    # so that the reconstruction passes through the samples, and that pre-filter is the
    # (circular) inverse of the kernel sampled on the grid. Its Fourier response is the
    # discrete-time transform of those samples, which divides out here. For orders 0
    # and 1 the samples are a delta function and this factor is identically one.
    samples = _cardinal_bspline_at_integers(order)
    prefilter = np.full(np.shape(k), samples[0], dtype=float)
    for offset, value in enumerate(samples[1:], start=1):
        prefilter = prefilter + 2 * value * np.cos(offset * k)

    return response / prefilter


def interpolation_window(kvec: Sequence[np.ndarray], order: int = 1) -> np.ndarray:
    r"""Fourier-space response of ``cosmotile``'s spline interpolation.

    Tiling reconstructs a continuous field from grid samples, and the reconstruction
    kernel suppresses power. A mode :math:`\mathbf{k}` of the coeval box therefore
    appears in the interpolated field multiplied by

    .. math::

        W(\mathbf{k}) = \prod_i \frac{\mathrm{sinc}^{p+1}(k_i / 2)}{b_p(k_i)},

    for spline order :math:`p`, where :math:`b_p` is the discrete-time transform of the
    B-spline sampled on the grid. For the default trilinear interpolation
    (``order=1``) :math:`b_p \equiv 1` and this reduces to the familiar
    :math:`\prod_i \mathrm{sinc}^2(k_i / 2)`.

    This is the response of the field *itself*, so the power spectrum is suppressed by
    ``W**2``.

    Parameters
    ----------
    kvec
        One array of angular wavenumbers per dimension, in inverse cell units. These
        need only be mutually broadcastable, so the ``(n,1,1)``, ``(1,n,1)``,
        ``(1,1,n)`` components of an FFT mode grid may be passed directly without
        materialising three full ``n**3`` arrays.
    order
        The spline order used for the interpolation, in the range 0-5. This must match
        the ``interpolation_order`` passed to
        :func:`~cosmotile.make_lightcone_slice_interpolator`.

    Returns
    -------
    window
        The response, broadcast over every element of ``kvec``.

    Raises
    ------
    ValueError
        If ``order`` is outside the range 0-5.

    Examples
    --------
    >>> import numpy as np
    >>> from cosmotile.theory import interpolation_window
    >>> k = 2 * np.pi * np.fft.fftfreq(4)
    >>> kvec = (k[:, None, None], k[None, :, None], k[None, None, :])
    >>> float(interpolation_window(kvec, order=1)[0, 0, 0])
    1.0
    """
    if order < 0 or order > _MAX_ORDER:
        raise ValueError(f"order must be in the range 0-{_MAX_ORDER}")

    window = np.asarray(1.0)
    for k in kvec:
        window = window * _interpolation_window_1d(np.asarray(k, dtype=float), order)
    return window


def discrete_angular_power(
    kmag: np.ndarray,
    weight: np.ndarray,
    volume: float,
    radius: float,
    ells: np.ndarray,
    nbin: int | None = None,
    *,
    radial_width: float = 0.0,
    n_radial_nodes: int | None = None,
) -> np.ndarray:
    r"""Evaluate the angular power spectrum of a shell through a periodic box.

    A periodic box contains only the discrete modes ``k = 2 pi j / L``, so the exact
    prediction for a shell cut through a tiled box is the mode sum

    .. math::

        C_\ell = \frac{4\pi}{V} \sum_{\mathbf{k}} P(k) j_\ell^2(kr),

    which is what this function evaluates. It agrees with the continuum integral of
    :func:`continuum_angular_power` in the limit of many modes; the sum is the right
    thing to compare a tiled box against, because it automatically encodes the power
    missing below the box fundamental.

    Parameters
    ----------
    kmag
        Magnitude of every mode to include, as a flat array, in inverse cell units.
        Exclude the ``k = 0`` mode.
    weight
        The weight ``P(k)`` carried by each mode, flat and matching ``kmag``. Multiply
        in :func:`interpolation_window` squared to predict the power of an interpolated
        shell rather than of the underlying field, and pass
        ``V * |delta_k|^2`` instead of the ensemble ``P(k)`` to predict a particular
        realisation.
    volume
        The box volume, in cells cubed.
    radius
        Shell radius, in cells.
    ells
        Multipoles at which to evaluate.
    radial_width
        Width of the radial top-hat the shell was averaged over, in cells; zero (the
        default) for a geometrically thin shell. Averaging turns the shell into a
        projection with a normalised radial kernel ``q(r)``, so ``j_l(kr)`` is replaced
        by its average over that kernel,

        .. math::

            C_\ell = \frac{4\pi}{V} \sum_{\mathbf{k}} P(k)
                     \left| \int \mathrm{d}r \, q(r) \, j_\ell(kr) \right|^2 ,

        with ``q`` proportional to ``r^2`` across the window, matching what
        :func:`cosmotile.make_lightcone_slice_interpolator` computes. This does *not*
        make Limber's approximation applicable: that needs ``dr >> r / l``, which one
        slice is nowhere near.

        Pass the top-hat that was actually applied. Where the box carries a cell window
        of its own the interpolator applies only the remainder, so that is
        ``cosmotile.residual_radial_width(radial_width, coeval_cell_width)`` rather than
        the ``radial_width`` you asked it for.
    n_radial_nodes
        Gauss-Legendre nodes used for the radial average, or ``None`` (the default) to
        choose from the data. ``j_ell(kr)`` runs through about ``k w / 2 pi`` periods
        across a window of width ``w``, and Gauss-Legendre integrates a polynomial of
        degree ``2n - 1`` exactly, so the nodes needed grow with ``k_max w``. The
        default takes the largest ``|k|`` present with a margin, which reproduces a
        far denser rule to round-off while staying much cheaper.
    nbin
        Number of ``|k|`` bins used to compress the mode sum, or ``None`` (the default)
        to choose it from ``radius``. ``j_ell`` depends on the modes only through
        ``|k|``, so they may be pre-summed in bins of ``|k|``, which turns an
        ``n**3``-term sum into an ``nbin``-term one.

        That compression is only lossless while ``j_ell^2(kr)`` is effectively constant
        across a bin, and it oscillates with period ``pi / radius`` in ``k``. The bins
        must therefore be narrower than that, so the number needed *grows with the
        shell radius*. The default places 16 bins per oscillation, which reproduces the
        unbinned sum to a few parts in ``10**4`` -- and exactly, once the bins are fine
        enough to separate the box's distinct ``|k|`` values, which for a shell beyond a
        few tens of cells they are. A fixed ``nbin`` that ignores ``radius`` does not:
        300 bins are ample at a radius of 40 cells and wrong by tens of percent at
        200.

    Returns
    -------
    cl
        The angular power spectrum, one value per entry of ``ells``.

    See Also
    --------
    continuum_angular_power : The infinite-box limit of this sum.
    """
    lo, hi = kmag.min() * 0.999, kmag.max() * 1.001
    if nbin is None:
        oscillations = (hi - lo) * radius / np.pi
        nbin = max(_MIN_BINS, int(np.ceil(_BINS_PER_OSCILLATION * oscillations)))
    edges = np.linspace(lo, hi, nbin + 1)

    # One pass gives both the summed weight in each bin and its weighted mean |k|.
    index = np.digitize(kmag, edges)
    summed = np.bincount(index, weights=weight, minlength=len(edges) + 1)[1:-1]
    moment = np.bincount(index, weights=weight * kmag, minlength=len(edges) + 1)[1:-1]

    good = summed > 0  # empty bins contribute nothing and are dropped
    wbin, kbin = summed[good], moment[good] / summed[good]

    if radial_width == 0:
        return np.array(
            [
                (4 * np.pi / volume) * np.sum(wbin * spherical_jn(int(ell), kbin * radius) ** 2)
                for ell in ells
            ]
        )

    # The r^2 volume element is what makes the average the mean over a shell of finite
    # thickness, so it is in the kernel here exactly as it is in the tiling.
    if n_radial_nodes is None:
        n_radial_nodes = _MIN_RADIAL_NODES + int(np.ceil(kbin.max() * radial_width / 2))
    nodes, node_weights = np.polynomial.legendre.leggauss(n_radial_nodes)
    radii = radius + 0.5 * radial_width * nodes
    kernel = node_weights * radii**2
    kernel = kernel / kernel.sum()
    kr = np.outer(radii, kbin)

    return np.array(
        [
            (4 * np.pi / volume)
            * np.sum(wbin * np.sum(kernel[:, None] * spherical_jn(int(ell), kr), axis=0) ** 2)
            for ell in ells
        ]
    )


def continuum_angular_power(
    pk: Callable[[np.ndarray], np.ndarray],
    radius: float,
    ells: np.ndarray,
    kmax: float,
    nk: int = 40000,
) -> np.ndarray:
    r"""Evaluate the continuum integral ``C_l = (2/pi) int dk k^2 P(k) j_l^2(kr)``.

    This is the infinite-box limit of :func:`discrete_angular_power`: what a tiled box
    *would* give if it contained every mode. The difference between the two is exactly
    the large-scale power a finite box is missing.

    Parameters
    ----------
    pk
        The three-dimensional power spectrum, a callable of the angular wavenumber
        ``k`` in inverse cell units.
    radius
        Shell radius, in cells.
    ells
        Multipoles at which to evaluate.
    kmax
        Upper limit of the integral, in inverse cell units. For a band-limited spectrum
        set this to the cut-off; otherwise take it well above ``max(ells) / radius``.
    nk
        Number of trapezoidal integration points. The integrand oscillates like
        ``j_ell^2``, so this must comfortably resolve ``kmax * radius / pi`` periods.

    Returns
    -------
    cl
        The angular power spectrum, one value per entry of ``ells``.

    See Also
    --------
    discrete_angular_power : The finite periodic-box mode sum.
    """
    k = np.linspace(kmax / nk, kmax, nk)
    integrand = k**2 * pk(k)
    return np.array(
        [
            (2 / np.pi) * trapezoid(integrand * spherical_jn(int(ell), k * radius) ** 2, k)
            for ell in ells
        ]
    )
