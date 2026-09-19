r"""Direct tests of the shipped theory module, :mod:`cosmotile.theory`.

These are unit tests of the predictions themselves: that the interpolation window really
is the Fourier response of what ``cosmotile`` does, that the binned mode sum reproduces
the sum it compresses, and that the continuum integral reproduces its closed form.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest
from scipy.special import spherical_jn

from cosmotile import _interpolate_coeval
from cosmotile.theory import (
    continuum_angular_power,
    discrete_angular_power,
    interpolation_window,
)

from .conftest import band_limited_powerlaw, mode_grid

ORDERS = (0, 1, 2, 3, 4, 5)


# ---------------------------------------------------------------------------------
# The interpolation window
# ---------------------------------------------------------------------------------
def measured_response(order: int, wavenumber: int, ncell: int = 16, subsample: int = 8) -> float:
    """Measure what ``cosmotile``'s interpolation does to a single Fourier mode.

    A box carrying the single mode ``cos(k x)`` is interpolated onto a grid ``subsample``
    times finer than the cells, along one axis. The reconstruction is a sum of the mode
    and its aliases at ``k + 2 pi m``, which the fine grid resolves separately, so the
    amplitude of the un-aliased component read off the FFT *is* the window function.
    """
    k = 2 * np.pi * wavenumber / ncell
    box = np.zeros((ncell, 2, 2))
    box[:] = np.cos(k * np.arange(ncell))[:, None, None]

    coords = np.zeros((3, ncell * subsample))
    coords[0] = np.arange(ncell * subsample) / subsample

    transform = np.fft.fft(_interpolate_coeval(box, coordinates=coords, order=order))
    return float(2 * transform[wavenumber].real / coords.shape[1])


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("wavenumber", [1, 3, 5])
def test_window_is_the_response_of_the_interpolation(order: int, wavenumber: int) -> None:
    """``interpolation_window`` must equal what ``cosmotile`` actually does to a mode.

    This is the whole point of the function, and it is the reason it lives in the
    package rather than in a test file: it describes ``cosmotile``'s own interpolation,
    pre-filter and all. Orders 0 and 1 use interpolating kernels directly; orders 2-5
    are pre-filtered, and the window has to divide out the pre-filter to match.

    The residual is the small alias contamination the finite sub-sampling folds back
    into the measured bin, which is largest for the bluntest kernels.
    """
    k = 2 * np.pi * wavenumber / 16
    predicted = float(interpolation_window([np.array(k)], order=order))
    assert predicted == pytest.approx(measured_response(order, wavenumber), rel=6e-3)


def test_window_is_unity_for_the_zero_mode() -> None:
    """Interpolation is exact for a constant field, at every order."""
    for order in ORDERS:
        window = interpolation_window([np.zeros(1)] * 3, order=order)
        np.testing.assert_allclose(window, 1.0)


def test_window_reduces_to_sinc_squared_at_order_one() -> None:
    """The default trilinear window is the familiar ``prod_i sinc^2(k_i / 2)``.

    The generalised form carries a pre-filter factor which must be identically one for
    the interpolating kernels (orders 0 and 1), or the default would silently change.
    """
    kmag, kvec = mode_grid(8)
    expected = np.ones_like(kmag)
    for k in kvec:
        expected = expected * np.sinc(k / (2 * np.pi)) ** 2

    np.testing.assert_allclose(interpolation_window(kvec), expected, rtol=1e-14)


def test_window_is_separable_across_dimensions() -> None:
    """The window is a product over dimensions, and takes any number of them."""
    k = np.linspace(0.0, np.pi, 17)
    one_d = interpolation_window([k], order=3)
    np.testing.assert_allclose(interpolation_window([k, k, k], order=3), one_d**3, rtol=1e-14)


def test_window_suppresses_less_at_higher_order() -> None:
    """Above order 1, each extra spline order reproduces a mode better.

    Order 0 is the exception and is excluded deliberately: nearest-neighbour has a
    *larger* amplitude response than trilinear, which is why it preserves the variance
    of the field. What it does not preserve is anything else -- the missing suppression
    reappears as aliasing.
    """
    k = np.array(np.pi / 2)
    responses = [float(interpolation_window([k], order=order)) for order in ORDERS[1:]]
    assert all(a < b for a, b in pairwise(responses))
    assert all(0 < r <= 1 for r in responses)


@pytest.mark.parametrize("order", [-1, 6])
def test_window_rejects_unsupported_orders(order: int) -> None:
    """Only the orders ``cosmotile`` can interpolate with are accepted."""
    with pytest.raises(ValueError, match="order must be in the range 0-5"):
        interpolation_window([np.zeros(1)], order=order)


# ---------------------------------------------------------------------------------
# The discrete mode sum
# ---------------------------------------------------------------------------------
def brute_force_mode_sum(
    kmag: np.ndarray, weight: np.ndarray, volume: float, radius: float, ells: np.ndarray
) -> np.ndarray:
    """``(4 pi / V) sum_k P(k) j_l^2(kr)``, summed mode by mode with no binning."""
    return np.array(
        [
            (4 * np.pi / volume) * np.sum(weight * spherical_jn(int(ell), kmag * radius) ** 2)
            for ell in ells
        ]
    )


def test_discrete_binning_is_exact_for_resolved_modes() -> None:
    """With every ``|k|`` in its own bin, the compression must be exact.

    ``discrete_angular_power`` pre-sums modes in fine bins of ``|k|`` because
    ``j_ell`` depends on nothing else. When the bins resolve the distinct ``|k|``
    values, that is an identity rather than an approximation -- and the empty bins in
    between must simply drop out.
    """
    kmag = np.array([0.4, 0.9, 1.7, 2.6])
    weight = np.array([3.0, 1.0, 0.5, 0.25])
    ells = np.arange(0, 12)

    np.testing.assert_allclose(
        discrete_angular_power(kmag, weight, 512.0, 6.0, ells),
        brute_force_mode_sum(kmag, weight, 512.0, 6.0, ells),
        rtol=1e-12,
    )


def grid_modes(ncell: int, order: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Flat ``(|k|, P(k) W(k)^2)`` for the non-zero FFT modes of an ``ncell**3`` box."""
    kmag, kvec = mode_grid(ncell)
    weight = band_limited_powerlaw(-2.0, np.pi / 4)(kmag) * interpolation_window(kvec, order) ** 2
    nonzero = kmag > 0
    return kmag[nonzero].ravel(), weight[nonzero].ravel()


@pytest.mark.parametrize("radius", [20.0, 40.0, 200.0])
def test_discrete_binning_is_accurate_on_a_real_mode_grid(radius: float) -> None:
    """On a full FFT mode grid the binning must reproduce the unbinned sum.

    The default bin count is chosen from ``radius`` precisely so that this holds at
    every radius. It is the one thing about the compression that can go wrong: bin
    widths that do not resolve the oscillation of ``j_ell^2(kr)`` average the Bessel
    function over the bin instead of evaluating it, which a fixed bin count silently
    starts doing as the shell is pushed outwards.
    """
    ncell = 48
    flat_k, flat_w = grid_modes(ncell)
    volume = float(ncell) ** 3
    ells = np.arange(2, min(int(0.8 * (np.pi / 4) * radius), 40))

    np.testing.assert_allclose(
        discrete_angular_power(flat_k, flat_w, volume, radius, ells),
        brute_force_mode_sum(flat_k, flat_w, volume, radius, ells),
        rtol=1e-3,
    )


def test_discrete_default_bin_count_beats_a_fixed_one() -> None:
    """A radius-independent bin count is not good enough, which is why the default is not.

    Pinned as a regression: 300 fixed bins -- the value this function shipped with
    before the default was made radius-aware -- are wrong by tens of percent for a shell
    at 200 cells, while the default is exact there.
    """
    ncell, radius = 64, 200.0
    flat_k, flat_w = grid_modes(ncell)
    volume = float(ncell) ** 3
    ells = np.arange(2, 40)

    exact = brute_force_mode_sum(flat_k, flat_w, volume, radius, ells)
    fixed = discrete_angular_power(flat_k, flat_w, volume, radius, ells, nbin=300)

    assert np.max(np.abs(fixed / exact - 1)) > 0.1
    np.testing.assert_allclose(
        discrete_angular_power(flat_k, flat_w, volume, radius, ells), exact, rtol=1e-3
    )


def test_discrete_approaches_the_continuum_for_a_large_box() -> None:
    """The mode sum tends to the continuum integral well above the box fundamental.

    This is the statement that makes the pair useful: the two differ only by the power
    a finite box is missing, so where the box is not missing any, they agree.

    Two conditions have to hold for that. The band must sit above the box fundamental
    (here it spans 2 to 6 times it), and the mode spacing ``2 pi / L`` must resolve the
    oscillation of ``j_ell^2(kr)``, whose period in ``k`` is ``pi / r`` -- that is,
    ``r`` must not much exceed ``L / 2``. Push the shell far beyond that and the sum
    starts to *alias* the Bessel oscillation instead of sampling it, and it becomes
    jagged around the integral even though both remain individually correct.
    """
    ncell, radius = 64, 20.0
    kcut = np.pi / 4
    pk = band_limited_powerlaw(-2.0, kcut)

    kmag, _ = mode_grid(ncell)
    nonzero = kmag > 0
    flat_k = kmag[nonzero].ravel()
    ells = np.arange(4, 12)

    discrete = discrete_angular_power(flat_k, pk(flat_k), float(ncell) ** 3, radius, ells)
    continuum = continuum_angular_power(pk, radius, ells, kcut)

    np.testing.assert_allclose(discrete, continuum, rtol=0.02)


# ---------------------------------------------------------------------------------
# The continuum integral
# ---------------------------------------------------------------------------------
def test_continuum_matches_its_closed_form() -> None:
    r"""For ``P(k) = A k^-2`` the integral is ``A / (r (2l + 1))``.

    Using :math:`\int_0^\infty j_\ell^2(x)\,\mathrm{d}x = \pi / [2(2\ell + 1)]`, with the
    correction for truncating the ``1/(2x^2)`` tail of :math:`j_\ell^2` at
    ``x = k_cut r``. This is the normalisation check for anyone comparing their own
    pipeline against the module.
    """
    amplitude, radius, kcut = 1.0, 1000.0, np.pi / 4
    pk = band_limited_powerlaw(-2.0, kcut, amplitude)
    ells = np.arange(2, 61)

    numeric = continuum_angular_power(pk, radius, ells, kcut)
    closed_form = (
        amplitude / (radius * (2 * ells + 1)) * (1 - (2 * ells + 1) / (np.pi * kcut * radius))
    )

    np.testing.assert_allclose(numeric, closed_form, rtol=2e-3)
