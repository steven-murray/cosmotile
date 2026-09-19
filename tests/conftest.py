r"""Shared machinery for the physical validation tests.

All of these helpers work in *cell units*: the coeval box is ``n`` cells on a side, the
cell size is unity, and so the box length is ``L = n`` and wavenumbers are
``k = 2 pi j / n``. Distances to shells are likewise in cells.

The Fourier convention used throughout (and matched by :func:`gaussian_box`) is

.. math::

    \delta(\mathbf{x}) = \sum_{\mathbf{k}} \delta_{\mathbf{k}}
                          e^{i \mathbf{k}\cdot\mathbf{x}},
    \qquad
    \langle |\delta_{\mathbf{k}}|^2 \rangle = P(k) / V,

with :math:`V = L^3`. This is the same convention as :mod:`cosmotile.theory`, which
ships the closed-form angular-power predictions these tests are compared against.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
import pytest


# ---------------------------------------------------------------------------------
# Random fields
# ---------------------------------------------------------------------------------
def _angular_average() -> Callable[..., tuple]:
    """Return ``powerbox.tools.angular_average``, skipping the test if it is absent.

    Imported lazily so that this module still loads -- and the tests that need neither
    ``powerbox`` nor ``healpy`` still run -- when the optional test extras are missing.
    """
    return pytest.importorskip("powerbox.tools").angular_average


def gaussian_box(n: int, pk: Callable[[np.ndarray], np.ndarray], seed: int) -> np.ndarray:
    """Make a periodic Gaussian random field of ``n**3`` cells with power spectrum ``pk``.

    ``pk`` is a function of the angular wavenumber ``k`` in inverse cell units, and the
    returned field satisfies ``<|delta_k|^2> = pk(k) / n**3`` in the convention
    documented at the top of this module. ``powerbox``'s defaults (``a=b=1``,
    ``vol_normalised_power=True``) already use exactly this convention once
    ``boxlength`` is given in cells.
    """
    pbx = pytest.importorskip("powerbox")
    return np.asarray(pbx.PowerBox(N=n, dim=3, pk=pk, boxlength=float(n), seed=seed).delta_x())


def band_limited_powerlaw(
    index: float = -2.0, kcut: float = np.pi / 4, amplitude: float = 1.0
) -> Callable[[np.ndarray], np.ndarray]:
    """Build a power law ``P(k) = A k**index``, truncated above ``kcut``.

    The truncation is what makes the angular-power comparison clean: it keeps all of the
    input power well below the Nyquist wavenumber ``pi``, so the aliased copies of each
    mode (which the interpolation unavoidably introduces at ``|k + 2 pi m|``) land far
    above the multipoles being tested.
    """

    def pk(k: np.ndarray) -> np.ndarray:
        k = np.asarray(k, dtype=float)
        inband = (k > 0) & (k <= kcut)
        return np.where(inband, amplitude * np.where(inband, k, 1.0) ** index, 0.0)

    return pk


# ---------------------------------------------------------------------------------
# Mode grids
# ---------------------------------------------------------------------------------
def mode_grid(n: int) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return ``(k_magnitude, (kx, ky, kz))`` for the FFT modes of an ``n**3`` box.

    The three components are returned *broadcastable* rather than materialised -- shapes
    ``(n,1,1)``, ``(1,n,1)``, ``(1,1,n)`` -- so only ``k_magnitude`` is a full ``n**3``
    array. A materialised ``meshgrid`` would quadruple the memory for no benefit, which
    matters as soon as anyone reaches for a finer reference grid.
    """
    k1 = 2 * np.pi * np.fft.fftfreq(n)
    kvec = (k1[:, None, None], k1[None, :, None], k1[None, None, :])
    return np.sqrt(kvec[0] ** 2 + kvec[1] ** 2 + kvec[2] ** 2), kvec


# ---------------------------------------------------------------------------------
# Band powers
# ---------------------------------------------------------------------------------
def band_average(cl: np.ndarray, lo: int, hi: int) -> float:
    """``(2l+1)``-weighted mean of ``cl`` over the multipole band ``[lo, hi)``."""
    weights = 2 * np.arange(len(cl)) + 1.0
    return float(np.sum(weights[lo:hi] * cl[lo:hi]) / np.sum(weights[lo:hi]))


# ---------------------------------------------------------------------------------
# Angular coordinates
# ---------------------------------------------------------------------------------
def unit_vectors_to_lonlat(vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert unit vectors of shape ``(n, 3)`` to ``(latitude, longitude)`` in radians."""
    lat = np.arcsin(np.clip(vec[:, 2], -1.0, 1.0))
    lon = np.mod(np.arctan2(vec[:, 1], vec[:, 0]), 2 * np.pi)
    return lat, lon


def random_directions(n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` isotropically-distributed unit vectors, shape ``(n, 3)``."""
    vec = rng.normal(size=(n, 3))
    return vec / np.linalg.norm(vec, axis=1)[:, None]


def pairs_at_separation(
    n: int, theta: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """``n`` isotropic pairs of unit vectors separated by exactly the angle ``theta``."""
    first = random_directions(n, rng)

    # A uniformly-random direction perpendicular to each of `first`.
    perp = rng.normal(size=(n, 3))
    perp -= np.sum(perp * first, axis=1)[:, None] * first
    perp /= np.linalg.norm(perp, axis=1)[:, None]

    return first, np.cos(theta) * first + np.sin(theta) * perp


# ---------------------------------------------------------------------------------
# Real-space statistics of the box
# ---------------------------------------------------------------------------------
def box_correlation_function(box: np.ndarray, rbins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Isotropic two-point correlation function of a periodic box, exactly.

    Computed from the full 3D correlation function (the inverse transform of the squared
    modulus of the box's Fourier transform), so it is the *realisation's* own
    correlation function over all pairs -- not an ensemble average and not an estimate.
    """
    n = box.shape[0]
    xi3d = np.real(np.fft.ifftn(np.abs(np.fft.fftn(box)) ** 2)) / n**3

    lag = np.fft.fftfreq(n) * n
    sep = np.sqrt(sum(g**2 for g in np.meshgrid(lag, lag, lag, indexing="ij")))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # empty bins are expected and dropped below
        xi, rbar, _, _ = _angular_average()(field=xi3d, coords=sep, bins=rbins)

    good = np.isfinite(xi) & np.isfinite(rbar)
    return rbar[good], xi[good]
