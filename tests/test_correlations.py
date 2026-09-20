"""Real-space two-point statistics of a tiled shell.

The angular power spectrum tests check the method in harmonic space against a formula
that has to be derived. These tests check it in configuration space against a statement
that needs no derivation at all: two points on the shell separated by a chord of length
``d`` are two points of the coeval box separated by ``d``, so their correlation *must*
be the box's own correlation function ``xi(d)``. Nothing about the shell geometry,
tiling or interpolation is allowed to change that.

Because ``xi`` here is computed exactly from the realisation (every pair in the box, via
FFT) rather than from an ensemble average, the tests stay at small separations, where
``xi`` is large compared with the noise.

That noise is dominated by *sample* variance, not by pair sub-sampling: a shell touches
only part of the box, so the pairs it offers are not a fair draw from the box's own
``xi``. Raising the pair count from thirty thousand to two hundred thousand leaves the
scatter unchanged at about 6% per (seed, separation), so these tests average over several
realisations instead, and their tolerances are set by that scatter.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.interpolate import interp1d

import cosmotile as cmt

from .conftest import (
    band_limited_powerlaw,
    box_correlation_function,
    gaussian_box,
    mode_grid,
    pairs_at_separation,
    random_directions,
    unit_vectors_to_lonlat,
)

NCELL = 64
KCUT = np.pi / 4
SEEDS = (0, 1, 2)
# The angular test averages over more realisations than the rest: its scatter is sample
# variance, so a handful of seeds leaves the mean itself uncertain at the several-percent
# level. See the module docstring.
ANGULAR_SEEDS = range(10)

# Separations at which xi is comfortably above the pair-sampling noise for this spectrum.
SEPARATIONS = (0.0, 1.0, 2.0, 4.0)


@pytest.fixture(scope="module")
def spectrum() -> object:
    """Return the band-limited power-law spectrum used throughout this module."""
    return band_limited_powerlaw(-2.0, KCUT)


def shell_values(box: np.ndarray, vectors: np.ndarray, radius: float, order: int = 1) -> np.ndarray:
    """Tile ``box`` onto the shell of the given radius, along the given sight-lines."""
    lat, lon = unit_vectors_to_lonlat(vectors)
    return next(
        cmt.make_lightcone_slice(
            coevals=box,
            latitude=lat,
            longitude=lon,
            distance_to_shell=radius,
            interpolation_order=order,
        )
    )


def exact_xi(box: np.ndarray) -> interp1d:
    """Interpolator over the box's exact isotropic correlation function."""
    sep, xi = box_correlation_function(box, np.arange(0.0, NCELL / 2, 0.5))
    return interp1d(sep, xi, kind="cubic", bounds_error=False, fill_value="extrapolate")


@pytest.mark.parametrize("radius", [40.0, 150.0])
def test_angular_correlation_equals_box_xi_at_the_chord_distance(
    spectrum: object, radius: float
) -> None:
    """``w(theta)`` on the shell must equal ``xi(2 r sin(theta/2))`` in the box.

    Two sight-lines separated by angle ``theta`` hit the shell at two points whose
    straight-line (chord) separation is ``2 r sin(theta/2)``. Those are just two points
    of the periodic box at that separation, so their expected product is the box's own
    ``xi`` there -- a statement with no approximation in it beyond the reconstruction
    kernel, which the tolerances below absorb. Comparing against the *box's* ``xi`` is
    the right thing whatever the cell values mean: were they cell averages, both sides
    would carry the cell window and it would cancel.

    This catches any error in the radius scaling or the spherical-to-Cartesian mapping
    that the single-realisation power spectrum test might absorb into its tolerance,
    and it does so at *two* radii, which fixes the radius as a genuine length rather
    than an arbitrary scale factor.
    """
    rng = np.random.default_rng(1234)
    ratios = []

    for seed in ANGULAR_SEEDS:
        box = gaussian_box(NCELL, spectrum, seed)
        xi = exact_xi(box)

        for chord in SEPARATIONS:
            theta = 2 * np.arcsin(min(chord / (2 * radius), 1.0))
            first, second = pairs_at_separation(30000, theta, rng)
            values = shell_values(box, np.vstack([first, second]), radius)
            measured = float(np.mean(values[:30000] * values[30000:]))
            ratios.append(measured / float(xi(chord)))

    # Tolerances are set by the measured sample variance -- about 6% per point, so a
    # little under 2% on the mean of forty -- plus room for the reconstruction kernel,
    # which suppresses the measured correlation by a few percent at ``radius = 40`` where
    # the shell is comparable to the box. They are not tuned to one realisation: a change
    # of ``powerbox`` version, which redraws every box, must not move the verdict.
    ratios = np.array(ratios)
    assert np.all(np.abs(ratios - 1) < 0.25), f"chord ratios: {np.round(ratios, 3)}"
    assert abs(ratios.mean() - 1) < 0.10, f"mean ratio {ratios.mean():.4f}"


def test_radial_correlation_between_shells_equals_box_xi(spectrum: object) -> None:
    """Two shells along the same sight-line must correlate as ``xi(r2 - r1)``.

    The angular test above constrains the transverse geometry; this one constrains the
    radial direction, which is the axis a lightcone is actually built along. Two points
    at radii ``r1`` and ``r2`` in the *same* direction are exactly ``r2 - r1`` apart, so
    their correlation is ``xi(r2 - r1)`` with no projection factor of any kind.
    """
    rng = np.random.default_rng(4321)
    vectors = random_directions(40000, rng)
    inner = 60.0
    ratios = []

    for seed in SEEDS:
        box = gaussian_box(NCELL, spectrum, seed)
        xi = exact_xi(box)
        near = shell_values(box, vectors, inner)

        for gap in SEPARATIONS:
            far = near if gap == 0 else shell_values(box, vectors, inner + gap)
            ratios.append(float(np.mean(near * far)) / float(xi(gap)))

    ratios = np.array(ratios)
    assert np.all(np.abs(ratios - 1) < 0.15), f"radial ratios: {np.round(ratios, 3)}"
    assert abs(ratios.mean() - 1) < 0.07, f"mean ratio {ratios.mean():.4f}"


def test_nearest_neighbour_tiling_preserves_the_one_point_statistics(
    spectrum: object,
) -> None:
    """Order-0 tiling must preserve the mean and variance of the box.

    With nearest-neighbour sampling the shell is a subsample of the box's cells, so its
    one-point moments are the box's own up to subsampling noise -- there is no kernel to
    suppress anything. That holds whatever the cells mean; it is a statement about
    resampling, not about the field the values represent. A systematic offset here would
    mean the shell is preferentially landing on particular cells, i.e. the sampling of
    the periodic lattice is biased.
    """
    for seed in SEEDS:
        box = gaussian_box(NCELL, spectrum, seed)
        shell = next(
            cmt.make_healpix_lightcone_slice(
                nside=64, coevals=box, distance_to_shell=90.0, interpolation_order=0
            )
        )
        assert abs(shell.var() / box.var() - 1) < 0.05
        assert abs(shell.mean() - box.mean()) < 0.1 * box.std()


def test_trilinear_variance_suppression_matches_its_analytic_value(
    spectrum: object,
) -> None:
    """Order-1 tiling suppresses the variance by a *predictable* amount.

    Trilinear interpolation of a single mode ``k`` at sub-cell offset ``t`` scales its
    amplitude by ``|(1 - t) + t e^{i k}|`` in each dimension. Averaging the square over a
    uniformly-distributed offset gives ``(2 + cos k_i) / 3`` per dimension, so a shell
    cutting a box at generic offsets has variance

    ``Var_shell / Var_box = sum_k P(k) prod_i (2 + cos k_i)/3 / sum_k P(k)``.

    For this spectrum that is a ~3% loss. The test matters because it says the
    suppression is *understood*: it is the interpolation kernel and nothing else. Users
    who need the small-scale variance preserved should raise ``interpolation_order`` or
    over-resolve the coeval box, not rescale the output.
    """
    kmag, (kx, ky, kz) = mode_grid(NCELL)
    response = ((2 + np.cos(kx)) / 3) * ((2 + np.cos(ky)) / 3) * ((2 + np.cos(kz)) / 3)
    nonzero = kmag > 0

    ratios, predictions = [], []
    for seed in SEEDS:
        box = gaussian_box(NCELL, spectrum, seed)
        modes = np.abs(np.fft.fftn(box)) ** 2
        predicted = np.sum(modes[nonzero] * response[nonzero]) / np.sum(modes[nonzero])
        predictions.append(predicted)

        shell = next(
            cmt.make_healpix_lightcone_slice(
                nside=64, coevals=box, distance_to_shell=90.0, interpolation_order=1
            )
        )
        ratios.append((shell.var() / box.var()) / predicted)

    assert all(0.9 < p < 1.0 for p in predictions), (
        f"expected a few percent of suppression for this spectrum, got {predictions}"
    )
    ratios = np.array(ratios)
    assert np.all(np.abs(ratios - 1) < 0.08), f"var ratios: {np.round(ratios, 4)}"


def test_shell_is_statistically_isotropic(spectrum: object) -> None:
    """The variance of the shell must not depend on where you look on it.

    A tiled shell is cut through a cubic lattice, so it is fair to worry that the
    statistics pick up the lattice axes -- for instance that sight-lines down a
    coordinate axis behave differently from diagonal ones. This compares the variance in
    caps around a box axis and around a body diagonal.
    """
    rng = np.random.default_rng(99)
    box = gaussian_box(NCELL, spectrum, 0)
    radius = 110.0

    axis = np.array([0.0, 0.0, 1.0])
    diagonal = np.ones(3) / np.sqrt(3)

    def cap_variance(centre: np.ndarray) -> float:
        vectors = random_directions(60000, rng)
        keep = vectors @ centre > np.cos(0.5)
        return float(shell_values(box, vectors[keep], radius).var())

    along_axis, along_diagonal = cap_variance(axis), cap_variance(diagonal)
    assert abs(along_axis / along_diagonal - 1) < 0.25
