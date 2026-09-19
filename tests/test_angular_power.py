r"""Angular power spectrum of a tiled coeval box (cf. GH #289).

The central physical claim of ``cosmotile`` is that cutting a spherical shell out of a
periodically-tiled coeval box gives you the angular statistics that shell *should* have.
That claim has a closed-form test. Expanding a plane wave in spherical harmonics gives,
for an infinitely thin shell at comoving radius ``r`` cut through a field with 3D power
spectrum ``P(k)``,

.. math::

    C_\ell = \frac{2}{\pi} \int \mathrm{d}k\, k^2\, P(k)\, j_\ell^2(kr),

whose discrete-box counterpart is ``C_l = (4 pi / V) sum_k P(k) j_l^2(kr)``. This is the
*exact* relation; a geometrically thin shell has no Limber limit, because Limber's
approximation needs a radial kernel of finite width to integrate over.

The tests below verify that relation in the regime where it should hold, and then pin
down each of the three ways it stops holding: the box fundamental at low ``l``, the
interpolation kernel and aliasing at high ``l``, and periodic replication of structure
at large ``r / L``.
"""

from __future__ import annotations

import numpy as np
import pytest

import cosmotile as cmt

from .conftest import (
    band_average,
    band_limited_powerlaw,
    continuum_angular_power,
    gaussian_box,
    linear_interp_window,
    mode_grid,
    theory_angular_power,
    unit_vectors_to_lonlat,
)

NCELL = 64
VOLUME = float(NCELL) ** 3
KCUT = np.pi / 4  # all input power below half-Nyquist
NSIDE = 64
LMAX = 80
SEEDS = (0, 1, 2, 3)

# Bands chosen to sit inside the window of validity at radius 80: the box fundamental
# puts the low edge at l ~ k_min r = 7.9, and the spectral cut-off the high edge at
# l ~ k_cut r = 62.8.
BANDS = ((10, 16), (16, 24), (24, 34), (34, 46), (46, 60))


def band_scatter(lo: int, hi: int) -> float:
    """Fractional 1-sigma scatter expected on a band power from ``a_lm`` randomness.

    Even with the box's mode amplitudes held fixed, the ``a_lm`` of the shell are random
    (the box phases are), so a band power built from ``N = sum (2l+1)`` harmonic
    coefficients is chi-squared distributed with ``N`` degrees of freedom and fluctuates
    by ``sqrt(2/N)``. Low-``l`` bands are therefore intrinsically much noisier, and the
    tolerances below are set from this rather than picked by hand.
    """
    return float(np.sqrt(2 / np.sum(2 * np.arange(lo, hi) + 1)))


def healpix_shell(box: np.ndarray, radius: float, order: int = 1) -> np.ndarray:
    """Tile ``box`` onto a HEALPix shell of the given radius."""
    return next(
        cmt.make_healpix_lightcone_slice(
            nside=NSIDE,
            coevals=box,
            distance_to_shell=radius,
            interpolation_order=order,
        )
    )


def measured_cl(box: np.ndarray, radius: float, order: int = 1) -> np.ndarray:
    """Angular power spectrum of the tiled shell."""
    hp = pytest.importorskip("healpy")
    return hp.anafast(healpix_shell(box, radius, order), lmax=LMAX)


def predicted_cl(
    box: np.ndarray | None,
    radius: float,
    pk: object,
    ells: np.ndarray,
    ncell: int = NCELL,
) -> np.ndarray:
    """Predicted ``C_l`` for a shell through the box.

    If ``box`` is given, the prediction uses that realisation's *own* mode amplitudes
    ``|delta_k|^2`` rather than the ensemble ``P(k)``. That removes the sample variance
    of the input field from the comparison, leaving only the irreducible chi-squared
    scatter of the ``a_lm``, so it is a far more sensitive test of the tiling itself.
    Pass ``box=None`` to use the ensemble ``P(k)`` instead.
    """
    kmag, kvec = mode_grid(ncell)
    window = linear_interp_window(kvec) ** 2
    volume = float(ncell) ** 3

    if box is None:
        weight = pk(kmag) * window
    else:
        weight = volume * np.abs(np.fft.fftn(box) / box.size) ** 2 * window

    nonzero = kmag > 0
    return theory_angular_power(
        kmag[nonzero].ravel(), weight[nonzero].ravel(), volume, radius, ells
    )


@pytest.mark.parametrize("radius", [80.0, 200.0])
def test_angular_power_matches_theory(radius: float) -> None:
    """The measured ``C_l`` must match ``(4 pi / V) sum_k P(k) W(k)^2 j_l^2(kr)``.

    This is the headline test of GH #289. The prediction uses each realisation's own
    mode amplitudes and the analytic trilinear window ``W(k) = prod_i sinc^2(k_i / 2)``,
    so the only thing left free is the tiling geometry itself.

    The residual scatter is the chi-squared scatter of the ``a_lm``: even with the mode
    amplitudes fixed, the phases are random, so each band power fluctuates by
    ``sqrt(2/N)`` (see :func:`band_scatter`). Each band is therefore checked at four
    times its own expected scatter, and the mean over all bands and realisations -- where
    that scatter largely averages down -- at 6%.
    """
    pk = band_limited_powerlaw(-2.0, KCUT)
    ells = np.arange(LMAX + 1)

    ratios = []
    for seed in SEEDS:
        box = gaussian_box(NCELL, pk, seed)
        cl = measured_cl(box, radius)
        theory = predicted_cl(box, radius, pk, ells)
        ratios.append([band_average(cl, lo, hi) / band_average(theory, lo, hi) for lo, hi in BANDS])

    ratios = np.array(ratios)
    tolerance = np.array([4 * band_scatter(lo, hi) for lo, hi in BANDS])
    assert np.all(np.abs(ratios - 1) < tolerance), (
        f"per-band ratios out of tolerance:\n{np.round(ratios, 3)}\n"
        f"tolerance: {np.round(tolerance, 3)}"
    )
    assert abs(ratios.mean() - 1) < 0.06, f"mean ratio {ratios.mean():.4f}"


def test_angular_power_matches_ensemble_theory() -> None:
    """Averaged over realisations, the measurement recovers the *input* ``P(k)``.

    The previous test conditions on each realisation's modes; this one does not, so it
    checks the whole chain -- ``P(k)`` in, ``C_l`` out -- against the analytic formula
    with nothing borrowed from the realisation. It is correspondingly noisier: at these
    multipoles only a few tens of box modes contribute to each band, so individual bands
    scatter by tens of percent and only the average over bands and seeds is tight.
    """
    pk = band_limited_powerlaw(-2.0, KCUT)
    ells = np.arange(LMAX + 1)
    radius = 120.0
    theory = predicted_cl(None, radius, pk, ells)

    ratios = np.array(
        [
            [
                band_average(measured_cl(gaussian_box(NCELL, pk, seed), radius), lo, hi)
                / band_average(theory, lo, hi)
                for lo, hi in BANDS
            ]
            for seed in range(8)
        ]
    )

    assert abs(ratios.mean() - 1) < 0.12, f"mean ratio {ratios.mean():.4f}"


def test_angular_power_is_insensitive_to_rotation_and_origin() -> None:
    """Rotating or translating the box must not change the angular power spectrum.

    Rotating and translating are how ``cosmotile`` builds independent-looking
    realisations from one coeval box. Statistical homogeneity and isotropy of the
    underlying field mean the shell's angular power spectrum must be unchanged --
    the shell samples the same field, just differently oriented.
    """
    from scipy.spatial.transform import Rotation

    pk = band_limited_powerlaw(-2.0, KCUT)
    box = gaussian_box(NCELL, pk, 42)
    radius = 120.0
    hp = pytest.importorskip("healpy")

    plain = hp.anafast(healpix_shell(box, radius), lmax=LMAX)
    variants = [
        hp.anafast(
            next(
                cmt.make_healpix_lightcone_slice(
                    nside=NSIDE,
                    coevals=box,
                    distance_to_shell=radius,
                    rotation=rot,
                    origin=org,
                )
            ),
            lmax=LMAX,
        )
        for rot, org in [
            (Rotation.from_rotvec([0.7, -0.2, 1.1]), None),
            (None, (13.0, -27.0, 5.0)),
            (Rotation.from_rotvec([2.0, 0.4, -0.8]), (100.0, 40.0, -60.0)),
        ]
    ]

    for lo, hi in BANDS:
        reference = band_average(plain, lo, hi)
        for cl in variants:
            assert abs(band_average(cl, lo, hi) / reference - 1) < 0.5

    # Averaged over all bands the agreement should be much better than band by band.
    for cl in variants:
        mean_ratio = np.mean(
            [band_average(cl, lo, hi) / band_average(plain, lo, hi) for lo, hi in BANDS]
        )
        assert abs(mean_ratio - 1) < 0.25


def test_thin_shell_integral_reduces_to_its_closed_form() -> None:
    r"""Check the theory machinery itself against an exact analytic result.

    Every other test in this module compares a measurement to
    ``C_l = (4 pi / V) sum_k P(k) j_l^2(kr)``, so that expression had better be right.
    For ``P(k) = A k^-2`` the continuum integral can be done in closed form, because
    ``int_0^inf j_l^2(x) dx = pi / (2 (2l+1))``:

    .. math::

        C_\ell = \frac{A}{r (2\ell + 1)}.

    Truncating the spectrum at ``k_cut`` removes the ``1/(2x^2)`` tail of ``j_l^2``
    beyond ``x = k_cut r``, giving the correction factor ``1 - (2l+1)/(pi k_cut r)``.
    Note that this is *not* the Limber approximation: a geometrically thin shell has no
    Limber limit (Limber needs a radial kernel of finite width), which is worth knowing
    before comparing a single lightcone slice against ``P(k)/r^2``.
    """
    amplitude = 1.0
    pk = band_limited_powerlaw(-2.0, KCUT, amplitude)
    radius = 1000.0
    ells = np.arange(2, 61)

    numeric = continuum_angular_power(pk, radius, ells, KCUT)
    closed_form = (
        amplitude / (radius * (2 * ells + 1)) * (1 - (2 * ells + 1) / (np.pi * KCUT * radius))
    )

    np.testing.assert_allclose(numeric, closed_form, rtol=2e-3)


# ---------------------------------------------------------------------------------
# Failure modes -- these assert the *limitations* of the method, so that they stay
# documented and cannot silently change.
# ---------------------------------------------------------------------------------
def test_power_is_missing_below_the_box_fundamental() -> None:
    """A tiled box cannot produce power at ``l`` below ``k_min r = 2 pi r / L``.

    The box only contains modes at multiples of the fundamental ``2 pi / L``, so the
    lowest multipole with any real signal is ``l ~ 2 pi r / L``. Below that, the shell
    has far less power than the same ``P(k)`` in a larger box would give. This is the
    single most important limitation of the method and is why lightcones tiled from
    small boxes must not be trusted on large angular scales.
    """
    pk = band_limited_powerlaw(-2.0, KCUT)
    radius = 300.0
    ells = np.arange(LMAX + 1)
    fundamental_ell = 2 * np.pi * radius / NCELL  # ~29
    assert fundamental_ell > 12, "the test band must sit below the box fundamental"

    # The infinite-box limit: what this P(k) would give with every mode present.
    unbounded = continuum_angular_power(pk, radius, ells, KCUT)

    deficits = [
        band_average(measured_cl(gaussian_box(NCELL, pk, seed), radius), 2, 12)
        / band_average(unbounded, 2, 12)
        for seed in SEEDS
    ]

    assert np.mean(deficits) < 0.5, (
        f"expected a large-scale power deficit, got {np.mean(deficits):.3f}"
    )


def test_aliasing_floor_above_the_spectral_cutoff() -> None:
    """Above ``l ~ k_cut r`` the shell shows spurious power the input does not contain.

    Interpolation reconstructs a continuous field from samples, and any interpolating
    kernel leaks each mode ``k`` into aliases at ``k + 2 pi m``. With a sharply
    band-limited input the true ``C_l`` falls to zero above ``l = k_cut r``, but the
    measured spectrum hits a floor instead. Users should treat multipoles beyond
    ``l ~ pi r / cell_size`` (the Nyquist limit) as meaningless, and band-limited inputs
    even sooner.
    """
    pk = band_limited_powerlaw(-2.0, KCUT)
    radius = 80.0
    cutoff_ell = int(KCUT * radius)  # ~62
    ells = np.arange(LMAX + 1)

    box = gaussian_box(NCELL, pk, 0)
    cl = measured_cl(box, radius)
    theory = predicted_cl(box, radius, pk, ells)

    # Well above the cut-off the true signal has fallen by ten orders of magnitude...
    above = slice(cutoff_ell + 14, LMAX + 1)
    assert theory[above].max() < 1e-8 * theory[10]

    # ...but the measurement has not: it flattens out onto an aliasing floor.
    assert cl[above].min() > 50 * theory[above].max(), (
        "expected a spurious aliasing floor above the spectral cut-off"
    )


@pytest.mark.parametrize("multiple", [1, 2])
def test_structure_is_exactly_replicated_at_lattice_separations(multiple: int) -> None:
    """Sight-lines separated by a lattice vector see *identical* values, not correlated ones.

    Once the shell is large enough to re-intersect a translated copy of the box, pairs of
    directions whose chord is a box lattice vector are perfectly, deterministically
    correlated. At radius ``m L / 2`` the two ends of a coordinate axis are exactly
    ``m L`` apart, so they return the same number bit for bit -- a correlation the
    underlying field does not have. This is the replication artefact, and it is why
    ``r / L`` should be kept modest (or the box rotated between shells).
    """
    rng = np.random.default_rng(3)
    box = rng.normal(size=(NCELL,) * 3)
    radius = multiple * NCELL / 2

    # +x and -x, +y and -y, +z and -z.
    axes = np.array(
        [[1.0, 0, 0], [-1.0, 0, 0], [0, 1.0, 0], [0, -1.0, 0], [0, 0, 1.0], [0, 0, -1.0]]
    )
    lat, lon = unit_vectors_to_lonlat(axes)
    shell = next(
        cmt.make_lightcone_slice(
            coevals=box,
            latitude=lat,
            longitude=lon,
            distance_to_shell=radius,
            interpolation_order=0,
        )
    )

    assert shell[0] == shell[1]
    assert shell[2] == shell[3]
    assert shell[4] == shell[5]

    # And it is not simply that the box is nearly constant along those directions.
    _ = rng  # keep the generator referenced for clarity
    assert box.std() > 0.5
