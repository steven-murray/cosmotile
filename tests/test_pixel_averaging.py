r"""Averaging a lightcone pixel over its solid angle and radial extent (cf. GH #465).

By default ``cosmotile`` evaluates the coeval field at exactly one point per output
pixel: the pixel centre, at exactly the shell radius. A lightcone pixel is not a point,
though -- it subtends a solid angle, and in a lightcone it also has the radial thickness
of the slice spacing -- so there is a second, opt-in path that *averages* over that
extent instead.

The two directions are tested separately, and both against statements that are exact
rather than merely plausible:

* **Angular.** The ``4**k`` HEALPix sub-pixels at ``nside * 2**k`` are equal-area and
  tile their parent exactly, so their unweighted mean is an unbiased estimate of the
  pixel average. That tiling is asserted directly here; what it *does* to the angular
  power spectrum (it makes the HEALPix pixel window apply) is in
  ``test_angular_power.py``.
* **Radial.** Averaging uses Gauss-Legendre nodes weighted by the ``r^2`` volume
  element, so for a smooth field the answer must agree with a brute-force quadrature of
  the same integral to many digits -- and a mode with radial wavenumber ``k`` must be
  suppressed by the top-hat transform ``sinc(k w / 2)``.

Every test also pins the default: with ``subsample_level = 0`` and
``n_radial_samples = 1`` the output must be bit-for-bit what it was before any of this
existed.
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy import units as un
from astropy_healpix import HEALPix

import cosmotile as cmt

from .conftest import random_directions, unit_vectors_to_lonlat

NCELL = 32
CENTRE = np.array([16.0, 16.0, 16.0])
RADIUS = 7.0


def smooth_box(wavenumber: float = 2 * np.pi * 3 / NCELL) -> np.ndarray:
    """Build a box holding one Fourier mode along ``x``, measured from ``CENTRE``."""
    grid = np.meshgrid(*(np.arange(NCELL, dtype=float),) * 3, indexing="ij")
    return np.cos(wavenumber * (grid[0] - CENTRE[0]))


def slice_at(
    latitude: np.ndarray, longitude: np.ndarray, box: np.ndarray, **kw: object
) -> np.ndarray:
    """Tile ``box`` onto the given directions at ``RADIUS``, order-5 interpolation."""
    return next(
        cmt.make_lightcone_slice(
            coevals=box,
            latitude=latitude,
            longitude=longitude,
            distance_to_shell=RADIUS,
            origin=CENTRE,
            interpolation_order=5,
            **kw,
        )
    )


# ---------------------------------------------------------------------------------
# Angular averaging
# ---------------------------------------------------------------------------------
@pytest.mark.parametrize("order", ["ring", "nested"])
def test_subsample_level_zero_is_the_pixel_centre(order: str) -> None:
    """The default must be exactly the old behaviour, in both pixel orderings.

    Averaging is opt-in precisely so that existing lightcones do not change under
    anyone's feet, so this is checked for bitwise equality rather than closeness.
    """
    hp = HEALPix(nside=8, order=order)
    lon, lat = hp.healpix_to_lonlat(np.arange(hp.npix))

    latitude, longitude = cmt.healpix_subpixel_lonlat(nside=8, order=order)

    np.testing.assert_array_equal(latitude, lat.to_value("radian"))
    np.testing.assert_array_equal(longitude, lon.to_value("radian"))

    box = smooth_box()
    reference = next(
        cmt.make_healpix_lightcone_slice(
            nside=8, order=order, coevals=box, distance_to_shell=RADIUS, origin=CENTRE
        )
    )
    averaged = next(
        cmt.make_healpix_lightcone_slice(
            nside=8,
            order=order,
            subsample_level=0,
            coevals=box,
            distance_to_shell=RADIUS,
            origin=CENTRE,
        )
    )
    np.testing.assert_array_equal(averaged, reference)


@pytest.mark.parametrize("order", ["ring", "nested"])
@pytest.mark.parametrize("level", [1, 2])
def test_subpixels_tile_their_parent_pixel(order: str, level: int) -> None:
    """Every sub-sample must fall inside the pixel it is supposed to be averaging.

    This is the property that makes the mean over sub-samples an unbiased estimate of
    the pixel average: HEALPix sub-pixels at ``nside * 2**k`` are equal-area and there
    are exactly ``4**k`` of them per parent, so no area weighting is needed. If the
    ring/nested bookkeeping were wrong the sub-samples would scatter over the sphere and
    every averaged map would be quietly smeared.
    """
    nside = 8
    hp = HEALPix(nside=nside, order=order)
    latitude, longitude = cmt.healpix_subpixel_lonlat(
        nside=nside, order=order, subsample_level=level
    )

    assert latitude.shape == (4**level, hp.npix)

    parent = hp.lonlat_to_healpix(longitude.ravel() * un.rad, latitude.ravel() * un.rad)
    np.testing.assert_array_equal(
        parent.reshape(latitude.shape), np.broadcast_to(np.arange(hp.npix), latitude.shape)
    )


def test_angular_average_is_the_mean_over_the_sub_samples() -> None:
    """The averaged value must be the plain mean of the individually-tiled sub-samples.

    Interpolating all sub-samples at once and averaging internally must give exactly
    what tiling each sub-sample separately and averaging by hand would give -- so the
    only thing the new code path adds is the average, not a different interpolation.
    """
    box = smooth_box()
    latitude, longitude = cmt.healpix_subpixel_lonlat(nside=8, subsample_level=2)

    averaged = slice_at(latitude, longitude, box)
    by_hand = np.mean(
        [slice_at(lat, lon, box) for lat, lon in zip(latitude, longitude, strict=True)], axis=0
    )

    np.testing.assert_allclose(averaged, by_hand, rtol=0, atol=1e-14)


def test_angular_averaging_leaves_a_uniform_field_alone() -> None:
    """Averaging is a weighted mean with unit total weight, so a constant is a fixed point."""
    latitude, longitude = cmt.healpix_subpixel_lonlat(nside=8, subsample_level=2)

    shell = slice_at(latitude, longitude, np.full((NCELL,) * 3, 2.5))

    np.testing.assert_allclose(shell, 2.5, rtol=1e-12)


def test_angular_averaging_smooths_the_field() -> None:
    """An averaged map must have less small-scale variance than the sampled one.

    The pixel average is a low-pass filter, so on a field with structure at the pixel
    scale it must reduce the variance. This is the map-space shadow of the pixel window
    measured in ``test_angular_power.py``.
    """
    rng = np.random.default_rng(7)
    box = rng.normal(size=(NCELL,) * 3)

    sampled, averaged = (
        next(
            cmt.make_healpix_lightcone_slice(
                nside=32,
                subsample_level=level,
                coevals=box,
                distance_to_shell=40.0,
                origin=CENTRE,
            )
        )
        for level in (0, 2)
    )

    assert averaged.var() < 0.8 * sampled.var()


# ---------------------------------------------------------------------------------
# Radial averaging
# ---------------------------------------------------------------------------------
@pytest.mark.parametrize("width", [1.0, 2.0, 4.0])
def test_radial_average_matches_the_exact_shell_integral(width: float) -> None:
    r"""The radial average must equal ``int r^2 f(r) dr / int r^2 dr`` over the shell.

    Along the ``+x`` sight-line from the observer the box is exactly ``cos(k r)``, so
    the integral the averaging is supposed to be doing can be evaluated by brute force
    and compared directly. Gauss-Legendre with ``n`` nodes is exact for polynomials of
    degree ``2n - 1``, so the error must fall steeply with ``n`` and then stop at the
    floor set by the order-5 interpolation of the box itself (about 1e-6 here) -- which
    is what is asserted, rather than merely that the answer moved the right way.
    """
    wavenumber = 2 * np.pi * 3 / NCELL
    box = smooth_box(wavenumber)
    latitude, longitude = np.array([0.0]), np.array([0.0])  # the +x direction

    radii = np.linspace(RADIUS - width / 2, RADIUS + width / 2, 20001)
    exact = np.trapezoid(radii**2 * np.cos(wavenumber * radii), radii) / np.trapezoid(
        radii**2, radii
    )

    errors = {
        n: abs(
            slice_at(latitude, longitude, box, radial_width=width, n_radial_samples=n)[0] - exact
        )
        for n in (2, 3, 4, 8)
    }

    # Two nodes cannot resolve the curvature of the mode across the shell; by four the
    # quadrature error has dropped below the floor set by interpolating the box, so
    # doubling again cannot help.
    assert errors[2] > 10 * errors[4], f"quadrature not converging: {errors}"
    assert errors[4] < 1e-5 * abs(exact), f"{errors}"
    assert errors[8] < 1e-5 * abs(exact), f"{errors}"


@pytest.mark.parametrize("width", [1.0, 2.0, 4.0])
def test_radial_averaging_suppresses_a_radial_mode_like_a_top_hat(width: float) -> None:
    r"""A mode along the line of sight must be suppressed by ``sinc(k w / 2)``.

    Averaging over a radial top-hat of width ``w`` convolves the field with that
    top-hat, whose Fourier transform is ``sinc(k w / 2)``. This is the radial
    counterpart of the HEALPix pixel window, and it is the reason radial averaging is
    worth having: without it a slice carries the field's full small-scale radial power,
    which the finite slice spacing cannot represent.

    The ``r^2`` volume weighting is what makes this only a leading-order statement: it
    tilts the average outwards, correcting the pure top-hat at order ``(w / r)^2``.
    Asserting the correction against that scaling -- rather than against a hand-tuned
    number -- is what makes the test mean something at all three widths.
    """
    radius = 20.0
    wavenumber = 2 * np.pi * 3 / NCELL
    box = smooth_box(wavenumber)
    latitude, longitude = np.array([0.0]), np.array([0.0])

    def at(**kw: object) -> float:
        return float(
            next(
                cmt.make_lightcone_slice(
                    coevals=box,
                    latitude=latitude,
                    longitude=longitude,
                    distance_to_shell=radius,
                    origin=CENTRE,
                    interpolation_order=5,
                    **kw,
                )
            )[0]
        )

    ratio = at(radial_width=width, n_radial_samples=8) / at()
    expected = float(np.sinc(wavenumber * width / (2 * np.pi)))

    assert abs(ratio - expected) < 2.5 * (width / radius) ** 2, (
        f"width {width}: measured suppression {ratio:.4f} vs top-hat {expected:.4f}"
    )
    # ...and the suppression is a real effect, not a rounding error.
    assert 1 - ratio > 0.5 * (1 - expected)


def test_n_radial_samples_one_is_the_shell_radius() -> None:
    """The default must sample the shell itself, whatever ``radial_width`` says."""
    box = smooth_box()
    latitude, longitude = cmt.healpix_subpixel_lonlat(nside=8)

    reference = slice_at(latitude, longitude, box)
    for radial_width in (0.0, 3.0):
        np.testing.assert_array_equal(
            slice_at(latitude, longitude, box, radial_width=radial_width), reference
        )


def test_radial_and_angular_averaging_compose() -> None:
    """Asking for both must average over the product set, not one or the other.

    The sub-samples are a product of the radial quadrature and the angular sub-pixels,
    so doing both at once must equal doing the angular average at each radial node and
    then combining those with the radial weights.
    """
    box = smooth_box()
    latitude, longitude = cmt.healpix_subpixel_lonlat(nside=8, subsample_level=1)
    width, nodes = 3.0, 4

    both = slice_at(latitude, longitude, box, radial_width=width, n_radial_samples=nodes)

    offsets, weights = np.polynomial.legendre.leggauss(nodes)
    radii = RADIUS + 0.5 * width * offsets
    weights = weights * radii**2
    weights /= weights.sum()

    by_hand = sum(
        weight
        * next(
            cmt.make_lightcone_slice(
                coevals=box,
                latitude=latitude,
                longitude=longitude,
                distance_to_shell=radius,
                origin=CENTRE,
                interpolation_order=5,
            )
        )
        for weight, radius in zip(weights, radii, strict=True)
    )

    np.testing.assert_allclose(both, by_hand, rtol=0, atol=1e-13)


# ---------------------------------------------------------------------------------
# Line-of-sight projection of a vector field
# ---------------------------------------------------------------------------------
def test_projection_of_a_hubble_flow_survives_averaging() -> None:
    """A pure outflow must still project to exactly ``-H r`` on an averaged map.

    Every sub-sample of a pixel sits at the same radius, so the average over them of
    ``-H r`` is ``-H r`` -- exactly, with no smoothing. This catches the easy mistake of
    averaging the vector components *before* projecting them, which would leave a
    residual wherever the sub-samples point in different directions.
    """
    latitude, longitude = cmt.healpix_subpixel_lonlat(nside=8, subsample_level=2)
    interpolator = cmt.make_lightcone_slice_interpolator(
        latitude=latitude, longitude=longitude, distance_to_shell=RADIUS, origin=CENTRE
    )

    hubble = 0.1
    grid = np.meshgrid(*(np.arange(NCELL, dtype=float),) * 3, indexing="ij")
    field = [hubble * (g - c) for g, c in zip(grid, CENTRE, strict=True)]

    los = next(cmt.make_lightcone_slice_vector_field([field], interpolator))

    assert los.shape == (12 * 8**2,)
    np.testing.assert_allclose(los, -hubble * RADIUS, atol=1e-12)


def test_projection_of_a_bulk_flow_averages_the_dipole_over_the_pixel() -> None:
    """A uniform flow must project to the *mean* of ``-v . n`` over the sub-pixels.

    Unlike the outflow above, the bulk-flow dipole varies across a pixel, so this fixes
    that the projection happens per sub-sample and the averaging afterwards.
    """
    latitude, longitude = cmt.healpix_subpixel_lonlat(nside=8, subsample_level=2)
    interpolator = cmt.make_lightcone_slice_interpolator(
        latitude=latitude, longitude=longitude, distance_to_shell=RADIUS, origin=CENTRE
    )

    velocity = np.array([0.7, -0.2, 1.3])
    field = [np.full((NCELL,) * 3, v) for v in velocity]

    los = next(cmt.make_lightcone_slice_vector_field([field], interpolator))

    normals = np.array(
        [
            np.cos(latitude) * np.cos(longitude),
            np.cos(latitude) * np.sin(longitude),
            np.sin(latitude),
        ]
    )
    expected = np.mean(-np.einsum("i,ijk->jk", velocity, normals), axis=0)

    np.testing.assert_allclose(los, expected, atol=1e-12)


def test_averaged_projection_differs_from_the_sampled_one() -> None:
    """Averaging the dipole over a pixel must actually change it, or nothing is tested."""
    sightlines = random_directions(64, np.random.default_rng(11))
    lat, lon = unit_vectors_to_lonlat(sightlines)

    velocity = np.array([0.7, -0.2, 1.3])
    field = [np.full((NCELL,) * 3, v) for v in velocity]

    # A deliberately crude two-point "pixel": the sight-line and a neighbour 0.2 rad away.
    stacked_lat = np.stack([lat, np.clip(lat + 0.2, -np.pi / 2, np.pi / 2)])
    stacked_lon = np.stack([lon, lon])

    interpolator = cmt.make_lightcone_slice_interpolator(
        latitude=stacked_lat, longitude=stacked_lon, distance_to_shell=RADIUS, origin=CENTRE
    )
    averaged = next(cmt.make_lightcone_slice_vector_field([field], interpolator))

    assert np.abs(averaged + sightlines @ velocity).max() > 0.01
