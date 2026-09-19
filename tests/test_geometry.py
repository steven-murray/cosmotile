"""Exact tests of the geometry of the tiling.

Everything in this module is checked either to machine precision or against an
*analytic* error bound, so nothing here depends on a statistical tolerance. If the
mapping from (latitude, longitude, radius, rotation, origin) to box pixel coordinates is
right, these pass; if any convention is flipped or transposed, they fail loudly.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import cosmotile as cmt

from .conftest import random_directions, unit_vectors_to_lonlat

NCELL = 64


@pytest.fixture
def directions() -> np.ndarray:
    """Return a fixed set of isotropic sight-lines, shared by the geometric tests."""
    return random_directions(2000, np.random.default_rng(20240617))


def plane_wave_box(nvec: tuple[int, int, int], phase: float, n: int = NCELL) -> np.ndarray:
    """Build a periodic box holding the single Fourier mode ``cos(k.x + phase)``."""
    grid = np.meshgrid(*(np.arange(n, dtype=float),) * 3, indexing="ij")
    kvec = 2 * np.pi * np.asarray(nvec, dtype=float) / n
    return np.cos(sum(k * x for k, x in zip(kvec, grid, strict=True)) + phase)


@pytest.mark.parametrize("nvec", [(1, 0, 0), (1, 2, -1), (4, 0, 3)])
@pytest.mark.parametrize("radius", [33.3, 150.0])
@pytest.mark.parametrize(
    ("origin", "rotation"),
    [
        (None, None),
        ((3.5, -2.0, 7.0), None),
        (None, Rotation.from_rotvec([0.3, -0.9, 0.2])),
        ((3.5, -2.0, 7.0), Rotation.from_rotvec([0.3, -0.9, 0.2])),
    ],
)
def test_plane_wave_is_reproduced_analytically(
    directions: np.ndarray,
    nvec: tuple[int, int, int],
    radius: float,
    origin: tuple[float, float, float] | None,
    rotation: Rotation | None,
) -> None:
    """Tiling a single Fourier mode must reproduce that mode evaluated on the shell.

    This is the strongest single statement we can make about the geometry: the value at
    sight-line ``n`` must be ``cos(k . (R r n + o) + phi)``, which pins down the radius,
    the rotation, the origin offset and the order in which they are applied, all at
    once.

    The residual is pure trilinear interpolation error, for which the textbook bound on
    a unit grid is ``|f - f_interp| <= (1/8) sum_i k_i^2`` (each dimension contributes
    ``h^2 |f''| / 8`` with ``h = 1``). The test checks that we sit *under* that bound and
    -- because the bound is attained somewhere on a dense set of sight-lines -- also
    that we come close to it, which would catch an accidental extra smoothing.
    """
    phase = 0.7
    box = plane_wave_box(nvec, phase)
    kvec = 2 * np.pi * np.asarray(nvec, dtype=float) / NCELL

    lat, lon = unit_vectors_to_lonlat(directions)
    shell = next(
        cmt.make_lightcone_slice(
            coevals=box,
            latitude=lat,
            longitude=lon,
            distance_to_shell=radius,
            origin=origin,
            rotation=rotation,
            interpolation_order=1,
        )
    )

    pos = radius * directions
    if rotation is not None:
        pos = pos @ rotation.as_matrix().T
    if origin is not None:
        pos = pos + np.asarray(origin)

    expected = np.cos(pos @ kvec + phase)
    max_error = np.abs(shell - expected).max()
    bound = np.sum(kvec**2) / 8

    assert max_error <= bound
    assert max_error > 0.9 * bound


@pytest.mark.parametrize("order", range(6))
def test_interpolation_error_falls_with_order(directions: np.ndarray, order: int) -> None:
    """Higher spline order must monotonically improve a well-resolved mode.

    Spline interpolation of order ``n`` converges as ``h**(n+1)``, so on a mode that is
    comfortably resolved each extra order should buy at least an order of magnitude.
    This is a regression guard on the spline pre-filter: without it, ``order >= 2``
    evaluates the B-spline basis directly against the samples, which *smooths* the field
    and makes the error grow with order instead of shrinking.
    """
    nvec = (2, 1, 0)
    box = plane_wave_box(nvec, 0.4)
    kvec = 2 * np.pi * np.asarray(nvec, dtype=float) / NCELL
    radius = 37.3

    lat, lon = unit_vectors_to_lonlat(directions)
    shell = next(
        cmt.make_lightcone_slice(
            coevals=box,
            latitude=lat,
            longitude=lon,
            distance_to_shell=radius,
            interpolation_order=order,
        )
    )
    expected = np.cos((radius * directions) @ kvec + 0.4)
    error = np.abs(shell - expected).max()

    # Measured errors are ~1.4e-1, 6.0e-3, 6.9e-5, 4.1e-6, 6.3e-8, 3.9e-9 for orders 0-5.
    tolerance = [2e-1, 1e-2, 2e-4, 1e-5, 2e-7, 1e-8][order]
    assert error < tolerance


@pytest.mark.parametrize("order", range(6))
def test_prefiltering_does_not_change_the_answer(directions: np.ndarray, order: int) -> None:
    """``prefilter_coeval`` must only move work, never change the result.

    The spline pre-filter depends on ``(coeval, order)`` alone, so hoisting it out of the
    per-shell loop has to be *bit*-identical to filtering inside it. Anything weaker --
    a different boundary condition, a dtype change, a filter applied twice -- shows up
    here immediately.
    """
    box = plane_wave_box((2, 1, 0), 0.4)
    lat, lon = unit_vectors_to_lonlat(directions)
    kw = {
        "latitude": lat,
        "longitude": lon,
        "distance_to_shell": 37.3,
        "interpolation_order": order,
    }

    raw = next(cmt.make_lightcone_slice(coevals=box, **kw))
    filtered = cmt.prefilter_coeval(box, order)
    hoisted = next(cmt.make_lightcone_slice(coevals=filtered, **kw))

    np.testing.assert_array_equal(hoisted, raw)


def test_one_prefilter_serves_every_shell(directions: np.ndarray) -> None:
    """The whole point: one filtered box, reused across many shells, at many radii."""
    box = plane_wave_box((1, 2, -1), 1.1)
    lat, lon = unit_vectors_to_lonlat(directions)
    filtered = cmt.prefilter_coeval(box, 3)

    for radius in (21.0, 37.3, 150.0):
        kw = {
            "latitude": lat,
            "longitude": lon,
            "distance_to_shell": radius,
            "interpolation_order": 3,
        }
        raw = next(cmt.make_lightcone_slice(coevals=box, **kw))
        hoisted = next(cmt.make_lightcone_slice(coevals=filtered, **kw))
        np.testing.assert_array_equal(hoisted, raw)


@pytest.mark.parametrize("order", [2, 3])
def test_prefiltering_twice_would_be_wrong(order: int) -> None:
    """Why the tag has to suppress the second filter, rather than it being harmless.

    Filtering twice is not a no-op: it corrupts the field. So the tag is load-bearing,
    not a convenience -- without it, handing a pre-filtered box back to
    ``make_lightcone_slice`` would silently return the wrong shell.
    """
    box = plane_wave_box((2, 1, 0), 0.4)
    once = np.asarray(cmt.prefilter_coeval(box, order))
    twice = np.asarray(cmt.prefilter_coeval(once, order))

    assert not np.allclose(once, twice)


def test_prefiltered_order_must_match_the_interpolation_order(directions: np.ndarray) -> None:
    """A box filtered for one order is not a valid input at another.

    This is the one mistake the tag cannot silently absorb: the box genuinely holds the
    wrong coefficients, and tiling it anyway would return a plausible but wrong shell.
    """
    lat, lon = unit_vectors_to_lonlat(directions)
    slices = cmt.make_lightcone_slice(
        coevals=cmt.prefilter_coeval(np.zeros((NCELL,) * 3), 3),
        latitude=lat,
        longitude=lon,
        distance_to_shell=37.3,
        interpolation_order=5,
    )
    with pytest.raises(ValueError, match="pre-filtered for order 3"):
        next(slices)


def test_prefilter_coeval_validates_its_order() -> None:
    box = np.zeros((8, 8, 8))
    with pytest.raises(TypeError, match="must be an integer"):
        cmt.prefilter_coeval(box, 3.0)
    with pytest.raises(ValueError, match="range 0-5"):
        cmt.prefilter_coeval(box, 6)


def test_prefilter_tag_does_not_survive_derived_arrays() -> None:
    """Anything derived from a pre-filtered box is a plain array again.

    The pre-filter of a slice is not the slice of the pre-filter, so a tag that
    propagated through views and arithmetic would license genuinely wrong results. Losing
    it is the safe direction: an untagged array is simply filtered on its own account, so
    a derived box is correct -- merely not hoisted.
    """
    filtered = cmt.prefilter_coeval(np.zeros((8, 8, 8)), 3)
    assert filtered.spline_order == 3
    assert filtered[::2].spline_order is None
    assert (filtered * 2.0).spline_order is None
    assert np.asarray(filtered).__class__ is np.ndarray


def test_order_zero_returns_exact_box_values(directions: np.ndarray) -> None:
    """Nearest-neighbour tiling may only ever return values that are *in* the box.

    With ``interpolation_order=0`` the shell is a resampling, not an average, so its
    one-point distribution is a subsample of the box's own. Any smoothing, wrap error or
    spurious extrapolation would introduce values that are not in the box.
    """
    rng = np.random.default_rng(11)
    box = rng.normal(size=(NCELL,) * 3)

    lat, lon = unit_vectors_to_lonlat(directions)
    shell = next(
        cmt.make_lightcone_slice(
            coevals=box,
            latitude=lat,
            longitude=lon,
            distance_to_shell=77.0,
            interpolation_order=0,
        )
    )

    assert np.isin(shell, box).all()


def test_rotation_is_equivalent_to_rotating_the_sightlines(directions: np.ndarray) -> None:
    """``rotation=R`` must be exactly the same as evaluating along the rotated directions.

    The documented meaning of ``rotation`` is "rotate the spherical coordinates before
    interpolation", so this identity has to hold to round-off, and it fixes the
    handedness: it is ``R n``, not ``R^-1 n``.
    """
    rng = np.random.default_rng(5)
    box = rng.normal(size=(NCELL,) * 3)
    rotation = Rotation.from_rotvec([0.31, -0.77, 1.2])
    radius = 51.7

    lat, lon = unit_vectors_to_lonlat(directions)
    rotated_lat, rotated_lon = unit_vectors_to_lonlat(directions @ rotation.as_matrix().T)

    with_rotation = next(
        cmt.make_lightcone_slice(
            coevals=box,
            latitude=lat,
            longitude=lon,
            distance_to_shell=radius,
            rotation=rotation,
        )
    )
    rotated_coords = next(
        cmt.make_lightcone_slice(
            coevals=box,
            latitude=rotated_lat,
            longitude=rotated_lon,
            distance_to_shell=radius,
        )
    )

    np.testing.assert_allclose(with_rotation, rotated_coords, atol=1e-10)


def test_rotations_compose(directions: np.ndarray) -> None:
    """Rotating by ``R2 R1`` must equal rotating the sight-lines by ``R1`` then by ``R2``."""
    rng = np.random.default_rng(6)
    box = rng.normal(size=(NCELL,) * 3)
    r1 = Rotation.from_rotvec([0.2, 0.4, -0.1])
    r2 = Rotation.from_rotvec([-0.9, 0.15, 0.6])

    lat, lon = unit_vectors_to_lonlat(directions)
    composed = next(
        cmt.make_lightcone_slice(
            coevals=box, latitude=lat, longitude=lon, distance_to_shell=44.0, rotation=r2 * r1
        )
    )

    once_lat, once_lon = unit_vectors_to_lonlat(directions @ r1.as_matrix().T)
    stepwise = next(
        cmt.make_lightcone_slice(
            coevals=box,
            latitude=once_lat,
            longitude=once_lon,
            distance_to_shell=44.0,
            rotation=r2,
        )
    )

    np.testing.assert_allclose(composed, stepwise, atol=1e-10)


@pytest.mark.parametrize("order", [0, 1, 3])
def test_origin_is_periodic_in_the_box_length(directions: np.ndarray, order: int) -> None:
    """Translating the origin by whole box lengths must change nothing at all.

    The box is periodic and is tiled by wrapping, so an origin shift of an integer
    number of box lengths in each direction is the identity. This exercises the
    ``grid-wrap`` handling far outside the fundamental domain -- the regime a real
    lightcone at ``r >> L`` lives in.
    """
    rng = np.random.default_rng(7)
    box = rng.normal(size=(NCELL,) * 3)
    lat, lon = unit_vectors_to_lonlat(directions)

    base = np.array([1.5, 2.5, -3.5])
    shifted = base + NCELL * np.array([3, -1, 2])

    kw = {
        "coevals": box,
        "latitude": lat,
        "longitude": lon,
        "distance_to_shell": 51.7,
        "interpolation_order": order,
    }
    a = next(cmt.make_lightcone_slice(origin=tuple(base), **kw))
    b = next(cmt.make_lightcone_slice(origin=tuple(shifted), **kw))

    np.testing.assert_allclose(a, b, atol=1e-10)


def test_origin_shift_equals_shifting_the_box(directions: np.ndarray) -> None:
    """Moving the origin by whole cells must equal rolling the box the other way."""
    rng = np.random.default_rng(8)
    box = rng.normal(size=(NCELL,) * 3)
    lat, lon = unit_vectors_to_lonlat(directions)
    shift = (5, -3, 11)

    moved_origin = next(
        cmt.make_lightcone_slice(
            coevals=box, latitude=lat, longitude=lon, distance_to_shell=29.0, origin=shift
        )
    )
    rolled_box = next(
        cmt.make_lightcone_slice(
            coevals=np.roll(box, [-s for s in shift], axis=(0, 1, 2)),
            latitude=lat,
            longitude=lon,
            distance_to_shell=29.0,
        )
    )

    np.testing.assert_allclose(moved_origin, rolled_box, atol=1e-10)


def test_uniform_box_is_exactly_uniform_everywhere(directions: np.ndarray) -> None:
    """A constant box must tile to a constant shell for every order and geometry.

    Interpolation kernels are partitions of unity, so this must hold exactly. It is the
    cheapest possible check that the weights are normalised at every order (including
    the pre-filtered high orders).
    """
    box = np.full((NCELL,) * 3, 2.5)
    lat, lon = unit_vectors_to_lonlat(directions)

    for order in range(6):
        shell = next(
            cmt.make_lightcone_slice(
                coevals=box,
                latitude=lat,
                longitude=lon,
                distance_to_shell=1234.5,
                origin=(1000.0, -1000.0, np.pi),
                rotation=Rotation.from_rotvec([0.1, 0.2, 0.3]),
                interpolation_order=order,
            )
        )
        np.testing.assert_allclose(shell, 2.5, rtol=1e-12)


def test_tiling_is_linear(directions: np.ndarray) -> None:
    """Tiling is a linear operator on the coeval field.

    Every downstream statistical argument -- the angular power spectrum prediction in
    particular -- assumes the shell is a linear functional of the box. This checks it
    directly, including for the pre-filtered orders where the operator is a composition
    of two linear filters.
    """
    rng = np.random.default_rng(9)
    a, b = rng.normal(size=(NCELL,) * 3), rng.normal(size=(NCELL,) * 3)
    lat, lon = unit_vectors_to_lonlat(directions)

    kw = {
        "latitude": lat,
        "longitude": lon,
        "distance_to_shell": 61.0,
        "interpolation_order": 3,
    }
    sa, sb, sab = cmt.make_lightcone_slice(coevals=[a, b, 3 * a - 2 * b], **kw)

    np.testing.assert_allclose(sab, 3 * sa - 2 * sb, atol=1e-10)
