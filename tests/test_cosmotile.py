"""Basic tests of cosmotile algorithm."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest
from astropy import units as un
from astropy.cosmology import Planck18
from scipy.spatial.transform import Rotation as R  # noqa: N817

import cosmotile as cmt
from cosmotile import make_healpix_lightcone_slice, make_lightcone_slice


def test_make_lightcone_slice_inputs() -> None:
    """Simple tests that bad inputs raise appropriate errors."""
    rng = np.random.default_rng()
    coeval = rng.uniform(size=(10, 10, 10))
    lat = rng.uniform(size=11) * np.pi - np.pi / 2
    lon = rng.uniform(size=11) * 2 * np.pi

    def call(
        coeval: np.ndarray | Sequence[np.ndarray] = coeval,
        latitude: np.ndarray = lat,
        longitude: np.ndarray = lon,
        distance_to_shell: float = 1.0,
        **kw: Any,
    ) -> None:
        next(
            make_lightcone_slice(
                coevals=[coeval] if isinstance(coeval, np.ndarray) else coeval,
                latitude=latitude,
                longitude=longitude,
                distance_to_shell=distance_to_shell,
                **kw,
            )
        )

    with pytest.raises(ValueError, match="all coevals must have three dimensions"):
        call(coeval=coeval[0])

    with pytest.raises(ValueError, match="latitude and longitude must have the same shape"):
        call(latitude=np.concatenate((lat, [0])))

    with pytest.raises(ValueError, match="latitude and longitude must be 1D arrays"):
        call(latitude=np.zeros((11, 11)), longitude=np.zeros((11, 11)))

    with pytest.raises(ValueError, match="latitude must be between -pi/2 and pi/2"):
        call(latitude=np.arange(11))

    with pytest.raises(ValueError, match="longitude must be between 0 and 2pi"):
        call(longitude=-1 * np.ones(11))

    with pytest.raises(ValueError, match="distance_to_shell must be positive"):
        call(distance_to_shell=-1)

    with pytest.raises(ValueError, match="interpolation_order must be in the range 0-5"):
        call(interpolation_order=1000)

    with pytest.raises(TypeError, match="interpolation_order must be an integer"):
        call(interpolation_order=1.0)

    with pytest.raises(ValueError, match="all coevals must have the same shape"):
        call(coeval=[coeval[:8, :8, :8], coeval])

    with pytest.raises(ValueError, match="origin must be a sequence of length 3"):
        call(origin=(1, 2, 3, 4))


@pytest.mark.parametrize("distance_to_shell", [5, 20, np.pi])
@pytest.mark.parametrize("origin", [(0, 0, 0), (3, 6, -1), (1000, -1000, np.pi)])
@pytest.mark.parametrize(
    "rotation",
    [
        R.from_quat([0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)]),
        None,
    ],
)
def test_uniform_box(
    distance_to_shell: float,
    origin: tuple[float, float, float],
    rotation: R | None,
) -> None:
    """Test that a uniform box gives uniform interpolated values."""
    rng = np.random.default_rng()
    coeval = np.ones((10, 10, 10))
    lat = rng.uniform(size=11) * np.pi - np.pi / 2
    lon = rng.uniform(size=11) * 2 * np.pi

    shell = next(
        make_lightcone_slice(
            coevals=coeval,
            distance_to_shell=distance_to_shell,
            origin=origin,
            rotation=rotation,
            latitude=lat,
            longitude=lon,
        )
    )
    assert np.allclose(shell, 1, rtol=1e-8)


@pytest.mark.parametrize("distance_to_shell", [5, 20, np.pi])
@pytest.mark.parametrize("origin", [(0, 0, 0), (3, 6, -1), (1000, -1000, np.pi)])
@pytest.mark.parametrize(
    "rotation",
    [
        R.from_quat([0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)]),
        None,
    ],
)
def test_random_uniform_box(
    distance_to_shell: float,
    origin: tuple[float, float, float],
    rotation: R | None,
) -> None:
    """Test that a random uniform box doesn't yield answers bigger than the maximum."""
    rng = np.random.default_rng()
    coeval = rng.uniform(size=(12, 13, 14))
    lat = rng.uniform(size=11) * np.pi - np.pi / 2
    lon = rng.uniform(size=11) * 2 * np.pi

    shell = next(
        make_lightcone_slice(
            coevals=coeval,
            distance_to_shell=distance_to_shell,
            origin=origin,
            rotation=rotation,
            latitude=lat,
            longitude=lon,
        )
    )
    assert np.all(shell <= 1)


@pytest.mark.parametrize("distance_to_shell", [5, 20, np.pi])
@pytest.mark.parametrize("origin", [(0, 0, 0), (3, 6, -1), (1000, -1000, np.pi)])
@pytest.mark.parametrize(
    "rotation",
    [
        R.from_quat([0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)]),
        None,
    ],
)
def test_healpix(
    distance_to_shell: float,
    origin: tuple[float, float, float],
    rotation: R | None,
) -> None:
    """Test that a random uniform box doesn't yield answers bigger than the maximum."""
    rng = np.random.default_rng()
    coeval = rng.uniform(size=(12, 13, 14))

    shell = next(
        make_healpix_lightcone_slice(
            coevals=coeval,
            distance_to_shell=distance_to_shell,
            origin=origin,
            rotation=rotation,
            nside=16,
        )
    )
    assert np.all(shell <= 1)


def test_distance_to_shell() -> None:
    """Test the get_distance_to_shell_from_redshift function."""
    assert cmt.get_distance_to_shell_from_redshift(
        z=1, cell_size=1 * un.Mpc, cosmo=Planck18
    ) < cmt.get_distance_to_shell_from_redshift(z=2, cell_size=1 * un.Mpc, cosmo=Planck18)

    assert cmt.get_distance_to_shell_from_redshift(
        z=1, cell_size=1 * un.Mpc, cosmo=Planck18
    ) == 2 * cmt.get_distance_to_shell_from_redshift(z=1, cell_size=2 * un.Mpc, cosmo=Planck18)


def test_lightcone_slice_vector_field() -> None:
    """Test the make_lightcone_slice_vector_field function."""
    coeval = np.ones((10, 10, 10))
    distance_to_shell = 1.0 * un.pixel
    lat = np.zeros(21, dtype=float)
    lon = np.linspace(0, 2 * np.pi, 21)

    # put points at the poles for testing
    lat = np.append(lat, np.array([np.pi / 2, -np.pi / 2]))
    lon = np.append(lon, np.array([0, 0]))

    # all displacement is in x-direction
    rsds = [
        [
            np.ones_like(coeval) * un.pixel,
            np.zeros_like(coeval) * un.pixel,
            np.zeros_like(coeval) * un.pixel,
        ]
    ]

    interpolator = cmt.make_lightcone_slice_interpolator(
        latitude=lat, longitude=lon, distance_to_shell=distance_to_shell
    )

    with pytest.raises(
        ValueError,
        match="coeval_vector_fields must be a sequence of 3-tuples. Got length",
    ):
        next(
            cmt.make_lightcone_slice_vector_field(
                interpolator=interpolator,
                coeval_vector_fields=[rsds[0][:2]],
            )
        )

    with pytest.raises(ValueError, match="all coeval vector fields must have the same shape."):
        next(
            cmt.make_lightcone_slice_vector_field(
                interpolator=interpolator,
                coeval_vector_fields=[[rsds[0][0][1:], rsds[0][1], rsds[0][2]]],
            )
        )

    los = next(
        cmt.make_lightcone_slice_vector_field(
            interpolator=interpolator,
            coeval_vector_fields=rsds,
        )
    )

    assert los[0] == -1 * un.pixel
    assert los[10] == 1 * un.pixel
    assert los[20] == -1 * un.pixel
    assert np.isclose(los[21], 0 * un.pixel)
    assert np.isclose(los[22], 0 * un.pixel)


def test_apply_rsds() -> None:
    """Test the apply_rsds function."""
    field = np.ones((2, 10))
    losv = np.ones((2, 10)) * un.pixel
    losv[1] = -1 * un.pixel
    distance = np.array([10, 11]) * un.pixel

    with pytest.raises(ValueError, match="field must have at least 2 slices"):
        cmt.apply_rsds(
            field=field[:1],
            los_displacement=losv[:1],
            distance=distance[:1],
            n_subcells=1,
        )

    with pytest.raises(ValueError, match="field and los_displacement must have the same shape"):
        cmt.apply_rsds(
            field=field[:, :9],
            los_displacement=losv,
            distance=distance,
            n_subcells=1,
        )

    with pytest.raises(ValueError, match="field and distance must have the same number"):
        cmt.apply_rsds(
            field=field,
            los_displacement=losv,
            distance=distance[:1],
            n_subcells=1,
        )

    # Here, the close slice all moves inward, and the far slice all moves outward,
    # so nothing left in the field.
    out = cmt.apply_rsds(
        field=field,
        los_displacement=losv,
        distance=distance,
        n_subcells=1,
    )

    np.all(out == 0)

    # Now, they both point inwards, so we should end up with all ones.
    losv = np.ones((2, 10)) * un.pixel
    losv[0] = -1 * un.pixel

    out = cmt.apply_rsds(
        field=field,
        los_displacement=losv,
        distance=distance,
        n_subcells=1,
    )

    np.all(out == 1)

    # In fact, any velocity field that is constant should keep the out constant.
    losv = 3 * np.ones((2, 10)) * un.pixel

    out = cmt.apply_rsds(
        field=field,
        los_displacement=losv,
        distance=distance,
        n_subcells=1,
    )

    np.all(out == 1)

    # Here, every pixel goes to the middle, but this *just* includes the two pixels
    # we put in.
    losv = (
        np.array(
            [
                -0.5 * np.ones(10),
                0.5 * np.ones(10),
            ]
        )
        * un.pixel
    )

    out = cmt.apply_rsds(
        field=field,
        los_displacement=losv,
        distance=distance,
        n_subcells=1,
    )

    np.all(out == 1)

    out = cmt.apply_rsds(
        field=field,
        los_displacement=losv,
        distance=distance,
        n_subcells=4,
    )

    np.all(out == 1)

    # Now, change the distance ever so slightly, so we use the other interpolator.
    field = np.ones((7, 10))

    losv = np.zeros((7, 10)) * un.pixel
    distance = np.array([10, 11.001, 12, 13, 14, 15, 16]) * un.pixel

    out = cmt.apply_rsds(
        field=field,
        los_displacement=losv,
        distance=distance,
        n_subcells=1,
    )

    np.allclose(out, 1, atol=2e-3)


def test_vector_field_nonzero_origin() -> None:
    """Test that moving the origin doesn't change the LoS component of a static vfield.

    This was originally a bug -- while the field is homogeneous, and so the origin
    can be arbitrary, when taking the line-of-sight component of a field, we have to make
    sure that the coordinates are correct with respect to an actual origin, otherwise
    the dot product of the vector field with the coordinates is wrong.
    """
    vx = np.ones((10, 10, 10))
    vy = np.zeros((10, 10, 10))
    vz = np.zeros((10, 10, 10))

    latitude = np.zeros(21)
    longitude = np.linspace(0, 2 * np.pi, 21)
    interpolator = cmt.make_lightcone_slice_interpolator(
        longitude=longitude,
        latitude=latitude,
        distance_to_shell=5.5,
    )

    interpolator_origin = cmt.make_lightcone_slice_interpolator(
        longitude=longitude,
        latitude=latitude,
        distance_to_shell=5.5,
        origin=(1, 2, 3),
    )

    los = next(cmt.make_lightcone_slice_vector_field([[vx, vy, vz]], interpolator))

    los_origin = next(cmt.make_lightcone_slice_vector_field([[vx, vy, vz]], interpolator_origin))

    np.testing.assert_allclose(los, los_origin, atol=1e-6, rtol=1e-6)


def test_transform_with_different_origin_types() -> None:
    """Test that passing the same origin using different types works as expected."""
    latitude = np.zeros(21)
    longitude = np.linspace(0, 2 * np.pi, 21)

    tuple_origin = cmt.transform_to_pixel_coords(
        comoving_radius=10,
        latitude=latitude,
        longitude=longitude,
        origin=(1, 2, 3),
    )

    array_origin = cmt.transform_to_pixel_coords(
        comoving_radius=10,
        latitude=latitude,
        longitude=longitude,
        origin=np.array([1, 2, 3]),
    )

    np.testing.assert_equal(tuple_origin, array_origin)
