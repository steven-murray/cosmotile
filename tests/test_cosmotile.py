"""Basic tests of cosmotile algorithm."""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R  # noqa: N817

from cosmotile import make_healpix_lightcone_slice
from cosmotile import make_lightcone_slice


def test_make_lightcone_slice_inputs() -> None:
    """Simple tests that bad inputs raise appropriate errors."""
    coeval = np.random.uniform(size=(10, 10, 10))
    lat = np.random.uniform(size=11) * np.pi - np.pi / 2
    lon = np.random.uniform(size=11) * 2 * np.pi

    def call(
        coeval: np.ndarray = coeval,
        coeval_res: float = 1.0,
        redshift: float | None = 10.0,
        latitude: np.ndarray = lat,
        longitude: np.ndarray = lon,
        **kw: Any,
    ) -> None:

        make_lightcone_slice(
            coeval=coeval,
            coeval_res=coeval_res,
            redshift=redshift,
            latitude=latitude,
            longitude=longitude,
            **kw,
        )

    with pytest.raises(ValueError, match="coeval must have three dimensions"):
        call(coeval=coeval[0])

    with pytest.raises(ValueError, match="coeval_res must be positive"):
        call(coeval_res=0)

    with pytest.raises(
        ValueError, match="latitude and longitude must have the same shape"
    ):
        call(latitude=np.concatenate((lat, [0])))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="latitude and longitude must be 1D arrays"):
        call(latitude=np.zeros((11, 11)), longitude=np.zeros((11, 11)))

    with pytest.raises(ValueError, match="latitude must be between -pi/2 and pi/2"):
        call(latitude=np.arange(11))

    with pytest.raises(ValueError, match="longitude must be between 0 and 2pi"):
        call(longitude=-1 * np.ones(11))

    with pytest.raises(ValueError, match="cosmo must be an astropy FLRW object"):
        call(cosmo=None)

    with pytest.raises(ValueError, match="redshift must be non-negative"):
        call(redshift=-1)

    with pytest.raises(ValueError, match="distance_to_shell must be positive"):
        call(distance_to_shell=-1)

    with pytest.raises(
        ValueError, match="either distance_to_shell or redshift must be specified"
    ):
        call(redshift=None, distance_to_shell=None)

    with pytest.raises(
        ValueError, match="interpolation_order must be in the range 0-5"
    ):
        call(interpolation_order=1000)


@pytest.mark.parametrize("distance_to_shell", [5, 20, np.pi, None])
@pytest.mark.parametrize("redshift", [1.0])
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
    redshift: float,
) -> None:
    """Test that a uniform box gives uniform interpolated values."""
    coeval = np.ones((10, 10, 10))
    lat = np.random.uniform(size=11) * np.pi - np.pi / 2
    lon = np.random.uniform(size=11) * 2 * np.pi

    shell = make_lightcone_slice(
        coeval=coeval,
        coeval_res=1.0,
        distance_to_shell=distance_to_shell,
        redshift=redshift,
        origin=origin,
        rotation=rotation,
        latitude=lat,
        longitude=lon,
    )
    assert np.allclose(shell, 1, rtol=1e-8)


@pytest.mark.parametrize("distance_to_shell", [5, 20, np.pi, None])
@pytest.mark.parametrize("redshift", [1.0])
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
    redshift: float,
) -> None:
    """Test that a random uniform box doesn't yield answers bigger than the maximum."""
    coeval = np.random.uniform(size=(12, 13, 14))
    lat = np.random.uniform(size=11) * np.pi - np.pi / 2
    lon = np.random.uniform(size=11) * 2 * np.pi

    shell = make_lightcone_slice(
        coeval=coeval,
        coeval_res=1.0,
        distance_to_shell=distance_to_shell,
        redshift=redshift,
        origin=origin,
        rotation=rotation,
        latitude=lat,
        longitude=lon,
    )
    assert np.all(shell <= 1)


@pytest.mark.parametrize("distance_to_shell", [5, 15])
def test_stripe(distance_to_shell: float) -> None:
    """Test a box with a single non-zero plane."""
    coeval = np.zeros((10, 10, 10))
    coeval[5] = 1.0  # a plane at x=5

    lat = np.array([0, 0, 0, 0, np.pi / 2, -np.pi / 2])  # points on cartesian axes
    lon = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2, 0, 0])

    shell = make_lightcone_slice(
        coeval=coeval,
        coeval_res=1.0,
        distance_to_shell=distance_to_shell,
        latitude=lat,
        longitude=lon,
    )
    print(shell)
    assert np.isclose(shell[0], 1.0)
    assert shell[1] == 0
    assert np.isclose(shell[2], 1.0)
    assert shell[3] == 0
    assert shell[4] == 0
    assert shell[5] == 0


@pytest.mark.parametrize("distance_to_shell", [5, 20, np.pi, None])
@pytest.mark.parametrize("redshift", [1.0])
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
    redshift: float,
) -> None:
    """Test that a random uniform box doesn't yield answers bigger than the maximum."""
    coeval = np.random.uniform(size=(12, 13, 14))

    shell = make_healpix_lightcone_slice(
        coeval=coeval,
        coeval_res=1.0,
        distance_to_shell=distance_to_shell,
        redshift=redshift,
        origin=origin,
        rotation=rotation,
        nside=16,
    )
    assert np.all(shell <= 1)
