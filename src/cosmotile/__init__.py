"""Cosmotile."""

from __future__ import annotations

from typing import Any
from typing import Literal

import numpy as np
from astropy.cosmology import FLRW
from astropy.cosmology import Planck18
from astropy_healpix import HEALPix
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation


def make_lightcone_slice(
    *,
    coeval: np.ndarray,
    coeval_res: float,
    latitude: np.ndarray,
    longitude: np.ndarray,
    redshift: float | None = None,
    distance_to_shell: float | None = None,
    cosmo: FLRW = Planck18,
    interpolation_order: int = 1,
    origin: tuple[float, float, float] = (0, 0, 0),
    rotation: Rotation | None = None,
) -> np.ndarray:
    """
    Create a lightcone slice in angular coordinates from two coeval simulations.

    Interpolates the input coeval box to angular coordinates.

    Parameters
    ----------
    coeval
        The rectangular coeval simulation to interpolate to the angular coordinates.
        Must have three dimensions (not necessarily the same size).
    coeval_res
        The resolution of the coeval box in each of its 3 dimensions.
    redshift
        The redshift of the coeval box.
    latitude
        An array of latitude coordinates onto which to tile the box. In radians from
        -pi/2 to pi/2
    longitude
        An array, same size as latitude, of longitude coordinates onto which to tile the
        box. In radians from 0 to 2pi.
    cosmo
        The cosmology.
    interpolation_order
        The order of interpolation. Must be in the range 0-5.
    origin
        Define the location of the centre of the spherical shell, assuming that the
        (0,0,0) pixel of the coeval box is at (0,0,0) in cartesian coordinates.
    rotation
        The rotation by which to rotate the spherical coordinates before interpolation.
        This is done before shifting the origin, and is equivalent to rotating the
        coeval box beforing tiling it.
    """
    if coeval.ndim != 3:
        raise ValueError("coeval must have three dimensions")

    if not isinstance(cosmo, FLRW):
        raise ValueError("cosmo must be an astropy FLRW object")

    if redshift is not None and redshift < 0:
        raise ValueError("redshift must be non-negative")

    if distance_to_shell is not None and distance_to_shell <= 0:
        raise ValueError("distance_to_shell must be positive")

    if distance_to_shell is None and redshift is None:
        raise ValueError("either distance_to_shell or redshift must be specified")

    if interpolation_order < 0 or interpolation_order > 5:
        raise ValueError("interpolation_order must be in the range 0-5")

    # Determine the radial comoving distance r to the comoving shell at the
    # frequency of interest.
    dc = distance_to_shell
    if dc is None:
        dc = cosmo.comoving_distance(redshift).value

    pixel_coords = transform_to_pixel_coords(
        coeval_res=coeval_res,
        comoving_radius=dc,
        latitude=latitude,
        longitude=longitude,
        origin=origin,
        rotation=rotation,
    )

    # Do the interpolation
    return map_coordinates(
        coeval,
        pixel_coords,
        order=int(interpolation_order),
        mode="grid-wrap",  # this wraps each dimension.
        prefilter=False,
    )


def transform_to_pixel_coords(
    *,
    coeval_res: float,
    comoving_radius: float,
    latitude: np.ndarray,
    longitude: np.ndarray,
    origin: tuple[float, float, float] = (0, 0, 0),
    rotation: Rotation | None = None,
) -> np.ndarray:
    """Transform input spherical coordinates to pixel coordinates wrt a coeval box.

    Parameters
    ----------
    coeval_res
        The resolution of the coeval box.
    comoving_radius
        The radius of the spherical coordinates (in comoving units).
    latitude
        An array of latitude coordinates onto which to tile the box. In radians from
        -pi/2 to pi/2
    longitude
        An array, same size as latitude, of longitude coordinates onto which to tile the
        box. In radians from 0 to 2pi.
    origin
        Define the location of the centre of the spherical shell, assuming that the
        (0,0,0) pixel of the coeval box is at (0,0,0) in cartesian coordinates.
    rotation
        The rotation by which to rotate the spherical coordinates before interpolation.
        This is done before shifting the origin, and is equivalent to rotating the
        coeval box beforing tiling it.
    """
    if coeval_res <= 0:
        raise ValueError("coeval_res must be positive")

    if latitude.shape != longitude.shape:
        raise ValueError("latitude and longitude must have the same shape")

    if latitude.ndim != 1:
        raise ValueError("latitude and longitude must be 1D arrays")

    if np.any(np.abs(latitude) > np.pi / 2):
        raise ValueError("latitude must be between -pi/2 and pi/2")

    if np.any((longitude < 0) | (longitude > 2 * np.pi)):
        raise ValueError("longitude must be between 0 and 2pi")

    # Get the cartesian coordinates (x, y, z) of the angular lightcone coords.
    phi = np.pi / 2 - latitude
    sinphi = np.sin(phi)
    cart_coords = comoving_radius * np.array(
        [
            sinphi * np.cos(longitude),
            sinphi * np.sin(longitude),
            np.cos(phi),
        ]
    )

    # Get a rotation matrix
    if rotation is not None:
        cart_coords = np.dot(rotation.as_matrix(), cart_coords)

    # Apply an offset transformation if desired.
    cart_coords += np.array(origin)[:, None]

    # Divide by the resolution so now the coordinates are in units of pixel number.
    cart_coords /= coeval_res
    return cart_coords


def make_healpix_lightcone_slice(
    nside: int, order: Literal["ring", "nested"] = "ring", **kwargs: Any
) -> np.ndarray:
    """
    Create a healpix lightcone slice in angular coordinates.

    This is a simple wrapper around :func:`make_lightcone_slice` that sets up angular
    co-ordinates from a healpix grid.

    Parameters
    ----------
    nside
        The Nside parameter of the healpix map.
    order
        The ordering of the pixels in the healpix map.

    Other Parameters
    ----------------
    All other parameters are passed through to :func:`make_lightcone_slice`.
    """
    hp = HEALPix(nside=nside, order=order)
    lon, lat = hp.healpix_to_lonlat(np.arange(hp.npix))

    return make_lightcone_slice(
        latitude=lat.to_value("radian"), longitude=lon.to_value("radian"), **kwargs
    )
