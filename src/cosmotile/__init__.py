"""Cosmotile."""

from __future__ import annotations

from functools import partial
from typing import Any
from typing import Callable
from typing import Iterable
from typing import Literal

import numpy as np
from astropy.cosmology import FLRW
from astropy.cosmology import Planck18
from astropy_healpix import HEALPix
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation

from .cic import cloud_in_cell_los


def make_lightcone_slice_interpolator(
    *,
    coeval_res: float,
    latitude: np.ndarray,
    longitude: np.ndarray,
    redshift: float | None = None,
    distance_to_shell: float | None = None,
    cosmo: FLRW = Planck18,
    interpolation_order: int = 1,
    origin: tuple[float, float, float] = (0, 0, 0),
    rotation: Rotation | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Create a callable interpolator for a lightcone slice.

    Parameters
    ----------
    coeval_res
        The resolution of the coeval box in each of its 3 dimensions.
    latitude
        An array of latitude coordinates onto which to tile the box. In radians from
        -pi/2 to pi/2
    longitude
        An array, same size as latitude, of longitude coordinates onto which to tile the
        box. In radians from 0 to 2pi.
    redshift
        The redshift of the coeval box.
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

    Returns
    -------
    interpolator
        A callable that takes a 3D array of coeval values and returns a 2D array of
        interpolated values on a redshift slice.
    """
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

    if not isinstance(interpolation_order, int):
        raise TypeError("interpolation_order must be an integer")

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

    coordmap = partial(
        map_coordinates,
        coordinates=pixel_coords,
        order=interpolation_order,
        mode="grid-wrap",  # this wraps each dimension.
        prefilter=False,
    )

    coordmap.__name__ = "lightcone_slice_interpolator"
    coordmap.__doc__ = """Interpolate a coeval box to a lightcone slice at a given redshift.

This function is a wrapper around :func:`scipy.ndimage.map_coordinates` created by
functools.partial.

Parameters
----------
coeval
    A 3D array of float coeval values to be interpolated to the lightcone slice.

Returns
-------
lightcone_slice
    A 2D array of float interpolated values on the lightcone slice.
"""


def make_lightcone_slice(
    *, coevals: Iterable[np.ndarray], **kwargs
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Create a lightcone slice in angular coordinates from two coeval simulations.

    Interpolates the input coeval box to angular coordinates.

    Parameters
    ----------
    coevals
        An iterable of rectangular coeval simulations to interpolate to the angular
        coordinates. Must have three dimensions (not necessarily the same size). Each
        box must have the same shape, and all are assumed to be at the same coordinates.
        Each coeval box can be a different simulated field.

    Other Parameters
    ----------------
    All other parameters are passed to :func:`make_lightcone_slice_interpolator`.

    Yields
    ------
    field
        Each interpolated field on the angular coordinates.
    """
    if hasattr(coevals, "ndim") and coevals.ndim == 3:
        coevals = [coevals]

    if any(cv.ndim != 3 for cv in coevals):
        raise ValueError("all coevals must have three dimensions")

    if any(cv.shape != coevals[0].shape for cv in coevals):
        raise ValueError("all coevals must have the same shape")

    coordmap = make_lightcone_slice_interpolator(**kwargs)

    for cv in coevals:
        yield coordmap(cv)


def make_lightcone_slice_vector_field(
    coeval_vector_fields: Iterable[Iterable[np.ndarray, np.ndarray, np.ndarray]],
    interpolator: Callable[[np.ndarray], np.ndarray],
):
    """
    Interpolate a 3D vector field to a lightcone slice as a line-of-sight component.

    This takes a sequence of 3D vector fields, eg. the velocity field, and interpolates
    each component to the lightcone slice. It then computes the line-of-sight component
    of each interpolated vector field, where positive values are oriented towards the
    observer.

    Parameters
    ----------
    coeval_vector_fields
        An iterable of 3D vector fields to interpolate to the lightcone slice. Each
        vector field must be an iterable of 3 3D arrays, each of the same shape.
    interpolator
        A callable that takes a 3D array of coeval values and returns a 2D array of
        interpolated values on a redshift slice. This should be created by
        :func:`make_lightcone_slice_interpolator` using the properties of the
        coeval vector fields.

    Yields
    ------
    los_component
        The line-of-sight component of each interpolated vector field.
    """
    pixel_coords = interpolator.keywords["coordinates"]
    coord_norm = np.sqrt(np.sum(np.square(pixel_coords), axis=0))
    for i, cvf in enumerate(coeval_vector_fields):
        if len(cvf) != 3:
            raise ValueError(
                "coeval_vector_fields must be an iterable of 3-tuples. Got length "
                f"{len(cvf)} at index {i}"
            )
        if any(c.shape != cvf[0].shape for c in cvf):
            raise ValueError(
                f"all coeval vector fields must have the same shape. For index {i}, "
                f"got shapes {[c.shape for c in cvf]}"
            )

        cvf = np.array([interpolator(c) for c in cvf])

        # Now take the dot product of the vector field with (negative) pixel coordinates to get
        # the LoS comp.
        cvf *= -pixel_coords
        yield np.sum(cvf, axis=0) / coord_norm


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


def apply_rsds(
    field: np.ndarray,
    los_displacement: np.ndarray,
    distance: np.ndarray,
    n_subcells: int = 4,
) -> np.ndarray:
    """Apply redshift-space distortions to a field.

    Notes
    -----
    To ensure that we cover all the slices in the field after the velocities have
    been applied, we extrapolate the densities and velocities on either end by the
    maximum velocity offset in the field.
    Then, to ensure we don't pick up cells with zero particles (after displacement),
    we interpolate the slices onto a finer regular grid (in comoving distance) and
    then displace the field on that grid, before interpolating back onto the original
    slices.

    Parameters
    ----------
    field
        The field to apply redshift-space distortions to, shape (nslices, ncoords).
    los_displacement
        The line-of-sight "apparent" displacement of the field, in comoving distance
        units, equal to ``v_los/H(z)``. Positive values are towards the observer, shape
        ``(nslices, ncoords)``.
    distance
        The comoving distance to each slice in the field, in the same units as
        ``los_displacement``, shape (nslices,).
    """
    regular = np.allclose(np.diff(np.diff(distance)), 0.0)
    interpolator = RegularGridInterpolator if regular else RectBivariateSpline

    # TODO: convert these to distances...
    vmax_towards_observer = np.max(los_displacement[0])
    vmax_away_from_observer = np.max(-los_displacement[-1])

    smallest_slice = np.min(np.diff(distance))
    rsd_dx = smallest_slice / n_subcells

    fine_grid = np.arange(
        distance - vmax_away_from_observer, distance[-1] + vmax_towards_observer, rsd_dx
    )
    ang_coords = np.arange(field.shape[1])
    fine_field = interpolator(distance, ang_coords, field)(fine_grid, ang_coords)
    fine_rsd = interpolator(distance, ang_coords, los_displacement / rsd_dx)(
        fine_grid, ang_coords
    )
    fine_field = cloud_in_cell_los(fine_field, fine_rsd)

    return RegularGridInterpolator(fine_grid, ang_coords, fine_field)(
        distance, ang_coords
    )
