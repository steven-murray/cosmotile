"""Cosmotile."""

from __future__ import annotations

from collections.abc import Generator, Iterator, Sequence
from functools import partial
from typing import Any, Callable, Literal

import numpy as np
from astropy import units as un
from astropy.cosmology import FLRW, Planck18
from astropy_healpix import HEALPix
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation

from . import _version
from .cic import cloud_in_cell_los

__version__ = _version.version

_LENGTH = "length"


def get_distance_to_shell_from_redshift(
    z: float, cell_size: un.Quantity[_LENGTH], cosmo: FLRW = Planck18
) -> un.Quantity[un.pixel]:
    """Get a distance to a shell, in units of cell size, from a given redshift.

    Parameters
    ----------
    z
        The redshift
    cell_size
        The resolution of the coeval simulation, in comoving units.
    cosmo
        The astropy cosmology.

    Returns
    -------
    distance
        The distance, in units of pixels, to the shell.
    """
    return (cosmo.comoving_distance(z)).to(un.pixel, un.pixel_scale(cell_size / un.pixel))


def make_lightcone_slice_interpolator(
    *,
    latitude: np.ndarray,
    longitude: np.ndarray,
    distance_to_shell: float,
    interpolation_order: int = 1,
    origin: np.ndarray | tuple[float, float, float] | None = None,
    rotation: Rotation | None = None,
) -> partial[np.ndarray]:
    """
    Create a callable interpolator for a lightcone slice.

    Parameters
    ----------
    latitude
        An array of latitude coordinates onto which to tile the box. In radians from
        -pi/2 to pi/2
    longitude
        An array, same size as latitude, of longitude coordinates onto which to tile the
        box. In radians from 0 to 2pi.
    distance_to_shell
        The distance to the spherical shell onto which to interpolate, in units of
        the cell-size of the coeval box(es) you wish to interpolate.
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
    if distance_to_shell <= 0:
        raise ValueError("distance_to_shell must be positive")

    if interpolation_order < 0 or interpolation_order > 5:
        raise ValueError("interpolation_order must be in the range 0-5")

    if not isinstance(interpolation_order, int):
        raise TypeError("interpolation_order must be an integer")

    if isinstance(origin, (tuple, list)):
        origin = np.array(origin)

    if origin is not None and origin.shape != (3,):
        raise ValueError("origin must be a sequence of length 3")

    pixel_coords = transform_to_pixel_coords(
        comoving_radius=distance_to_shell,
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

    # Save the origin to the coordmap because it's useful for getting
    # line-of-sight vectors.
    coordmap.origin = origin

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
    return coordmap


def make_lightcone_slice(
    *, coevals: Sequence[np.ndarray] | np.ndarray, **kwargs: Any
) -> Iterator[np.ndarray]:
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
    if isinstance(coevals, np.ndarray) and coevals.ndim == 3:
        coevals = [coevals]

    if any(cv.ndim != 3 for cv in coevals):
        raise ValueError("all coevals must have three dimensions")

    if any(cv.shape != coevals[0].shape for cv in coevals):
        raise ValueError("all coevals must have the same shape")

    coordmap = make_lightcone_slice_interpolator(**kwargs)
    return map(coordmap, coevals)


def make_lightcone_slice_vector_field(
    coeval_vector_fields: Sequence[Sequence[np.ndarray]],
    interpolator: Callable[[np.ndarray], np.ndarray],
) -> Iterator[np.ndarray]:
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
    if interpolator.origin is not None:
        pixel_coords = pixel_coords - interpolator.origin[:, None]

    coord_norm = np.sqrt(np.sum(np.square(pixel_coords), axis=0))

    def _doit(cvf: Sequence[np.ndarray]) -> np.ndarray:
        if len(cvf) != 3:
            raise ValueError(
                f"coeval_vector_fields must be a sequence of 3-tuples. Got length {len(cvf)}"
            )
        if any(c.shape != cvf[0].shape for c in cvf):
            raise ValueError(
                f"all coeval vector fields must have the same shape. "
                f"Got shapes {[c.shape for c in cvf]}"
            )
        unit = getattr(cvf[0], "unit", 1)
        cvf_interp = np.array([interpolator(c) for c in cvf]) * unit

        # Now take the dot product of the vector field with (negative) pixel coordinates to get
        # the LoS comp.
        cvf_interp *= -pixel_coords
        return np.sum(cvf_interp, axis=0) / coord_norm

    return map(_doit, coeval_vector_fields)


def transform_to_pixel_coords(
    *,
    comoving_radius: un.Quantity[un.pixel],
    latitude: np.ndarray,
    longitude: np.ndarray,
    origin: (un.Quantity[un.pixel, (3,), float] | tuple[float, float, float] | None) = None,
    rotation: Rotation | None = None,
) -> np.ndarray:
    """Transform input spherical coordinates to pixel coordinates wrt a coeval box.

    Parameters
    ----------
    comoving_radius
        The radius of the spherical coordinates (in units of the cell size).
    latitude
        An array of latitude coordinates onto which to tile the box. In radians from
        -pi/2 to pi/2
    longitude
        An array, same size as latitude, of longitude coordinates onto which to tile the
        box. In radians from 0 to 2pi.
    origin
        Define the location of the centre of the spherical shell, assuming that the
        (0,0,0) pixel of the coeval box is at (0,0,0) in cartesian coordinates.
        In units of the cell size.
    rotation
        The rotation by which to rotate the spherical coordinates before interpolation.
        This is done before shifting the origin, and is equivalent to rotating the
        coeval box beforing tiling it.
    """
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
    if origin is not None:
        if not isinstance(origin, np.ndarray):
            origin = np.array(origin)
        cart_coords += origin[:, None]

    return cart_coords


def make_healpix_lightcone_slice(
    nside: int, order: Literal["ring", "nested"] = "ring", **kwargs: Any
) -> Generator:
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

    yield from make_lightcone_slice(
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
        The line-of-sight "apparent" displacement of the field, in pixel coordinates.
        Equal to ``v / H(z) / cell_size``.
        Positive values are towards the observer, shape ``(nslices, ncoords)``.
    distance
        The comoving distance to each slice in the field, in units of the cell size.
        shape (nslices,).
    """
    if field.shape != los_displacement.shape:
        raise ValueError("field and los_displacement must have the same shape")
    if field.shape[0] < 2:
        raise ValueError("field must have at least 2 slices")
    if field.shape[0] != distance.size:
        raise ValueError("field and distance must have the same number of slices")

    is_regular = np.allclose(np.diff(np.diff(distance)), 0.0)
    interpolator = RegularGridInterpolator if is_regular else RectBivariateSpline

    # TODO: convert these to distances...
    vmax_towards_observer = max(np.max(los_displacement[0]), 0)
    vmax_away_from_observer = min(0, np.min(los_displacement[-1]))

    smallest_slice = np.min(np.diff(distance))
    rsd_dx = smallest_slice / n_subcells

    fine_grid = np.arange(
        (distance.min() - vmax_towards_observer).to_value(rsd_dx.unit),
        (distance.max() - vmax_away_from_observer).to_value(rsd_dx.unit),
        rsd_dx.value,
    )
    if fine_grid.max() < distance.max().to_value(rsd_dx.unit):
        fine_grid = np.append(fine_grid, distance.max().to_value(rsd_dx.unit))

    ang_coords = np.arange(field.shape[1])
    if is_regular:
        x, y = np.meshgrid(fine_grid, ang_coords, indexing="ij")
        grid = (x.flatten(), y.flatten())
        fine_field = interpolator(
            (distance, ang_coords), field, bounds_error=False, fill_value=None
        )(grid).reshape(x.shape)
        fine_rsd = interpolator(
            (distance, ang_coords),
            los_displacement / rsd_dx,
            bounds_error=False,
            fill_value=None,
        )(grid).reshape(x.shape)
    else:
        fine_field = interpolator(distance, ang_coords, field)(fine_grid, ang_coords)
        fine_rsd = interpolator(distance, ang_coords, los_displacement / rsd_dx)(
            fine_grid, ang_coords
        )
    fine_field = cloud_in_cell_los(fine_field, fine_rsd)

    x, y = np.meshgrid(distance, ang_coords, indexing="ij")

    return RegularGridInterpolator((fine_grid, ang_coords), fine_field)(
        (
            x.flatten(),
            y.flatten(),
        )
    ).reshape(x.shape)
