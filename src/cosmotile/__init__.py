"""Cosmotile."""

from __future__ import annotations

from collections.abc import Callable, Generator, Iterator, Sequence
from functools import partial
from typing import Any, Literal

import numpy as np
from astropy import units as un
from astropy.cosmology import FLRW, Planck18
from astropy_healpix import HEALPix
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator
from scipy.ndimage import map_coordinates, spline_filter
from scipy.spatial.transform import Rotation

from . import _version
from . import theory as theory
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


class PrefilteredCoeval(np.ndarray):
    """A coeval box to which the spline pre-filter has already been applied.

    Instances are produced by :func:`prefilter_coeval`, and carry the spline order they
    were filtered for in :attr:`spline_order`. That tag is the whole mechanism: it is
    what tells :func:`make_lightcone_slice` to skip the filter it would otherwise apply,
    and what lets it reject a box filtered for the wrong order.

    The tag is deliberately not propagated through views, slices or arithmetic: any
    array derived from a `PrefilteredCoeval` is a plain array again, because the
    pre-filter of a derived array is not in general the derived pre-filtered array.
    """

    #: The interpolation order the box was pre-filtered for, or ``None`` if the tag was
    #: dropped (e.g. on a slice or the result of an arithmetic operation).
    spline_order: int | None = None

    def __array_finalize__(self, obj: Any) -> None:
        """Drop the pre-filter tag on any array derived from this one."""
        self.spline_order = None


def prefilter_coeval(coeval: np.ndarray, order: int) -> PrefilteredCoeval:
    """Apply the spline pre-filter to a coeval box once, for re-use across many shells.

    Interpolating at ``order >= 2`` requires the coeval box to be converted to B-spline
    coefficients first (see :func:`_interpolate_coeval`). Those coefficients depend only
    on the box and the order -- not on the shell radius, rotation or origin -- so for a
    lightcone of many shells the filter need only be computed once.

    .. code-block:: python

        filtered = cosmotile.prefilter_coeval(coeval, order=3)
        for radius in radii:
            (shell,) = cosmotile.make_lightcone_slice(
                coevals=filtered,
                latitude=lat,
                longitude=lon,
                distance_to_shell=radius,
                interpolation_order=3,
            )

    Parameters
    ----------
    coeval
        The coeval box to pre-filter.
    order
        The interpolation order the box is being prepared for. Must be in the range 0-5,
        and must match the ``interpolation_order`` it is later tiled with. Orders 0 and 1
        use interpolating kernels and need no filter, so for them this only tags the box.

    Returns
    -------
    prefiltered
        The pre-filtered box, tagged with ``order``. Pass it to
        :func:`make_lightcone_slice` (or :func:`make_lightcone_slice_interpolator`)
        in place of the raw box; no further flag is needed, since the tag is what tells
        the interpolator the filter has already been applied.

    Notes
    -----
    The result of tiling a pre-filtered box is bit-identical to tiling the raw box at the
    same order; this only moves the work out of the per-shell loop.

    Mutating ``coeval`` after calling this does **not** update the returned array for
    ``order > 1`` (it is a fresh array); re-run this function if the box changes.
    """
    if not isinstance(order, int):
        raise TypeError("order must be an integer")

    if order < 0 or order > 5:
        raise ValueError("order must be in the range 0-5")

    arr = np.asarray(coeval)
    if order > 1:
        arr = spline_filter(arr, order=order, mode="grid-wrap", output=np.float64)

    out: PrefilteredCoeval = arr.view(PrefilteredCoeval)
    out.spline_order = order
    return out


def _average_subsamples(values: np.ndarray, weights: np.ndarray | None) -> np.ndarray:
    """Collapse per-sub-sample values onto one value per output pixel.

    ``values`` holds ``weights.size`` sub-samples for every output pixel, laid out
    sub-sample-major; the result is their weighted mean. Written as a broadcast product
    rather than a matrix product so that it carries ``astropy`` units through.
    """
    if weights is None:
        return values
    return (weights[:, None] * values.reshape(weights.size, -1)).sum(axis=0)


def _interpolate_coeval(
    coeval: np.ndarray,
    *,
    coordinates: np.ndarray,
    order: int,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Interpolate a coeval box at the given (pixel) coordinates.

    This is a thin wrapper around :func:`scipy.ndimage.map_coordinates` that applies
    the spline pre-filter itself, using the same periodic boundary condition as the
    interpolation. Without the pre-filter, ``order >= 2`` evaluates the B-spline basis
    directly against the data, which *smooths* the field rather than interpolating it
    (the resulting curve does not pass through the input samples).

    Orders 0 and 1 use interpolating kernels and need no pre-filter, so they are
    passed straight through.

    The filtered array depends only on ``(coeval, order)``, so when tiling one box onto
    many shells it can be computed once with :func:`prefilter_coeval` instead of once per
    shell. Such a box arrives here tagged with the order it was filtered for, and that tag
    -- which only :func:`prefilter_coeval` can produce -- is what suppresses the filter
    here. No flag is taken on trust and nothing is cached between calls, so this cannot
    silently filter twice, nor silently reuse a stale filter for a box that has since been
    mutated. A box tagged for a different order than it is being tiled at is an error.
    """
    tagged_order = getattr(coeval, "spline_order", None)

    if tagged_order is not None and tagged_order != order:
        raise ValueError(
            f"coeval was pre-filtered for order {tagged_order}, but is being "
            f"interpolated at order {order}. Pre-filter at the order you will tile with."
        )

    coeval = np.asarray(coeval)
    if order > 1 and tagged_order is None:
        coeval = spline_filter(coeval, order=order, mode="grid-wrap", output=np.float64)

    return _average_subsamples(
        map_coordinates(
            coeval,
            coordinates=coordinates,
            order=order,
            mode="grid-wrap",  # this wraps each dimension.
            prefilter=False,  # we have already pre-filtered above, if required.
        ),
        weights,
    )


def _radial_quadrature(
    distance_to_shell: float, radial_width: float, n_radial_samples: int
) -> tuple[Sequence[float], np.ndarray]:
    """Gauss-Legendre nodes and weights across the radial extent of a shell.

    A lightcone shell is not geometrically thin -- it has the thickness of the slice
    spacing -- so the value a pixel should carry is the average of the field over
    ``[r - w/2, r + w/2]``, weighted by the ``r^2`` volume element.

    With ``n_radial_samples == 1`` this returns the shell radius itself and a unit
    weight, which is the point-sampling behaviour.
    """
    if n_radial_samples == 1:
        return [distance_to_shell], np.ones(1)

    nodes, weights = np.polynomial.legendre.leggauss(n_radial_samples)
    radii = distance_to_shell + 0.5 * radial_width * nodes
    weights = weights * radii**2
    return list(radii), weights / weights.sum()


def make_lightcone_slice_interpolator(
    *,
    latitude: np.ndarray,
    longitude: np.ndarray,
    distance_to_shell: float,
    interpolation_order: int = 1,
    origin: np.ndarray | tuple[float, float, float] | None = None,
    rotation: Rotation | None = None,
    radial_width: float = 0.0,
    n_radial_samples: int = 1,
) -> partial[np.ndarray]:
    """
    Create a callable interpolator for a lightcone slice.

    By default each output pixel is a single *point sample* of the coeval field, at the
    pixel centre and at exactly the shell radius. Both directions can instead be
    *averaged* over the extent the pixel really subtends -- see ``latitude`` and
    ``radial_width`` below. Averaging costs one interpolation per sub-sample, so it is
    opt-in; nothing changes unless you ask for it.

    Parameters
    ----------
    latitude
        An array of latitude coordinates onto which to tile the box. In radians from
        -pi/2 to pi/2.

        May instead be two-dimensional, with shape ``(n_subsamples, npix)``. The box is
        then interpolated at every sub-sample and the results averaged, so that each
        output pixel is an average over the directions given rather than a sample at
        one of them. :func:`make_healpix_lightcone_slice` builds such an array from
        HEALPix sub-pixels.
    longitude
        An array, same shape as latitude, of longitude coordinates onto which to tile
        the box. In radians from 0 to 2pi.
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
    radial_width
        The radial thickness of the shell, in units of the cell size -- in a lightcone,
        the spacing between neighbouring slices. Only used when
        ``n_radial_samples > 1``.
    n_radial_samples
        The number of Gauss-Legendre nodes used to average the field over
        ``radial_width``, weighted by the ``r^2`` volume element. The default of 1
        samples the shell radius itself.

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

    if latitude.ndim > 2:
        raise ValueError("latitude and longitude must be 1D or 2D arrays")

    if latitude.shape != longitude.shape:
        raise ValueError("latitude and longitude must have the same shape")

    if n_radial_samples < 1:
        raise ValueError("n_radial_samples must be at least 1")

    if n_radial_samples > 1 and not 0 < radial_width < 2 * distance_to_shell:
        raise ValueError(
            "radial_width must be positive and less than twice distance_to_shell "
            "when averaging radially"
        )

    if isinstance(origin, (tuple, list)):
        origin = np.array(origin)

    if origin is not None and origin.shape != (3,):
        raise ValueError("origin must be a sequence of length 3")

    n_angular_samples = latitude.shape[0] if latitude.ndim == 2 else 1
    radii, radial_weights = _radial_quadrature(distance_to_shell, radial_width, n_radial_samples)

    # Sub-samples are laid out radial-major, then angular, then pixel, so that the
    # weights are simply the outer product of the two sets.
    pixel_coords = np.concatenate(
        [
            transform_to_pixel_coords(
                comoving_radius=radius,
                latitude=latitude.ravel(),
                longitude=longitude.ravel(),
                origin=origin,
                rotation=rotation,
            )
            for radius in radii
        ],
        axis=1,
    )

    weights = None
    if n_radial_samples > 1 or n_angular_samples > 1:
        weights = np.repeat(radial_weights, n_angular_samples) / n_angular_samples

    coordmap = partial(
        _interpolate_coeval,
        coordinates=pixel_coords,
        order=interpolation_order,
        weights=weights,
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
        Each coeval box can be a different simulated field. If you are tiling the same
        box onto many shells at ``interpolation_order >= 2``, pass boxes pre-filtered by
        :func:`prefilter_coeval`, so that the spline pre-filter is computed once rather
        than once per shell.

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
    keywords = dict(interpolator.keywords)

    # Each sub-sample has its own line of sight, so the projection must be done
    # sub-sample by sub-sample and only then averaged. Strip the averaging off the
    # interpolator and re-apply it at the end.
    weights = keywords.pop("weights", None)
    raw_interpolator = partial(
        _interpolate_coeval, coordinates=keywords["coordinates"], order=keywords["order"]
    )

    pixel_coords = keywords["coordinates"]
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
        cvf_interp = np.array([raw_interpolator(c) for c in cvf]) * unit

        # Now take the dot product of the vector field with (negative) pixel coordinates to get
        # the LoS comp.
        cvf_interp *= -pixel_coords
        return _average_subsamples(np.sum(cvf_interp, axis=0) / coord_norm, weights)

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


def healpix_subpixel_lonlat(
    nside: int, order: Literal["ring", "nested"] = "ring", subsample_level: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Angular coordinates of the HEALPix pixels, optionally sub-divided.

    With ``subsample_level = 0`` this returns the ``(npix,)`` pixel centres. With
    ``subsample_level = k`` it returns ``(4**k, npix)`` arrays holding the centres of
    the ``4**k`` sub-pixels of side ``nside * 2**k`` that exactly tile each pixel. Since
    HEALPix sub-pixels are equal-area and tile their parent exactly, the unweighted mean
    over them is an unbiased estimate of the mean of the field over the pixel.

    Parameters
    ----------
    nside
        The Nside parameter of the healpix map.
    order
        The ordering of the pixels in the healpix map.
    subsample_level
        How many times to halve the pixel side. Each level costs four times as many
        interpolations.

    Returns
    -------
    latitude, longitude
        In radians, of shape ``(npix,)`` or ``(4**k, npix)``.
    """
    if subsample_level < 0:
        raise ValueError("subsample_level must be non-negative")

    hp = HEALPix(nside=nside, order=order)

    if subsample_level == 0:
        lon, lat = hp.healpix_to_lonlat(np.arange(hp.npix))
        return lat.to_value("radian"), lon.to_value("radian")

    # Sub-pixels of a pixel are contiguous in NESTED ordering, which is the whole
    # reason the sub-division is exact: sub-pixel ``m`` of parent ``p`` is simply
    # ``p * 4**k + m``.
    nsub = 4**subsample_level
    parent = np.arange(hp.npix)
    if order == "ring":
        parent = hp.ring_to_nested(parent)

    subgrid = HEALPix(nside=nside * 2**subsample_level, order="nested")
    lon, lat = subgrid.healpix_to_lonlat(
        (parent[None, :] * nsub + np.arange(nsub)[:, None]).ravel()
    )

    shape = (nsub, hp.npix)
    return lat.to_value("radian").reshape(shape), lon.to_value("radian").reshape(shape)


def make_healpix_lightcone_slice(
    nside: int,
    order: Literal["ring", "nested"] = "ring",
    subsample_level: int = 0,
    **kwargs: Any,
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
    subsample_level
        If zero (the default), each pixel takes a single sample of the field at its
        centre, which is the historical behaviour: the resulting map is a *sampled*
        field, not a pixelised one, so the HEALPix pixel window does not apply to it and
        must not be divided out.

        If ``k > 0``, each pixel is instead averaged over the ``4**k`` sub-pixels of a
        map with ``nside * 2**k``. The map is then genuinely pixelised, so its angular
        power spectrum carries the usual pixel window (``healpy.pixwin(nside)**2``) and
        power above ``l ~ 2 Nside`` is suppressed rather than aliased down. Cost grows
        as ``4**k``; ``k = 2`` is usually enough.

    Other Parameters
    ----------------
    All other parameters are passed through to :func:`make_lightcone_slice`.
    """
    lat, lon = healpix_subpixel_lonlat(nside=nside, order=order, subsample_level=subsample_level)

    yield from make_lightcone_slice(latitude=lat, longitude=lon, **kwargs)


def _slice_edges(distance: np.ndarray) -> np.ndarray:
    """Return the ``nslice + 1`` cell edges implied by the slice centres ``distance``.

    Interior edges sit midway between neighbouring slices; the two outer edges are
    placed so that the first and last cells are symmetric about their own centres. For
    a regular grid this is simply ``distance +/- delta / 2``.
    """
    mid = 0.5 * (distance[1:] + distance[:-1])
    return np.concatenate(([2 * distance[0] - mid[0]], mid, [2 * distance[-1] - mid[-1]]))


def _average_over_cells(
    fine_field: np.ndarray, fine_lo: float, fine_dx: float, edges: np.ndarray
) -> np.ndarray:
    """Average a piecewise-constant radial field over a coarser set of cells.

    ``fine_field`` holds the mean value of the field in each of a contiguous set of
    cells of width ``fine_dx``, the first of which starts at ``fine_lo``. The result is
    the mean of that same field over the cells delimited by ``edges``, computed exactly
    by differencing the cumulative integral -- which is piecewise linear, so
    interpolating it linearly is not an approximation.

    This is the step that makes the sub-cell refinement in :func:`apply_rsds` a genuine
    convergence parameter: *sampling* the fine grid at the output slice centres throws
    away everything that landed between them, whereas integrating over the output cell
    keeps all of it.
    """
    nfine = fine_field.shape[0]

    # Cumulative integral of the field up to each fine-cell edge.
    cumulative = np.concatenate(
        (np.zeros((1, fine_field.shape[1])), np.cumsum(fine_field, axis=0) * fine_dx)
    )

    # Position of each output edge in units of fine cells. The fine grid is built so
    # that output edges fall on fine-cell edges whenever the output grid is regular;
    # snapping to the integer recovers that exactly in the face of round-off, which is
    # what makes a zero displacement an exact round trip.
    position = (edges - fine_lo) / fine_dx
    whole = np.round(position)
    position = np.clip(np.where(np.abs(position - whole) < 1e-8, whole, position), 0.0, nfine)

    # Split into the index of the fine cell each edge falls in, and the fraction of the
    # way across that cell.
    index = np.clip(position.astype(np.int64), 0, nfine - 1)
    frac = position - index

    integral = cumulative[index] + frac[:, None] * (cumulative[index + 1] - cumulative[index])
    return np.diff(integral, axis=0) / np.diff(edges)[:, None]


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
    then displace the field on that grid.

    The displaced fine grid is finally *integrated* over the radial extent of each
    output slice, rather than sampled at its centre. Every parcel therefore lands in
    exactly one output cell (in the proportions in which it straddles them), mass is
    conserved, and ``n_subcells`` is a real convergence parameter: raising it shrinks
    the cloud-in-cell kernel without anything falling between the output slices.

    Parameters
    ----------
    field
        The field to apply redshift-space distortions to, shape (nslices, ncoords).
    los_displacement
        The line-of-sight "apparent" displacement of the field, in pixel coordinates.
        Equal to ``v / H(z) / cell_size``.
        Positive values are towards the observer, shape ``(nslices, ncoords)``.
        This is the same sign convention as the output of
        :func:`make_lightcone_slice_vector_field`, so the two can be chained directly.
        A parcel at comoving distance ``d`` with displacement ``u`` is observed at
        apparent distance ``d - u``.
    distance
        The comoving distance to each slice in the field, in units of the cell size.
        shape (nslices,).
    n_subcells
        The number of sub-cells per (smallest) output slice used for the displacement.
        Larger values resolve the displacement field more finely and give a more
        accurate answer, at proportionally greater cost.
    """
    if field.shape != los_displacement.shape:
        raise ValueError("field and los_displacement must have the same shape")
    if field.shape[0] < 2:
        raise ValueError("field must have at least 2 slices")
    if field.shape[0] != distance.size:
        raise ValueError("field and distance must have the same number of slices")

    is_regular = np.allclose(np.diff(np.diff(distance)), 0.0)
    interpolator = RegularGridInterpolator if is_regular else RectBivariateSpline

    smallest_slice = np.min(np.diff(distance))
    rsd_dx = smallest_slice / n_subcells
    dist = distance.to_value(rsd_dx.unit)
    dx = rsd_dx.value

    # The output slices are cells, not points: they run from edge to edge.
    edges = _slice_edges(dist)

    # Displacement in units of the fine cell, which is what cloud-in-cell wants.
    los_cells = np.asarray(los_displacement / rsd_dx)

    # Pad the fine grid by a whole number of fine cells at each end, so that material
    # displaced off either end of the output grid still has somewhere to land -- and so
    # that a regular output grid lines up exactly with the fine one.
    n_near = int(np.ceil(max(np.max(los_cells[0]), 0.0)))
    n_far = int(np.ceil(-min(np.min(los_cells[-1]), 0.0)))
    n_body = int(np.ceil((edges[-1] - edges[0]) / dx - 1e-8))
    nfine = n_near + n_body + n_far
    fine_lo = edges[0] - n_near * dx
    fine_grid = fine_lo + (np.arange(nfine) + 0.5) * dx

    # Refine the field conservatively: every fine cell takes the value of the output
    # slice it lies in (and the end slices' values beyond the grid). Unlike
    # interpolating it, this leaves the mean over each output slice untouched, so with
    # no displacement the refine-displace-average round trip is the identity.
    fine_field = np.asarray(field)[np.clip(np.searchsorted(edges, fine_grid) - 1, 0, dist.size - 1)]

    # The displacement, by contrast, is a smooth function sampled at the slice centres,
    # so interpolate it.
    ang_coords = np.arange(field.shape[1])
    if is_regular:
        x, y = np.meshgrid(fine_grid, ang_coords, indexing="ij")
        fine_rsd = interpolator(
            (dist, ang_coords),
            los_cells,
            bounds_error=False,
            fill_value=None,
        )((x.flatten(), y.flatten())).reshape(x.shape)
    else:
        fine_rsd = interpolator(dist, ang_coords, los_cells)(fine_grid, ang_coords)

    # ``fine_grid`` runs from near to far, but ``los_displacement`` is positive towards
    # the observer, so the displacement *along the grid axis* is the negative of it.
    fine_field = cloud_in_cell_los(fine_field, -fine_rsd)

    # Integrate over each output slice rather than sampling it at the centre.
    return _average_over_cells(fine_field, fine_lo, dx, edges)
