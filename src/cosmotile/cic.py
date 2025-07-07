"""An implementation of Cloud-in-Cell interpolation in 3D."""

from __future__ import annotations

import warnings

import numpy as np

try:
    from numba import njit

    NUMBA = True
except ImportError:
    NUMBA = False


def cloud_in_cell_coeval(
    field: np.ndarray, dx: np.ndarray, dy: np.ndarray, dz: np.ndarray
) -> np.ndarray:
    """
    Interpolates a point in 3D space.

    Note that the implementation was derived largely from
    https://astro.uchicago.edu/~andrey/Talks/PM/pm.pdf, specially slide 13.

    Parameters
    ----------
    field
        The original field before displacement.
    dx, dy, dz
        Displacement of each coordinate in the field in the x,y, and z direction.
        The displacement must be in units of the cell size. Displacements can represent,
        for example, redshift-space distortions (RSDs) or peculiar velocities. If
        velocities are in km / s, then input displacement as v / H(z) / cell_size.
    """
    if not NUMBA:  # pragma: no cover
        warnings.warn("Install numba for a speedup of cloud_in_cell", stacklevel=2)

    if not field.shape == dx.shape == dy.shape == dz.shape:
        raise ValueError("Field and displacement must have the same shape.")

    out = np.zeros_like(field)

    nx, ny, nz = dx.shape
    for ii in range(nx):
        for jj in range(ny):
            for kk in range(nz):
                weight = field[ii, jj, kk]

                # Get the offset of this grid cell
                ddx = dx[ii, jj, kk]
                ddy = dy[ii, jj, kk]
                ddz = dz[ii, jj, kk]

                # adding a value of nx pre-mod ensures we are still within the range [0, nx]
                x = (ii + ddx + nx) % nx
                y = (jj + ddy + ny) % ny
                z = (kk + ddz + nz) % nz

                i = int(x)
                j = int(y)
                k = int(z)

                ip = (i + 1) % nx
                jp = (j + 1) % ny
                kp = (k + 1) % nz

                tx = (i + 1) - x
                ty = (j + 1) - y
                tz = (k + 1) - z

                ddx = 1 - tx
                ddy = 1 - ty
                ddz = 1 - tz

                # Do the trilinear interpolation to surrounding 8 cells
                out[i, j, k] += tx * ty * tz * weight
                out[ip, j, k] += ddx * ty * tz * weight
                out[i, jp, k] += tx * ddy * tz * weight
                out[ip, jp, k] += ddx * ddy * tz * weight
                out[i, j, kp] += tx * ty * ddz * weight
                out[ip, j, kp] += ddx * ty * ddz * weight
                out[i, jp, kp] += tx * ddy * ddz * weight
                out[ip, jp, kp] += ddx * ddy * ddz * weight

    return out


def cloud_in_cell_los(
    field: np.ndarray, delta_los: np.ndarray, periodic: bool = False
) -> np.ndarray:
    """
    Interpolate in the line-of-sight direction using cloud-in-cell algorithm.

    Note that the implementation was derived largely from
    https://astro.uchicago.edu/~andrey/Talks/PM/pm.pdf, specially slide 13.

    Notes
    -----
    The ``field`` here is assumed to be regularly spaced in comoving distance along the
    line-of-sight (the angular coordinates can be arbitrarily arranged). For each angular
    point, we first displace the regular grid along the line-of-sight by ``delta_los``,
    to create a a new, non-regular grid (which we can consider to be "particles"). We
    then use the regular cloud-in-cell interpolation to interpolate the particles back
    on to the regular grid.

    Parameters
    ----------
    field
        The regularly-spaced (along LoS) field before displacement by delta_los.
        Shape ``(nlos_slices, nangles)``.
    delta_los
        Displacement of each coordinate in the field along the LoS.
        The displacement must be in units of the regular grid size, i.e.
        ``v / H(z) / grid_resolution``. Same shape as ``field``.
    periodic
        Whether the field is periodic along the line-of-sight axis.
    """
    if not NUMBA:  # pragma: no cover
        warnings.warn("Install numba for a speedup of cloud_in_cell", stacklevel=2)

    if field.shape != delta_los.shape:
        raise ValueError("Field and displacement must have the same shape.")

    out = np.zeros_like(field)

    nslice, nangles = delta_los.shape
    for ii in range(nslice):
        weight = field[ii]

        # Get the offset of this grid cell
        ddx = delta_los[ii]
        x = ii + ddx

        i = x.astype(np.int32)
        ip = i + 1

        tx = ip - x
        ddx = 1 - tx

        if not periodic:
            for jj in range(nangles):
                if 0 <= i[jj] < nslice:
                    out[i[jj], jj] += tx[jj] * weight[jj]
                if 0 <= ip[jj] < nslice:
                    out[ip[jj], jj] += ddx[jj] * weight[jj]
        else:
            for jj in range(nangles):
                out[i[jj] % nslice, jj] += tx[jj] * weight[jj]
                out[ip[jj] % nslice, jj] += ddx[jj] * weight[jj]
    return out


if NUMBA:
    cloud_in_cell_coeval_nojit = cloud_in_cell_coeval
    cloud_in_cell_coeval = njit(cloud_in_cell_coeval)
    cloud_in_cell_los_nojit = cloud_in_cell_los
    cloud_in_cell_los = njit(cloud_in_cell_los)
