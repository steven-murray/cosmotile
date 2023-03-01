"""An implementation of Cloud-in-Cell interpolation in 3D."""

from __future__ import annotations

import warnings

import numpy as np


try:
    from numba import njit

    NUMBA = True
except ImportError:
    NUMBA = False


def cloud_in_cell(
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


if NUMBA:
    cloud_in_cell_nojit = cloud_in_cell
    cloud_in_cell = njit(cloud_in_cell)
