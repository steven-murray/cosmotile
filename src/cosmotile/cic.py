"""An implementation of Cloud-in-Cell interpolation in 3D."""

from __future__ import annotations

import numpy as np
from numba import njit


@njit  # type: ignore
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
        The displacement must be in units of the cell size.
    """
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

                # Get the cell into which the cell was shifted.
                # modulo the number of cells so we wrap around.
                # Note that i,j,k is the coordinates of the centre of the cell
                # just *below* the offset position. So the cloud only hits
                # this cell and cells *above* it, not below.
                i = int(ii + ddx) % nx
                j = int(jj + ddy) % ny
                k = int(kk + ddz) % nz

                ip = (i + 1) % nx
                jp = (j + 1) % ny
                kp = (k + 1) % nz

                # Get the offset of the cell within the parent cell, assuming
                # a resolution of unity. If the offset is negative, then ddx is defined
                # as the positive distance between the cell before it and the shifted
                # point (distance between each point is unity).
                #   o            o            o            o            o
                #   |      <-------ddx--------x
                #   i      <-----| = ddx % 1
                #   <-ddx-><-tx->
                #
                # In the positive ddx case:
                #                                          i
                #                                          |
                #   o            o            o            o            o
                #                             x-------ddx------->
                #                                ddx % 1 = |---->
                #                                          <-ddx-><-tx->
                # We have to put a clause in to check when a negative ddx takes you
                # to exactly a node, because then we get the mod wrong.
                if ddx < 0:
                    tx = ddx % 1
                    if tx == 0:
                        tx = 1
                    ddx = 1 - tx
                else:
                    ddx %= 1
                    tx = 1 - ddx

                if ddy < 0:
                    ty = ddy % 1
                    if ty == 0:
                        ty = 1
                    ddy = 1 - ty
                else:
                    ddy %= 1
                    ty = 1 - ddy

                if ddz < 0:
                    tz = ddz % 1
                    if tz == 0:
                        tz = 1
                    ddz = 1 - tz
                else:
                    ddz %= 1
                    tz = 1 - ddz

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
