"""Host-side machinery for redshift-space distortions.

:func:`apply_rsds` is the one operation in ``cosmotile`` whose *shapes* depend on its
input values: how far the grid must be padded is set by the largest displacement in the
field, and a shape that depends on a value cannot be traced. Everything that does not
depend on the field values is therefore precomputed here, into an :class:`RsdPlan`, and
the padding becomes an argument rather than a measurement.

Two of those precomputations deserve a note, because they are what makes a JAX version
possible at all.

**The displacement interpolation is a linear operator in radius.** Both branches of the
original code are: on a regular grid, :class:`scipy.interpolate.RegularGridInterpolator`
is *bit-identical* to one-dimensional linear interpolation with linear extrapolation,
with the angular axis a complete no-op since it is evaluated at its own integer nodes.
On an irregular grid, :class:`scipy.interpolate.RectBivariateSpline` is a cubic spline
that is emphatically *not* close to the linear one -- but it is still linear and
column-separable, so probing it with unit impulses recovers its matrix to about
``1e-13``. Carrying that matrix reproduces the original numerics exactly while leaving
FITPACK on the host, where it belongs.

**The re-binning weights are fixed by the grids alone.** ``searchsorted`` runs here, on
edges that are known before any field arrives, so the traced code is a gather and a
difference.
"""

import dataclasses
from typing import Any, Literal

import numpy as np
from scipy.interpolate import RectBivariateSpline


@dataclasses.dataclass(frozen=True)
class RsdPlan:
    """Everything about an RSD application that does not depend on the field values.

    Built by :func:`make_rsd_plan`. Reusable across as many fields as share a radial
    grid, which for a lightcone is all of them.
    """

    #: ``(nfine,)`` width of each sub-cell, in the units of ``distance``.
    fine_widths: Any

    #: ``(nfine,)`` index of the input slice each sub-cell takes its value from.
    refine_index: Any

    #: ``(nfine,)`` how much of the padding is real field: zero for ``outside="empty"``
    #: and one for ``outside="edge"``. One inside the grid either way.
    source_mask: Any

    #: ``(nfine, k)`` radial stencil for the displacement. ``k`` is 2 on a regular
    #: grid, and ``nslice`` on an irregular one, where the spline is global.
    interp_index: Any

    #: ``(nfine, k)`` matching stencil weights.
    interp_weight: Any

    #: ``(nslice + 1,)`` sub-cell each output edge falls in, for the re-binning.
    rebin_index: Any

    #: ``(nslice + 1,)`` how far across that sub-cell it falls.
    rebin_frac: Any

    #: ``(nslice,)`` width of each output slice.
    out_widths: Any

    #: ``(nfine,)`` cumulative width up to each sub-cell's near edge.
    fine_cumulative: Any

    #: Number of output slices.
    nslice: int

    #: Sub-cells of padding at the near and far ends.
    n_near: int
    n_far: int

    #: Sub-cells per output slice.
    n_subcells: int

    #: What the field does beyond the grid: ``"empty"`` or ``"edge"``.
    outside: str


def slice_edges(distance: np.ndarray) -> np.ndarray:
    """Return the ``nslice + 1`` cell edges implied by the slice centres ``distance``.

    Interior edges sit midway between neighbouring slices; the two outer edges are
    placed so that the first and last cells are symmetric about their own centres. For
    a regular grid this is simply ``distance +/- delta / 2``.
    """
    mid = 0.5 * (distance[1:] + distance[:-1])
    return np.concatenate(([2 * distance[0] - mid[0]], mid, [2 * distance[-1] - mid[-1]]))


def is_regular(distance: np.ndarray) -> bool:
    """Whether the slice centres are evenly spaced."""
    return bool(np.allclose(np.diff(np.diff(distance)), 0.0))


def _linear_stencil(distance: np.ndarray, fine_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two-point stencil for linear interpolation in radius, extrapolating linearly.

    Bit-identical to what :class:`scipy.interpolate.RegularGridInterpolator` does with
    ``fill_value=None``, which is the branch the original code takes on a regular grid.
    """
    upper = np.clip(np.searchsorted(distance, fine_grid, side="left"), 1, distance.size - 1)
    lower = upper - 1
    span = distance[upper] - distance[lower]
    frac = (fine_grid - distance[lower]) / span
    return (
        np.stack([lower, upper], axis=1).astype(np.int32),
        np.stack([1.0 - frac, frac], axis=1),
    )


#: Angular nodes used to probe the spline. ``RectBivariateSpline`` defaults to a cubic
#: in both directions and needs more nodes than its degree, so four is the minimum that
#: builds. The probe is constant along that axis and a tensor-product spline reproduces
#: a constant exactly, so the radial operator it recovers does not depend on how many
#: angular nodes were used -- four gives the same matrix as four hundred.
_PROBE_ANGLES = 4


def _spline_matrix(distance: np.ndarray, fine_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Recover the ``RectBivariateSpline`` operator by probing it with unit impulses.

    The spline is linear in its data and separable across the angular axis, so feeding
    it an impulse in radius -- held constant in angle -- gives one column of its matrix.
    That is exact to roundoff, and it keeps FITPACK, which has no JAX equivalent, on the
    host.

    Worth being explicit about why the matrix is needed at all: on an irregular grid this
    branch is a *cubic* spline, and it is nowhere near the linear interpolation the
    regular branch does -- they differ by order unity on the same data. Replacing it with
    linear interpolation would silently change the physics, so the operator is carried
    across instead.
    """
    nslice = distance.size
    angular = np.arange(_PROBE_ANGLES)
    impulses = np.eye(nslice)
    matrix = np.stack(
        [
            RectBivariateSpline(
                distance, angular, np.tile(impulses[:, i : i + 1], (1, _PROBE_ANGLES))
            )(fine_grid, angular)[:, 0]
            for i in range(nslice)
        ],
        axis=1,
    )
    index = np.ascontiguousarray(np.broadcast_to(np.arange(nslice, dtype=np.int32), matrix.shape))
    return index, matrix


def make_rsd_plan(
    distance: np.ndarray,
    *,
    n_subcells: int = 4,
    max_displacement: float,
    outside: Literal["empty", "edge"] = "edge",
) -> RsdPlan:
    """Precompute everything about an RSD application that the field values do not set.

    Parameters
    ----------
    distance
        ``(nslice,)`` comoving distance to each slice, in cells. Plain numbers, not a
        :class:`~astropy.units.Quantity`: units do not survive a traced computation, so
        strip them before you get here.
    n_subcells
        Sub-cells per output slice used for the displacement. A genuine convergence
        parameter: raising it shrinks the cloud-in-cell kernel.
    max_displacement
        The largest line-of-sight displacement the plan must accommodate, in cells, and
        the reason a JAX version is possible: it replaces the original's measurement of
        the actual displacement, which made the padded shapes depend on the data.

        Pass a physical bound -- ``max |v| / H / cell_size`` -- or, if you have the field
        already, ``float(np.abs(los_displacement).max())``. Too large merely wastes
        memory.

        With ``outside="empty"`` too small a bound does not corrupt the interior:
        material simply leaves the grid sooner than it would have, which is already what
        happens at the true edges. With ``outside="edge"`` it does matter, because the
        padding is also where material flows *in* from -- make it comfortably larger than
        the displacement at the first and last slices.
    outside
        What the field does beyond the range ``distance`` covers: ``"edge"`` (the
        default) continues it at the first and last slice values, ``"empty"`` takes it
        to be zero. See :func:`cosmotile.apply_rsds`, whose keyword this mirrors.

    Returns
    -------
    plan
        Pass to :func:`cosmotile.jax.apply_rsds`.
    """
    distance = np.asarray(distance, dtype=float)
    if distance.ndim != 1:
        raise ValueError("distance must be a 1D array")
    if distance.size < 2:
        raise ValueError("distance must have at least 2 slices")
    if not isinstance(n_subcells, (int, np.integer)) or n_subcells < 1:
        raise ValueError("n_subcells must be a positive integer")
    if max_displacement < 0:
        raise ValueError("max_displacement must be non-negative")
    if outside not in ("empty", "edge"):
        raise ValueError("outside must be 'empty' or 'edge'")

    nslice = distance.size
    edges = slice_edges(distance)
    widths = np.diff(edges)

    # Subdivide each output cell, so every output edge is also a fine edge. On an
    # irregular grid that is what keeps the refine-average round trip exact.
    body = np.repeat(widths / n_subcells, n_subcells)

    n_near = int(np.ceil(max_displacement / body[0]))
    n_far = int(np.ceil(max_displacement / body[-1]))

    fine_widths = np.concatenate((np.full(n_near, body[0]), body, np.full(n_far, body[-1])))
    fine_edges = np.empty(fine_widths.size + 1)
    fine_edges[0] = edges[0] - n_near * body[0]
    fine_edges[1:] = fine_edges[0] + np.cumsum(fine_widths)
    fine_grid = 0.5 * (fine_edges[:-1] + fine_edges[1:])

    refine_index = np.concatenate(
        (
            np.zeros(n_near, dtype=np.int32),
            np.repeat(np.arange(nslice, dtype=np.int32), n_subcells),
            np.full(n_far, nslice - 1, dtype=np.int32),
        )
    )

    # Outside the grid the displacement is held at its boundary value rather than
    # extrapolated: a linearly extrapolated velocity grows without bound, so the far
    # padding would be flung arbitrarily far and the answer would depend on how much of
    # it there was. Fine cells inside the grid still extrapolate.
    probe = np.clip(fine_grid, edges[0], edges[-1]) if outside == "edge" else fine_grid

    stencil = _linear_stencil if is_regular(distance) else _spline_matrix
    interp_index, interp_weight = stencil(distance, probe)

    # Locate each output edge in the fine grid, exactly as _rebin does.
    clipped = np.clip(edges, fine_edges[0], fine_edges[-1])
    rebin_index = np.clip(
        np.searchsorted(fine_edges, clipped, side="right") - 1, 0, fine_widths.size - 1
    )
    rebin_frac = (clipped - fine_edges[rebin_index]) / fine_widths[rebin_index]

    # 'edge' carries the first and last slice values outward; 'empty' says there is
    # nothing out there. Applied as a mask so the traced executor needs no branch.
    pad = 1.0 if outside == "edge" else 0.0
    source_mask = np.concatenate(
        (np.full(n_near, pad), np.ones(nslice * n_subcells), np.full(n_far, pad))
    )

    return RsdPlan(
        fine_widths=fine_widths,
        refine_index=refine_index,
        source_mask=source_mask,
        interp_index=interp_index,
        interp_weight=interp_weight,
        rebin_index=rebin_index.astype(np.int32),
        rebin_frac=rebin_frac,
        out_widths=widths,
        fine_cumulative=fine_edges[:-1] - fine_edges[0],
        nslice=nslice,
        n_near=n_near,
        n_far=n_far,
        n_subcells=n_subcells,
        outside=outside,
    )
