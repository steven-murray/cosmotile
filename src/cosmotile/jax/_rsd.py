"""Redshift-space distortions, as a traced and differentiable computation."""

import jax.numpy as jnp
from jax import Array
from jax.typing import DTypeLike

from .._rsd import RsdPlan


def _deposit(out: Array, index: Array, angle: Array, value: Array, nfine: int) -> Array:
    """Scatter-add ``value`` at ``index``, discarding anything off either end of the grid.

    ``mode="drop"`` alone is not enough. JAX drops indices at or above the axis length,
    but a *negative* index is still interpreted Python-style and wraps to the far end --
    so material displaced off the near edge would silently reappear at the far edge,
    which looks entirely plausible and is completely wrong. Mapping negatives to
    ``nfine`` puts them in the range ``drop`` really does drop.
    """
    return out.at[jnp.where(index < 0, nfine, index), angle].add(value, mode="drop")


def _cloud_in_cell(field: Array, shift: Array) -> Array:
    """Deposit each fine cell at its displaced position, splitting between neighbours.

    The NumPy original loops and guards each index with ``0 <= i < nfine``; here the
    same thing is two masked scatter-adds. Mass that leaves the padded grid is gone,
    which is the physically correct outcome and what makes the padding meaningful.
    """
    nfine = field.shape[0]
    position = jnp.arange(nfine)[:, None] + shift
    lower = jnp.floor(position).astype(jnp.int32)
    frac = position - lower

    angle = jnp.broadcast_to(jnp.arange(field.shape[1]), field.shape)
    out = jnp.zeros_like(field)
    out = _deposit(out, lower, angle, (1.0 - frac) * field, nfine)
    return _deposit(out, lower + 1, angle, frac * field, nfine)


def apply_rsds(field: Array, los_displacement: Array, plan: RsdPlan) -> Array:
    """Apply redshift-space distortions to a field, on a fixed plan.

    Computes the same thing as :func:`cosmotile.apply_rsds`, but with every shape fixed
    by ``plan`` rather than by the data, so it can be jitted, vmapped and
    differentiated. It is differentiable in **both** arguments: linearly in ``field``,
    and piecewise-linearly in ``los_displacement`` through the cloud-in-cell kernel.

    Parameters
    ----------
    field
        ``(nslice, nangles)`` radial *cell averages* -- see the convention note on
        :func:`cosmotile.apply_rsds`, which applies here unchanged.
    los_displacement
        ``(nslice, nangles)`` apparent displacement in cells, positive towards the
        observer, matching :func:`cosmotile.make_lightcone_slice_vector_field`.
    plan
        From :func:`cosmotile._rsd.make_rsd_plan`, built for this radial grid and a
        bound on the displacement.

    Returns
    -------
    distorted
        ``(nslice, nangles)``, again as radial cell averages. Mass that leaves the
        padded grid is gone, as it should be.
    """
    if field.shape != los_displacement.shape:
        raise ValueError("field and los_displacement must have the same shape")
    if field.shape[0] != plan.nslice:
        raise ValueError(
            f"field has {field.shape[0]} slices but the plan was built for {plan.nslice}"
        )

    dtype = jnp.result_type(field, los_displacement)
    fine_widths = jnp.asarray(plan.fine_widths, dtype=dtype)

    # Refine conservatively: each sub-cell takes the value of the slice it lies in, so
    # the mean over each output slice is untouched and a zero displacement is exact. What
    # the padding holds is the plan's ``outside`` choice, applied here as a mask: zero
    # beyond the grid, or the edge value carried outward.
    fine_field = jnp.asarray(field, dtype=dtype)[jnp.asarray(plan.refine_index)]
    fine_field = fine_field * jnp.asarray(plan.source_mask, dtype=dtype)[:, None]

    # The displacement is smooth, so it is interpolated rather than repeated -- through
    # the operator the plan precomputed, then expressed in local sub-cells.
    weight = jnp.asarray(plan.interp_weight, dtype=dtype)
    displacement = jnp.einsum(
        "fk,fka->fa", weight, jnp.asarray(los_displacement, dtype=dtype)[plan.interp_index]
    )
    displacement = displacement / fine_widths[:, None]

    # The grid runs from near to far but the displacement is positive towards the
    # observer, so along the grid axis it is the negative of it.
    fine_field = _cloud_in_cell(fine_field, -displacement)

    return _rebin(fine_field, fine_widths, plan, dtype)


def _rebin(fine_field: Array, fine_widths: Array, plan: RsdPlan, dtype: DTypeLike) -> Array:
    """Integrate the displaced fine grid over each output slice.

    Integrating rather than sampling the slice centre is what makes ``n_subcells`` a
    real convergence parameter: nothing that landed between the output centres is
    thrown away.
    """
    cumulative = jnp.concatenate(
        (
            jnp.zeros((1, fine_field.shape[1]), dtype),
            jnp.cumsum(fine_field * fine_widths[:, None], axis=0),
        )
    )
    index = jnp.asarray(plan.rebin_index)
    frac = jnp.asarray(plan.rebin_frac, dtype=dtype)[:, None]
    integral = cumulative[index] + frac * (cumulative[index + 1] - cumulative[index])
    return jnp.diff(integral, axis=0) / jnp.asarray(plan.out_widths, dtype=dtype)[:, None]
