"""Tests of the shared cardinal B-spline mathematics.

Everything in :mod:`cosmotile._spline` is backend-agnostic arithmetic, and every claim
made about it is exact rather than statistical. The stencil it produces must reproduce
:func:`scipy.ndimage.map_coordinates` to roundoff for every order in range -- that is
the whole basis on which a second backend can be trusted to agree with the first -- and
the integer samples it returns must match the B-spline ``scipy`` builds from knots.

The two are deliberately the same code: ``cardinal_bspline_at_integers`` is
``spline_weights`` evaluated at the offset that lands the stencil on integer arguments.
A test that they agree with independent ``scipy`` constructions is therefore a test that
the one recursion is right, not that two implementations were kept in step by hand.
"""

import numpy as np
import pytest
from scipy.interpolate import BSpline
from scipy.ndimage import map_coordinates, spline_filter

from cosmotile._spline import (
    MAX_ORDER,
    base_index,
    bspline_dtft,
    cardinal_bspline_at_integers,
    spline_weights,
)

ORDERS = list(range(MAX_ORDER + 1))


@pytest.mark.parametrize("order", ORDERS)
def test_weights_sum_to_one(order: int) -> None:
    """A partition of unity, or the interpolation would not preserve a constant field."""
    offset = np.linspace(0.0, 1.0, 101, endpoint=False)
    np.testing.assert_allclose(np.asarray(spline_weights(offset, order)).sum(axis=0), 1.0)


@pytest.mark.parametrize("order", ORDERS)
def test_weights_are_non_negative(order: int) -> None:
    """B-splines are non-negative, so the stencil can never overshoot its inputs."""
    offset = np.linspace(0.0, 1.0, 101, endpoint=False)
    assert np.asarray(spline_weights(offset, order)).min() >= 0.0


@pytest.mark.parametrize("order", ORDERS)
def test_integer_samples_match_scipy_bspline(order: int) -> None:
    """The recursion reproduces the kernel ``scipy`` builds from its knot vector.

    This is the one place the two could drift apart, because the pre-filter inverts
    these samples while the interpolation uses the recursion: if they disagreed, tiling
    at ``order >= 2`` would not pass through the grid values at all.
    """
    knots = np.arange(order + 2) - (order + 1) / 2
    reference = BSpline.basis_element(knots, extrapolate=False)(
        np.arange(order // 2 + 1, dtype=float)
    )
    np.testing.assert_array_equal(cardinal_bspline_at_integers(order), reference)


@pytest.mark.parametrize("order", ORDERS)
def test_stencil_reproduces_map_coordinates(order: int) -> None:
    """The end-to-end claim: base index plus weights *is* ``map_coordinates``.

    Checked at coordinates well outside the grid, since the wrap is part of the
    contract, and against the same ``grid-wrap`` pre-filter the library uses.
    """
    rng = np.random.default_rng(20260919)
    n = 16
    box = rng.standard_normal((n, n, n))
    coeff = (
        spline_filter(box, order=order, mode="grid-wrap", output=np.float64) if order > 1 else box
    )
    coords = rng.uniform(-3 * n, 4 * n, size=(3, 200))

    expected = map_coordinates(coeff, coords, order=order, mode="grid-wrap", prefilter=False)

    base, offset = base_index(coords, order)
    weights = np.asarray(spline_weights(offset, order))
    got = np.zeros(coords.shape[1])
    for a in range(order + 1):
        for b in range(order + 1):
            for c in range(order + 1):
                got += (
                    weights[a, 0]
                    * weights[b, 1]
                    * weights[c, 2]
                    * coeff[
                        (base[0] + a) % n,
                        (base[1] + b) % n,
                        (base[2] + c) % n,
                    ]
                )

    np.testing.assert_allclose(got, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("order", ORDERS)
def test_base_index_offset_is_a_unit_fraction(order: int) -> None:
    """The offset must land in ``[0, 1)``: that is what keeps float32 usable."""
    rng = np.random.default_rng(3)
    coords = rng.uniform(-50.0, 50.0, size=1000)
    base, offset = base_index(coords, order)
    assert offset.min() >= 0.0
    assert offset.max() < 1.0
    np.testing.assert_allclose(base + offset, coords - order / 2 + 0.5)


@pytest.mark.parametrize("order", [0, 1])
def test_interpolating_orders_need_no_prefilter(order: int) -> None:
    """Orders 0 and 1 have a delta-function sampled kernel, so ``b_p`` is identically one."""
    np.testing.assert_array_equal(cardinal_bspline_at_integers(order), [1.0])
    k = np.linspace(-np.pi, np.pi, 33)
    np.testing.assert_allclose(bspline_dtft(k, order), 1.0)


@pytest.mark.parametrize("order", ORDERS)
def test_dtft_is_positive_so_dividing_by_it_is_stable(order: int) -> None:
    """The pre-filter divides by ``b_p``; if it vanished anywhere, that would blow up."""
    k = np.linspace(-np.pi, np.pi, 257)
    assert np.asarray(bspline_dtft(k, order)).min() > 0.0


@pytest.mark.parametrize("order", ORDERS)
def test_dtft_is_the_transform_of_the_integer_samples(order: int) -> None:
    """``b_p(k)`` really is the discrete-time transform of the sampled kernel."""
    samples = cardinal_bspline_at_integers(order)
    k = np.linspace(-np.pi, np.pi, 65)
    expected = samples[0] + sum(
        2 * value * np.cos(offset * k) for offset, value in enumerate(samples[1:], start=1)
    )
    np.testing.assert_allclose(bspline_dtft(k, order), expected)
