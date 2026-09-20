"""Tests of the parallel NumPy-backend gather.

The kernel in :mod:`cosmotile._gather` exists only to be faster than
:func:`scipy.ndimage.map_coordinates`, so the whole of its contract is that it computes
the same thing. ``scipy`` is what the physics tests in this suite were calibrated
against, and swapping the default implementation underneath them is only safe because
of what is checked here.

Agreement is to roundoff rather than bit for bit -- a parallel sum and a serial one add
in different orders -- and exact at order 0, where the answer is a single box value and
there is nothing to round.

The short-axis cases are a regression test. The stencil for order 5 spans six samples,
so on an axis of two cells it wraps around the box three times; an implementation that
wrapped by subtracting a single period would index off the end of the array. The
window tests in ``test_theory.py`` tile a ``(16, 2, 2)`` box and caught exactly that.
"""

import numpy as np
import pytest
from scipy.ndimage import map_coordinates, spline_filter

import cosmotile as cmt
from cosmotile import _gather
from cosmotile._spline import MAX_ORDER

ORDERS = list(range(MAX_ORDER + 1))


def _reference(box: np.ndarray, coords: np.ndarray, order: int) -> np.ndarray:
    coefficients = (
        spline_filter(box, order=order, mode="grid-wrap", output=np.float64) if order > 1 else box
    )
    return map_coordinates(
        coefficients, coords, order=order, mode="grid-wrap", prefilter=False
    ), coefficients


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((16, 16, 16), id="cubic"),
        pytest.param((16, 2, 2), id="axes-shorter-than-the-stencil"),
        pytest.param((1, 8, 8), id="single-cell-axis"),
    ],
)
def test_gather_agrees_with_scipy(order: int, shape: tuple[int, ...]) -> None:
    """The whole contract, including where the stencil wraps the box several times."""
    if not _gather.NUMBA:  # pragma: no cover - the nojit session
        pytest.skip("numba is not installed")

    rng = np.random.default_rng(20260920)
    box = rng.standard_normal(shape)
    # Well outside the box, since the periodic wrap is part of what is being checked.
    coords = rng.uniform(-40.0, 40.0, size=(3, 300))

    expected, coefficients = _reference(box, coords, order)
    np.testing.assert_allclose(
        _gather.gather(coefficients, coords, order), expected, atol=1e-12, rtol=1e-12
    )


def test_order_zero_is_bit_exact() -> None:
    """Nearest neighbour returns a value that is *in* the box, so nothing is rounded."""
    if not _gather.NUMBA:  # pragma: no cover - the nojit session
        pytest.skip("numba is not installed")

    rng = np.random.default_rng(5)
    box = rng.standard_normal((12, 12, 12))
    coords = rng.uniform(-40.0, 40.0, size=(3, 200))
    expected, _ = _reference(box, coords, 0)
    np.testing.assert_array_equal(_gather.gather(box, coords, 0), expected)


@pytest.mark.parametrize("order", [1, 3])
def test_the_scipy_fallback_is_reachable_and_agrees(order: int) -> None:
    """The escape hatch, and the reason it exists.

    ``scipy`` is the reference the physics tests were calibrated against. If one of them
    ever moves, being able to switch this kernel off in a single call is how you find
    out whether it was responsible -- so the switch is tested, not merely present.
    """
    rng = np.random.default_rng(11)
    box = rng.standard_normal((16, 16, 16))
    latitude = rng.uniform(-np.pi / 2, np.pi / 2, 200)
    longitude = rng.uniform(0, 2 * np.pi, 200)
    kwargs = {
        "latitude": latitude,
        "longitude": longitude,
        "distance_to_shell": 9.3,
        "interpolation_order": order,
    }

    fast = next(cmt.make_lightcone_slice(coevals=box, **kwargs))
    try:
        _gather.use_scipy_gather(True)
        assert not _gather.available()
        slow = next(cmt.make_lightcone_slice(coevals=box, **kwargs))
    finally:
        _gather.use_scipy_gather(False)

    assert _gather.available() == _gather.NUMBA
    np.testing.assert_allclose(fast, slow, atol=1e-12, rtol=1e-12)
