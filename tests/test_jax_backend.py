"""Tests of the optional JAX backend.

The governing claim is **parity**: :mod:`cosmotile.jax` must compute the same thing as
the NumPy path, to double-precision roundoff, for every order and every combination of
sub-sampling, radial averaging, rotation and origin. The NumPy path is calibrated
against ``scipy`` by the rest of this suite, so agreeing with it is what makes the
second backend trustworthy rather than merely fast.

That is not a soft claim. At order 0 the two agree *exactly*, because nearest-neighbour
tiling returns values that are in the box; elsewhere they agree to about ``1e-14``
relative, which is the accumulated roundoff of a different summation order and nothing
else.

Two things beyond parity are checked here, because they are the reason the backend
exists. The **adjoint identity** ``<Ac, y> == <c, A'y>`` holds to machine precision,
which says the gradient really is the transpose of the interpolation and not an
approximation to it. And the **geometry gradient** vanishes identically at order 0 and
is well defined from order 3, which is the rule users need in order to know which order
to differentiate at.

Single precision gets its own tier with its own explicitly stated tolerances, since
JAX defaults to it and the numbers are worth writing down rather than discovering.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from scipy.ndimage import map_coordinates, spline_filter  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

import cosmotile as cmt  # noqa: E402
from cosmotile import jax as cjax  # noqa: E402

ORDERS = list(range(6))
NCELL = 24
RADIUS = 13.7


@pytest.fixture(autouse=True)
def _double_precision():
    """Run every test in double precision, so parity is a real claim.

    Scoped rather than global: flipping ``jax_enable_x64`` for the whole process would
    change the numerics of anything else the user has imported, which is exactly why
    the library itself never does it.
    """
    with jax.enable_x64():
        yield


def _box(seed: int = 7) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((NCELL,) * 3)


def _angles(n: int = 400) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1234)
    return rng.uniform(-np.pi / 2, np.pi / 2, n), rng.uniform(0, 2 * np.pi, n)


# --------------------------------------------------------------------------- parity


@pytest.mark.parametrize("order", ORDERS)
def test_matches_the_numpy_backend(order: int) -> None:
    """The governing claim, at every order."""
    box = _box()
    lat, lon = _angles()

    sampling = cjax.make_shell_sampling(latitude=lat, longitude=lon, order=order)
    got = np.asarray(cjax.shell(cjax.prefilter_coeval(box, order), sampling, RADIUS))

    expected = next(
        cmt.make_lightcone_slice(
            coevals=box,
            latitude=lat,
            longitude=lon,
            distance_to_shell=RADIUS,
            interpolation_order=order,
        )
    )
    np.testing.assert_allclose(got, expected, atol=1e-12, rtol=1e-12)


def test_order_zero_is_bit_exact() -> None:
    """Nearest neighbour returns values that are *in* the box, so there is nothing to round."""
    box = _box()
    lat, lon = _angles()
    sampling = cjax.make_shell_sampling(latitude=lat, longitude=lon, order=0)
    got = np.asarray(cjax.shell(cjax.prefilter_coeval(box, 0), sampling, RADIUS))
    expected = next(
        cmt.make_lightcone_slice(
            coevals=box,
            latitude=lat,
            longitude=lon,
            distance_to_shell=RADIUS,
            interpolation_order=0,
        )
    )
    np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("order", [1, 3, 5])
@pytest.mark.parametrize(
    "geometry",
    [
        pytest.param({"subsample_level": 1}, id="subsampled"),
        pytest.param({"radial_width": 5.0, "n_radial_samples": 4}, id="radial"),
        pytest.param(
            {"subsample_level": 1, "radial_width": 5.0, "n_radial_samples": 4},
            id="subsampled-and-radial",
        ),
    ],
)
def test_matches_the_numpy_backend_for_averaged_pixels(order: int, geometry: dict) -> None:
    """Sub-samples must be laid out and weighted identically, or the averages diverge."""
    box = _box()
    level = geometry.pop("subsample_level", 0)
    lat, lon = cmt.healpix_subpixel_lonlat(8, subsample_level=level)

    sampling = cjax.make_shell_sampling(latitude=lat, longitude=lon, order=order, **geometry)
    got = np.asarray(cjax.shell(cjax.prefilter_coeval(box, order), sampling, RADIUS))

    expected = next(
        cmt.make_lightcone_slice(
            coevals=box,
            latitude=lat,
            longitude=lon,
            distance_to_shell=RADIUS,
            interpolation_order=order,
            **geometry,
        )
    )
    np.testing.assert_allclose(got, expected, atol=1e-11, rtol=1e-11)


@pytest.mark.parametrize("order", [1, 3, 5])
def test_matches_the_numpy_backend_with_rotation_and_origin(order: int) -> None:
    """The rotation matrix crosses the backend boundary, not the ``Rotation`` object."""
    box = _box()
    lat, lon = _angles()
    rotation = Rotation.from_euler("xyz", [0.3, -0.7, 1.1])
    origin = np.array([3.0, -2.0, 11.0])

    sampling = cjax.make_shell_sampling(latitude=lat, longitude=lon, order=order)
    got = np.asarray(
        cjax.shell(
            cjax.prefilter_coeval(box, order),
            sampling,
            RADIUS,
            rotation=jnp.asarray(rotation.as_matrix()),
            origin=jnp.asarray(origin),
        )
    )

    expected = next(
        cmt.make_lightcone_slice(
            coevals=box,
            latitude=lat,
            longitude=lon,
            distance_to_shell=RADIUS,
            interpolation_order=order,
            rotation=rotation,
            origin=origin,
        )
    )
    np.testing.assert_allclose(got, expected, atol=1e-11, rtol=1e-11)


@pytest.mark.parametrize("order", [3, 5])
@pytest.mark.parametrize(
    "shape",
    [pytest.param((16, 2, 2), id="short-axes"), pytest.param((1, 8, 8), id="single-cell-axis")],
)
def test_an_axis_shorter_than_the_stencil_still_wraps_correctly(
    order: int, shape: tuple[int, ...]
) -> None:
    """The order-5 stencil spans six samples, so a two-cell axis wraps three times.

    Wrapping by subtracting a single period -- which is what the fast path does, and is
    correct for every realistic box -- would index off the end here. The window tests in
    ``test_theory.py`` tile a ``(16, 2, 2)`` box, so this is not a hypothetical shape.
    """
    rng = np.random.default_rng(20260920)
    box = rng.standard_normal(shape)
    coords = rng.uniform(-40.0, 40.0, size=(3, 200))

    coefficients = spline_filter(box, order=order, mode="grid-wrap", output=np.float64)
    expected = map_coordinates(coefficients, coords, order=order, mode="grid-wrap", prefilter=False)
    got = cjax.shell_from_coordinates(
        cjax.PrefilteredCoeval(jnp.asarray(coefficients), order),
        jnp.asarray(coords),
        order=order,
    )
    np.testing.assert_allclose(np.asarray(got), expected, atol=1e-12, rtol=1e-12)


# ------------------------------------------------------------------------ prefilter


@pytest.mark.parametrize("order", [2, 3, 4, 5])
def test_fft_prefilter_reproduces_the_iir_filter(order: int) -> None:
    """JAX has no recursive spline filter, so this replaces it -- exactly, not nearly.

    Under the periodic boundary condition the filter is a circular deconvolution, so a
    division in Fourier space is the same operator rather than an approximation to it.
    """
    box = _box()
    expected = spline_filter(box, order=order, mode="grid-wrap", output=np.float64)
    got = np.asarray(cjax.prefilter_coeval(box, order).coefficients)
    np.testing.assert_allclose(got, expected, atol=1e-11, rtol=1e-11)


@pytest.mark.parametrize("order", [0, 1])
def test_interpolating_orders_are_passed_through_unfiltered(order: int) -> None:
    """Orders 0 and 1 need no filter, so the box comes back untouched."""
    box = _box()
    np.testing.assert_array_equal(np.asarray(cjax.prefilter_coeval(box, order)), box)


def test_prefilter_validates_its_order() -> None:
    box = _box()
    with pytest.raises(TypeError, match="must be an integer"):
        cjax.prefilter_coeval(box, 3.0)
    with pytest.raises(ValueError, match="range 0-5"):
        cjax.prefilter_coeval(box, 6)


def test_a_box_filtered_for_the_wrong_order_is_rejected() -> None:
    """The one mistake the tag cannot absorb: genuinely wrong coefficients."""
    lat, lon = _angles(50)
    sampling = cjax.make_shell_sampling(latitude=lat, longitude=lon, order=5)
    with pytest.raises(ValueError, match="pre-filtered for order 3"):
        cjax.shell(cjax.prefilter_coeval(_box(), 3), sampling, RADIUS)


def test_raw_boxes_are_refused_above_order_one() -> None:
    """Filtering inside a shell would redo it per shell, so the backend will not guess."""
    lat, lon = _angles(50)
    sampling = cjax.make_shell_sampling(latitude=lat, longitude=lon, order=3)
    with pytest.raises(ValueError, match="needs B-spline coefficients"):
        cjax.shell(jnp.asarray(_box()), sampling, RADIUS)


# ------------------------------------------------------------------------ gradients


@pytest.mark.parametrize("order", ORDERS)
def test_the_gradient_is_the_transpose_of_the_interpolation(order: int) -> None:
    r"""The adjoint identity, :math:`\langle Ac, y\rangle = \langle c, A^T y\rangle`.

    The interpolation is linear in the box, so this must hold to roundoff. It is a
    stronger statement than a finite-difference check: it says the reverse pass is the
    exact transpose operator, not merely close to the right derivative.
    """
    rng = np.random.default_rng(99)
    box = jnp.asarray(rng.standard_normal((12,) * 3))
    lat, lon = _angles(40)
    sampling = cjax.make_shell_sampling(latitude=lat, longitude=lon, order=order)

    def forward(coefficients):
        return cjax.shell(cjax.PrefilteredCoeval(coefficients, order), sampling, 7.3)

    cotangent = jnp.asarray(rng.standard_normal(sampling.npix))
    out, vjp = jax.vjp(forward, box)
    (gradient,) = vjp(cotangent)

    np.testing.assert_allclose(
        float(jnp.vdot(out, cotangent)), float(jnp.vdot(box, gradient)), rtol=1e-12
    )


def test_gradient_with_respect_to_the_box_matches_finite_differences() -> None:
    """The gradient field-level inference actually wants."""
    rng = np.random.default_rng(4)
    box = jnp.asarray(rng.standard_normal((12,) * 3))
    lat, lon = _angles(40)
    order = 3
    sampling = cjax.make_shell_sampling(latitude=lat, longitude=lon, order=order)

    def loss(coefficients):
        return jnp.sum(cjax.shell(cjax.prefilter_coeval(coefficients, order), sampling, 7.3) ** 2)

    gradient = jax.grad(loss)(box)
    step = 1e-6
    for index in [(2, 3, 4), (7, 1, 9)]:
        difference = (loss(box.at[index].add(step)) - loss(box.at[index].add(-step))) / (2 * step)
        np.testing.assert_allclose(float(gradient[index]), float(difference), rtol=1e-5)


def test_the_geometry_gradient_vanishes_at_order_zero() -> None:
    """Nearest neighbour is piecewise constant, so moving the shell changes nothing.

    This is the rule users need: differentiate with respect to a radius, origin or
    rotation only at order 3 or above, where the reconstruction is smooth.
    """
    box = jnp.asarray(_box(3))
    lat, lon = _angles(40)
    sampling = cjax.make_shell_sampling(latitude=lat, longitude=lon, order=0)
    coefficients = cjax.prefilter_coeval(box, 0)

    def loss(radius):
        return jnp.sum(cjax.shell(coefficients, sampling, radius) ** 2)

    assert float(jax.grad(loss)(RADIUS)) == 0.0


@pytest.mark.parametrize("order", [3, 5])
def test_the_geometry_gradient_is_right_from_order_three(order: int) -> None:
    """Where the reconstruction is smooth, the radius gradient is a real derivative."""
    box = jnp.asarray(_box(3))
    lat, lon = _angles(40)
    sampling = cjax.make_shell_sampling(latitude=lat, longitude=lon, order=order)
    coefficients = cjax.prefilter_coeval(box, order)

    def loss(radius):
        return jnp.sum(cjax.shell(coefficients, sampling, radius) ** 2)

    step = 1e-6
    difference = (loss(RADIUS + step) - loss(RADIUS - step)) / (2 * step)
    np.testing.assert_allclose(float(jax.grad(loss)(RADIUS)), float(difference), rtol=1e-6)


# ----------------------------------------------------------------------- lightcones


def test_lightcone_scan_folds_the_shells() -> None:
    """Scanning must give exactly what looping gives, one shell live at a time."""
    box = jnp.asarray(_box(5))
    lat, lon = cmt.healpix_subpixel_lonlat(8, subsample_level=0)
    order = 3
    sampling = cjax.make_shell_sampling(latitude=lat, longitude=lon, order=order)
    coefficients = cjax.prefilter_coeval(box, order)
    radii = jnp.linspace(6.0, 20.0, 12)

    scanned = cjax.lightcone_scan(
        coefficients, sampling, radii, lambda carry, shell: carry + jnp.mean(shell**2), 0.0
    )
    looped = sum(float(jnp.mean(cjax.shell(coefficients, sampling, float(r)) ** 2)) for r in radii)
    np.testing.assert_allclose(float(scanned), looped, rtol=1e-12)


def test_rematerialising_the_scan_gives_the_same_gradient() -> None:
    """``remat`` trades memory for recomputation; it must not trade accuracy."""
    box = jnp.asarray(_box(5))
    lat, lon = cmt.healpix_subpixel_lonlat(8, subsample_level=0)
    order = 3
    sampling = cjax.make_shell_sampling(latitude=lat, longitude=lon, order=order)
    radii = jnp.linspace(6.0, 20.0, 8)

    def total(coefficients, remat):
        return cjax.lightcone_scan(
            cjax.PrefilteredCoeval(coefficients, order),
            sampling,
            radii,
            lambda carry, shell: carry + jnp.mean(shell**2),
            0.0,
            remat=remat,
        )

    plain = jax.grad(lambda c: total(c, False))(box)
    remade = jax.grad(lambda c: total(c, True))(box)
    np.testing.assert_allclose(np.asarray(plain), np.asarray(remade), atol=1e-14)


def test_shells_can_be_jitted_and_vmapped() -> None:
    """The whole reason for a pure API: the user wraps it, the library does not."""
    box = jnp.asarray(_box(5))
    lat, lon = _angles(60)
    order = 3
    sampling = cjax.make_shell_sampling(latitude=lat, longitude=lon, order=order)

    def one(coefficients):
        return cjax.shell(cjax.PrefilteredCoeval(coefficients, order), sampling, RADIUS)

    eager = np.asarray(one(box))
    np.testing.assert_allclose(np.asarray(jax.jit(one)(box)), eager, atol=1e-13)

    boxes = jnp.stack([box, 2 * box, -box])
    batched = np.asarray(jax.vmap(one)(boxes))
    assert batched.shape == (3, sampling.npix)
    np.testing.assert_allclose(batched[1], 2 * eager, atol=1e-12)


# ------------------------------------------------------------------------ interfaces


def test_shell_from_coordinates_needs_npix_to_average() -> None:
    """Weights say how many sub-samples there are, not how many pixels they fall in."""
    with pytest.raises(ValueError, match="npix is required"):
        cjax.shell_from_coordinates(
            jnp.zeros((4, 4, 4)), jnp.zeros((3, 8)), order=1, weights=jnp.ones(2) / 2
        )


def test_make_shell_sampling_validates_its_arguments() -> None:
    lat, lon = _angles(10)
    with pytest.raises(TypeError, match="must be an integer"):
        cjax.make_shell_sampling(latitude=lat, longitude=lon, order=1.0)
    with pytest.raises(ValueError, match="range 0-5"):
        cjax.make_shell_sampling(latitude=lat, longitude=lon, order=6)
    with pytest.raises(ValueError, match="same shape"):
        cjax.make_shell_sampling(latitude=lat, longitude=lon[:-1])
    with pytest.raises(ValueError, match="1D or 2D"):
        cjax.make_shell_sampling(latitude=lat.reshape(1, 1, -1), longitude=lon.reshape(1, 1, -1))
    with pytest.raises(ValueError, match="n_radial_samples must be at least 1"):
        cjax.make_shell_sampling(latitude=lat, longitude=lon, n_radial_samples=0)
    with pytest.raises(ValueError, match="coeval_cell_width must be non-negative"):
        cjax.make_shell_sampling(latitude=lat, longitude=lon, coeval_cell_width=-1.0)
    with pytest.raises(ValueError, match="averaging cannot sharpen"):
        cjax.make_shell_sampling(latitude=lat, longitude=lon, radial_width=0.5)


def test_the_plan_objects_are_pytrees() -> None:
    """``order`` must be static metadata, since the gather unrolls over it."""
    filtered = cjax.prefilter_coeval(_box(), 3)
    leaves, treedef = jax.tree_util.tree_flatten(filtered)
    assert len(leaves) == 1
    assert jax.tree_util.tree_unflatten(treedef, leaves).order == 3

    lat, lon = _angles(10)
    sampling = cjax.make_shell_sampling(latitude=lat, longitude=lon, order=4)
    leaves, treedef = jax.tree_util.tree_flatten(sampling)
    assert jax.tree_util.tree_unflatten(treedef, leaves).order == 4


# --------------------------------------------------------------------- precision


def test_single_precision_costs_about_a_part_in_a_million() -> None:
    """JAX defaults to float32, so the price of that is worth stating rather than finding.

    The gather stays accurate because the coordinate is split into an integer base and
    a fraction in ``[0, 1)``, so a shell radius of a thousand cells never passes through
    a ``float32``. The pre-filter is the looser of the two: it is a *global*
    deconvolution, so its error does not stay local the way the gather's does.
    """
    box = _box()
    lat, lon = _angles()
    order = 3
    sampling = cjax.make_shell_sampling(latitude=lat, longitude=lon, order=order)

    expected = next(
        cmt.make_lightcone_slice(
            coevals=box,
            latitude=lat,
            longitude=lon,
            distance_to_shell=RADIUS,
            interpolation_order=order,
        )
    )

    with jax.enable_x64(False):
        single = np.asarray(
            cjax.shell(cjax.prefilter_coeval(box.astype(np.float32), order), sampling, RADIUS)
        )
    assert single.dtype == np.float32
    assert np.abs(single - expected).max() / expected.std() < 1e-4
