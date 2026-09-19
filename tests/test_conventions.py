r"""What a grid value means, and what window the output therefore carries.

A simulation cell holds the *mean* of the field over that cell. Written out, the box is
``g_n = (T * f)(x_n)`` for an underlying field ``f`` and a cell top-hat ``T`` -- which is
simultaneously "the cell average of ``f``" and "a point sample of the smoothed field
``s = T * f``". The same number, under two names: there is no second arithmetic hiding
behind the choice of words, which is why ``cosmotile`` has no switch for it.

What there *is* is a window chain. The value on the shell is

.. math:: Q * \Lambda * T * f

evaluated at the pixel, where ``T`` is the cell window that came with the data,
``\Lambda`` the reconstruction kernel set by ``interpolation_order``, and ``Q`` whatever
output window was requested (nothing, by default). The tests here pin each factor:

* ``T`` is invertible, and :func:`~cosmotile.deconvolve_cell_window` inverts it exactly
  for a band-limited periodic field.
* ``\Lambda`` converges on ``s``, **not** on ``f``. Raising ``interpolation_order`` on a
  cell-averaged box drives the error towards the cell window and stops there; only
  deconvolving first makes the order ladder converge on the underlying field. This is the
  one place where the convention silently changes what a user should expect, so it is
  asserted from both sides.
* ``Q`` is delivered *convolved with* ``T``, never on its own, which is what
  :func:`~cosmotile.residual_radial_width` exists to correct for.

Every box in this module is built by applying the analytic cell window to a known mode,
so "the exact cell averages of ``f``" is available in closed form rather than by
quadrature.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

import cosmotile as cmt

from .conftest import random_directions, unit_vectors_to_lonlat

NCELL = 64
CENTRE = np.array([32.0, 32.0, 32.0])

# A mode at a quarter of the grid Nyquist: well enough resolved that the interpolation
# error can fall far below the cell window, which is the whole point of the order tests.
HARMONIC = 8
WAVENUMBER = 2 * np.pi * HARMONIC / NCELL
CELL_WINDOW = float(np.sinc(WAVENUMBER / (2 * np.pi)))  # sinc(k/2) = 0.9745


def plane_wave_boxes() -> tuple[np.ndarray, np.ndarray]:
    """Return point samples of a mode along ``x``, and its exact cell averages.

    Averaging a single mode over a cell multiplies it by ``sinc(k/2)`` and leaves the
    phase alone, so the cell-averaged box is the point-sampled one times a constant --
    no quadrature needed, and the comparison below is exact.
    """
    grid = np.meshgrid(*(np.arange(NCELL, dtype=float),) * 3, indexing="ij")
    samples = np.cos(WAVENUMBER * (grid[0] - CENTRE[0]))
    return samples, samples * CELL_WINDOW


# ---------------------------------------------------------------------------------
# The cell window itself
# ---------------------------------------------------------------------------------
@pytest.mark.parametrize("width", [0.5, 1.0, 1.5])
def test_cell_window_matches_the_analytic_top_hat_transform(width: float) -> None:
    """``cell_window`` must be ``prod_i sinc(k_i w / 2)`` on the grid's own modes.

    Checked against an independently written product over materialised frequency grids,
    in both the full and half-spectrum layouts, on a deliberately non-cubic grid so that
    a transposed axis or a misplaced ``rfftfreq`` cannot pass.
    """
    shape = (8, 10, 12)

    for rfft in (False, True):
        freqs = [np.fft.fftfreq(n) for n in shape]
        if rfft:
            freqs[-1] = np.fft.rfftfreq(shape[-1])
        expected = np.einsum("i,j,k->ijk", *(np.sinc(f * width) for f in freqs), optimize=True)

        np.testing.assert_allclose(
            cmt.cell_window(shape, width, rfft=rfft) * np.ones(expected.shape),
            expected,
            rtol=1e-14,
        )


def test_deconvolution_recovers_point_samples_of_a_band_limited_field() -> None:
    """Deconvolving exact cell averages must return the point samples they came from.

    For a field band-limited to the grid Nyquist the cell window is a non-zero factor on
    every mode the grid carries, so dividing it out is exact -- not an approximation, and
    not a regularised inverse. That is asserted to round-off here, because it is the
    property the rest of the module leans on.
    """
    rng = np.random.default_rng(0)
    grid = np.meshgrid(*(np.arange(NCELL, dtype=float),) * 3, indexing="ij")

    samples = np.zeros((NCELL,) * 3)
    averages = np.zeros((NCELL,) * 3)
    for harmonic in ([1, 0, 0], [0, 3, 0], [2, 0, 5], [7, -4, 1]):
        kvec = 2 * np.pi * np.asarray(harmonic, dtype=float) / NCELL
        mode = np.cos(sum(k * x for k, x in zip(kvec, grid, strict=True)) + rng.uniform(0, 7))
        samples += mode
        averages += mode * np.prod(np.sinc(kvec / (2 * np.pi)))

    np.testing.assert_allclose(cmt.deconvolve_cell_window(averages), samples, atol=1e-13)


def test_deconvolution_inverts_the_cell_window() -> None:
    """Convolving with the cell window and deconvolving must be the identity, both ways.

    White noise is the harder direction: it puts power right up against Nyquist, where
    the window is smallest (``sinc(pi/2) = 0.64`` per axis) and the inverse largest.
    """
    rng = np.random.default_rng(1)
    box = rng.normal(size=(NCELL,) * 3)

    window = cmt.cell_window(box.shape)
    averaged = np.real(np.fft.ifftn(np.fft.fftn(box) * window))

    np.testing.assert_allclose(cmt.deconvolve_cell_window(averaged), box, atol=1e-11)

    sharpened = cmt.deconvolve_cell_window(box)
    np.testing.assert_allclose(
        np.real(np.fft.ifftn(np.fft.fftn(sharpened) * window)), box, atol=1e-11
    )

    # The inverse amplifies near-Nyquist power; users need to know by how much.
    assert 1.5 < sharpened.std() / box.std() < 1.8


@pytest.mark.parametrize("width", [0.0, -1.0, 2.0, 3.0])
def test_deconvolution_rejects_a_window_it_cannot_invert(width: float) -> None:
    """``sinc(k w / 2)`` first vanishes at ``k = pi``, ``w = 2``, so two cells is the limit."""
    with pytest.raises(ValueError, match="width must be positive and less than two"):
        cmt.deconvolve_cell_window(np.zeros((4, 4, 4)), width=width)


# ---------------------------------------------------------------------------------
# What the interpolation order actually converges on
# ---------------------------------------------------------------------------------
def shell_errors(box: np.ndarray, radius: float = 17.3) -> dict[str, list[float]]:
    """Max error of the tiled shell against ``f`` and against ``s``, per spline order."""
    vectors = random_directions(3000, np.random.default_rng(0))
    latitude, longitude = unit_vectors_to_lonlat(vectors)
    kvec = np.array([WAVENUMBER, 0.0, 0.0])

    underlying = np.cos((radius * vectors) @ kvec)
    smoothed = underlying * CELL_WINDOW

    errors: dict[str, list[float]] = {"f": [], "s": []}
    for order in range(6):
        shell = next(
            cmt.make_lightcone_slice(
                coevals=box,
                latitude=latitude,
                longitude=longitude,
                distance_to_shell=radius,
                origin=CENTRE,
                interpolation_order=order,
            )
        )
        errors["f"].append(float(np.abs(shell - underlying).max()))
        errors["s"].append(float(np.abs(shell - smoothed).max()))
    return errors


def test_interpolation_order_converges_on_the_cell_averaged_field_not_the_true_one() -> None:
    """On a cell-averaged box, raising the order stops helping at the cell window.

    This is the claim most likely to mislead. The accuracy page's order ladder -- each
    extra spline order buying about an order of magnitude -- is a statement about
    reconstructing whatever the grid holds. If the grid holds cell averages, that is the
    *smoothed* field ``s``, and the error against the *underlying* field ``f`` bottoms out
    at ``1 - sinc(k/2)`` no matter how high the order goes. The residual is not an
    interpolation error and no interpolation order removes it.

    Both halves are asserted: convergence against ``s``, and a hard floor against ``f``
    sitting at the analytically known cell-window deficit.
    """
    _, averaged = plane_wave_boxes()
    errors = shell_errors(averaged)

    # Against the field the box actually samples, the usual ladder holds.
    against_smoothed = errors["s"][1:]
    assert all(b < a for a, b in itertools.pairwise(against_smoothed)), (
        f"error against s must fall with order, got {against_smoothed}"
    )
    assert against_smoothed[-1] < against_smoothed[0] / 1000

    # Against the underlying field it does not: orders 3-5 all sit on the cell window.
    floor = 1 - CELL_WINDOW
    for order in (3, 4, 5):
        assert abs(errors["f"][order] / floor - 1) < 0.05, (
            f"order {order} error against f is {errors['f'][order]:.3e}, "
            f"expected the cell-window floor {floor:.3e}"
        )


def test_deconvolution_restores_convergence_towards_the_underlying_field() -> None:
    """Remove the cell window first and the order ladder works against ``f`` again.

    The companion to the test above: the floor is the cell window and nothing else, so
    dividing it out must recover the same convergence the smoothed field enjoyed.
    """
    _, averaged = plane_wave_boxes()
    errors = shell_errors(cmt.deconvolve_cell_window(averaged))["f"][1:]

    assert all(b < a for a, b in itertools.pairwise(errors)), (
        f"error against f must fall with order once deconvolved, got {errors}"
    )
    assert errors[-1] < errors[0] / 1000
    assert errors[-1] < 0.01 * (1 - CELL_WINDOW), "must fall far below the old floor"


# ---------------------------------------------------------------------------------
# Composition of the output window with the cell window
# ---------------------------------------------------------------------------------
def sightline_amplitude(box: np.ndarray, radius: float, **kwargs: object) -> float:
    """Amplitude of the mode on a ``+x`` sight-line, relative to the underlying field."""
    value = next(
        cmt.make_lightcone_slice(
            coevals=box,
            latitude=np.array([0.0]),
            longitude=np.array([0.0]),
            distance_to_shell=radius,
            origin=CENTRE,
            interpolation_order=5,
            **kwargs,
        )
    )[0]
    return float(value / np.cos(WAVENUMBER * radius))


@pytest.mark.parametrize("width", [1.0, 2.0, 3.0])
def test_the_requested_window_is_delivered_convolved_with_the_cell_window(
    width: float,
) -> None:
    """Asking for a radial top-hat ``Q`` on a cell-averaged box gives ``Q * T``, not ``Q``.

    The radius is large enough that the ``r^2`` volume weighting is negligible (it
    perturbs the top-hat at order ``(w/r)^2``, tested separately in
    ``test_pixel_averaging.py``), which isolates the composition being asserted here:
    the output carries the product of both windows, and deconvolving first leaves
    exactly the one that was requested.
    """
    radius = 60.0
    samples, averaged = plane_wave_boxes()
    requested = float(np.sinc(WAVENUMBER * width / (2 * np.pi)))
    averaging = {"radial_width": width, "n_radial_samples": 8}

    np.testing.assert_allclose(
        sightline_amplitude(averaged, radius, **averaging), requested * CELL_WINDOW, rtol=1e-3
    )
    np.testing.assert_allclose(
        sightline_amplitude(cmt.deconvolve_cell_window(averaged), radius, **averaging),
        requested,
        rtol=1e-3,
    )
    # And on a genuinely point-sampled box there is no cell window to compose with.
    np.testing.assert_allclose(
        sightline_amplitude(samples, radius, **averaging), requested, rtol=1e-3
    )


@pytest.mark.parametrize("target", [1.5, 2.0, 4.0])
def test_residual_radial_width_composes_to_the_requested_window(target: float) -> None:
    """``sqrt(dr^2 - D^2)`` must reproduce a top-hat of ``dr`` once the cell window is on.

    Both windows expand as ``1 - k^2 x^2 / 24``, so widths add in quadrature to leading
    order. Asserted as window algebra across every multipole up to the target's own
    Nyquist -- where the rule has to hold, and where the naive choice of the full slice
    width is badly wrong.
    """

    def tophat(k: np.ndarray, w: float) -> np.ndarray:
        return np.asarray(np.sinc(k * w / (2 * np.pi)))

    residual = cmt.residual_radial_width(target)
    wavenumbers = np.linspace(1e-6, np.pi / target, 400)

    wanted = tophat(wavenumbers, target)
    delivered = tophat(wavenumbers, residual) * tophat(wavenumbers, 1.0)
    naive = tophat(wavenumbers, target) * tophat(wavenumbers, 1.0)

    error = np.abs(delivered / wanted - 1).max()
    assert error < 0.025, f"residual width off by {error:.2%}"
    assert error < 0.25 * np.abs(naive / wanted - 1).max(), "must beat the naive choice"


def test_residual_radial_width_is_zero_when_the_cell_already_supplies_it() -> None:
    """A slice no coarser than a cell needs no extra averaging, and asking gives zero.

    The zero is meant to be passed straight through, so it must also be accepted as a
    ``radial_width`` and leave the output untouched.
    """
    assert cmt.residual_radial_width(0.5) == 0.0
    assert cmt.residual_radial_width(1.0) == 0.0
    assert cmt.residual_radial_width(2.0, cell_size=2.0) == 0.0
    assert cmt.residual_radial_width(2.0) == pytest.approx(np.sqrt(3.0))

    _, averaged = plane_wave_boxes()
    sampled = sightline_amplitude(averaged, 60.0)

    # The cell window alone is what a one-cell slice wanted in the first place.
    np.testing.assert_allclose(sampled, CELL_WINDOW, rtol=1e-3)
    np.testing.assert_allclose(
        sightline_amplitude(
            averaged, 60.0, radial_width=cmt.residual_radial_width(1.0), n_radial_samples=8
        ),
        sampled,
        rtol=0,
        atol=1e-14,
    )
