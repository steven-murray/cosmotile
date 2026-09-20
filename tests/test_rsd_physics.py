"""Physical tests of the line-of-sight projection and redshift-space distortions.

Two separate pieces of physics live here.

:func:`~cosmotile.make_lightcone_slice_vector_field` projects a 3D vector field (a
peculiar velocity field, in practice) onto the line of sight. That projection is pure
geometry, so every test of it below is exact to round-off: a pure outflow, a pure
rotation and a uniform flow each have an unambiguous right answer at every point of the
shell.

:func:`~cosmotile.apply_rsds` then moves the field along the line of sight. That is a
one-dimensional continuity problem with a closed-form solution, ``rho_s(s) ds =
rho_r(r) dr``, which the tests compare against directly.

**Sign convention.** Both functions take positive to mean *towards the observer*. A
parcel at comoving distance ``d`` moving towards us with displacement ``u`` is observed
at ``d - u``. The two therefore chain directly, and the tests below assert that
end-to-end rather than each half in isolation.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from astropy import units as un
from scipy.interpolate import interp1d

import cosmotile as cmt

from .conftest import random_directions, unit_vectors_to_lonlat

NCELL = 32
CENTRE = np.array([16.0, 16.0, 16.0])
# Small enough that the shell stays strictly inside the box: the analytic velocity
# fields below are not periodic, so wrapping would compare against the wrong thing.
RADIUS = 7.0


@pytest.fixture
def sightlines() -> np.ndarray:
    """Isotropic sight-lines used by the projection tests."""
    return random_directions(500, np.random.default_rng(1))


@pytest.fixture
def interpolator(sightlines: np.ndarray) -> object:
    """Return an interpolator onto a shell of radius ``RADIUS`` inside the box."""
    lat, lon = unit_vectors_to_lonlat(sightlines)
    return cmt.make_lightcone_slice_interpolator(
        latitude=lat, longitude=lon, distance_to_shell=RADIUS, origin=CENTRE
    )


def offsets_from_centre() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cell-centre coordinates of the box, measured from ``CENTRE``."""
    grid = np.meshgrid(*(np.arange(NCELL, dtype=float),) * 3, indexing="ij")
    return tuple(g - c for g, c in zip(grid, CENTRE, strict=True))


# ---------------------------------------------------------------------------------
# Line-of-sight projection
# ---------------------------------------------------------------------------------
def test_hubble_outflow_projects_to_a_uniform_recession(interpolator: object) -> None:
    """A pure Hubble flow about the observer must give ``-H r`` at every point.

    ``v = H (x - o)`` points radially outwards from the observer with magnitude ``H r``,
    so its line-of-sight component -- positive towards the observer -- must be exactly
    ``-H r`` on the whole shell, with no angular structure whatsoever.

    This is the sharpest available test of the projection, because it fixes both the
    sign and the fact that the dot product is taken against the vector *from the
    observer*, not from the corner of the box. Getting the origin wrong leaves a dipole.
    """
    hubble = 0.1
    dx, dy, dz = offsets_from_centre()
    field = [hubble * dx, hubble * dy, hubble * dz]

    los = next(cmt.make_lightcone_slice_vector_field([field], interpolator))

    np.testing.assert_allclose(los, -hubble * RADIUS, atol=1e-12)


def test_solid_body_rotation_has_no_line_of_sight_component(interpolator: object) -> None:
    """A velocity field ``omega x (x - o)`` is everywhere transverse, so it must project to zero.

    Rotation about the observer produces no radial motion anywhere, so it can produce no
    redshift-space distortion. Any leakage would mean the projection is mixing the
    transverse components into the radial one.
    """
    omega = np.array([0.13, -0.4, 0.2])
    dx, dy, dz = offsets_from_centre()
    field = [
        omega[1] * dz - omega[2] * dy,
        omega[2] * dx - omega[0] * dz,
        omega[0] * dy - omega[1] * dx,
    ]

    los = next(cmt.make_lightcone_slice_vector_field([field], interpolator))

    assert np.abs(los).max() < 1e-12


def test_uniform_flow_projects_to_a_pure_dipole(
    interpolator: object, sightlines: np.ndarray
) -> None:
    """A constant velocity field must project to ``-v . n``, a pure dipole.

    This is the bulk-flow limit, and it is the one case where the answer depends only on
    direction and not at all on radius -- so it isolates the angular part of the
    projection from the radial part tested above.
    """
    velocity = np.array([0.7, -0.2, 1.3])
    field = [np.full((NCELL,) * 3, v) for v in velocity]

    los = next(cmt.make_lightcone_slice_vector_field([field], interpolator))

    np.testing.assert_allclose(los, -sightlines @ velocity, atol=1e-12)


def test_projection_is_linear_in_the_vector_field(interpolator: object) -> None:
    """Superposing velocity fields must superpose their line-of-sight components."""
    rng = np.random.default_rng(2)
    first = [rng.normal(size=(NCELL,) * 3) for _ in range(3)]
    second = [rng.normal(size=(NCELL,) * 3) for _ in range(3)]
    combined = [3 * a - 2 * b for a, b in zip(first, second, strict=True)]

    a, b, ab = cmt.make_lightcone_slice_vector_field([first, second, combined], interpolator)

    np.testing.assert_allclose(ab, 3 * a - 2 * b, atol=1e-10)


def test_projection_carries_units_through(interpolator: object) -> None:
    """A velocity field with units must yield a line-of-sight component with those units."""
    dx, dy, dz = offsets_from_centre()
    field = [0.1 * dx * un.pixel, 0.1 * dy * un.pixel, 0.1 * dz * un.pixel]

    los = next(cmt.make_lightcone_slice_vector_field([field], interpolator))

    assert los.unit == un.pixel
    np.testing.assert_allclose(los.value, -0.1 * RADIUS, atol=1e-12)


# ---------------------------------------------------------------------------------
# Redshift-space distortions
# ---------------------------------------------------------------------------------
def make_los_grid(nslice: int = 256, start: float = 1000.0) -> un.Quantity:
    """Build a regular radial grid of ``nslice`` slices, in cell units."""
    return (start + np.arange(nslice, dtype=float)) * un.pixel


def test_positive_displacement_moves_structure_towards_the_observer() -> None:
    """The sign convention must match the line-of-sight projection's.

    ``make_lightcone_slice_vector_field`` returns positive for motion towards the
    observer, and users chain it straight into ``apply_rsds``; if the two disagree the
    resulting lightcone has its distortions inverted, which is a silent and physically
    serious error. A localised feature displaced by ``+u`` must move ``u`` slices
    *nearer*, and by ``-u`` must move ``u`` slices further away.
    """
    nslice = 21
    distance = make_los_grid(nslice, 10.0)
    field = np.zeros((nslice, 1))
    field[10] = 1.0

    for displacement, expected in ((2.0, 8), (-2.0, 12)):
        out = cmt.apply_rsds(
            field=field,
            los_displacement=np.full((nslice, 1), displacement) * un.pixel,
            distance=distance,
            n_subcells=4,
        )
        assert np.argmax(out[:, 0]) == expected


def test_uniform_displacement_translates_the_field_rigidly() -> None:
    """A displacement that is the same everywhere must translate, not distort.

    With no velocity gradient there is no compression, so continuity says the profile
    must come back unchanged in shape and amplitude, just shifted. This separates the
    translation from the density response tested next.
    """
    nslice = 128
    distance = make_los_grid(nslice, 500.0)
    radial = np.arange(nslice, dtype=float)
    profile = 1 + 0.5 * np.sin(2 * np.pi * radial / 32)
    field = profile[:, None]
    shift = 3.0

    out = cmt.apply_rsds(
        field=field,
        los_displacement=np.full((nslice, 1), shift) * un.pixel,
        distance=distance,
        n_subcells=4,
    )

    # Compare away from the ends, where the grid is padded by extrapolation.
    interior = slice(20, nslice - 20)
    expected = 1 + 0.5 * np.sin(2 * np.pi * (radial + shift) / 32)
    np.testing.assert_allclose(out[interior, 0], expected[interior], atol=0.02)


@pytest.mark.parametrize("amplitude", [0.5, 1.0, 2.0])
def test_density_response_follows_one_dimensional_continuity(amplitude: float) -> None:
    """A sheared displacement must compress and rarefy the field as continuity requires.

    For a uniform field displaced by ``u(r)``, the observed distance is ``s = r - u(r)``
    and mass conservation gives ``rho_s(s) = 1 / (1 - u'(r))`` -- overdense where the
    flow converges, underdense where it diverges. This is the actual physics of
    redshift-space distortions in one dimension, and it is what ``apply_rsds`` exists to
    reproduce.

    The residual is the cloud-in-cell kernel, which smooths the output by about one
    slice; it therefore scales with the amplitude of the response, which is what the
    amplitude-dependent tolerance below encodes.
    """
    nslice, wavelength = 256, 64.0
    distance = make_los_grid(nslice)
    radial = np.arange(nslice, dtype=float)

    displacement = amplitude * np.sin(2 * np.pi * radial / wavelength)
    gradient = amplitude * (2 * np.pi / wavelength) * np.cos(2 * np.pi * radial / wavelength)
    assert np.abs(gradient).max() < 1, "no shell crossing, so the mapping is invertible"

    out = cmt.apply_rsds(
        field=np.ones((nslice, 1)),
        los_displacement=displacement[:, None] * un.pixel,
        distance=distance,
        n_subcells=4,
    )

    # The exact solution, evaluated on the observed (redshift-space) grid.
    exact = interp1d(
        radial - displacement, 1.0 / (1.0 - gradient), bounds_error=False, fill_value=np.nan
    )(radial)

    interior = slice(40, nslice - 40)
    residual = np.abs(out[interior, 0] - exact[interior])
    assert residual.max() < 0.1 * amplitude
    assert np.sqrt(np.mean(residual**2)) < 0.03 * amplitude


def test_mass_is_conserved_by_the_displacement() -> None:
    """Displacing the field must move mass around, not create or destroy it.

    Cloud-in-cell assignment conserves the deposited weight exactly, so the integral of
    the field along the line of sight must be preserved wherever material does not leave
    the grid.
    """
    nslice = 256
    distance = make_los_grid(nslice)
    radial = np.arange(nslice, dtype=float)
    rng = np.random.default_rng(3)

    field = 1 + 0.3 * np.sin(2 * np.pi * radial / 50)[:, None]
    displacement = 1.5 * np.sin(2 * np.pi * radial / 70 + 0.3)[:, None] * un.pixel

    out = cmt.apply_rsds(
        field=field, los_displacement=displacement, distance=distance, n_subcells=4
    )

    interior = slice(40, nslice - 40)
    assert abs(out[interior].sum() / field[interior].sum() - 1) < 0.02
    _ = rng


def test_zero_displacement_is_the_identity() -> None:
    """With no peculiar velocity, redshift space is real space."""
    nslice = 64
    distance = make_los_grid(nslice, 200.0)
    rng = np.random.default_rng(4)
    field = 1 + 0.2 * rng.normal(size=(nslice, 8))

    out = cmt.apply_rsds(
        field=field,
        los_displacement=np.zeros((nslice, 8)) * un.pixel,
        distance=distance,
        n_subcells=4,
    )

    np.testing.assert_allclose(out, field, atol=1e-10)


def test_hubble_flow_chains_from_projection_to_distortion() -> None:
    """A pure outflow, projected then applied, must push structure *away* from us.

    This is the end-to-end statement that matters in practice: the output of
    ``make_lightcone_slice_vector_field`` can be fed straight into ``apply_rsds``. An
    outflow projects to a negative line-of-sight displacement, which must in turn place
    structure at larger apparent distance.
    """
    nslice = 32
    radius = RADIUS
    vectors = random_directions(64, np.random.default_rng(5))
    lat, lon = unit_vectors_to_lonlat(vectors)
    interpolator = cmt.make_lightcone_slice_interpolator(
        latitude=lat, longitude=lon, distance_to_shell=radius, origin=CENTRE
    )

    hubble = 0.2
    dx, dy, dz = offsets_from_centre()
    los = next(
        cmt.make_lightcone_slice_vector_field(
            [[hubble * dx * un.pixel, hubble * dy * un.pixel, hubble * dz * un.pixel]],
            interpolator,
        )
    )
    # An outflow recedes from us, so the projected displacement is negative.
    assert np.all(los < 0)

    distance = make_los_grid(nslice, 100.0)
    field = np.zeros((nslice, len(vectors)))
    field[16] = 1.0

    out = cmt.apply_rsds(
        field=field,
        los_displacement=np.tile(los, (nslice, 1)),
        distance=distance,
        n_subcells=4,
    )

    assert np.all(np.argmax(out, axis=0) > 16)


# ---------------------------------------------------------------------------------
# ``n_subcells`` convergence (GH #465)
# ---------------------------------------------------------------------------------
def continuity_solution(amplitude: float, wavelength: float, nslice: int) -> np.ndarray:
    """Evaluate ``rho_s(s) = rho_r(r) / (1 - u'(r))`` for a sinusoidal displacement.

    Returned on the observed (redshift-space) grid of slice centres, with NaN wherever
    the mapping does not reach.
    """
    radial = np.arange(nslice, dtype=float)
    displacement = amplitude * np.sin(2 * np.pi * radial / wavelength)
    gradient = amplitude * (2 * np.pi / wavelength) * np.cos(2 * np.pi * radial / wavelength)
    return interp1d(
        radial - displacement, 1.0 / (1.0 - gradient), bounds_error=False, fill_value=np.nan
    )(radial)


@pytest.mark.parametrize("amplitude", [1.0, 2.0])
def test_n_subcells_is_a_convergence_parameter(amplitude: float) -> None:
    """Refining the sub-cell grid must monotonically improve the answer.

    ``n_subcells`` refines the grid the displacement is applied on. It should therefore
    behave like any other discretisation parameter: raising it must buy accuracy against
    the exact continuity solution, and keep buying it until some other error dominates.

    It did not. The final step used to *sample* the refined grid at the output slice
    centres, so everything that landed between them was simply discarded -- the finer
    the grid, the more there was to discard. Measured RMS error was flat or worse as
    ``n_subcells`` rose (GH #465: 0.039, 0.048, 0.047, 0.041, 0.039 at 1, 2, 4, 8, 16
    for ``amplitude = 2``). Integrating over the output cell instead makes the
    refinement do what its name says.
    """
    nslice, wavelength = 256, 64.0
    distance = make_los_grid(nslice)
    radial = np.arange(nslice, dtype=float)
    displacement = amplitude * np.sin(2 * np.pi * radial / wavelength)

    exact = continuity_solution(amplitude, wavelength, nslice)
    interior = slice(40, nslice - 40)

    errors = []
    for n_subcells in (1, 2, 4, 8, 16, 32):
        out = cmt.apply_rsds(
            field=np.ones((nslice, 1)),
            los_displacement=displacement[:, None] * un.pixel,
            distance=distance,
            n_subcells=n_subcells,
        )
        residual = out[interior, 0] - exact[interior]
        errors.append(float(np.sqrt(np.mean(residual**2))))

    assert all(b < a for a, b in itertools.pairwise(errors)), (
        f"RMS error must fall monotonically with n_subcells, got {np.round(errors, 5)}"
    )
    # Roughly first order in the sub-cell size, so 32x refinement must buy an order of
    # magnitude. (It is not exactly first order: the cloud-in-cell kernel narrows with
    # the sub-cell, but the output cell average stays a top-hat of one slice.)
    assert errors[-1] < errors[0] / 10, f"only {errors[0] / errors[-1]:.1f}x better"


def test_refinement_conserves_mass() -> None:
    """Refining and re-averaging must move mass about, never create or destroy it.

    Cloud-in-cell deposition conserves the weight it deposits and the final step
    integrates the deposited field over the output cells, so with a displacement that
    keeps every parcel on the grid the total must be preserved to round-off -- at any
    refinement.

    The displacement is chosen to point *inwards* at both ends (away from the observer
    at the near end, towards it at the far end), so no parcel leaves and, equally
    importantly, the grid needs no extrapolated padding -- which would bring mass in
    from outside and is a separate, documented behaviour.
    """
    nslice = 128
    distance = make_los_grid(nslice, 500.0)
    radial = np.arange(nslice, dtype=float)
    rng = np.random.default_rng(12)

    field = 1 + 0.3 * rng.normal(size=(nslice, 4))
    inward = -2.0 * np.cos(np.pi * radial / (nslice - 1))
    displacement = np.tile(inward[:, None], (1, 4)) * un.pixel

    for n_subcells in (1, 4, 16):
        out = cmt.apply_rsds(
            field=field,
            los_displacement=displacement,
            distance=distance,
            n_subcells=n_subcells,
        )
        assert abs(out.sum() / field.sum() - 1) < 1e-12, f"n_subcells={n_subcells}"


def test_the_output_is_an_average_over_the_slice_not_a_sample() -> None:
    """A spike that lands between two slices must be shared, not lost.

    This is the mechanism behind the non-convergence, in isolation. A single parcel
    displaced to exactly the boundary between two output cells straddles both, so
    cloud-in-cell splits it evenly -- whatever the sub-cell refinement. Sampling the
    refined grid instead returned the parcel intact at ``n_subcells = 1`` and nothing at
    all at ``n_subcells = 4``, because no output slice sat on the sub-cell it landed in.
    """
    nslice = 21
    distance = make_los_grid(nslice, 10.0)
    field = np.zeros((nslice, 1))
    field[10] = 1.0

    for n_subcells in (1, 2, 4, 8):
        out = cmt.apply_rsds(
            field=field,
            los_displacement=np.full((nslice, 1), 2.5) * un.pixel,
            distance=distance,
            n_subcells=n_subcells,
        )
        np.testing.assert_allclose(out[7, 0], 0.5, atol=1e-12)
        np.testing.assert_allclose(out[8, 0], 0.5, atol=1e-12)
        np.testing.assert_allclose(out.sum(), 1.0, atol=1e-12)


@pytest.mark.parametrize(
    "distance",
    [
        np.arange(7.0) + 10,
        np.array([10.0, 11.001, 12, 13, 14, 15, 16]),
        np.array([10.0, 10.3, 11.7, 13.9, 14.2, 17.0, 21.5]),
    ],
    ids=["regular", "mildly irregular", "strongly irregular"],
)
@pytest.mark.parametrize("n_subcells", [1, 4, 16])
def test_zero_displacement_is_the_identity_on_any_grid(
    distance: np.ndarray, n_subcells: int
) -> None:
    """The refine-displace-average round trip must be exact however the slices are spaced.

    Each output cell is subdivided into ``n_subcells`` equal parts, so every output edge
    is also a sub-cell edge and both the refinement and the final average are exact. A
    *globally* uniform sub-grid cannot manage this on an irregular slice grid: its cells
    straddle the output edges and mix neighbouring slices together, which left a residual
    of up to 4% that ``n_subcells`` barely improved.
    """
    rng = np.random.default_rng(6)
    field = 1 + 0.5 * rng.normal(size=(distance.size, 8))

    out = cmt.apply_rsds(
        field=field,
        los_displacement=np.zeros_like(field) * un.pixel,
        distance=distance * un.pixel,
        n_subcells=n_subcells,
    )

    np.testing.assert_allclose(out, field, rtol=0, atol=1e-12)
    assert abs(out.sum() / field.sum() - 1) < 1e-12


def test_n_subcells_must_be_a_positive_integer() -> None:
    """A zero would divide by zero; a float or a negative would build a nonsense grid."""
    field = np.ones((3, 2))
    distance = make_los_grid(3, 10.0)

    for bad in (0, -1, 2.5):
        with pytest.raises(ValueError, match="n_subcells must be a positive integer"):
            cmt.apply_rsds(
                field=field,
                los_displacement=np.zeros((3, 2)) * un.pixel,
                distance=distance,
                n_subcells=bad,
            )


def test_an_empty_outside_cannot_create_mass() -> None:
    """With ``outside="empty"`` displacement can only ever remove material.

    Before version 2.0 the padding always held replicated copies of the first and last
    slices *and* extrapolated the displacement across them without bound, so material
    that was never there flowed back in: the total could exceed what went in, and the
    answer depended on how many sub-cells of padding happened to be allocated -- a
    quantity set by the data, not by physics.
    """
    nslice = 96
    distance = make_los_grid(nslice)
    radial = np.arange(nslice, dtype=float)

    # A strictly positive field, so "more mass came out than went in" is unambiguous,
    # and a displacement pointing inward at both ends -- the case that used to leak.
    field = 1 + 0.5 * np.sin(2 * np.pi * radial / 31)[:, None] * np.ones((1, 5))
    displacement = np.full((nslice, 5), -2.0)
    displacement[nslice // 2 :] = 2.0

    out = cmt.apply_rsds(
        field=field,
        los_displacement=displacement * un.pixel,
        distance=distance,
        n_subcells=4,
        outside="empty",
    )

    assert out.sum() <= field.sum() + 1e-9
    assert (out >= -1e-12).all()


def test_the_two_outside_conventions_differ_in_the_direction_they_should() -> None:
    """``"edge"`` lets material flow in as well as out; ``"empty"`` only out.

    Which is right is the caller's to decide -- the data say nothing about what lies
    beyond the range they cover -- so the point here is that the choice is real and
    points the way it claims to.
    """
    nslice = 64
    distance = make_los_grid(nslice)
    field = np.ones((nslice, 4))
    # Everything moves towards the observer, so material leaves the near end and, if
    # there is anything out there, arrives at the far end.
    displacement = np.full((nslice, 4), 2.0) * un.pixel

    empty = cmt.apply_rsds(
        field=field, los_displacement=displacement, distance=distance, outside="empty"
    )
    edge = cmt.apply_rsds(
        field=field, los_displacement=displacement, distance=distance, outside="edge"
    )

    assert empty.sum() < field.sum()
    assert edge.sum() > empty.sum()
    # A uniform field translated through a uniform field is unchanged, so 'edge' is the
    # convention that reproduces it.
    np.testing.assert_allclose(edge, field, rtol=1e-6)


def test_apply_rsds_rejects_an_unknown_outside() -> None:
    """A typo must not silently pick a convention."""
    distance = make_los_grid(8)
    with pytest.raises(ValueError, match="outside must be"):
        cmt.apply_rsds(
            field=np.ones((8, 2)),
            los_displacement=np.zeros((8, 2)) * un.pixel,
            distance=distance,
            outside="reflect",
        )
