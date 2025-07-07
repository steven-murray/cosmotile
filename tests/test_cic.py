"""Tests of the cloud-in-cell function."""

import numpy as np
import pytest

from cosmotile.cic import cloud_in_cell_coeval as cloud_in_cell
from cosmotile.cic import cloud_in_cell_los as cic_los


class TestCIC:
    """Tests of the cloud_in_cell method."""

    def test_bad_input(self) -> None:
        """Test that different-shaped inputs to CIC raise appropriate error."""
        fld = np.zeros((10, 10, 10))
        dx = np.zeros_like(fld)
        dy = np.zeros_like(fld)
        dz = np.zeros_like(fld)

        with pytest.raises(ValueError, match="Field and displacement must have the same shape."):
            cloud_in_cell(fld, dx, dy, dz[1:])

    def test_cic_single_nonzero_dx(self) -> None:
        """Test the cloud-in-cell function."""
        x = np.zeros((10, 10, 10))
        x[::2] = 1

        dx = np.zeros_like(x)
        dx[::2] = 1.0
        dy = np.zeros_like(x)
        dz = np.zeros_like(x)

        out = cloud_in_cell(x, dx, dy, dz)

        assert np.allclose(out[1::2], 1.0)

        out = cloud_in_cell(x, -dx, dy, dz)
        assert np.allclose(out[1::2], 1.0)

        out = cloud_in_cell(x, -1.5 * dx, dy, dz)
        assert np.allclose(out, 0.5)

    def test_cic_single_nonzero_dy(self) -> None:
        """Test the cloud-in-cell function."""
        x = np.zeros((10, 10, 10))
        x[:, ::2] = 1

        dx = np.zeros_like(x)
        dy = np.zeros_like(x)
        dy[:, ::2] = 1.0
        dz = np.zeros_like(x)

        out = cloud_in_cell(x, dx, dy, dz)
        assert np.allclose(out[:, 1::2], 1.0)
        out = cloud_in_cell(x, dx, -dy, dz)
        assert np.allclose(out[:, 1::2], 1.0)
        out = cloud_in_cell(x, dx, -1.5 * dy, dz)
        assert np.allclose(out, 0.5)

    def test_cic_single_nonzero_dz(self) -> None:
        """Test the cloud-in-cell function."""
        x = np.zeros((10, 10, 10))
        x[:, :, ::2] = 1

        dx = np.zeros_like(x)
        dy = np.zeros_like(x)
        dz = np.zeros_like(x)
        dz[:, :, ::2] = 1.0

        out = cloud_in_cell(x, dx, dy, dz)
        assert np.allclose(out[:, :, 1::2], 1.0)
        out = cloud_in_cell(x, dx, dy, -dz)
        assert np.allclose(out[:, :, 1::2], 1.0)
        out = cloud_in_cell(x, dx, dy, -1.5 * dz)
        assert np.allclose(out, 0.5)


class TestCICLoS:
    """Tests of the cloud_in_cell_los function."""

    @pytest.mark.parametrize("shift", [-11, -10, -1, 0, 1, 10, 11])
    def test_integer_shift_periodic(self, shift: int):
        """Test that shifting periodic box by integers results in a simple roll."""
        nangles = 1
        nslices = 10
        rng = np.random.default_rng(12345)
        box_in = rng.random((nslices, nangles))
        delta_los = shift * np.ones_like(box_in)
        box_out = cic_los(
            field=box_in,
            delta_los=delta_los,
            periodic=True,
        )

        box_in_shifted = np.roll(box_in, shift, axis=0)
        np.testing.assert_allclose(box_out, box_in_shifted)

    def test_mass_conservation_periodic(self):
        """Test that mass is conserved when shifting periodic boxes."""
        nangles = 5
        nslices = 10
        rng = np.random.default_rng(12345)
        box_in = rng.random((nangles, nslices))
        delta_los = rng.uniform(0, nslices * 2, size=box_in.shape)
        box_out = cic_los(
            field=box_in,
            delta_los=delta_los,
            periodic=True,
        )
        assert np.isclose(np.sum(box_out), np.sum(box_in))

    def test_huge_shift_non_periodic(self):
        """If the velocities are large, all mass is shifted out of the box."""
        nangles = 5
        nslices = 10
        rng = np.random.default_rng(12345)
        box_in = rng.random((nangles, nslices))
        delta_los = nslices * 2 * np.ones_like(box_in)
        box_out = cic_los(
            field=box_in,
            delta_los=delta_los,
        )
        assert np.allclose(box_out, 0)

    @pytest.mark.parametrize("periodic", [True, False])
    def test_zero_shift_identity(self, periodic: bool):
        """Test whether a zero-shift does nothing."""
        nangles = 5
        nslices = 10
        rng = np.random.default_rng(12345)
        box_in = rng.random((nangles, nslices))
        delta_los = np.zeros_like(box_in)

        box_out = cic_los(field=box_in, delta_los=delta_los, periodic=periodic)
        assert np.allclose(box_out, box_in)

    def test_bad_inputs(self) -> None:
        """Test the cloud-in-cell line-of-sight function."""
        field = np.zeros((10, 20))
        los = np.zeros((11, 20))

        with pytest.raises(ValueError, match="Field and displacement must have the same shape."):
            cic_los(field, los)
