"""Tests of the cloud-in-cell function."""
import numpy as np
import pytest

from cosmotile.cic import cloud_in_cell_coeval as cloud_in_cell
from cosmotile.cic import cloud_in_cell_los as cic_los


def test_cic_bad_input() -> None:
    """Test that different-shaped inputs to CIC raise appropriate error."""
    fld = np.zeros((10, 10, 10))
    dx = np.zeros_like(fld)
    dy = np.zeros_like(fld)
    dz = np.zeros_like(fld)

    with pytest.raises(
        ValueError, match="Field and displacement must have the same shape."
    ):
        cloud_in_cell(fld, dx, dy, dz[1:])


def test_cic_dx() -> None:
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


def test_cic_dy() -> None:
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


def test_cic_dz() -> None:
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


def test_cic_los() -> None:
    """Test the cloud-in-cell line-of-sight function."""
    field = np.zeros((10, 20))
    los = np.zeros((11, 20))

    with pytest.raises(
        ValueError, match="Field and displacement must have the same shape."
    ):
        cic_los(field, los)
