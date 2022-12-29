"""Tests of the cloud-in-cell function."""
import numpy as np

from cosmotile.cic import cloud_in_cell


def test_cic_dx() -> None:
    """Test the cloud-in-cell function."""
    x = np.zeros((10, 10, 10))
    x[::2] = 1

    dx = np.zeros_like(x)
    dx[::2] = 1.0
    dy = np.zeros_like(x)
    dz = np.zeros_like(x)

    out = cloud_in_cell(x, dx, dy, dz)

    print(np.any(out > 0.0))
    assert np.allclose(out[1::2], 1.0)

    out = cloud_in_cell(x, -dx, dy, dz)
    assert np.allclose(out[1::2], 1.0)


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
