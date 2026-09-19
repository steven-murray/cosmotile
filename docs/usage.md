# Usage

`cosmotile` interpolates a regular, periodic coeval box onto a set of angular
coordinates on a single spherical shell. To build a full lightcone, call it once per
shell radius.

Everything is measured in **cell units**: `distance_to_shell` and `origin` are in units
of the coeval box's cell size, not in Mpc. Use
{func}`~cosmotile.get_distance_to_shell_from_redshift` to convert.

## A single shell

```python
import numpy as np

import cosmotile

# An artificial co-eval "simulation", periodic on its boundaries: zeros everywhere
# except for ones on the three coordinate planes.
box = np.zeros((100, 100, 100))
box[0] = 1
box[:, 0] = 1
box[:, :, 0] = 1

# Evaluate at a latitude of zero, right around the full longitude.
lat = np.zeros(1000)
lon = np.linspace(0, 2 * np.pi, 1000, endpoint=False)

(shell,) = cosmotile.make_lightcone_slice(
    coevals=box,
    latitude=lat,
    longitude=lon,
    distance_to_shell=100.0,  # in units of the cell size
)
```

`make_lightcone_slice` takes an *iterable* of coeval boxes and returns an iterator over
the corresponding shells, so several co-located fields (density, ionisation fraction,
spin temperature, ...) can share one set of interpolation coordinates:

```python
density_shell, xhi_shell = cosmotile.make_lightcone_slice(
    coevals=[density, xhi],
    latitude=lat,
    longitude=lon,
    distance_to_shell=100.0,
)
```

## On a HEALPix grid

{func}`~cosmotile.make_healpix_lightcone_slice` sets up the angular coordinates for you:

```python
(shell,) = cosmotile.make_healpix_lightcone_slice(
    nside=128,
    coevals=box,
    distance_to_shell=100.0,
)
```

## A full lightcone

Convert each redshift to a comoving distance in cell units, then loop:

```python
import numpy as np
from astropy import units as un
from astropy.cosmology import Planck18

import cosmotile

cell_size = 2.0 * un.Mpc
redshifts = np.linspace(7.0, 9.0, 51)
distances = [
    cosmotile.get_distance_to_shell_from_redshift(z=z, cell_size=cell_size, cosmo=Planck18)
    for z in redshifts
]

nside = 128
lightcone = np.array(
    [
        next(
            cosmotile.make_healpix_lightcone_slice(
                nside=nside, coevals=box, distance_to_shell=d.value
            )
        )
        for d in distances
    ]
)
```

Passing a different `rotation` and `origin` per shell yields an apparently different
realisation from the same box, at the cost of breaking the correlation along the line of
sight. See [Accuracy and Limitations](accuracy) for when that trade is worth making.

## Redshift-space distortions

Project the peculiar velocity field onto the line of sight, then displace the field
along it. Both steps use the convention that **positive means towards the observer**.

```python
from astropy_healpix import HEALPix

healpix = HEALPix(nside=nside, order="ring")
hp_lon, hp_lat = healpix.healpix_to_lonlat(np.arange(healpix.npix))

displacements = []
for d in distances:
    interpolator = cosmotile.make_lightcone_slice_interpolator(
        latitude=hp_lat.to_value("radian"),
        longitude=hp_lon.to_value("radian"),
        distance_to_shell=d.value,
    )
    # vx, vy, vz in units of the cell size, i.e. v / H(z) / cell_size.
    (los,) = cosmotile.make_lightcone_slice_vector_field([[vx, vy, vz]], interpolator)
    displacements.append(los)

distorted = cosmotile.apply_rsds(
    field=lightcone,  # shape (nslices, ncoords)
    los_displacement=np.array(displacements) * un.pixel,
    distance=un.Quantity(distances),  # shape (nslices,)
)
```

A positive `los_displacement` moves a parcel *towards* the observer, so it appears at
smaller comoving distance. That is the same sign convention
{func}`~cosmotile.make_lightcone_slice_vector_field` returns, so the two chain directly
as above.

## Before you trust the output

The tiling geometry is exact, but a finite periodic box cannot reproduce every angular
scale. Roughly, the angular power spectrum is reliable for

$$
\frac{2\pi r}{L} \lesssim \ell \lesssim \frac{\pi r}{\Delta},
$$

for a box of length $L$ and cell size $\Delta$ at shell radius $r$.
[Accuracy and Limitations](accuracy) works through where this comes from, how badly it
fails outside that window, and how to choose `nside` and `interpolation_order`.
