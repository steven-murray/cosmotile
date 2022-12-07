# Usage

A simple example of using cosmotile is as follows:

```python
import cosmotile
import numpy as np

# Create an artificial co-eval "simulation" that is periodic on its boundaries.
# In this case, the simulation is simply zeros with ones on all three planes:
box = np.zeros((100, 100, 100))
box[0] = 1
box[:, 0] = 1
box[:, :, 0] = 1

# cosmotile interpolates a regular, periodic box onto an arbitrary set of angular
# coordinates on a single spherical shell (to get a full 'lightcone', just re-call
# the function for different shell radii).
# Evaluate at a latitude of zero, around the full longitude.
lat = np.zeros(1000)
lon = np.linspace(0, 2*np.pi, 1000, endpoint=False)

lc_slice = cosmotile.make_lightcone_slice(
    coeval = box,
    coeval_res = 1.0,         # resolution of each cell of the box (in cMpc)
    latitude=lat,
    longitude=lon,
    distance_to_shell = 100,  # the radius of the shell (in cMpc)
)

# To get a lightcone between 100-200 Mpc:
distances = np.linspace(100, 200, 101)
lc = np.zeros((len(lat), len(distances)))


```
