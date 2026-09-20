# Differentiable and GPU tiling

`cosmotile.jax` is an optional backend that computes the same thing as the NumPy API,
but as pure JAX functions. Install it with

```console
$ pip install cosmotile[jax]
```

It exists for two reasons, and the first is the one that cannot be had any other way.

**Gradients.** Tiling is *linear in the box values*: every output pixel is a fixed
weighted sum of a few dozen cells. So the derivative of a shell with respect to the
coeval box is exactly the transpose of that sum — a scatter-add — and it is cheap,
exact and well conditioned. That lets `cosmotile` sit inside a differentiable forward
model, so a lightcone observation can be fitted back to whatever produced the box.

**Speed.** The gather is limited by memory *latency*, not arithmetic — which is precisely
the workload a GPU hides best. Measured on a $384^3$ box at `nside=256`, order 3:

| backend | throughput |
| --- | --- |
| NumPy path, `scipy` fallback | 3.5 Mpix/s |
| NumPy path with `numba` (the default) | 30 Mpix/s |
| `cosmotile.jax`, GPU, float64 | 161 Mpix/s |
| `cosmotile.jax`, GPU, float32 | 544 Mpix/s |

Note what it has to beat: the NumPy path is not `scipy` unless `numba` is missing. See
[Performance](performance) for the full picture, including when this is *not* worth it.

## It is not a mirror of the NumPy API

The NumPy API returns lazy iterators, carries `astropy` units, and validates its inputs
by looking at their values. All three are good things there and impossible under
`jax.jit`, so rather than offer functions with shapes that cannot be traced, this backend
offers one pure function per operation.

```python
import numpy as np
import cosmotile
from cosmotile import jax as cjax

lat, lon = cosmotile.healpix_subpixel_lonlat(nside=256, subsample_level=0)

sampling = cjax.make_shell_sampling(latitude=lat, longitude=lon, order=3)
coeff = cjax.prefilter_coeval(box, order=3)

shell = cjax.shell(coeff, sampling, radius=150.0)
```

`make_shell_sampling` holds everything about the geometry that does *not* depend on the
radius — in particular the unit vectors, which are the expensive part. One of these
serves a whole lightcone, and each shell then costs one scalar.

Nothing in the backend is decorated with `jax.jit`. Wrap the call site yourself;
pre-jitting a library primitive would compile it a second time when you jit the caller.

Unlike {func}`~cosmotile.prefilter_coeval`, the JAX `prefilter_coeval` is **not**
optional above order 1. The NumPy path will filter a raw box for you; this one refuses,
because doing it inside a jitted shell would redo it for every shell of the lightcone.

## A whole lightcone: scan, never vmap

A thousand shells at `nside=256` is 3.1 GB of output, and building all their coordinates
at once with `jax.vmap` would need a further 19 GB. Neither fits on a GPU. Use
{func}`~cosmotile.jax.lightcone_scan`, which folds a function over the shells with
`lax.scan` so that only one is ever live, rebuilding its coordinates from the shared unit
vectors:

```python
import jax.numpy as jnp

radii = jnp.linspace(100.0, 400.0, 1000)

total_power = cjax.lightcone_scan(
    coeff,
    sampling,
    radii,
    lambda carry, shell: carry + jnp.mean(shell**2),
    init=0.0,
)
```

Folding the shells into the statistic you want is the point. If you make the shells
*outputs* of the scan instead, reverse mode needs every shell's cotangent live and the
3.1 GB is back.

`vmap` is still the right tool across *fields* — many boxes on one geometry — since those
share their coordinates.

## What you can differentiate

```python
import jax

gradient = jax.grad(
    lambda b: jnp.sum(cjax.shell(cjax.prefilter_coeval(b, 3), sampling, 150.0) ** 2)
)(box)
```

**With respect to the box**, at every order, exactly. The reverse pass is the transpose
of the interpolation, not an approximation to it — the test suite asserts
$\langle Ac, y\rangle = \langle c, A^{\mathsf{T}} y\rangle$ to machine precision. This is
the gradient field-level inference wants.

**With respect to the geometry** — radius, origin, rotation — **use order 3 or above.**
Order 0 is piecewise constant, so its gradient is identically zero: `jax.grad` will
cheerfully hand you a field of zeros. Order 1 is piecewise linear, so its derivative
exists almost everywhere but jumps at every cell boundary. Only from order 3 is the
reconstruction smooth enough for the derivative to mean what you want it to.

One caveat specific to geometry gradients: see the note on precision below.

The `remat` flag on `lightcone_scan` re-materialises each step instead of saving its
residuals. It is rarely worth it for the interpolation itself, whose residuals are nearly
free because the map is linear with constant indices — reach for it when your own `body`
is the expensive part.

## Precision

JAX defaults to **single precision**, and this backend does not change that. Flipping
`jax_enable_x64` is global process state, and a library that flipped it would silently
change the numerics of every other JAX library you had imported. Use the scoped context
manager if you want double:

```python
with jax.enable_x64():
    shell = cjax.shell(coeff, sampling, radius=150.0)
```

In double precision the backend agrees with the NumPy path to about $10^{-14}$ relative,
and *exactly* at order 0. In single precision, expect around $2 \times 10^{-6}$ for the
gather and $3 \times 10^{-6}$ for the pre-filter — the pre-filter being the looser of the
two because it is a *global* deconvolution, so its error does not stay local the way the
gather's does.

The gather stays that accurate in `float32` only because of a deliberate choice: a
coordinate is never carried as a single float. It is split on the host into an integer
cell index and a fraction in $[0, 1)$, so the quantity that has to survive `float32` is
always of order one. Carrying a shell radius of a thousand cells directly would already
be uncertain by $10^{-4}$ cells, which would swamp everything you paid for by using a
high order.

That split is also the caveat on geometry gradients: differentiating with respect to a
radius means the split has to happen *inside* the trace, where `float32` reintroduces
exactly that error. Differentiate geometry under `jax.enable_x64()`, or at radii below a
few thousand cells.

## Memory

The box, its coefficients and — under reverse mode — its cotangent all have to fit on the
device at once. In `float32` that is three copies of $4N^3$ bytes; on a 4 GB card, $384^3$
fits comfortably and $448^3$ does not. Single precision is not only faster here, it is
what makes the larger box fit at all.

The FFT pre-filter is the peak: about 460 MB of workspace for a $384^3$ box in `float32`,
920 MB in `float64`. It runs once per box, so if you are tight on memory, pre-filter on
the host with {func}`~cosmotile.prefilter_coeval` and move the coefficients across — at
the cost of your gradient then being with respect to the coefficients rather than the raw
box.
