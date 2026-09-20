# Differentiable and GPU tiling

`cosmotile.jax` is an optional backend that computes the same thing as the NumPy API,
but as pure JAX functions. Install it with

```console
$ pip install cosmotile[jax]
```

It exists for two reasons:

**Gradients.** Tiling is *linear in the box values*, so its derivatives are automatic,
cheap, exact and well conditioned. This lets `cosmotile` be part of a differentiable
forward model, so a lightcone observation can be fitted back to whatever produced the
box.

**Speed.** The gather step is well suited to GPU acceleration. Measured on a $256^3$ box
at `nside=256`, order 3:

| backend | throughput |
| --- | --- |
| NumPy path, `scipy` fallback | 4.5 Mpix/s |
| NumPy path with `numba` (the default) | 47 Mpix/s |
| `cosmotile.jax`, GPU, float64 | 208 Mpix/s |
| `cosmotile.jax`, GPU, float32 | 409 Mpix/s |

See [Performance](performance) for the full picture, including when this is *not* worth it.

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

These are ordinary functions, so `jax.jit` them at whatever granularity suits you — the
whole forward model is usually the right level:

```python
import jax
from functools import partial


@partial(jax.jit, static_argnames="sampling")
def one_shell(coeff, radius):
    return cjax.shell(coeff, sampling, radius)
```

`sampling` is static because it carries the interpolation order, which fixes the shape of
the computation. The internal kernels are already jitted; nesting `jit` costs nothing, as
JAX inlines the inner one.

Unlike {func}`~cosmotile.prefilter_coeval`, the JAX `prefilter_coeval` is **not**
optional above order 1. The NumPy path will filter a raw box for you; this one refuses,
because doing it inside a jitted shell would redo it for every shell of the lightcone.

## Performance tips for creating a whole lightcone

A thousand shells at `nside=256` is 3.1 GB, and building all their coordinates at once
with `jax.vmap` would need a further 19 GB. Neither fits on a typical GPU.

If you only want a *summary* of the lightcone — a likelihood, an angular power spectrum,
a sum over shells — you never need it all at once.
{func}`~cosmotile.jax.lightcone_scan` makes one shell at a time and hands it to a function
that combines it with a running result:

```python
import jax.numpy as jnp

radii = jnp.linspace(100.0, 400.0, 1000)

total_power = cjax.lightcone_scan(
    coeff,
    sampling,
    radii,
    lambda running_total, shell: running_total + jnp.mean(shell**2),
    init=0.0,
)
```

This matters most when you are differentiating: the backward pass would otherwise have to
keep every shell in memory at once. If you want the shells themselves and they fit, just
call {func}`~cosmotile.jax.shell` in a loop.

`vmap` is the right tool across *fields* — many boxes on one geometry — since those share
their coordinates.

## What you can differentiate

```python
import jax

gradient = jax.grad(
    lambda b: jnp.sum(cjax.shell(cjax.prefilter_coeval(b, 3), sampling, 150.0) ** 2)
)(box)
```

**The box, and anything upstream of it.** This is the case that matters, and it works at
every order, exactly. Tiling is linear in the box values, so the reverse pass is the exact
transpose of the interpolation rather than an approximation to it — the test suite asserts
$\langle Ac, y\rangle = \langle c, A^{\mathsf{T}} y\rangle$ to machine precision.

Because it is exact and unconditional, gradients with respect to whatever *produced* the
box — cosmological parameters, an emulator's weights, an astrophysical model — flow
straight through by the chain rule, at any interpolation order:

```python
def loss(params):
    box = my_simulator(params)  # differentiable, in JAX
    coeff = cjax.prefilter_coeval(box, order=1)
    return jnp.sum((cjax.shell(coeff, sampling, 150.0) - observed) ** 2)


gradient = jax.grad(loss)(params)
```

Note the pre-filter has to be inside the traced function for this. If you pre-filter on
the host and pass the coefficients in, your gradient is with respect to the coefficients.

**The shell geometry itself** — differentiating with respect to `radius`, `origin` or
`rotation`, i.e. asking how the output changes if you move the shell through the box — is
a separate and much rarer thing, and it needs **order 3 or above**. Order 0 is piecewise
constant, so this derivative is identically zero and `jax.grad` will hand you zeros
without complaint; order 1 is piecewise linear, so it jumps at every cell boundary. None
of that affects the parameter gradients above.

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

One exception: if you differentiate with respect to the shell *geometry* (see above), do
it under `jax.enable_x64()` or at radii below a few thousand cells. Everything else is
fine in single precision.

## Memory

The box, its coefficients and — under reverse mode — its cotangent all have to fit on the
device at once, which is three copies of $4N^3$ bytes in `float32` (twice that in
`float64`). Budget about another $2 N^3$ complex values for the FFT pre-filter, which is
the transient peak.

So a rough estimate for the whole forward-and-backward pass is $20 N^3$ bytes in
`float32`: around 1.3 GB for $512^3$, 10 GB for $1024^3$. If that does not fit,
pre-filter on the host with {func}`~cosmotile.prefilter_coeval` and move the coefficients
across — at the cost of your gradient then being with respect to the coefficients rather
than the raw box.
