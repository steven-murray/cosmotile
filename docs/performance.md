# Performance

Tiling is one operation repeated an enormous number of times: for every output sample,
gather $(p+1)^3$ cells out of the coeval box and add them up. A lightcone of 51 shells at
`nside=128` with `subsample_level=1` and `n_radial_samples=4` is about 160 million of
those. So it is worth knowing what that costs, and what changes it.

The numbers come from `benchmarks/run_benchmarks.py`, whose output is committed at
`benchmarks/results/latest.json`; re-run it with `nox -s benchmarks`. Every row times a
public `cosmotile` entry point, not the library it calls underneath. They were measured
on a 16-core CPU and an RTX A2000 Laptop GPU — a small, power-limited card, which affects
the results considerably. **Measure on your own hardware before planning around any of
this.**

## The one number

A $256^3$ box tiled onto an `nside=256` shell at order 3:

| path | throughput | relative |
| --- | --- | --- |
| `scipy.ndimage.map_coordinates` (the fallback) | 4.5 Mpix/s | 1× |
| **NumPy backend with `numba`** (the default) | **47 Mpix/s** | 10× |
| `cosmotile.jax`, CPU, float64 | 2.8 Mpix/s | 0.6× |
| `cosmotile.jax`, CPU, float32 | 5.1 Mpix/s | 1.1× |
| `cosmotile.jax`, GPU, float64 | 208 Mpix/s | 46× |
| `cosmotile.jax`, GPU, float32 | 409 Mpix/s | 90× |

Against the `numba` backend, the GPU gains a speedup of about **9×** here.

The JAX **CPU** backend is not a speedup — it is roughly `scipy`'s speed and well behind
`numba`. Its value on a CPU is that it is differentiable.

## Why the GPU wins

Because the kernel is bound by memory *latency*, not by arithmetic, and a GPU keeps tens
of thousands of gathers in flight so the latency is never waited on.

Locality shows the same thing from the other side. The same 196 608 pixels, the same box,
order 3, varying only *where* the samples land:

| coordinates | throughput |
| --- | --- |
| HEALPix **nested**, $r = 60$ cells | 2.67 Mpix/s |
| HEALPix **ring**, $r = 60$ cells | 1.72 Mpix/s |
| HEALPix nested, $r = 300$ cells | 0.90 Mpix/s |
| random | 0.32 Mpix/s |

**Nested ordering is about 1.7 times faster than ring ordering** on the same pixels,
purely because neighbouring pixels in the nested scheme land nearer each other in the
box. A larger shell is slower for the same reason: its samples are spread more thinly, so
fewer share a cache line.

It is also why the benchmark tiles real HEALPix shells rather than random coordinates.
Random coordinates understate throughput by up to eight times and would make every
backend look artificially alike.

### If you are not using HEALPix

The principle is that samples adjacent in your coordinate array should be near each other
*in the box*. HEALPix nested ordering happens to do this well; latitude–longitude grids do
it moderately well along rows and badly between them; an arbitrary list of directions does
it not at all.

If you build coordinates yourself and tiling is your bottleneck, sort them before tiling —
by the Morton (Z-order) code of the integer cell each sample falls in, or failing that
simply by one cartesian axis — and un-sort the result afterwards. The factor available is
the one in the table above: several-fold, not a few percent.

## How it scales with interpolation order

$256^3$ box, `nside=256`, Mpix/s:

| order | `scipy` | `numba` | GPU float32 | GPU float64 |
| --- | --- | --- | --- | --- |
| 0 | 29.5 | 178 | 6989 | 5223 |
| 1 | 15.2 | 122 | 1957 | 1003 |
| 3 | 4.5 | 47 | 409 | 208 |
| 5 | 1.6 | 14 | 19 | 47 |

```{figure} figures/throughput_by_order.svg
:alt: Throughput against interpolation order for each backend, log scale

Every backend falls off with order at roughly the rate the work grows — order 5 gathers
216 cells per sample against order 1's 8. Orders 0 and 1 are so cheap per sample that
these measure dispatch overhead as much as the kernel; unfilled bars are measurements the
harness flagged as unreliable.
```

Against `scipy` the GPU advantage grows with order. However, against the default `numba`
implementation it diminishes with order:

```{figure} figures/gpu_speedup.svg
:alt: GPU speedup over the numba path, falling from about 40x at order 0 to parity at order 5

The GPU in single precision, relative to the default `numba` path. The advantage falls
from roughly 40× at order 0 to about 9× at order 3, and by order 5 it is gone. `numba`
scales with order about as well as the GPU does.
```

So if you tile at order 5, a GPU may buy you nothing at all.

Note the order-5 float32 row: on this card single precision is about **2.5 times slower**
than double there, reproducibly. That is the opposite of every other row and is a
property of how the compiler handles the 216-tap kernel, not something to rely on either
way — another reason to measure your own hardware.

## The pre-filter

Orders above 1 need the box converted to B-spline coefficients first. `scipy` does this
with a recursive filter; the JAX backend does it as an FFT deconvolution, which is
*exactly* the same operator under the periodic boundary condition `cosmotile` uses,
agreeing to $7 \times 10^{-15}$ relative in double precision.

Time to pre-filter a whole box at order 3:

| box | `scipy` | GPU float32 | GPU float64 |
| --- | --- | --- | --- |
| $256^3$ | 0.80 s | 0.008 s | 0.074 s |
| $384^3$ | 2.60 s | 0.029 s | — |

A one-off per box either way, and hoisted out of the shell loop by
{func}`~cosmotile.prefilter_coeval` on both backends — but it is worth having when you are
fitting and the box changes every iteration. On a CPU the FFT route is about three times
*slower* than `scipy`'s recursive filter; it is a GPU win specifically.

## Memory

The box, its coefficients and — under reverse-mode autodiff — its cotangent all have to
sit on the device together, and the FFT pre-filter needs a transient workspace on top.
A rough estimate for the whole forward-and-backward pass is

$$
\text{bytes} \approx 20 N^3 \quad \text{(float32)},
$$

so about 340 MB for $256^3$, 1.3 GB for $512^3$ and 10 GB for $1024^3$; double that in
`float64`. Single precision is not only faster, it is often what makes the larger box fit.

The output is its own problem. A thousand-shell lightcone at `nside=256` is 3.1 GB, and
building all of its coordinates at once with `jax.vmap` would want a further 19 GB. That
is what {func}`~cosmotile.jax.lightcone_scan` is for: it makes one shell at a time, so the
peak is one shell rather than all of them. See
[Differentiable and GPU tiling](jax-backend).

## What this means in practice

The full-lightcone example from [Usage](usage) — 51 shells, `nside=128`,
`subsample_level=1`, `n_radial_samples=4`, order 3 — is about 160 million interpolations:

| | time |
| --- | --- |
| `scipy` fallback | ~36 s |
| NumPy backend with `numba` | ~3 s |
| `cosmotile.jax` on a GPU, float32 | well under a second |

At that point the pre-filter and the host-to-device transfer, not the gather, are what
you are paying for.

## A note on measurement

These numbers came from a power-limited laptop GPU, which idles at a fraction of its
clock and ramps under load. Measured carelessly the same kernel differed by a factor of
seven between runs, so the harness warms up to a steady state, reports the median
alongside the best of nine runs, and flags rows where the two disagree. Treat a flagged
row as an upper bound.

## When not to use the `jax` backend

- **Order 5.** The GPU is at parity with the `numba` path there.
- **CPU only.** The JAX CPU backend is slower than `numba` at the orders you would
  actually use. Its value there is that it is differentiable, not that it is quick.
- **One shell, once.** Compilation is not free, and neither is moving a box to a device.
  The win is in loops.
- **A box that does not fit.** Once you are chunking or spilling to host memory, the
  transfers can cost more than the gather saves.

## Reproducing these numbers

```bash
python benchmarks/run_benchmarks.py --out benchmarks/results/latest.json
python benchmarks/make_performance_figures.py
```

Both need the `all` extra. They write committed files, so the documentation builds
without a GPU, `jax` or `matplotlib`.
