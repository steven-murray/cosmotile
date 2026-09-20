# Performance

Tiling is one operation repeated an enormous number of times: for every output sample,
gather $(p+1)^3$ cells out of the coeval box and add them up. A lightcone of 51 shells at
`nside=128` with `subsample_level=1` and `n_radial_samples=4` is about 160 million of
those. So it is worth knowing what that costs, and what changes it.

The numbers come from `benchmarks/run_benchmarks.py`, whose output is committed at
`benchmarks/results/latest.json`. Re-run it with `nox -s benchmarks`. The machine was a
16-core CPU and an RTX A2000 Laptop GPU with **4 GB** — a small, power-limited card,
which turns out to matter twice over.

## The one number

A $256^3$ box tiled onto an `nside=256` shell at order 3, which is the setup you should
probably be using:

| path | throughput | relative |
| --- | --- | --- |
| `scipy.ndimage.map_coordinates` (the fallback) | 3.5 Mpix/s | 1× |
| **NumPy backend with `numba`** (the default) | **30 Mpix/s** | **8.7×** |
| `cosmotile.jax`, CPU, float64 | 2.2 Mpix/s | 0.6× |
| `cosmotile.jax`, CPU, float32 | 3.8 Mpix/s | 1.1× |
| `cosmotile.jax`, GPU, float64 | 161 Mpix/s | 46× |
| `cosmotile.jax`, GPU, float32 | 544 Mpix/s | **155×** |

Three things to read off that.

`scipy` is the *fallback*, not the default. With `numba` installed — it is the `perf`
extra, and has been for a long time — the NumPy backend gathers through its own parallel
kernel instead, which is about nine times faster and agrees with `scipy` to roundoff.
**That is free and needs no code change**; it is the row most users are actually on, and
it is what the GPU has to beat.

Against that, the GPU is about **18×**, not 155×. Still decisive, but worth stating
honestly.

And the JAX **CPU** backend is not a speedup at all — it is slower than `scipy`, let
alone than the `numba` kernel. Its value on a CPU is that it is differentiable.

## Why the GPU wins

Because the kernel is bound by memory *latency*, not by arithmetic.

The evidence is that single precision buys nothing on a CPU. The `numba` gather runs at
essentially identical speed in float64 and float32, even though float32 halves the bytes
moved. The cores are not waiting for bandwidth or for the FPU; they are waiting for DRAM,
one cache miss at a time.

Locality confirms it from the other side. The same 196 608 pixels, the same box, order 3,
varying only *where* the samples land:

| coordinates | throughput |
| --- | --- |
| HEALPix **nested**, $r = 60$ cells | 2.67 Mpix/s |
| HEALPix **ring**, $r = 60$ cells | 1.72 Mpix/s |
| HEALPix nested, $r = 300$ cells | 0.90 Mpix/s |
| random | 0.32 Mpix/s |

**Nested ordering is about 1.7 times faster than ring ordering** on the same pixels,
purely because neighbouring pixels in the nested scheme land nearer each other in the
box. A larger shell is slower for the same reason: its samples are spread more thinly, so
fewer share a cache line. This is free performance if your pipeline does not care about
the ordering.

It is also why the benchmark tiles real HEALPix shells rather than random coordinates.
Random coordinates understate throughput by up to eight times and would make every
backend look artificially alike — and on a GPU the same mistake goes the other way, which
is how a casual measurement of this kernel came out at 62 Mpix/s when the real figure was
seven times higher.

A GPU fixes precisely this problem. It does not make any single gather faster; it keeps
tens of thousands of them in flight so the latency never has to be waited on.

## How it scales with order

$256^3$ box, `nside=256`, Mpix/s:

| order | `scipy` | `numba` | GPU float32 | GPU float64 |
| --- | --- | --- | --- | --- |
| 0 | 23.5 | 104 | 2943 | 1557 |
| 1 | 11.4 | 68 | 1589 | 762 |
| 3 | 3.5 | 30 | 544 | 161 |
| 5 | 1.3 | 13 | 85 | 52 |

```{figure} figures/throughput_by_order.svg
:alt: Throughput against interpolation order for each backend, log scale

Throughput for a $256^3$ box on an `nside=256` shell. Every backend falls off with
order at roughly the rate the work grows — order 5 gathers 216 cells per sample against
order 1's 8 — and the ordering between them is stable across orders. Hollow bars, where
they appear, are measurements the harness flagged as unreliable.
```

Against `scipy` the GPU advantage looks like it grows with order. Against the path you
are actually on it does the opposite:

```{figure} figures/gpu_speedup.svg
:alt: GPU speedup over the numba path, falling from about 28x at order 0 to 7x at order 5

The GPU in single precision, relative to the default `numba` path. The advantage *falls*
with order — about 28× at order 0, 18× at order 3 and under 7× at order 5 — and barely
depends on the box size. `numba` scales with order about as well as the GPU does, so the
high orders are where the CPU path closes the gap, not where it loses it.
```

That is worth knowing before you reach for a GPU: if you tile at order 5, the honest
figure is closer to 7× than to 100×.

## The pre-filter

Orders above 1 need the box converted to B-spline coefficients first. `scipy` does this
with a recursive filter; the JAX backend does it as an FFT deconvolution, which is
*exactly* the same operator under the periodic boundary condition `cosmotile` uses,
agreeing to $7 \times 10^{-15}$ relative in double precision.

Time to pre-filter a whole box at order 3:

| box | `scipy` | GPU float32 | GPU float64 |
| --- | --- | --- | --- |
| $256^3$ | 1.20 s | 0.008 s | 0.074 s |
| $384^3$ | 4.02 s | 0.029 s | out of memory |

A one-off per box either way, and hoisted out of the shell loop by
{func}`~cosmotile.prefilter_coeval` on both backends — but it is worth having when you
are fitting and the box changes every iteration.

Note this reverses on a CPU: the FFT route is about three times *slower* than `scipy`'s
recursive filter there. It is a GPU win specifically.

## Memory, and why 4 GB is the real constraint

The box, its coefficients and — under reverse-mode autodiff — its cotangent all have to
sit on the device together. In float32 that is $4N^3$ bytes apiece:

| box | float32 | float64 |
| --- | --- | --- |
| $256^3$ | 67 MB | 134 MB |
| $384^3$ | 226 MB | 453 MB |
| $512^3$ | 537 MB | 1.07 GB |

On this 4 GB card, $384^3$ at order 3 runs in float32 and **runs out of memory in
float64**; at order 5 it does not fit in either. Single precision is not merely faster
here — it is what makes the larger box possible at all. The FFT pre-filter is the peak:
roughly 460 MB of workspace for $384^3$ in float32, 920 MB in float64.

The output is its own problem. A thousand-shell lightcone at `nside=256` is 3.1 GB, and
building all of its coordinates at once with `jax.vmap` would want a further 19 GB.
Neither fits. That is what {func}`~cosmotile.jax.lightcone_scan` is for: it folds over the
shells one at a time, rebuilding each shell's coordinates from shared unit vectors, so the
peak is one shell rather than all of them. See
[Differentiable and GPU tiling](jax-backend).

## What this means in practice

The full-lightcone example from [Usage](usage) — 51 shells, `nside=128`,
`subsample_level=1`, `n_radial_samples=4`, order 3 — is about 160 million interpolations:

| | gather time |
| --- | --- |
| `scipy` fallback | ~46 s |
| NumPy backend with `numba` | ~5 s |
| `cosmotile.jax` on a GPU, float32 | well under a second |

Which is the difference between a thing you wait for and a thing you put inside a fitting
loop. At that point the pre-filter and the host-to-device transfer, not the gather,
are what you are paying for.

## A caveat about these numbers

The GPU here is a power-limited laptop part. It idles at a tenth of its maximum clock,
ramps under sustained load and is then pulled back by its power cap — it had accumulated
half a minute of power capping by the time these were taken. Measured carelessly, the
same kernel differed by a factor of seven between runs minutes apart.

The harness therefore warms up until at least half a second of wall clock has passed
rather than for a fixed number of calls, and records the median alongside the best of
nine runs, flagging any row where the two disagree by more than half. Treat a flagged row
as an upper bound rather than a measurement, and treat all of these as the shape of the
effect rather than a specification of your hardware.

## When not to bother

- **Order 0 or 1 on small shells.** The kernel is too cheap to amortise the launch, and
  the `numba` path is already doing 70–100 Mpix/s.
- **Order 5, unless the box is large.** The GPU is under 7× the `numba` path there, which
  may not repay moving the box across.
- **CPU only.** The JAX CPU backend is slower than both `scipy` and `numba` at the orders
  you would actually use. Its value there is that it is differentiable, not that it is
  quick.
- **One shell, once.** Compilation is not free, and neither is moving a box to a device.
  The win is in loops.
- **A box that does not fit.** Above roughly $384^3$ on a 4 GB card you will be chunking
  or dropping to float32, and at some point the transfers cost more than the gather saves.

## Reproducing these numbers

```bash
python benchmarks/run_benchmarks.py --out benchmarks/results/latest.json
python benchmarks/make_performance_figures.py
```

The first needs the `all` extra; the second needs `matplotlib`. Both write committed
files, so the documentation builds without either — and without a GPU.
