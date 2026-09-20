# Accuracy and Limitations

`cosmotile` cuts a spherical shell out of a periodically-tiled coeval box. The geometry
of that operation is exact, and the tests in `tests/test_geometry.py` check it to
machine precision. What is *not* exact is the statistics of the resulting map: a finite,
periodic, discretely-sampled box cannot reproduce every angular scale, and this page
says precisely which ones it can.

Read the summary, then use the rest of the page to work out the numbers for your own
setup.

## Summary: the window of validity

For a coeval box of $N$ cells on a side with cell size $\Delta$ (so box length
$L = N\Delta$), tiled onto a shell at comoving distance $r$, the angular power spectrum
is trustworthy for

$$
\frac{2\pi r}{L} \;\lesssim\; \ell \;\lesssim\; \frac{\pi r}{\Delta}
$$

with the additional requirement that $\ell \ll 2 N_{\rm side}$ if you are working on a
HEALPix grid. In words:

| Limit | Set by | Symptom if you ignore it |
|---|---|---|
| $\ell \lesssim 2\pi r / L$ | The box fundamental $k_{\min} = 2\pi/L$ | Power is **missing**, by up to 100% |
| $\ell \gtrsim \pi r / \Delta$ | The Nyquist wavenumber of the box grid | Power is **spurious** (aliasing floor) |
| $\ell \gtrsim 2 N_{\rm side}$ | The HEALPix grid | Small-scale power aliases down (mitigated by `subsample_level`) |
| $r / L \gtrsim 1$ | Periodic replication | The same structures recur across the sky |

```{figure} figures/validity_window.svg
:alt: Angular power spectrum of a tiled shell compared with theory

The angular power spectrum of a shell tiled from a $128^3$ box at $r/L = 1.56$, averaged
over four realisations. The white region is the window of validity. Inside it the
measurement tracks the discrete box-mode prediction to a few percent. To the left, the
infinite-box prediction (dotted) runs above both: that gap is power the box does not
contain. To the right, the true signal has fallen away but the measurement flattens onto
an aliasing floor.
```

## What a cell holds, and what comes out

Before any of the numbers below mean anything, you need to know what a grid value *is*.

A simulation cell almost always holds the **mean of the field over that cell**, not a
sample at its centre. Writing the underlying continuous field as $f$ and the cell top-hat
as $T$, the box is

$$
g_{\mathbf n} = (T \ast f)(\mathbf x_{\mathbf n}),
\qquad
\tilde T(\mathbf k) = \prod_i \mathrm{sinc}(k_i \Delta / 2).
$$

That single number is *simultaneously* "the cell average of $f$" and "a point sample of
the smoothed field $s = T \ast f$". The value on the shell is

$$
Q \ast \Lambda \ast T \ast f
$$

evaluated at the pixel. In detail:

| Factor | What it is | Who supplies it |
|---|---|---|
| $T$ | the **cell window** of the input box | your simulation — it is already in the data |
| $\Lambda$ | the **reconstruction kernel** | `interpolation_order` |
| $Q$ | the **output window** | nothing by default; `subsample_level` and/or `radial_width` |

The cell window function, **$T$**, remains embedded in the interpolated lightcone slices
that `cosmotile` computes. This has two major consequences:

*First*, increasing `interpolation_order` converges on $s$, not $f$. That is, improving
the interpolation does not remove the cell-size averaging of the input simulation.

```{figure} figures/cell_window.svg
:alt: Interpolation error against wavenumber for spline orders 1 to 5, before and after deconvolution

Maximum error in reproducing a single mode on the shell, for a box holding cell averages.
On the left, orders 2, 3 and 5 lie on top of one another along the dashed cell-window
line — they are not missing, they have all converged on the *cell-averaged* field and the
residual against the underlying one is the cell window. On the right the same box after
`deconvolve_cell_window`, where the order ladder separates by orders of magnitude again.
```

*Second*, any output window you request stacks *on top of* $T$ rather than replacing it.
Naively applying a radial top-hat of the slice spacing $\Delta r$ would deliver
$\mathrm{sinc}(k\Delta r/2)\,\mathrm{sinc}(k\Delta/2)$ — at $\Delta r = \Delta$, twice the
smoothing you asked for:

| $k_\parallel$ / slice Nyquist | 0.25 | 0.50 | 0.75 | 1.00 |
|---|---|---|---|---|
| power delivered / power naively requested | 0.950 | 0.811 | 0.615 | 0.405 |

So `radial_width` is the **total** window you want, not an extra one to pile on:
`cosmotile` subtracts what the box already has (`coeval_cell_width`, one cell by default)
and applies only the remainder. Its default of 1.0 is therefore a no-op, and a value
below one cell is an error. See "Radial averaging" below.

### Output resolution is the coarser of the two

The short version of all this:

1. **The output already contains the input simulation's smoothing.** You cannot get a
   lightcone of higher resolution than the box it came from — no interpolation order, no
   sub-sampling, nothing but a deconvolution will recover scales the simulation averaged
   away.
2. **The output resolution is usually not the input resolution.** In the angular
   direction it is whatever you choose — `nside`, or your own coordinates — and in the
   radial direction it is unspecified unless you say. Neither is tied to the cell size.

Put together: **the smoothing scale of a `cosmotile` output is the coarser of the input
and output resolutions.** Asking for output cells finer than the input buys you nothing
but interpolation; asking for coarser ones does real averaging, and the window you get is
the one you asked for.

### If you really do want $f$

{func}`~cosmotile.deconvolve_cell_window`, applied to the box once before the shell loop,
divides $\tilde T$ out — exactly, for a periodic field band-limited to the grid Nyquist.
Then set `coeval_cell_width=0`, and the output carries only the window you requested.

It is not free: it sharpens structure the simulation never resolved and amplifies
whatever aliased power and noise sit near Nyquist, by $1/\tilde T$ — a factor of 3.9 at
the Nyquist corner of a 3D grid, and about 1.6 in RMS on a white-noise box. Reach for it
when you need a lightcone whose window is exactly something you have specified, for
instance to compare a line-of-sight power spectrum against $P(k)$ times a known channel
response. If your box genuinely holds point samples, there is nothing to remove: just
pass `coeval_cell_width=0`.

{func}`~cosmotile.cell_window` returns $\tilde T$ on a grid's own modes, if you would
rather put it into your theory prediction than take it out of your data.

## Computing the theoretical expectation

This section derives the two theory curves in the figure above. The machinery is
{mod}`cosmotile.theory`, whose three functions are the three ingredients below, so you
can reproduce them for your own box rather than taking the numbers here on trust.

Throughout, $P(k)$ is **the gridded box's own power spectrum** — what you would measure by
FFT-ing the array you passed in. That is why no cell window appears in $W$ below: if your
box holds cell averages, $\tilde T$ is already in its measured $P(k)$. Starting instead
from a continuum $P(k)$ — a theory curve, say — you must put $\prod_i
\mathrm{sinc}^2(k_i\Delta/2)$ in yourself.

Expand a plane wave in spherical harmonics and project onto a shell of radius $r$. For a
field with three-dimensional power spectrum $P(k)$, the angular power spectrum of the
values on that shell is

$$
C_\ell = \frac{2}{\pi} \int \mathrm{d}k \; k^2 \, P(k) \, j_\ell^2(kr),
$$

which is {func}`~cosmotile.theory.continuum_angular_power`. This is exact for a
geometrically thin shell, and it is the **infinite box** (dotted) curve in the figure
above: every mode contributes, including the arbitrarily long wavelengths no finite
simulation contains.

A *periodic box* contains only the discrete modes $\mathbf{k} = 2\pi\mathbf{j}/L$, so the
prediction for a tiled box is the corresponding sum,

$$
C_\ell = \frac{4\pi}{V} \sum_{\mathbf{k}} P(k) \, |W(\mathbf{k})|^2 \, j_\ell^2(kr),
\qquad V = L^3,
$$

which is {func}`~cosmotile.theory.discrete_angular_power`. Here $W$ is the interpolation
kernel's Fourier response, {func}`~cosmotile.theory.interpolation_window`, given under
"The interpolation kernel" below. This is the **box modes** (dashed) curve. The two
expressions differ only in replacing an integral over all $k$ by a sum over the modes the
box actually has, and that difference *is* the finite-box error — everything in the next
section follows from it.

### Why not Limber?

Limber's approximation replaces $j_\ell^2(kr)$ by a delta function at
$k = (\ell + 1/2)/r$, which for a field projected along the line of sight with a
normalised radial kernel $q(r)$ gives the familiar

$$
C_\ell \approx \int \mathrm{d}r \; q^2(r) \, \frac{P\!\left((\ell + 1/2)/r\right)}{r^2}.
$$

It is worth spelling out why that form does **not** apply here, because
$P(\ell/r)/r^2$ is the first thing most people reach for when sanity-checking a
lightcone. For a top-hat kernel of width $\Delta r$ the integral evaluates to
$P((\ell + 1/2)/r) / (r^2 \Delta r)$, which diverges as $\Delta r \to 0$. A `cosmotile`
slice is a geometrically thin shell, so there is no $\Delta r$ to put there and no limit
in which the exact expression above reduces to $P(\ell/r)/r^2$. The dimensions give the
same warning: $P/r^2$ carries units of length, whereas $C_\ell$ for a dimensionless field
is dimensionless.

So compare a single slice against the exact integral, not against Limber. If you stack
many slices into a genuine projection with a normalised radial kernel, Limber applies
again in the usual way.

### A closed form for a power law

For $P(k) = A k^{-2}$ the integral can be done analytically, using
$\int_0^\infty j_\ell^2(x)\,\mathrm{d}x = \pi/[2(2\ell+1)]$:

$$
C_\ell = \frac{A}{r\,(2\ell+1)}.
$$

This is *not* the curve plotted above. The figure uses a power law truncated at
$k_{\rm cut} = \pi/4$ to keep the input power well below Nyquist, whereas this result
integrates over all $k$. Truncation removes the $1/(2x^2)$ tail of $j_\ell^2$ beyond
$x = k_{\rm cut} r$, multiplying the result by roughly

$$
1 - \frac{2\ell + 1}{\pi \, k_{\rm cut} r},
$$

which in the figure's configuration ($k_{\rm cut} r \approx 157$) is already a 16%
correction by $\ell = 40$. The closed form is still useful as an independent check on
your pipeline's normalisation in the regime $\ell \ll k_{\rm cut} r$, and
`tests/test_angular_power.py` verifies both it and the truncation correction directly.

### Working it out for your own box

All three functions work in **cell units**: the cell size is unity, so the box length is
$L = N$, the volume is $V = N^3$, wavenumbers are $k = 2\pi j / N$ and the distance to
the shell is in cells.

```python
import numpy as np
from cosmotile.theory import (
    continuum_angular_power,
    discrete_angular_power,
    interpolation_window,
)

ncell = 128  # box size, in cells
radius = 200.0  # distance to the shell, in cells
order = 1  # matches interpolation_order
ells = np.arange(2, 301)


def pk(k):  # your 3D power spectrum, in cell units
    return k**-2.0


k1 = 2 * np.pi * np.fft.fftfreq(ncell)
kvec = (k1[:, None, None], k1[None, :, None], k1[None, None, :])
kmag = np.sqrt(sum(k**2 for k in kvec))
window = interpolation_window(kvec, order=order) ** 2

nonzero = kmag > 0  # the k = 0 mode carries no C_l
kflat = kmag[nonzero].ravel()
weight = pk(kflat) * window[nonzero].ravel()

predicted = discrete_angular_power(kflat, weight, float(ncell) ** 3, radius, ells)
unbounded = continuum_angular_power(pk, radius, ells, kmax=np.pi)

deficit = predicted / unbounded  # what your box is missing, per multipole
```

`predicted` is what a shell tiled from *any* box with that $P(k)$ should have, and
`unbounded` is what an infinite box would give; their ratio is the first figure on this
page. Passing $V\,|\delta_\mathbf{k}|^2$ from an actual box as `weight`, instead of the
ensemble $P(k)$, predicts that individual realisation and removes its sample variance
from the comparison — which is how the tests in `tests/test_angular_power.py` pin the
tiling down to a few percent.

## Large scales: the box fundamental

The box has no modes below $k_{\min} = 2\pi/L$, and $j_\ell^2(kr)$ peaks at $kr \approx
\ell$, so multipoles below $\ell_{\rm box} = 2\pi r / L$ are sourced by modes that simply
do not exist.

```{figure} figures/large_scale_deficit.svg
:alt: Ratio of measured to infinite-box angular power, against multipole in units of the box fundamental

Measured $C_\ell$ divided by the infinite-box prediction, for four shell radii. Plotted
against $\ell / \ell_{\rm box}$ the curves collapse: the deficit is governed solely by the
ratio of the multipole to the box fundamental.
```

Practical thresholds from that figure:

- $\ell < 0.5\,\ell_{\rm box}$: **more than half** the power is missing. Unusable.
- $\ell \approx \ell_{\rm box}$: the discrete mode shell over-weights the fundamental,
  giving a 20–40% *excess*.
- $\ell > 1.5\,\ell_{\rm box}$: accurate to the few-percent level.

Example: a 300 cMpc box observed at $z = 8$ ($r \approx 9200$ cMpc) has
$\ell_{\rm box} \approx 190$, so nothing below $\ell \approx 300$ can be believed. If you
need lower multipoles you need a bigger box.

## Small scales: interpolation and aliasing

### The interpolation kernel

Interpolation reconstructs a continuous field from the grid values, and the
reconstruction kernel suppresses power. This is the $\Lambda$ of the window chain above, and it is
separate from — and multiplies — whatever cell window the box already carries. For the
default trilinear interpolation (`interpolation_order=1`), the kernel is the triangle
function, whose Fourier response is

$$
W(\mathbf{k}) = \prod_i \mathrm{sinc}^2(k_i/2).
$$

For a general spline order $p$ the kernel is the order-$p$ B-spline, and orders above 1
are additionally pre-filtered so that the reconstruction passes through the samples, so

$$
W(\mathbf{k}) = \prod_i \frac{\mathrm{sinc}^{p+1}(k_i/2)}{b_p(k_i)},
\qquad
b_p(k) = \sum_n \beta^p(n)\, e^{-ikn},
$$

with $\beta^p$ the B-spline itself; $b_p \equiv 1$ for $p = 0$ and $p = 1$, which
recovers the expression above. {func}`~cosmotile.theory.interpolation_window` evaluates
this for any order `cosmotile` can interpolate with, and `tests/test_theory.py` checks it
against what the interpolation actually does to a single Fourier mode.

The mean-square response of the trilinear kernel, averaged over sub-cell offsets, is
$\prod_i (2 + \cos k_i)/3$, which is what the variance of a tiled shell is suppressed by.
`tests/test_correlations.py` checks that this prediction holds to a few percent, so the
suppression is understood rather than merely tolerated: if you need the small-scale
variance preserved, raise the interpolation order or over-resolve the box — do not
rescale the output.

### Aliasing

The same reconstruction leaks each mode $\mathbf{k}$ into aliases at
$\mathbf{k} + 2\pi\mathbf{m}$. Those aliases appear at $\ell \gtrsim 2\pi r / \Delta$, and
on a shell whose input spectrum falls steeply they show up as a floor rather than as a
small correction — visible at the right of the first figure, where the measurement sits
many orders of magnitude above the truth. Treat $\ell > \pi r/\Delta$ as meaningless, and
if your input spectrum is band-limited well below Nyquist, treat everything above
$k_{\rm cut} r$ as meaningless.

Part of that floor is not the box at all but the *pixel*: structure smaller than a
HEALPix pixel, folded down by sampling at pixel centres. Averaging over the pixel removes
that part (see below) and drops the floor by roughly a factor of four; what is left is
power the interpolation genuinely places at those multipoles, which no amount of
averaging can help with.

### Choosing the interpolation order

```{figure} figures/interpolation_order.svg
:alt: Maximum interpolation error against cells per wavelength, for spline orders 0 to 5

Maximum error in reproducing a single Fourier mode on the shell, against how well that
mode is resolved by the coeval grid. Each extra spline order buys roughly an order of
magnitude on well-resolved modes. **Read against the field the box actually holds**: on
a box of cell averages the same curves flatten out at the cell window, as the figure in
"What a cell holds, and what comes out" shows.
```

Orders 0 and 1 use interpolating kernels directly. Orders 2–5 require a spline pre-filter
first, which `cosmotile` applies internally with the same periodic boundary condition as
the interpolation.

- **Order 0** (nearest neighbour) returns exact box values, so it preserves the one-point
  distribution and the variance up to sampling noise. Use it when you care about the PDF
  of the field (for example a strongly non-Gaussian ionisation field) more than about
  smoothness.
- **Order 1** (trilinear) is the default and the right choice for most work.
- **Orders 3–5** cost more and are worth it only if your field is smooth on the cell scale
  and you need sub-percent accuracy. On a field with power near Nyquist they buy little,
  because the information genuinely is not there.

One caveat that catches people: if your box holds cell averages — it probably does —
then raising the order converges on the *cell-averaged* field and the error against the
underlying one stops falling at the cell window. Orders past 2 or 3 then buy nothing you
can see; {func}`~cosmotile.deconvolve_cell_window` is what restores the ladder. If you
use both, deconvolve *first* and pre-filter the result, since the pre-filter is computed
from the values it is given.

Orders above 1 also carry a real cost, but you only need to pay it once. By default the
spline pre-filter is applied to the whole coeval box on every call, so a lightcone of 100
shells filters the box 100 times — around 1.1 s each for a $256^3$ box, which utterly
dominates the 0.02 s of interpolation itself. The filtered box depends only on the box
and the order, not on the shell radius, rotation or origin, so hoist it out of the loop
with {func}`~cosmotile.prefilter_coeval`:

```python
filtered = cosmotile.prefilter_coeval(coeval, order=3)

for radius in radii:
    (shell,) = cosmotile.make_lightcone_slice(
        coevals=filtered,
        latitude=lat,
        longitude=lon,
        distance_to_shell=radius,
        interpolation_order=3,
    )
```

This is bit-identical to the default path — it only moves the work — and takes the
$256^3$/`nside=256` example from about 1.16 s per shell to 0.02 s, i.e. a 100-shell
lightcone from two minutes to a couple of seconds. Orders 0 and 1 need no filter, so they
are unaffected either way. (Those interpolation times assume `numba` is installed, which
makes the gather about nine times faster; see [Performance](performance).)

There is no flag to set and nothing to keep in sync: `prefilter_coeval` returns a
{class}`~cosmotile.PrefilteredCoeval`, which carries the order it filtered for, and that
— which nothing else can produce — is what tells the interpolator to skip the filter. A
raw box is filtered as before, so the two paths cannot be mixed up. Nothing is cached
between calls either, so a box you mutate in place can never yield a stale shell.

The one thing you must get right is the order: a box filtered for order 3 and tiled at
order 5 holds the wrong coefficients, and raises `ValueError` rather than returning a
plausible-looking shell.

A `PrefilteredCoeval` is deliberately *not* an array: it supports neither indexing nor
arithmetic, because the pre-filter of a slice is not the slice of the pre-filter and the
pre-filter of twice a box is not twice the pre-filter. Both would be quietly wrong, so
both raise `TypeError`. To derive a new box, take the coefficients out with
`np.asarray` and run `prefilter_coeval` on the result.

### Sampling versus averaging: `subsample_level`

By default a lightcone value is a **point sample** of the reconstructed field at the
pixel centre. Such a map carries the input box's cell window, but not the HEALPix pixel
window: you must **not** divide $w_\ell$ out of it, and
power above $\ell \sim 2 N_{\rm side}$ does not vanish but aliases back down.

Passing `subsample_level=k` to {func}`~cosmotile.make_healpix_lightcone_slice` instead
averages each pixel over the $4^k$ sub-pixels of an $N_{\rm side} 2^k$ map. Those
sub-pixels are equal-area and tile their parent exactly, so the mean over them converges
on the true pixel average, and the pixel window then applies **on top of** the cell
window.

```{figure} figures/pixel_window.svg
:alt: Ratio of averaged to sampled angular power, against multipole, for three sub-sample levels

The angular power of an averaged map divided by that of the sampled map built from the
same box. By $k=2$ the ratio is the HEALPix pixel window $w_\ell^2$ to about 1%, and by
$k=3$ to a few parts in a thousand. The comparison is between two maps of the same
realisation, so the input field's sample variance cancels out of it entirely.
```

What that buys you:

- **The pixel window applies.** Compare a measured $C_\ell$ against
  $C_\ell^{\rm theory} w_\ell^2$, or divide $w_\ell^2$ out of the measurement. On a
  sampled map both are wrong.
- **Less aliasing.** The spurious floor above the spectral cut-off (above) drops by
  roughly a factor of four at $k=2$ — the part of it that was sub-pixel structure folded
  down by sampling, as distinct from the part the interpolation genuinely puts there.
- **Cost.** $4^k$ interpolations per pixel.

### Choosing $k$

The sub-pixel mean is a quadrature rule for the mean over the pixel, so its error is set
by how finely the sub-pixels sample the scale the field varies on — the cell size.
Measured against the exact pixel average, with the sub-pixel arc in cells:

| sub-pixel arc / $\Delta$ | 0.65 | 0.33 | 0.16 | 0.08 |
|---|---|---|---|---|
| error on $C_\ell$ | 6% | 1.3% | 0.3% | 0.08% |

It falls as the square of the arc, and — perhaps surprisingly — it does **not** saturate
once the sub-pixels are smaller than a cell. This is quadrature error, not a sampling
limit: the reconstructed field is smooth rather than structureless below a cell, so a
finer rule keeps paying. A sub-pixel arc of about a quarter of a cell is needed for 1%,
not one cell.

A HEALPix pixel subtends roughly $0.52/N_{\rm side}$ radians, so that works out at
$k \approx \log_2\!\left(2r / N_{\rm side}\Delta\right)$.
{func}`~cosmotile.recommended_subsample_level` does this for you:

```python
k = cosmotile.recommended_subsample_level(nside, radius, tolerance=0.01)
```

The default stays $k = 0$, so averaging never happens unless you ask.

A large $k$ is itself a diagnostic. Cost grows as $4^k$, so needing more than two or
three means the pixel spans many cells — that is, `nside` is below the
$N_{\rm side} \gtrsim r/2\Delta$ of the previous section and you are discarding
resolution the box has. Raise `nside` instead; it is cheaper and you get the small
scales back.

Averaging does not remove the need to choose `nside` sensibly. A good rule is still to
make the pixel scale comparable to the cell size projected onto the shell,

$$
N_{\rm side} \gtrsim \frac{r}{2\Delta},
$$

which follows from a HEALPix pixel subtending roughly $0.52/N_{\rm side}$ radians. Then
trust only $\ell \ll 2 N_{\rm side}$ — beyond that the map cannot represent the signal
whether or not it is aliased.

### Radial averaging

The same argument applies along the line of sight: a lightcone shell has the thickness
of the slice spacing, not zero. Pass `radial_width` — the **total** window you want, in
cells, which is the slice spacing or the width of a frequency channel — together with
`n_radial_samples`:

```python
(shell,) = cosmotile.make_healpix_lightcone_slice(
    nside=nside,
    coevals=box,
    distance_to_shell=radius,
    radial_width=slice_spacing,
    n_radial_samples=4,
)
```

`cosmotile` subtracts what the box already carries and applies only the remainder. Your
box supplies about one cell of radial smoothing (the cubic cell window is within 3% of
isotropic even at Nyquist, so along any sight-line it acts as a radial top-hat of one
cell), so the extra top-hat applied is $\sqrt{\Delta r^2 - \Delta^2}$: widths add in
quadrature to leading order, since both windows expand as $1 - k^2x^2/24$. That
reproduces the window you asked for to better than 2.5% out to its own Nyquist, against
up to 36% if the full $\Delta r$ were applied on top. The arithmetic is
{func}`~cosmotile.residual_radial_width` if you want it directly.

The consequences of that convention are worth stating plainly:

- **`radial_width=1.0` is the default and does nothing** — one cell is what you already
  have.
- **`radial_width` below `coeval_cell_width` is an error.** Averaging cannot sharpen.
- **If your box holds point samples**, or you have run
  {func}`~cosmotile.deconvolve_cell_window` on it, pass `coeval_cell_width=0` so the full
  width is applied.

The averaging is real: the Gauss–Legendre nodes interpolate the coeval box at their own
radii and those values are combined, so a window spanning several cells genuinely
averages several cells of your simulation. The weights carry the $r^2$ volume element,
always — it is what makes the result the mean over a shell of that thickness. Four nodes
integrate the window exactly for anything a cell-scale field can hold; a window spanning
many cells of a field with power near Nyquist wants roughly $n \gtrsim \pi w / 2$.

A radial mode $k$ is then suppressed by the top-hat transform $\mathrm{sinc}(k\Delta r/2)$,
with a correction of order $(\Delta r/r)^2$ from the volume weighting.

Averaging radially turns the slice into a projection with a normalised radial kernel
$q(r)$ — the $q$ of "Why not Limber?" above — so the thin-shell $C_\ell$ no longer
describes it: $j_\ell^2(kr)$ must be replaced by
$\left|\int \mathrm{d}r \, q(r) \, j_\ell(kr)\right|^2$. Pass `radial_width` to
{func}`~cosmotile.theory.discrete_angular_power` and it evaluates exactly that — give it
the top-hat actually applied, which is
`residual_radial_width(radial_width, coeval_cell_width)` when the box carries a cell
window of its own. `tests/test_angular_power.py` checks a radially averaged shell
against it.

This does **not** bring Limber back: Limber needs the radial kernel to be wide compared
with the oscillation scale of $j_\ell$, i.e. $\Delta r \gg r/\ell$, which for a one-cell
slice at $r = 80$ cells means $\ell \gg 80$. A single slice is nowhere near that however
it is averaged; stack many into a genuine projection and Limber applies in the usual way.

One thing to watch: the radial window does not only suppress. Below the spectral cut-off
it does, monotonically in the width. *Above* it, where a thin shell has almost no signal
left, the window reaches radii at which $j_\ell(kr)$ is larger than at the shell itself
and the predicted power goes *up*. Both regimes are pinned in `tests/test_theory.py`.

Both defaults (`subsample_level=0`, `n_radial_samples=1`) reproduce point sampling
exactly, so nothing changes unless you ask for it — but "point sampling" always means
point sampling of the reconstructed *cell-averaged* field, never of the underlying one.

## Replication

A periodic box tiled onto a large shell repeats. Two sight-lines whose chord is a box
lattice vector return *identical* values, not merely correlated ones — this is asserted
directly in `tests/test_angular_power.py`. A shell of radius $r$ has area $4\pi r^2$,
whereas the box offers a cross-section of only $L^2$, so the sky is covered by roughly
$4\pi (r/L)^2$ box-sized patches drawn from a single simulation — about 200 of them by
$r/L = 4$.

```{figure} figures/angular_scale.png
:alt: The same coeval box tiled onto shells at two radii

The same $64^3$ box at $r/L = 0.45$ and $r/L = 4$. At large $r/L$ the box subtends a small
angle, so all of its power moves to high $\ell$ and the largest angular scales are empty
— the low-$\ell$ deficit of the previous section, seen in map space.
```

Replication does not show up as an *error* in $C_\ell$, which is why the tests above
still pass at $r/L = 4$: the periodicity is already built into the discrete mode sum the
measurement is compared against, and the angular power spectrum averages over exactly the
pair separations at which the repetition lives. What it does corrupt is anything that
depends on the *distribution* of structures across the sky — counts of rare peaks,
cross-correlations between widely separated patches, or covariance matrices estimated
from a single lightcone. Mitigate it by applying a different `rotation` and `origin` to
each shell, which decorrelates successive radii at the cost of introducing discontinuities
along the line of sight.

## Redshift-space distortions

{func}`~cosmotile.apply_rsds` displaces the field along the line of sight. It is a
one-dimensional continuity problem: for a displacement field $u(r)$, the observed
distance is $s = r - u(r)$ and

$$
\rho_s(s) = \frac{\rho_r(r)}{1 - u'(r)}.
$$

`tests/test_rsd_physics.py` checks that directly. Four things to know.

**It works in cell averages, throughout.** Unlike the tiling functions, `apply_rsds`
treats a slice as a *cell* running from edge to edge whose value is the mean of the field
over it — both on input and on output. That is not a style choice: moving material about
is only meaningful for a quantity with extent, and it is what makes the displacement
conserve mass and a zero displacement an exact round trip.

That is consistent with a lightcone built by point-sampling shells **provided your slice
spacing is comparable to your cell size**, because the box's own cell window then already
supplies about one slice of radial smoothing (see "What a cell holds, and what comes
out"). If your slices are much coarser than your cells, a point-sampled lightcone is not
a radial cell average and `apply_rsds` will assume smoothing that is not there — build
the lightcone with `radial_width` first, choosing it with
{func}`~cosmotile.residual_radial_width`.

**Sign convention.** Positive displacement means *towards the observer*, so a parcel at
distance $d$ with displacement $u$ is observed at $d - u$. This matches the output of
{func}`~cosmotile.make_lightcone_slice_vector_field`, and the two chain directly.

**Shell crossing.** The mapping $r \mapsto s$ is only invertible while $u' < 1$. Beyond
that the flow multi-streams; the cloud-in-cell deposition still conserves mass, but the
result is no longer the continuity solution.

**`n_subcells` is a convergence knob.** It sets how finely the line-of-sight grid is
refined before the displacement is applied. The displaced grid is then *integrated* over
the radial extent of each output slice, so every parcel lands in exactly one output cell
(split in proportion where it straddles two) and mass is conserved. Raising `n_subcells`
therefore shrinks the cloud-in-cell kernel without anything falling between the slices.

```{figure} figures/n_subcells_convergence.svg
:alt: RMS error against the continuity solution, falling with n_subcells

RMS error of `apply_rsds` against $\rho_s = \rho_r / (1 - u')$ for a sinusoidal
displacement of amplitude $A$, over 256 slices, interior only. The error falls roughly
as $1/n$. Before this was fixed the same measurement was flat — 0.039, 0.048, 0.047,
0.041, 0.039 at $n = 1, 2, 4, 8, 16$ for $A = 2$ — because the final step sampled the
refined grid instead of integrating over the output cell
([issue #465](https://github.com/steven-murray/cosmotile/issues/465)).
```

The default of 4 gives about 1% RMS accuracy for a displacement of one or two cells.
Raise it if your velocity field varies on the scale of a single slice; there is no
reason to go past ~16 unless the rest of your pipeline is that accurate.

The refinement is *conservative*: each fine cell takes the value of the output slice it
lies in, so with zero displacement the whole refine–displace–average round trip is
exactly the identity, at any `n_subcells`. The residual error is therefore the
cloud-in-cell kernel alone, which is a smoothing of roughly one sub-cell.

**Padding at the ends.** To catch material displaced in from beyond the grid, the field
and the displacement are extrapolated past the first and last slices. Mass is conserved
exactly for parcels that stay on the grid, but the end slices can gain material from
that extrapolated region — so treat the outermost few slices of a lightcone as you would
any other boundary.

## Reproducing these figures

```bash
python docs/make_accuracy_figures.py
```

Requires the `dev` extra (which pulls in `powerbox` and `healpy`) plus `matplotlib`. The
predictions themselves come from {mod}`cosmotile.theory` and need neither.
