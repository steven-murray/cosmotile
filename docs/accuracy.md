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
| $\ell \gtrsim 2 N_{\rm side}$ | Sampling at HEALPix pixel centres | Small-scale power aliases into large scales |
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

## Computing the theoretical expectation

This section derives the two theory curves in the figure above. The machinery is
{mod}`cosmotile.theory`, whose three functions are the three ingredients below, so you
can reproduce them for your own box rather than taking the numbers here on trust.

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

Tiling reconstructs a continuous field from grid samples, and the reconstruction kernel
suppresses power. For the default trilinear interpolation (`interpolation_order=1`), the
kernel is the triangle function, whose Fourier response is

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

### Choosing the interpolation order

```{figure} figures/interpolation_order.svg
:alt: Maximum interpolation error against cells per wavelength, for spline orders 0 to 5

Maximum error in reproducing a single Fourier mode on the shell, against how well that
mode is resolved by the coeval grid. Each extra spline order buys roughly an order of
magnitude on well-resolved modes.
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

Orders above 1 also carry a real cost per shell. The spline pre-filter is applied to the
whole coeval box on every call, so a lightcone of 100 shells filters the box 100 times —
around 0.7 s each for a $256^3$ box, which dominates the interpolation itself. Budget for
that before reaching past order 1; reusing a precomputed filter is tracked as
[issue #466](https://github.com/steven-murray/cosmotile/issues/466).

### Choosing `nside`

The tiling samples the field *at pixel centres*; it does not average over pixels
([issue #465](https://github.com/steven-murray/cosmotile/issues/465)). So you
should **not** divide out the HEALPix pixel window — instead, choose `nside` high enough
that you are not aliasing. A good rule is to make the pixel scale comparable to the cell
size projected onto the shell,

$$
N_{\rm side} \gtrsim \frac{r}{2\Delta},
$$

which follows from a HEALPix pixel subtending roughly $0.52/N_{\rm side}$ radians. Then
trust only $\ell \ll 2 N_{\rm side}$.

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

`tests/test_rsd_physics.py` checks that directly. Three things to know.

**Sign convention.** Positive displacement means *towards the observer*, so a parcel at
distance $d$ with displacement $u$ is observed at $d - u$. This matches the output of
{func}`~cosmotile.make_lightcone_slice_vector_field`, and the two chain directly.

**Shell crossing.** The mapping $r \mapsto s$ is only invertible while $u' < 1$. Beyond
that the flow multi-streams; the cloud-in-cell deposition still conserves mass, but the
result is no longer the continuity solution.

**`n_subcells` is not a convergence knob.** It refines the grid used for the
displacement, but the final step *samples* that fine grid at the output slices rather
than averaging over them. Raising it therefore sharpens the deposited field without
improving the answer, and in extreme cases makes it worse: material concentrated into a
single sub-cell that no output slice sits on disappears entirely. The default of 4 is
reasonable; there is nothing to gain from 32. This is tracked as
[issue #465](https://github.com/steven-murray/cosmotile/issues/465), along with the
related point that the angular direction samples pixel centres rather than averaging over
the pixel.

Residual error from the cloud-in-cell kernel is a smoothing of roughly one output slice,
and it scales with the amplitude of the density response — so oversample the line of
sight if you need the small-scale redshift-space structure.

## Reproducing these figures

```bash
python docs/make_accuracy_figures.py
```

Requires the `dev` extra (which pulls in `powerbox` and `healpy`) plus `matplotlib`. The
predictions themselves come from {mod}`cosmotile.theory` and need neither.
