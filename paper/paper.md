---
title: 'cosmotile: Differentiable lightcones from periodic cosmological simulations'
tags:
  - Python
  - astronomy
  - cosmology
  - 21 cm
  - reionization
  - simulations
  - lightcones
authors:
  - name: Steven G. Murray
    orcid: 0000-0003-3059-3823
    affiliation: 1
  - name: Paul La Plante
    orcid: 0000-0002-4693-0102
    affiliation: 2
  - name: Nithyanandan Thyagarajan
    orcid: 0000-0003-1602-7868
    affiliation: 3
  - name: Piyanat Kittiwisit
    orcid: 0000-0003-2410-1424
    affiliation: 4, 5
affiliations:
  - name: Stellenbosch University, Stellenbosch, South Africa
    index: 1
  - name: Google
    index: 2
  - name: CSIRO, Space and Astronomy, Perth, Australia
    index: 3
  - name: University of the Western Cape, Cape Town, South Africa
    index: 4
  - name: South African Radio Astronomy Observatory, Cape Town, South Africa
    index: 5
date: TODO
bibliography: paper.bib
---

# Summary

Astrophysical and cosmological inference from large-scale observations such as
galaxy surveys and line-intensity maps, including high-redshift 21cm intensity maps,
requires forward-models of the observations.
At the core of such models are typically cosmological simulators that produce
snapshot (coeval, comoving) periodic cubes of a given observable quantity.
However, observations form a *lightcone* in which each successive line-of-sight slice
traces both distance and cosmic time. `cosmotile` implements a simple algorithm to
convert a series of evolving coeval periodic snapshots, in the form of regular grids,
into a past lightcone of arbitrary angular shape.
It is extremely general, and can act as a post-processor for any simulation code
that produces grid-based outputs (and is already incorporated into recent versions of
`21cmFAST` [@Mesinger2011; @Murray2020; @Davies2025]). Furthermore, it includes an
optional auto-differentiable JAX backend so that the conversion to observational
coordinates can sit inside gradient-dependent inference pipelines, and benefit from easy
GPU acceleration.

A summary list of features of `cosmotile` is as follows:

* Format-agnostic input and output (works with `numpy` arrays [@Harris2020] rather
  than relying on specific file formats).
* Highly efficient interpolation from the 3D boxes to the 2D shells.
* Support for efficient lightconing over multiple fields with the same geometry.
* Both CPU and GPU backends (via `scipy`, `numba` and `jax`).
* Auto-differentiation through `jax`.
* Support for creating lightcones of vector quantities, including projection of the
  vector onto the radial coordinate (e.g. creating a line-of-sight velocity field
  from an input 3D velocity field)
* Functions for perturbing lightcone shells due to redshift-space distortions.
* Rigorous tests and benchmarks, including a `theory` module that helps in predicting
  the statistics of the shell from those of the 3D inputs.


# Statement of need

`cosmotile` aims to solve a single important step in the production of mock observations
for pipeline validation and inference in cosmology: the conversion of coeval grid-based
simulations at a series of cosmic times (redshifts) into a continuous observable
lightcone. It was originally developed in the context of 21cm cosmology, in which
simulators (e.g. `21cmFAST`; `SimFast21` @Santos2010; `zreion`, @Battaglia2013)
typically produce 3D fields of various quantities like the 21cm brightness temperature
and neutral hydrogen fraction. Nevertheless, the interface of `cosmotile` is rather
generic, and can work just as well to produce any curved-sky observable out of a
periodic box (e.g. tSZ/kSZ maps, arbitrary line-intensity maps or weak-lensing shells).

Related packages tend to fall into one of a few categories.
Firstly, simulation codes often include a bespoke lightcone-maker.
These often come with limitations; for example the built-in lightcone routine of
`21cmFAST` (up to v3) relied on a flat-sky approximation.
Beyond this, the code to perform the lightcone conversion in these cases is typically
entangled with the simulation machinery itself and not easily reusable.
Secondly, the essential algorithm behind `cosmotile` has been implemented previously,
but on an ad-hoc basis in support of particular research goals (e.g. [@AstroUtils3;
@Kittiwisit2018]) and with insufficient attention to software best practices like
modularity, testing, documentation, packaging and efficiency.
Thirdly (and somewhat adjacently), there exist packages that directly generate
statistical simulations on a lightcone (e.g. CoLoRe [@Ramirez-Perez2022]).
These do not support conversion of outputs of existing (coeval) simulators.
Finally, there are lightcone generators that support *discrete* particle simulations
(e.g. `LightGen` [@Shreeram2025]), rather than the grid-based fields that `cosmotile`
tackles.

With several semi-numerical simulators available in the space of 21cm cosmology and
line intensity mapping, all of which have similar output data structures (a series of regular 3D grids over redshift), the utility of a standalone package dedicated to
efficiently converting these simulations into observational coordinates is clear.
Adding to this value, `cosmotile` ships with a large test suite that ensures that
angular power spectra on the lightcone are reproduced given known 3D Gaussian random
fields.
Beyond this, `cosmotile` comes with two interchangeable backends: a `numpy/numba` backend that operates on the CPU, and a JAX backend that can be used either on the CPU
or GPU and is auto-differentiable.
With this, differentiable simulators (e.g. simulators written in JAX, or 3D field
emulators) can be combined with `cosmotile` to produce fully differentiable observational forward models.
To our knowledge, this is the first package to offer this functionality.

# Method

The basic methodology of `cosmotile` is to utilise the periodic nature of the input
coeval simulations, tiling them in 3D such that they cover a requested 2D spherical
shell, and then interpolating the 3D coordinates onto the requested angular coordinates
on the shell (see Fig. A1 from [@Kittiwisit2018]).
That is, for each requested sky direction $\hat{n}$ and shell radius $r$, we
compute a coordinate $\mathbf{x} = \mathbf{x}_0 + r \mathbf{R}\hat{n}\ ({\rm mod} L)$,
where $\mathbf{x}_0$ and $\mathbf{R}$ are an arbitrary origin and rotation matrix chosen
once for all coordinates, and $L$ is the length of the periodic input boxes.
The field at $\mathbf{x}$ is computed via B-spline interpolation from the input.

By default, the outputs are *samples* of the field at the spherical coordinates
$(\hat{n}, r)$ requested. These coordinates can be arbitrary and need not be regular.
Such a sample inherits whatever smoothing the input already carries: a simulation
voxel almost always holds the mean of the field over it, and that cell window
propagates to the output. It is not, however, an average over the output pixel, and
the distinction matters in interpreting the result, since the pixel window should not
be divided out of the angular power spectrum of a sampled map.
For convenience, `cosmotile` also provides a function
that generates the samples on a HEALPix grid [@Gorski2005].
In this case, the regularity allows for sub-sampling the HEALPix pixels, so that the
result approaches a true pixelisation of the sky at the requested `NSIDE`, whose pixel
window then does apply.

In either case (unstructured coordinates or HEALPix) a radial width can be specified for
the shell such that if the input simulation has a finer resolution than the shell width,
the shell value will be the integrated field within the shell (accounting for the $r^2$
Jacobian of the integration).

Once a lightcone has been built, `cosmotile` can additionally apply redshift-space
distortions to it. This is a one-dimensional continuity problem: for a line-of-sight
displacement $u(r)$, a parcel at comoving distance $r$ is observed at $s = r - u(r)$,
so that $\rho_s(s) = \rho_r(r)/[1 - u'(r)]$. Rather than evaluate this directly,
`cosmotile` refines the line-of-sight grid, displaces it, and integrates the displaced
grid over the radial extent of each output slice, so that mass is conserved by
construction and a zero displacement is an exact round trip. The mapping holds up to
shell crossing ($u' \geq 1$), beyond which the flow multi-streams and the continuity
solution no longer exists.

Since the input simulation is tiled across the sky, the output may contain repeated
copies of the structures in the simulation. Conversely, while the output may be of any
resolution, the input box defines the smallest valid scales. In general, the output
is valid for
$$
  2\pi r/L \lesssim \ell \lesssim \pi r/\Delta,
$$
where $\Delta = L/N$ is the linear size of each voxel in the input simulation.

Unit tests within `cosmotile` verify that a periodic 3D Gaussian random field
(produced with `powerbox` [@Murray2018]) sampled onto a HEALPix shell produces the
analytically expected angular power spectrum within this range of validity (see Figure 1).
Separately, the tiling geometry is checked against analytic sample positions to machine
precision, the redshift-space distortions against the closed-form continuity solution
above (with the error falling as the inverse of the line-of-sight refinement factor),
and the NumPy and JAX backends against each other. The JAX backend computes the B-spline
pre-filter as an FFT deconvolution rather than by `scipy`'s recursive filter; under the
periodic boundary conditions `cosmotile` uses these are the same operator, and they
agree to a relative $7\times 10^{-15}$ at double precision.

![Comparison of a 3D periodic Gaussian Random Field with band-limited power-law power
spectrum (spectral index -2) sampled onto a HEALPix shell against the theoretical
prediction of the angular power spectrum in the thin-shell
limit. Within the window of validity, the `cosmotile` conversion matches the prediction
very well. At larger scales the input box does not contain the required
modes, so power is missing; at smaller scales the true signal has fallen away and the
measurement flattens onto an aliasing floor.](../docs/figures/validity_window.pdf)


# Performance

The key computational operation in `cosmotile` is the interpolation of the 3D box to
the output coordinates $\mathbf{x}$.
This is a gather operation: a weighted sum of $(p+1)^3$ cells per output sample, where
$p$ is the B-spline order. A modest lightcone of 50 HEALPix shells with `NSIDE=128`
requires about 10 million such samples (without sub-sampling), which at third order is
some 600 million multiply-adds.
In `cosmotile` there are three separate backends to perform this gather operation: a
base `scipy` [@Virtanen2020] implementation, an optimized `numba` [@Lam2015] path which
is used by default on the CPU, and a JAX [@Jax2018] implementation that may be
used on the CPU or GPU.

Figure 2 shows the measured throughput (in Mpix/s, where the pixel count is dependent
only on the *output* size), as timed on a mid-grade laptop with power-limited GPU
(NVIDIA A2000). While the GPU provides substantial acceleration, it is not always the
right choice: its benefit degrades for higher-order interpolation and small input/output
sizes. Furthermore, due to the need to compile the JAX code, it is unlikely to be
beneficial for single-shell computations.

![Performance of `cosmotile` over various axes, as measured on a mid-grade laptop
with power-limited NVIDIA A2000 GPU. Throughput is measured in output coordinates per
second (Mpix/s). Throughput decreases with increasing interpolation order, and is
higher on GPU than the default `numba` CPU algorithm by a factor of about 10. Throughput
remains essentially constant with increasing input and
output size.](../docs/figures/throughput_scaling.pdf)


# Acknowledgements

SGM has received funding from the European Union’s Horizon 2020 research and innovation
programme under the Marie Skłodowska-Curie grant agreement No 101067043.

# References
