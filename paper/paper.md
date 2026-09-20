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
  # TODO: decide the co-author list. Candidates, and what they'd be credited for:
  #   - Nithyanandan Thyagarajan (orcid 0000-0003-1602-7868) -- original algorithm
  #     (AstroUtils `cosmotile` module).
  #   - Piyanat Kittiwisit (orcid 0000-0003-2410-1424) -- first standalone packaging.
  # JOSS wants authors who made a substantial *software* contribution to the
  # submitted version. If they decline authorship, credit them in Acknowledgements
  # instead and cite Thyagarajan et al. (2017) for the algorithm.
affiliations:
  - name: Scuola Normale Superiore, Pisa, Italy
    index: 1
date: TODO
bibliography: paper.bib
---

# Summary

<!--
POINTS TO MAKE (roughly 150-250 words; JOSS wants this accessible to a
non-specialist scientific reader):

- The problem in one sentence: cosmological simulations are *coeval* -- a cube of
  the universe at one instant -- but a telescope sees a *lightcone*, where distance
  from the observer is time.
- The standard fix is to tile the periodic box: for each direction on the sky and
  each radial shell, find where that point falls in the (replicated) box and
  interpolate. Conceptually simple, fiddly to get right, and easy to get subtly
  wrong in ways that only show up in the power spectrum.
- `cosmotile` does exactly that operation, and only that operation: box + angular
  coordinates + shell radius -> values on the sky. It is deliberately not a
  simulation code and not a lightcone *pipeline*; it is the geometric kernel those
  pipelines need.
- What it adds over a hand-rolled `map_coordinates` call: exact periodic wrapping,
  arbitrary rotation/origin per shell, spline orders 0-5 with the pre-filter hoisted
  out of the shell loop, angular and radial pixel averaging (not just point
  sampling), mass-conserving redshift-space distortions, closed-form theory
  predictions to validate against, and a differentiable/GPU backend.
- One-line statement of the headline capability: the whole thing is linear in the
  box values, so it is differentiable, and a GPU does it ~9x faster than the
  optimised CPU path.
-->

# Statement of need

<!--
POINTS TO MAKE:

1. WHO NEEDS THIS. 21 cm cosmology and line-intensity mapping are the obvious
   users -- 21cmFAST, SimFast21, zreion-style semi-numerical boxes are all coeval
   and all need to be turned into an observable lightcone before they can be
   compared to data from HERA, MWA, LOFAR, SKA. But the operation is generic:
   anything that needs a curved-sky observable out of a periodic box (tSZ/kSZ maps,
   line-intensity mapping of CO/CII, weak-lensing shells) has the same problem.

2. WHAT EXISTS ALREADY, AND WHY IT ISN'T ENOUGH. Be concrete and fair here --
   JOSS reviewers care about this section more than any other.
   - Most codes bury the tiling inside their own pipeline: `21cmFAST` produces
     flat-sky lightcones along one axis; `Lightcone`/`LightconeMaker`-style helpers
     in various survey pipelines do a similar thing. Their tiling is not reusable,
     not separately tested, and usually flat-sky.
   - Curved-sky tiling has been done repeatedly, one-off, per paper. The direct
     ancestor here is Thyagarajan's `AstroUtils` implementation, later modularised
     by Kittiwisit; this package is a rewrite of that lineage.
   - General-purpose interpolators (`scipy.ndimage.map_coordinates`) do the
     arithmetic but none of the domain work: no periodic tiling of the sphere, no
     HEALPix pixel handling, no window bookkeeping, no RSDs, and no way to know
     whether the answer you got is trustworthy at the scale you care about.
   - Adjacent-but-different: `healpy`/`astropy-healpix` (pixelisation only),
     `powerbox` (makes the boxes, doesn't project them), `CoLoRe`/`lognormal`
     lightcone generators (generate directly on the lightcone rather than tiling
     an existing box -- a genuine alternative worth naming, with the trade-off
     stated: they can't take *your* simulation as input).
   - TODO: check current state of these before submission. Don't assert anything
     about another package you haven't verified against its current docs.

3. THE ARGUMENT FOR A SEPARATE PACKAGE. The tiling is the easy part; what's hard,
   and what gets skipped when it lives inside a pipeline, is knowing when the
   answer is wrong. A finite periodic box cannot produce every angular scale.
   `cosmotile` documents the window of validity, ships closed-form predictions
   (`cosmotile.theory`) so a user can check their own setup against it, and tests
   the geometry to machine precision.

4. THE DIFFERENTIABILITY ARGUMENT. Simulation-based inference and gradient-based
   fitting both want the forward model differentiable end to end. Tiling is linear
   in the box, so its gradients are exact, cheap and well conditioned -- there is
   no reason for this step to be the one that breaks the chain. As far as we know
   this is the first curved-sky tiling implementation to offer that.
-->

# Method

<!--
POINTS TO MAKE (keep this short -- JOSS is not a methods paper; point at the docs
for the derivations. Two or three paragraphs at most, plus Figure 1.)

- The geometry in three lines: for each sky direction $\hat{n}$ and shell radius
  $r$, the sample point is $\mathbf{x} = \mathbf{x}_0 + r\,R\,\hat{n}$, reduced
  modulo the box, then evaluated by B-spline interpolation of order $p$. Rotation
  $R$ and origin $\mathbf{x}_0$ are free per shell.
- Say plainly what the output *is*: by default a point sample of the reconstructed
  field at the pixel centre -- not a pixel average. Optional angular subsampling
  (`subsample_level`) and radial averaging (`radial_width`, `n_radial_samples`)
  make it a genuine cell average, which changes which windows apply to the measured
  $C_\ell$. This is the point most users get wrong and it is worth a sentence.
- The window of validity, stated as the one equation worth putting in the paper:
  $2\pi r / L \lesssim \ell \lesssim \pi r / \Delta$. Below the lower limit power
  is *missing* (the box has no modes there); above the upper it is *spurious*
  (aliasing floor).
- FIGURE 1: reuse `docs/figures/validity_window.svg` -- measured $C_\ell$ against
  the discrete-mode prediction, with the valid window shaded. This single figure
  makes the whole argument of the previous paragraph and is the strongest thing
  in the repo. (Convert to PDF/PNG for the JOSS build.)
- RSDs in two sentences: 1D continuity, $\rho_s(s) = \rho_r(r)/(1-u')$, applied by
  mass-conserving deposition onto the output slices; exact round trip at zero
  displacement; valid until shell crossing.
-->

# Performance and differentiability

<!--
POINTS TO MAKE (one or two paragraphs; the full picture is in docs/performance.md):

- The kernel is a gather: $(p+1)^3$ cells per output sample. A modest lightcone
  (51 shells, nside=128, subsample, order 3) is ~160 million of them.
- Three paths, one set of numbers, with the caveat that they're from one
  power-limited laptop GPU: scipy fallback 4.5 Mpix/s, numba 47 Mpix/s, JAX GPU
  float32 409 Mpix/s. Worth a small table.
- Be honest about where the GPU does *not* help -- order 5, CPU-only, a single
  shell, a box that doesn't fit. Reviewers respect this and it is already written
  in the docs.
- Differentiability: `cosmotile.jax` is a set of pure functions (no iterators, no
  units, no value-dependent validation) so the whole thing composes under
  `jit`/`grad`/`vmap`. `lightcone_scan` keeps the peak memory at one shell so a
  thousand-shell lightcone fits on a GPU.
- Possible FIGURE 2 if one is wanted: `docs/figures/gpu_speedup.svg`. Optional --
  JOSS papers are short and the validity figure matters more.
-->

# Validation

<!--
POINTS TO MAKE (can be folded into Method if the paper is running long):

- Geometry is tested to machine precision (`tests/test_geometry.py`): known
  analytic sample positions, periodic wrapping, rotation composition.
- Statistics are tested against `cosmotile.theory.discrete_angular_power` with the
  interpolation window folded in -- agreement to a few percent inside the window
  of validity (`tests/test_angular_power.py`).
- The JAX and NumPy backends are checked against each other; the FFT pre-filter
  agrees with scipy's recursive filter to 7e-15 relative in double precision.
- RSDs are checked against the closed-form continuity solution, with the error
  falling as 1/`n_subcells`.
-->

# Acknowledgements

<!--
POINTS TO MAKE:
- The algorithm originates with Nithyanandan Thyagarajan's implementation in
  AstroUtils, modularised by Piyanat Kittiwisit -- credit this explicitly whether
  or not they end up as authors.
- TODO: funding. Scuola Normale Superiore / whatever grant covers this.
- TODO: anyone who reported issues or drove features (check the issue tracker
  before submitting).
-->

# References
