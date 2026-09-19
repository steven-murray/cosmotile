"""Regenerate the figures used by ``docs/accuracy.md``.

Run from the repository root with the test extras installed::

    python docs/make_accuracy_figures.py

The figures are committed, so that building the documentation needs neither
``powerbox``, ``healpy`` nor ``matplotlib``. Re-run this only when the underlying
behaviour changes.
"""

from __future__ import annotations

import sys
from itertools import pairwise
from pathlib import Path

import healpy as hp
import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

import cosmotile as cmt
from cosmotile.theory import (
    continuum_angular_power,
    discrete_angular_power,
    interpolation_window,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from conftest import band_limited_powerlaw, gaussian_box, mode_grid

OUT = Path(__file__).parent / "figures"
mpl.rcParams.update({"figure.dpi": 110, "font.size": 9, "savefig.bbox": "tight"})


def figure_validity_window() -> None:
    """C_l of a tiled shell against theory, marking where each limit bites."""
    ncell, kcut, nside, lmax = 128, np.pi / 4, 128, 260
    radius = 200.0
    pk = band_limited_powerlaw(-2.0, kcut)
    ells = np.arange(lmax + 1)

    kmag, kvec = mode_grid(ncell)
    window = interpolation_window(kvec) ** 2
    nonzero = kmag > 0

    measured = np.zeros(lmax + 1)
    nseed = 4
    for seed in range(nseed):
        box = gaussian_box(ncell, pk, seed)
        shell = next(
            cmt.make_healpix_lightcone_slice(nside=nside, coevals=box, distance_to_shell=radius)
        )
        measured += hp.anafast(shell, lmax=lmax) / nseed

    discrete = discrete_angular_power(
        kmag[nonzero].ravel(),
        (pk(kmag) * window)[nonzero].ravel(),
        float(ncell) ** 3,
        radius,
        ells,
    )
    continuum = continuum_angular_power(pk, radius, ells, kcut)

    ell_box = 2 * np.pi * radius / ncell
    ell_cut = kcut * radius
    ell_nyq = np.pi * radius

    fig, (top, bot) = plt.subplots(
        2, 1, figsize=(6.4, 5.4), sharex=True, height_ratios=[2.2, 1], layout="constrained"
    )

    good = ells >= 2
    top.loglog(ells[good], measured[good], lw=1.2, label="tiled shell (mean of 4 boxes)")
    top.loglog(ells[good], discrete[good], lw=1.2, ls="--", label=r"box modes, $\sum_k$")
    top.loglog(ells[good], continuum[good], lw=1.2, ls=":", label=r"infinite box, $\int dk$")
    top.set_ylabel(r"$C_\ell$")
    top.set_ylim(1e-11, None)
    top.legend(frameon=False, fontsize=8)

    def logbin(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        edges = np.unique(np.round(np.geomspace(2, lmax, 26)).astype(int))
        centres = np.sqrt(edges[:-1] * edges[1:])
        weights = 2 * ells + 1.0
        binned = [
            np.sum(weights[lo:hi] * values[lo:hi]) / np.sum(weights[lo:hi])
            for lo, hi in pairwise(edges)
        ]
        return centres, np.array(binned)

    for reference, label, style in (
        (discrete, r"vs box modes, $\sum_k$", {"ls": "--", "color": "C1"}),
        (continuum, r"vs infinite box, $\int dk$", {"ls": ":", "color": "C2"}),
    ):
        ratio = np.divide(
            measured, reference, out=np.full_like(measured, np.nan), where=reference > 0
        )
        centres, binned = logbin(ratio)
        bot.semilogx(centres, binned, lw=1.3, label=label, **style)

    bot.axhline(1.0, color="k", lw=0.6)
    bot.set_ylim(0, 1.6)
    bot.set_ylabel("measured / theory")
    bot.set_xlabel(r"multipole $\ell$")
    bot.legend(frameon=False, fontsize=8, loc="lower right")

    for axis in (top, bot):
        axis.axvspan(1, ell_box, color="0.88", zorder=0)
        axis.axvspan(ell_cut, lmax, color="0.88", zorder=0)
        axis.axvline(ell_box, color="C3", lw=0.8)
        axis.axvline(ell_cut, color="C3", lw=0.8)
        axis.set_xlim(2, lmax)
    top.text(ell_box * 1.1, 3e-4, r"$\ell=2\pi r/L$", color="C3", fontsize=8)
    top.text(ell_cut * 0.36, 3e-4, r"$\ell=k_{\rm cut}r$", color="C3", fontsize=8)
    top.set_title(
        rf"$N={ncell}$, $r/L={radius / ncell:.2f}$, "
        rf"Nyquist at $\ell={ell_nyq:.0f}$",
        fontsize=9,
    )

    fig.savefig(OUT / "validity_window.svg")
    plt.close(fig)


def figure_interpolation_order() -> None:
    """Plane-wave interpolation error against wavenumber, for each spline order."""
    ncell = 64
    grid = np.meshgrid(*(np.arange(ncell, dtype=float),) * 3, indexing="ij")

    rng = np.random.default_rng(0)
    vec = rng.normal(size=(4000, 3))
    vec /= np.linalg.norm(vec, axis=1)[:, None]
    lat = np.arcsin(vec[:, 2])
    lon = np.mod(np.arctan2(vec[:, 1], vec[:, 0]), 2 * np.pi)
    radius = 37.3

    harmonics = [1, 2, 3, 4, 6, 8, 11, 14]
    kmags, errors = [], {order: [] for order in range(6)}
    for harmonic in harmonics:
        kvec = 2 * np.pi * np.array([harmonic, 0.0, 0.0]) / ncell
        box = np.cos(kvec[0] * grid[0] + 0.4)
        truth = np.cos((radius * vec) @ kvec + 0.4)
        kmags.append(np.linalg.norm(kvec))
        for order in range(6):
            shell = next(
                cmt.make_lightcone_slice(
                    coevals=box,
                    latitude=lat,
                    longitude=lon,
                    distance_to_shell=radius,
                    interpolation_order=order,
                )
            )
            errors[order].append(np.abs(shell - truth).max())

    fig, axis = plt.subplots(figsize=(5.6, 4.0), layout="constrained")
    cells_per_wavelength = 2 * np.pi / np.array(kmags)
    for order in range(6):
        axis.loglog(cells_per_wavelength, errors[order], "o-", ms=3, lw=1.1, label=f"order {order}")
    axis.invert_xaxis()
    axis.set_xlabel("cells per wavelength")
    axis.set_ylabel("max absolute error on the shell")
    axis.axhline(1e-2, color="k", lw=0.6, ls=":")
    axis.text(50, 1.4e-2, "1% error", fontsize=8)
    axis.set_xticks([64, 32, 16, 8, 4])
    axis.set_xticklabels(["64", "32", "16", "8", "4"])
    axis.minorticks_off()
    axis.legend(frameon=False, fontsize=8, ncols=2, loc="lower left")
    axis.set_title("Interpolating a single Fourier mode", fontsize=9)

    fig.savefig(OUT / "interpolation_order.svg")
    plt.close(fig)


def figure_deficit_versus_radius() -> None:
    """Where the large-scale power deficit sets in, as a function of ``r / L``."""
    ncell, kcut, nside, lmax = 64, np.pi / 4, 128, 220
    pk = band_limited_powerlaw(-2.0, kcut)
    ells = np.arange(lmax + 1)
    weights = 2 * ells + 1.0
    edges = np.unique(np.round(np.geomspace(2, lmax, 22)).astype(int))
    centres = np.sqrt(edges[:-1] * edges[1:])

    fig, axis = plt.subplots(figsize=(5.8, 4.0), layout="constrained")
    for radius_over_l in (0.5, 1.0, 2.0, 4.0):
        radius = radius_over_l * ncell
        measured = np.zeros(lmax + 1)
        nseed = 4
        for seed in range(nseed):
            shell = next(
                cmt.make_healpix_lightcone_slice(
                    nside=nside,
                    coevals=gaussian_box(ncell, pk, seed),
                    distance_to_shell=radius,
                )
            )
            measured += hp.anafast(shell, lmax=lmax) / nseed

        continuum = continuum_angular_power(pk, radius, ells, kcut)
        # Only compare where the input spectrum actually has power.
        usable = ells < 0.7 * kcut * radius
        ratio = np.divide(measured, continuum, out=np.full_like(measured, np.nan), where=usable)
        binned = np.array(
            [
                np.sum(weights[lo:hi] * ratio[lo:hi]) / np.sum(weights[lo:hi])
                if usable[hi - 1]
                else np.nan
                for lo, hi in pairwise(edges)
            ]
        )
        axis.semilogx(
            centres / (2 * np.pi * radius / ncell),
            binned,
            "o-",
            ms=2.5,
            lw=1.1,
            label=f"$r/L = {radius_over_l:g}$",
        )

    axis.axhline(1.0, color="k", lw=0.6)
    axis.axvline(1.0, color="C3", lw=0.8)
    axis.text(1.1, 0.12, "box fundamental", color="C3", fontsize=8, rotation=90)
    axis.set_xlabel(r"$\ell \,/\, (2\pi r / L)$")
    axis.set_ylabel(r"measured $C_\ell$ / infinite-box $C_\ell$")
    axis.set_ylim(0, 1.5)
    axis.set_xlim(0.08, 6)
    axis.legend(frameon=False, fontsize=8, loc="lower right")
    axis.set_title("Large-scale power lost to the finite box", fontsize=9)

    fig.savefig(OUT / "large_scale_deficit.svg")
    plt.close(fig)


def figure_angular_scale() -> None:
    """Plot the same coeval box tiled onto shells at two radii."""
    ncell, nside = 64, 128
    pk = band_limited_powerlaw(-2.5, np.pi / 6)
    box = gaussian_box(ncell, pk, 7)

    fig = plt.figure(figsize=(7.4, 3.2))
    for index, radius in enumerate((0.45 * ncell, 4.0 * ncell)):
        shell = next(
            cmt.make_healpix_lightcone_slice(nside=nside, coevals=box, distance_to_shell=radius)
        )
        limit = 2.5 * shell.std()
        hp.mollview(
            shell,
            fig=fig.number,
            sub=(1, 2, index + 1),
            title=f"$r/L = {radius / ncell:.2f}$",
            cbar=False,
            min=-limit,
            max=limit,
            cmap="RdBu_r",
        )
    fig.savefig(OUT / "angular_scale.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def figure_pixel_window() -> None:
    """Angular averaging makes the HEALPix pixel window apply, instead of aliasing."""
    ncell, kcut, nside, lmax = 64, np.pi / 4, 32, 64
    radius = 80.0
    pk = band_limited_powerlaw(-2.0, kcut)
    ells = np.arange(lmax + 1)
    pixwin = hp.pixwin(nside)[: lmax + 1] ** 2

    nseed = 4
    spectra = {level: np.zeros(lmax + 1) for level in (0, 1, 2, 3)}
    for seed in range(nseed):
        box = gaussian_box(ncell, pk, seed)
        for level, total in spectra.items():
            shell = next(
                cmt.make_healpix_lightcone_slice(
                    nside=nside,
                    subsample_level=level,
                    coevals=box,
                    distance_to_shell=radius,
                )
            )
            total += hp.anafast(shell, lmax=lmax) / nseed

    fig, axis = plt.subplots(figsize=(5.8, 4.0), layout="constrained")
    for level in (1, 2, 3):
        axis.plot(
            ells[2:],
            (spectra[level] / spectra[0])[2:],
            lw=1.2,
            label=rf"$k={level}$ ($4^{level}$ sub-samples)",
        )
    axis.plot(ells[2:], pixwin[2:], "k--", lw=1.2, label=r"HEALPix $w_\ell^2$")

    # Stop at l = 2 Nside: beyond it ``anafast`` cannot measure the map anyway.
    axis.set_xlabel(r"multipole $\ell$")
    axis.set_ylabel(r"averaged $C_\ell$ / sampled $C_\ell$")
    axis.set_xlim(2, lmax)
    axis.set_ylim(0.6, 1.05)
    axis.legend(frameon=False, fontsize=8, loc="lower left")
    axis.set_title(rf"Averaging over the pixel, $N_{{\rm side}}={nside}$", fontsize=9)

    fig.savefig(OUT / "pixel_window.svg")
    plt.close(fig)


def figure_n_subcells_convergence() -> None:
    """RMS error of ``apply_rsds`` against the 1D continuity solution."""
    from astropy import units as un
    from scipy.interpolate import interp1d

    nslice, wavelength = 256, 64.0
    distance = (1000 + np.arange(nslice, dtype=float)) * un.pixel
    radial = np.arange(nslice, dtype=float)
    interior = slice(40, nslice - 40)
    subcells = np.array([1, 2, 4, 8, 16, 32, 64])

    fig, axis = plt.subplots(figsize=(5.6, 4.0), layout="constrained")
    for amplitude in (1.0, 2.0):
        displacement = amplitude * np.sin(2 * np.pi * radial / wavelength)
        gradient = amplitude * (2 * np.pi / wavelength) * np.cos(2 * np.pi * radial / wavelength)
        exact = interp1d(
            radial - displacement, 1.0 / (1.0 - gradient), bounds_error=False, fill_value=np.nan
        )(radial)

        errors = []
        for n in subcells:
            out = cmt.apply_rsds(
                field=np.ones((nslice, 1)),
                los_displacement=displacement[:, None] * un.pixel,
                distance=distance,
                n_subcells=int(n),
            )
            errors.append(np.sqrt(np.mean((out[interior, 0] - exact[interior]) ** 2)))
        axis.loglog(subcells, errors, "o-", ms=3.5, lw=1.2, label=f"$A = {amplitude:g}$ cells")

    axis.loglog(subcells, 0.04 / subcells, "k:", lw=0.9, label=r"$\propto 1/n$")
    axis.set_xlabel("n_subcells")
    axis.set_ylabel("RMS error vs continuity solution")
    axis.set_xticks(subcells)
    axis.set_xticklabels([str(n) for n in subcells])
    axis.minorticks_off()
    axis.legend(frameon=False, fontsize=8)
    axis.set_title("Convergence of the redshift-space mapping", fontsize=9)

    fig.savefig(OUT / "n_subcells_convergence.svg")
    plt.close(fig)


def figure_cell_window() -> None:
    """Interpolation order converges on the cell-averaged field, not the underlying one."""
    ncell, radius = 64, 17.3
    grid = np.meshgrid(*(np.arange(ncell, dtype=float),) * 3, indexing="ij")
    centre = np.array([ncell / 2] * 3)

    rng = np.random.default_rng(0)
    vec = rng.normal(size=(3000, 3))
    vec /= np.linalg.norm(vec, axis=1)[:, None]
    lat = np.arcsin(vec[:, 2])
    lon = np.mod(np.arctan2(vec[:, 1], vec[:, 0]), 2 * np.pi)

    harmonics = [2, 4, 8, 12, 16]
    orders = [1, 2, 3, 5]
    as_is = {order: [] for order in orders}
    deconvolved = {order: [] for order in orders}
    floors = []

    for harmonic in harmonics:
        kvec = np.array([2 * np.pi * harmonic / ncell, 0.0, 0.0])
        window = float(np.sinc(kvec[0] / (2 * np.pi)))
        floors.append(1 - window)

        averaged = window * np.cos(kvec[0] * (grid[0] - centre[0]))
        sharpened = cmt.deconvolve_cell_window(averaged)
        truth = np.cos((radius * vec) @ kvec)

        for order in orders:
            for box, store in ((averaged, as_is), (sharpened, deconvolved)):
                shell = next(
                    cmt.make_lightcone_slice(
                        coevals=box,
                        latitude=lat,
                        longitude=lon,
                        distance_to_shell=radius,
                        origin=centre,
                        interpolation_order=order,
                    )
                )
                store[order].append(np.abs(shell - truth).max())

    kmags = 2 * np.pi * np.array(harmonics) / ncell / np.pi  # in units of the Nyquist

    fig, (left, right) = plt.subplots(1, 2, figsize=(8.0, 3.8), sharey=True, layout="constrained")
    for axis, store, title in (
        (left, as_is, "as given (cell averages)"),
        (right, deconvolved, "after deconvolve_cell_window"),
    ):
        # Distinct markers: on the left, orders 2-5 land on top of one another, which is
        # the whole point and should not read as missing curves.
        for order, marker in zip(orders, "os^D", strict=True):
            axis.loglog(
                kmags, store[order], marker=marker, ls="-", ms=4, lw=1.1, label=f"order {order}"
            )
        axis.loglog(kmags, floors, "k--", lw=1.4, label=r"cell window $1-\mathrm{sinc}(k/2)$")
        axis.set_xlabel(r"$k \,/\, k_{\rm Nyq}$")
        axis.set_title(title, fontsize=9)
        axis.grid(alpha=0.25, which="both", lw=0.4)
        axis.set_xticks(kmags)
        axis.set_xticklabels([f"{v:.2f}" for v in kmags])
        axis.minorticks_off()
    left.set_ylabel("max error against the underlying field")
    left.legend(frameon=False, fontsize=8, loc="lower right")

    fig.savefig(OUT / "cell_window.svg")
    plt.close(fig)


def main() -> None:
    """Build every figure."""
    OUT.mkdir(exist_ok=True)
    figure_validity_window()
    figure_interpolation_order()
    figure_deficit_versus_radius()
    figure_angular_scale()
    figure_pixel_window()
    figure_n_subcells_convergence()
    figure_cell_window()
    print(f"wrote figures to {OUT}")


if __name__ == "__main__":
    main()
