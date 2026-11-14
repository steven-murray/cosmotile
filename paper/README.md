# JOSS paper

Drafting notes for [#288](https://github.com/steven-murray/cosmotile/issues/288).
`paper.md` is a skeleton: every section has an HTML comment listing the points to
make. Delete each comment as you write the prose over it.

Push anything under `paper/` and the `Draft JOSS paper` workflow builds the PDF and
attaches it as a run artifact, so there is no need for a local pandoc/xelatex setup.

## Decisions still to make

**Author list.** The algorithm is Nithyanandan Thyagarajan's, modularised by Piyanat
Kittiwisit; this package is a rewrite of that lineage. JOSS asks for authors who made
a substantial contribution to *the submitted software*, so ask both whether they want
authorship, and fall back to Acknowledgements plus a citation of
[Kittiwisit et al. (2018)](https://doi.org/10.1093/mnras/stx3099) if they decline.

**Figures.** The strongest figure in the repo is `docs/figures/validity_window.svg` —
measured $C_\ell$ against the discrete-mode prediction with the valid window shaded. It
makes the whole argument of the paper in one panel. The JOSS build is LaTeX, so it needs
PDF or PNG: add a PDF output to `docs/make_accuracy_figures.py` rather than converting
the committed SVG. `docs/figures/gpu_speedup.svg` is a plausible second figure, but JOSS
papers are short and one may be enough.

**Statement of need.** This is the section reviewers weigh most heavily, and it is the
one place the draft asserts things about other packages. Check every such claim against
that package's current documentation before submitting — some of the comparisons in the
comments are from memory.

**Archive.** JOSS wants a tagged release deposited somewhere with a DOI (Zenodo), whose
metadata title and author list match the paper. Do the tag last, after the paper text
settles.

## Bibliography

`paper.bib` entries were pulled from the arXiv API or from a JOSS DOI, except those
marked `% VERIFY`, which were written from memory and need checking. Two entries are
still missing and are flagged with `TODO`: the other coeval simulation codes worth
naming, and at least one genuine alternative that generates directly on the lightcone
rather than tiling a box.

## Checklist before submitting

- [ ] Summary reads for a non-specialist; Statement of Need reads for the field
- [ ] Every package named in the Statement of Need verified against its current docs
- [ ] All `% VERIFY` and `TODO` markers cleared from `paper.bib`
- [ ] Figure(s) rendered as PDF and referenced from `paper.md`
- [ ] Author list, ORCIDs and affiliations confirmed with each author
- [ ] Funding statement in Acknowledgements
- [ ] Under ~1000 words
- [ ] Tagged release archived with a DOI, metadata matching the paper
