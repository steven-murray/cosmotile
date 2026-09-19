"""Sphinx configuration."""

project = "Cosmotile"
author = "Steven Murray"
copyright = "2022, Steven Murray"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_click",
    "myst_parser",
]
autodoc_typehints = "description"
html_theme = "furo"

# `accuracy.md` uses $...$ / $$...$$ for the angular power spectrum derivation.
myst_enable_extensions = ["dollarmath"]
