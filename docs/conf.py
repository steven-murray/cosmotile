"""Sphinx configuration."""

project = "Cosmotile"
author = "Steven Murray"
copyright = "2022, Steven Murray"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "myst_parser",
]
# `cosmotile.jax` is an optional backend; mocking it keeps the docs build from
# needing jax installed, in the same spirit as the committed accuracy figures.
autodoc_mock_imports = ["jax"]
autodoc_typehints = "description"
html_theme = "furo"

# `accuracy.md` uses $...$ / $$...$$ for the angular power spectrum derivation.
myst_enable_extensions = ["dollarmath"]
