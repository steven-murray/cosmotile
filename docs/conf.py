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
