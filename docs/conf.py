import os
import sys

sys.path.insert(0, os.path.abspath(".."))
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Spherical-Implicit-Neural-Representation"
copyright = "2025, Theo Hanon"
author = "Theo Hanon"
release = "latest"
version = "latest"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx_design",
]


templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_logo = "_static/atoms.png"
html_static_path = ["_static"]

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/TheoHanon/Spherical-Implicit-Neural-Representation",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        }
    ],
    "use_edit_page_button": True,
}

html_context = {
    "github_user": "TheoHanon",
    "github_repo": "Spherical-Implicit-Neural-Representation",
    "github_version": "main",
    "doc_path": "docs",
}

html_sidebars = {
    "_api": [],
    "_install": [],
    "_tutorial": [],
    "_reference": [],
}

napoleon_numpy_docstring = True
napoleon_use_ivar = True