# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

# Configuration file for the Sphinx documentation builder.

# -- Project information

project = "GFloat"
copyright = "2024, Graphcore Ltd"
author = "Andrew Fitzgibbon"
release = "0.5.2"  # Set version in package.sh
version = "0.5.2"  # Set version in package.sh

# -- General configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx_paramlinks",
    "myst_nb",
]

autodoc_typehints = "none"  # We have them in the parameter descriptors
autodoc_typehints_format = "short"
python_use_unqualified_type_names = True

autodoc_type_aliases = {
    "Iterable": "Iterable",
    "npt.ArrayLike": "ArrayLike",
    "npt.NDArray": "NDArray",
}

autodoc_default_options = {
    "member-order": "bysource",
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

# -- Options for HTML output

html_theme = "sphinx_rtd_theme"

# -- Options for EPUB output
epub_show_urls = "footnote"

# -- Options for myst_nb
nb_execution_mode = "off"
