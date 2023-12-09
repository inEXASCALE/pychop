import sys
import os
import sphinx_rtd_theme


sys.path.insert(0, os.path.abspath('../..'))

project = 'pychop'
copyright = '2023, inexascale computing'
author = 'inexascale computing'
release = '0.0.1'

extensions = [
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx_rtd_theme',
]
templates_path = ['_templates']
source_suffix = '.rst'
exclude_patterns = []
locale_dirs = ['locale/']
gettext_compact = False

exclude_patterns = []

pygments_style = 'lovelace'

html_theme = "sphinx_rtd_theme" # html_theme = 'alabaster'
html_theme_options = {
    'logo_only': False,
    'navigation_depth': 5,
}
html_static_path = []