import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

project = 'pychop'
copyright = '2026, InEXASCALE computing'
author = 'InEXASCALE computing'
release = '0.6.2'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
]

templates_path = ['_templates']
source_suffix = '.rst'
locale_dirs = ['locale/']
gettext_compact = False
exclude_patterns = []

pygments_style = 'lovelace'
pygments_dark_style = "monokai"

html_theme = "furo"
html_logo = "../imgs/pychop_logo.png"

html_theme_options = {
    "sidebar_hide_name": False, 
    "navigation_with_keys": True,
}


html_static_path = []

# Document attributes once, through autodoc members.
napoleon_use_ivar = True
