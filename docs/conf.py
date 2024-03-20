import sys
import os

sys.path.insert(0, os.path.abspath('./../../src/'))
sys.path.insert(0, os.path.abspath('./src/'))

# # Load modules that may conflict with autodoc
# # https://github.com/pydantic/pydantic/discussions/7763
# from prompttrail.models import google_cloud # noqa: E402 # type: ignore

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'promttrail'
copyright = '2023, combinatrix.ai'
author = 'combinatrix.ai'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    # 'sphinx_fontawesome',
    'sphinx.ext.napoleon',
    'sphinx_rtd_theme',
    # 'sphinx_multiversion',
    'sphinxcontrib.autodoc_pydantic',
    'myst_parser'
]

autosummary_generate = True  # Turn on sphinx.ext.autosummary
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
autodoc_mock_imports = ["openai", "google_cloud"]
autodoc_pydantic_model_show_json = False
autodoc_pydantic_model_show_config_summary = False
autodoc_default_options = {
    'special-members': '__init__',
}

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}
# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['../source/_static']
html_theme_options = {
    'collapse_navigation': False,
}