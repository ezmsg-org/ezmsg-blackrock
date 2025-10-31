# Configuration file for the Sphinx documentation builder.

import os
import sys

# Add the source directory to the path
sys.path.insert(0, os.path.abspath("../../src"))

# -- Project information --------------------------

project = "ezmsg.blackrock"
copyright = "2024, ezmsg Contributors"
author = "ezmsg Contributors"

# The version is managed by hatch-vcs and stored in __version__.py
try:
    from ezmsg.blackrock.__version__ import version as release
except ImportError:
    release = "unknown"

# For display purposes, extract the base version without git commit info
version = release.split("+")[0] if release != "unknown" else release

# -- General configuration --------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.duration",
    "sphinx_copybutton",
    "myst_parser",  # For markdown files
]

templates_path = ["_templates"]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The toctree master document
master_doc = "index"

# -- Autodoc configuration ------------------------------

# Auto-generate API docs
autosummary_generate = True
autosummary_imported_members = False
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_typehints_format = "short"
python_use_unqualified_type_names = True
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "show-inheritance": True,
}

# Don't show the full module path in the docs
add_module_names = False

# -- Intersphinx configuration --------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "ezmsg": ("https://www.ezmsg.org/ezmsg/", None),
    "ezmsg.event": ("https://www.ezmsg.org/ezmsg-event/", None),
}
intersphinx_disabled_domains = ["std"]

# -- Options for HTML output -----------------------------

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

# Set the base URL for the documentation
html_baseurl = "https://www.ezmsg.org/ezmsg-blackrock/"

html_theme_options = {
    "logo": {
        "text": f"ezmsg.blackrock {version}",
        "link": "https://ezmsg.org",  # Link back to main site
    },
    "header_links_before_dropdown": 4,
    "navbar_start": ["navbar-logo"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/ezmsg-org/ezmsg-blackrock",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "ezmsg.org",
            "url": "https://www.ezmsg.org",
            "icon": "fa-solid fa-house",
        },
    ],
}

# Timestamp is inserted at every page bottom in this strftime format.
html_last_updated_fmt = "%Y-%m-%d"

# -- Options for linkcode -----------------------------

branch = "main"
code_url = f"https://github.com/ezmsg-org/ezmsg-blackrock/blob/{branch}/"


def linkcode_resolve(domain, info):
    if domain != "py":
        return None
    if not info["module"]:
        return None
    filename = info["module"].replace(".", "/")
    return f"{code_url}src/{filename}.py"
