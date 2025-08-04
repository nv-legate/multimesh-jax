import os
import sys

sys.path.insert(0, os.path.abspath(".."))


project = "MultiMesh for JAX"
copyright = "2025, The MultiMesh for JAX Authors."
author = "The MultiMesh for JAX authors"

version = ""
release = ""


needs_sphinx = "2.1"

sys.path.append(os.path.abspath("sphinxext"))
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "matplotlib.sphinxext.plot_directive",
    "myst_nb",
    "sphinx_remove_toctrees",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxext.rediraffe",
    "sphinx.ext.githubpages",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
}

suppress_warnings = [
    "ref.citation",  # Many duplicated citations in numpy/scipy docstrings.
    "ref.footnote",  # Many unreferenced footnotes in numpy/scipy docstrings
    "myst.header",
]

templates_path = ["_templates"]

source_suffix = [".rst", ".ipynb", ".md"]

main_doc = "index"

language = "en"

exclude_patterns = []

pygments_style = None

autosummary_generate = True
napolean_use_rtype = False

html_theme = "sphinx_book_theme"

html_theme_options = {
    "show_toc_level": 2,
    "repository_url": "https://github.com/nv-legate/multimesh-jax",
    "use_repository_button": True,  # add a "link to repository" button
    "navigation_with_keys": False,
}

html_static_path = ["_static"]

html_css_files = [
    "style.css",
]

# -- Options for myst ----------------------------------------------
myst_heading_anchors = 3  # auto-generate 3 levels of heading anchors
myst_enable_extensions = ["dollarmath"]
nb_execution_mode = "force"
nb_execution_allow_errors = False
nb_merge_streams = True
nb_execution_show_tb = True

# Notebook cell execution timeout; defaults to 30.
nb_execution_timeout = 100

# List of patterns, relative to source directory, that match notebook
# files that will not be executed.
nb_execution_excludepatterns = []

# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "MultiMeshJAXdoc"


# -- Options for LaTeX output ------------------------------------------------

latex_elements = {}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = []


# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = []


# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = []


# -- Options for Epub output -------------------------------------------------

# Bibliographic Dublin Core info.
epub_title = project

# The unique identifier of the text. This can be a ISBN number
# or the project homepage.
#
# epub_identifier = ''

# A list of files that should not be packed into the epub file.
epub_exclude_files = ["search.html"]


# -- Extension configuration -------------------------------------------------

# Tell sphinx autodoc how to render type aliases.
autodoc_typehints = "description"
autodoc_typehints_description_target = "all"
autodoc_type_aliases = {}
autoclass_content = "class"

# Remove auto-generated API docs from sidebars. They take too long to build.
remove_from_toctrees = ["_autosummary/*"]

# Customize code links via sphinx.ext.linkcode


def linkcode_resolve(domain, info):
    if domain != "py":
        return None
    if not info["module"]:
        return None
    if not info["fullname"]:
        return None
    if not info["module"].startswith("multimesh.jax"):
        return None

    return None


# Generate redirects from deleted files to new sources
rediraffe_redirects = {}
