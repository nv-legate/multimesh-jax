MultiMesh for JAX: MPMD workflows for JAX
======================================

MultiMesh for JAX provides a framework for creating `task` contexts within jitted computations,
allowing different subcomputations to be placed on different GPU submeshes. These
`task` computations can be combined inside a global `jit` with data resharding across submeshes
occurring automatically. MultiMesh therefore enables pipeline parallelism to be easily expressed.
Standard Jax SPMD sharding idioms can be used within each `task`,
enabling full N-dimensional parallelism.
Readers can find more `architecture details <architecture.html>`_
or `get started <getting-started.html>`_ using it.

.. grid:: 2
   :margin: 0
   :padding: 0
   :gutter: 0

   .. grid-item-card:: Familiar Transformation API
      :columns: 12 6 6 4
      :class-card: sd-border-0
      :shadow: None

      MultiMesh for JAX uses idiomatic JAX transforms to assign submesh contexts to functions or Flax modules

   .. grid-item-card:: Plugin Installation
      :columns: 12 6 6 4
      :class-card: sd-border-0
      :shadow: None

      MultiMesh for JAX adds compiler and runtime functionality via the `jax_plugins` interface

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Architecture Overview

   architecture

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Getting Started

   getting-started

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: API Reference

   api

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Tutorials

   tutorials

