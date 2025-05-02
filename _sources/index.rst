MultiMesh for JAX: MPMD auto-sharding for JAX
======================================

MultiMesh for JAX provides a framework for creating `task` contexts within large computations.
This allows different subcomputations to be placed on different GPU submeshes to 
implement, e.g. pipeline parallelism. It also enables flexibly resharding tensors
based on logical names.

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

