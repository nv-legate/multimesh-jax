.. currentmodule:: multimesh.jax

Public API: multimesh.jax package
==============================

Subpackages
-----------

.. toctree::
   :maxdepth: 1

   multimesh.jax.flax

Task transformations
--------------------

.. autosummary::
   :toctree: _autosummary

   task
   microbatch
   parallelize
   with_sharding_constraint
   register_task
   MultiMesh
   Task
   TaskMesh

Task classes
------------
.. autoclass:: MultiMesh
   :members:
.. autoclass:: Task
.. autoclass:: TaskMesh
   :members:


Compiler contexts
-----------------

.. autosummary::
   :toctree: _autosummary

   autoshard
   context
   enable_fast_path
   enable_recomputation
   ignore_transforms

