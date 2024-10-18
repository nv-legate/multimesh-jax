.. currentmodule:: legate.jax

Public API: legate.jax package
==============================

Subpackages
-----------

.. toctree::
   :maxdepth: 1

   legate.jax.flax

Task transformations
--------------------

.. autosummary::
   :toctree: _autosummary

   task
   microbatch
   parallelize
   with_sharding_constraint
   register_task

Compiler contexts
-----------------

.. autosummary::
   :toctree: _autosummary

   autoshard
   context
   enable_fast_path
   enable_recomputation
   enable_task_fusion
   only_fuse_loop_tasks
   ignore_transforms
   host_offload_min_reuse_distance
   store_cache_min_parallelism

