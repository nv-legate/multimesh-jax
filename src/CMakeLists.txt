
add_subdirectory(base)

if (LegateXLA_ENABLE_PYTHON)
  add_subdirectory(hlo_runner)
else()
  add_subdirectory(runtime)
  add_subdirectory(plugin)
endif()
