
add_subdirectory(base)

if (LegateXLA_HLO_RUNNER)
  add_subdirectory(hlo_runner)
else()
  add_subdirectory(runtime)
  add_subdirectory(plugin)
endif()
