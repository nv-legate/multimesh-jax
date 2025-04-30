# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

load("//xla/tsl:tsl.default.bzl", "tsl_pybind_extension")

load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda", "cuda_default_copts", "cuda_library")

load(
    "//xla:xla.bzl",
    "xla_cc_test",
)

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "legate_utils",
    srcs = ["legate_utils.cc"],
    hdrs = ["legate_utils.h"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_set",
        "//xla/stream_executor:stream_executor_h",
        "//xla/service:backend",
        "//xla/pjrt/distributed:distributed",
        "//xla/pjrt/gpu:nccl_id_store",
        "//xla/service/gpu:gpu_executable_run_options",
        "@zuku",
        ":json_utils",
        ":allocator",
    ],
    alwayslink = True
)

cc_library(
    name = "legate_computation",
    srcs = ["legate_computation.cc"],
    hdrs = ["legate_computation.h"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "//xla/client:local_client",
        "//xla/hlo/ir:hlo",
        "//xla/service:backend",
        "//xla/pjrt:tfrt_cpu_pjrt_client",
        "//xla/pjrt:pjrt_stream_executor_client",
        "//xla/service/cpu:cpu_executable",
        "//xla/service/gpu:gpu_executable",
        "//xla/service/gpu/model:gpu_hlo_cost_analysis",
        "//xla/pjrt:pjrt_executable",
        "//xla/tools:hlo_module_loader",
        "@tsl//tsl/platform:status",
        ":legate_utils",
        ":cuda_utils",
        ":hlo_partition",
        ":legate_device_assignment",
    ],
    alwayslink = True,
)

cc_library(
    name = "legate_sharding",
    srcs = ["legate_sharding.cc"],
    hdrs = ["legate_sharding.h"],
    deps = [
        "@tsl//tsl/platform:status",
        "@tsl//tsl/platform:errors",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service/spmd:stateful_rng_spmd_partitioner",
        "@zuku",
    ],
)

cc_library(
    name = "mpmd_hoist_loop_convert",
    srcs = ["mpmd_hoist_loop_convert.cc"],
    hdrs = ["mpmd_hoist_loop_convert.h"],
    deps = [
        "//xla/hlo/pass:hlo_pass_pipeline",
        "@com_google_absl//absl/status",
        "@zuku",
        ":mpmd_instruction",
        ":hlo_partition",
    ],
)

cc_library(
    name = "mpmd_loop_unroll",
    srcs = ["mpmd_loop_unroll.cc"],
    hdrs = ["mpmd_loop_unroll.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@zuku",
        "//xla/hlo/pass:hlo_pass_pipeline",
        ":hlo_partition",
        ":loop_scheduler",
        ":mpmd_instruction",
        ":mpmd_loop",
        ":mpmd_utils",
    ],
)

cc_library(
    name = "mpmd_buffer_scheduling_name",
    srcs = ["mpmd_buffer_scheduling_name.cc"],
    hdrs = ["mpmd_buffer_scheduling_name.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@jsoncpp_git//:jsoncpp",
        "@zuku",
        "//xla:status_macros",
        ":mpmd_instruction",
        ":legate_sharding",
        ":hlo_partition",
    ],
)

cc_library(
    name = "mpmd_insert_reshard",
    srcs = ["mpmd_insert_reshard.cc"],
    hdrs = ["mpmd_insert_reshard.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@zuku",
        "//xla/hlo/pass:hlo_pass_pipeline",
        ":hlo_partition",
        ":legate_sharding",
        ":mpmd_instruction",
    ]
)

cc_library(
    name = "mpmd_shard_map_loop_reduce",
    srcs = ["mpmd_shard_map_loop_reduce.cc"],
    hdrs = ["mpmd_shard_map_loop_reduce.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@zuku",
        "//xla/service/gpu/transforms:reduce_scatter_creator",
        "//xla/service:all_reduce_folder",
        "//xla/service/gpu/transforms:all_reduce_splitter",
        ":hlo_partition",
    ],
)

cc_library(
    name = "mpmd_hoist_shard_map_reduce",
    srcs = ["mpmd_hoist_shard_map_reduce.cc"],
    hdrs = ["mpmd_hoist_shard_map_reduce.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@zuku",
        "//xla/hlo/pass:hlo_pass_pipeline",
        ":hlo_partition",
        ":mpmd_instruction",
        ":mpmd_utils",
    ],
)

cc_library(
    name = "mpmd_reorder_shard_map_transpose",
    srcs = ["mpmd_reorder_shard_map_transpose.cc"],
    hdrs = ["mpmd_reorder_shard_map_transpose.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@zuku",
        "//xla/hlo/pass:hlo_pass_pipeline",
        ":hlo_partition",
        ":mpmd_instruction",
        ":mpmd_utils",
    ],
)

cc_library(
    name = "mpmd_logical_sharding_propagation",
    srcs = ["mpmd_logical_sharding_propagation.cc"],
    hdrs = ["mpmd_logical_sharding_propagation.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@zuku",
        "//xla/hlo/pass:hlo_pass_pipeline",
        ":mpmd_instruction",
        ":mpmd_utils",
        ":logical_sharding_context",
    ],
)

cc_library(
    name = "mpmd_microbatch_loop_canonicalizer",
    srcs = ["mpmd_microbatch_loop_canonicalizer.cc"],
    hdrs = ["mpmd_microbatch_loop_canonicalizer.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@zuku",
        "//xla/hlo/pass:hlo_pass_pipeline",
        "//xla/service:tuple_simplifier",
        "//xla/service:call_inliner",
        ":hlo_partition",
        ":mpmd_instruction",
    ],
)

cc_library(
    name = "mpmd_cut_size_minimizer",
    srcs = ["mpmd_cut_size_minimizer.cc"],
    hdrs = ["mpmd_cut_size_minimizer.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@zuku",
        "//xla/hlo/pass:hlo_pass_pipeline",
        ":hlo_partition",
        ":legate_sharding",
        ":mpmd_instruction",
    ],
)

cc_library(
    name = "mpmd_color_uniquifier",
    srcs = ["mpmd_color_uniquifier.cc"],
    hdrs = ["mpmd_color_uniquifier.h"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@tsl//tsl/platform:status",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:variant",
        "@jsoncpp_git//:jsoncpp",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass_pipeline",
        "//xla/service:hlo_pass_pipeline",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:path",
        "@tsl//tsl/platform:platform_port",
        "@tsl//tsl/platform:statusor",
        "//xla:status",
        "@com_google_absl//absl/status",
        "@tsl//tsl/platform:protobuf",
        "//xla:debug_options_flags",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/types:span",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo_module_group",
        "@tsl//tsl/platform:numbers",
        "@zuku",
        ":mpmd_instruction",
        ":hlo_partition",
        ":legate_sharding",
    ],
)

cc_library(
    name = "mpmd_uniquify_colors",
    srcs = ["mpmd_uniquify_colors.cc"],
    hdrs = ["mpmd_uniquify_colors.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@zuku",
        "//xla/hlo/pass:hlo_pass_pipeline",
        ":hlo_partition",
        ":legate_sharding",
        ":mpmd_instruction",
    ],
)

cc_library(
    name = "mpmd_instruction_delay_recolor",
    srcs = ["mpmd_instruction_delay_recolor.cc"],
    hdrs = ["mpmd_instruction_delay_recolor.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@zuku",
        "//xla/hlo/pass:hlo_pass_pipeline",
        "//xla/hlo/transforms/simplifiers:hlo_dce",
        ":hlo_partition",
        ":mpmd_instruction",
    ],
)

cc_library(
    name = "mpmd_argument_recompute",
    srcs = ["mpmd_argument_recompute.cc"],
    hdrs = ["mpmd_argument_recompute.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@tsl//tsl/platform:logging",
        "@zuku",
        ":mpmd_instruction",
        ":hlo_partition",
        ":legate_sharding",
    ],
)

cc_library(
    name = "mpmd_simple_loop_increment_coloring",
    srcs = ["mpmd_simple_loop_increment_coloring.cc"],
    hdrs = ["mpmd_simple_loop_increment_coloring.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@zuku",
        "//xla/hlo/pass:hlo_pass_pipeline",
        ":hlo_partition",
        ":legate_sharding",
        ":mpmd_instruction",
    ],
)

cc_library(
    name = "mpmd_unpack_optimization_barrier",
    srcs = ["mpmd_unpack_optimization_barrier.cc"],
    hdrs = ["mpmd_unpack_optimization_barrier.h"],
    deps = [
        "@com_google_absl//absl/status",
        "//xla/hlo/pass:hlo_pass_pipeline",
        ":mpmd_instruction",
        ":mpmd_utils",
    ],
)

cc_library(
    name = "mpmd_repack_optimization_barrier",
    srcs = ["mpmd_repack_optimization_barrier.cc"],
    hdrs = ["mpmd_repack_optimization_barrier.h"],
    deps = [
        "@com_google_absl//absl/status",
        "//xla/hlo/pass:hlo_pass_pipeline",
        ":mpmd_instruction",
        ":mpmd_utils",
        ":mpmd_unpack_optimization_barrier",
    ],
)

cc_library(
    name = "mpmd_cross_task_barrier_remover",
    srcs = ["mpmd_cross_task_barrier_remover.cc"],
    hdrs = ["mpmd_cross_task_barrier_remover.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@zuku",
        "//xla/hlo/pass:hlo_pass_pipeline",
        ":hlo_partition",
        ":mpmd_instruction",
    ],
)

cc_library(
    name = "iota_sharding_sanitizer",
    srcs = ["iota_sharding_sanitizer.cc"],
    hdrs = ["iota_sharding_sanitizer.h"],
    deps = [
        "@com_google_absl//absl/status",
        ":hlo_partition",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_pass_pipeline",
        "@jsoncpp_git//:jsoncpp",
        "@zuku",
    ],
)

cc_library(
    name = "mpmd_computation_grouper",
    srcs = ["mpmd_computation_grouper.cc"],
    hdrs = ["mpmd_computation_grouper.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/container:flat_hash_map",
        "@jsoncpp_git//:jsoncpp",
        "@zuku",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass_pipeline",
        "//xla/service:hlo_pass_pipeline",
        ":color_dfs",
        ":mpmd_instruction",
        ":hlo_partition",
    ],
)

cc_library(
    name = "mpmd_computation_fusion",
    srcs = ["mpmd_computation_fusion.cc"],
    hdrs = ["mpmd_computation_fusion.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@jsoncpp_git//:jsoncpp",
        "@zuku",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass_pipeline",
        "//xla/service:hlo_pass_pipeline",
        ":mpmd_instruction",
        ":hlo_partition",
        ":color_dfs",
    ],
)

cc_library(
    name = "mpmd_unused_param_output_remover",
    srcs = ["mpmd_unused_param_output_remover.cc"],
    hdrs = ["mpmd_unused_param_output_remover.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@zuku",
        "//xla/hlo/pass:hlo_pass_pipeline",
        ":mpmd_instruction",
        ":hlo_partition",
    ],
)

cc_library(
    name = "mpmd_unused_loop_output_remover",
    srcs = ["mpmd_unused_loop_output_remover.cc"],
    hdrs = ["mpmd_unused_loop_output_remover.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@zuku",
        "//xla/hlo/pass:hlo_pass_pipeline",
        ":hlo_partition",
        ":mpmd_instruction",
    ],
)

cc_library(
    name = "mpmd_concatenate_grouper",
    srcs = ["mpmd_concatenate_grouper.cc"],
    hdrs = ["mpmd_concatenate_grouper.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@zuku",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass_pipeline",
        ":hlo_partition",
        ":mpmd_instruction",
    ],
)

cc_library(
    name = "mpmd_logical_to_gspmd_sharding",
    srcs = ["mpmd_logical_to_gspmd_sharding.cc"],
    hdrs = ["mpmd_logical_to_gspmd_sharding.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@zuku",
        "//xla/hlo/pass:hlo_pass_pipeline",
        ":mpmd_instruction",
        ":hlo_partition",
    ],
)

cc_library(
    name = "mpmd_parameter_replication",
    srcs = ["mpmd_parameter_replication.cc"],
    hdrs = ["mpmd_parameter_replication.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@zuku",
        ":hlo_partition",
        ":mpmd_instruction",
        "//xla/hlo/pass:hlo_pass_pipeline",
    ],
)

cc_library(
    name = "mpmd_computation_inliner",
    srcs = ["mpmd_computation_inliner.cc"],
    hdrs = ["mpmd_computation_inliner.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass_pipeline",
        "//xla/service:tuple_simplifier",
        "//xla/service:call_inliner",
        "@com_google_absl//absl/status",
        "@zuku",
        ":mpmd_instruction",
        ":hlo_partition",
    ],
)

cc_library(
    name = "mpmd_inline_explicit_tasks",
    srcs = ["mpmd_inline_explicit_tasks.cc"],
    hdrs = ["mpmd_inline_explicit_tasks.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@jsoncpp_git//:jsoncpp",
        "@zuku",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass_pipeline",
        "//xla/service:hlo_pass_pipeline",
        "//xla/service:tuple_simplifier",
        "//xla/service:call_inliner",
        ":mpmd_instruction",
        ":hlo_partition",
    ],
)

cc_library(
    name = "mpmd_insert_root_tuple_shardings",
    srcs = ["mpmd_insert_root_tuple_shardings.cc"],
    hdrs = ["mpmd_insert_root_tuple_shardings.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@zuku",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass_pipeline",
        "//xla/service:hlo_pass_pipeline",
        ":mpmd_instruction",
        ":hlo_partition",
    ],
)

cc_library(
    name = "mpmd_compute_assigned_colors",
    srcs = ["mpmd_compute_assigned_colors.cc"],
    hdrs = ["mpmd_compute_assigned_colors.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@zuku",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass_pipeline",
        "//xla/service:hlo_pass_pipeline",
        ":mpmd_instruction",
        ":hlo_partition",
    ],
)

cc_library(
    name = "mpmd_coloring",
    srcs = ["mpmd_coloring.cc"],
    hdrs = ["mpmd_coloring.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@zuku",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass_pipeline",
        "//xla/service:hlo_pass_pipeline",
        ":mpmd_instruction",
        ":hlo_partition",
        ":mpmd_utils",
        ":mpmd_inline_explicit_tasks",
        ":mpmd_insert_root_tuple_shardings",
        ":mpmd_compute_assigned_colors",
        ":mpmd_unpack_optimization_barrier",
        ":mpmd_repack_optimization_barrier",
    ],
)

cc_library(
    name = "mpmd_constant_output_copy",
    srcs = ["mpmd_constant_output_copy.cc"],
    hdrs = ["mpmd_constant_output_copy.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass_pipeline",
        "@com_google_absl//absl/status",
        ":hlo_partition",
        "@zuku",
    ],
)

cc_library(
    name = "mpmd_repeated_output_copy",
    srcs = ["mpmd_repeated_output_copy.cc"],
    hdrs = ["mpmd_repeated_output_copy.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@zuku",
        "//xla/hlo/pass:hlo_pass_pipeline",
        ":hlo_partition",
    ],
)

cc_library(
    name = "mpmd_sharding_propagation",
    srcs = ["mpmd_sharding_propagation.cc"],
    hdrs = ["mpmd_sharding_propagation.h"],
    deps = [
        "@com_google_absl//absl/status",
        "//xla/service:sharding_propagation",
        ":hlo_partition",
        "@zuku",
    ],
)

cc_library(
    name = "loop_scheduler",
    srcs = ["loop_scheduler.cc"],
    hdrs = ["loop_scheduler.h"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@tsl//tsl/platform:status",
        "@tsl//tsl/platform:errors",
        "//xla/hlo/ir:hlo",
        ":hlo_partition",
        ":legate_computation",
    ],
)

cc_library(
    name = "json_utils",
    srcs = ["json_utils.cc"],
    hdrs = ["json_utils.h"],
    deps = [
        "@tsl//tsl/platform:status",
        "@tsl//tsl/platform:errors",
        "@jsoncpp_git//:jsoncpp",
        "//xla:util",
    ],
    alwayslink = True,
)

cc_library(
    name = "mpmd_store",
    hdrs = ["mpmd_store.h"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@tsl//tsl/platform:status",
        "//xla/hlo/ir:hlo",
        "@zuku",
    ],
    alwayslink = True,
)

cc_library(
    name = "mpmd_loop",
    srcs = ["mpmd_loop.cc"],
    hdrs = ["mpmd_loop.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/container:flat_hash_map",
        "@zuku",
        ":json_utils",
    ],
    alwayslink = True,
)

cc_library(
    name = "color_dfs",
    hdrs = ["color_dfs.h"],
    srcs = ["color_dfs.cc"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@tsl//tsl/platform:status",
        "//xla/hlo/ir:hlo",
        ":mpmd_instruction",
    ],
)

cc_library(
    name = "mpmd_utils",
    hdrs = ["mpmd_utils.h"],
    srcs = ["mpmd_utils.cc"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@tsl//tsl/platform:status",
        "@jsoncpp_git//:jsoncpp",
        "//xla/hlo/ir:hlo",
        "@zuku",
        ":json_utils",
    ],
    alwayslink = True,
)

cc_library(
    name = "logical_sharding_context",
    hdrs = [":logical_sharding_context.h"],
    srcs = [":logical_sharding_context.cc"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@tsl//tsl/platform:status",
        "@zuku",
        ":json_utils",
        ":mpmd_loop",
        ":mpmd_utils",
    ],
    alwayslink = True,
)

cc_library(
    name = "mpmd_instruction",
    hdrs = ["mpmd_instruction.h"],
    srcs = ["mpmd_instruction.cc"],
    deps = [
        "@com_google_absl//absl/status",
        "//xla/hlo/ir:hlo",
        "@zuku",
        ":json_utils",
        ":mpmd_loop",
        ":logical_sharding_context",
    ],
    alwayslink = True,
)

cc_library(
    name = "hlo_partition",
    srcs = ["hlo_partition.cc"],
    hdrs = ["hlo_partition.h"],
    deps = [
        "@com_google_absl//absl/status",
        ":json_utils",
        ":legate_sharding",
        ":logical_sharding_context",
        ":mpmd_instruction",
        ":mpmd_utils",
        "//xla/hlo/ir:hlo",
        "@tsl//tsl/platform:regexp",
        "@jsoncpp_git//:jsoncpp",
        "@zuku",
    ],
    alwayslink = True,
)

cc_library(
    name = "mpmd_partition",
    srcs = ["mpmd_partition.cc"],
    hdrs = ["mpmd_partition.h"],
    deps = [
        ":mpmd_instruction",
        ":mpmd_store",
        ":mpmd_utils",
        ":legate_computation",
        "//xla/service:hlo_module_util",
        "//xla/tools:hlo_module_loader",
        "//xla/pjrt:compile_options_proto_cc",
        "//xla/pjrt:execute_options_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/transforms/simplifiers:hlo_dce",
        ":legate_sharding",
        ":legate_utils",
        ":mpmd_input_output_buffer_alias",
        ":mpmd_sharding_propagation",
        ":mpmd_logical_to_gspmd_sharding",
        ":mpmd_logical_sharding_propagation",
        ":mpmd_microbatch_loop_canonicalizer",
        ":mpmd_computation_fusion",
        ":mpmd_unused_loop_output_remover",
        ":mpmd_computation_grouper",
        ":mpmd_computation_inliner",
        ":mpmd_cut_size_minimizer",
        ":mpmd_cross_task_barrier_remover",
        ":mpmd_uniquify_colors",
        ":mpmd_instruction_delay_recolor",
        ":mpmd_concatenate_grouper",
        ":mpmd_coloring",
        ":mpmd_constant_output_copy",
        ":mpmd_argument_recompute",
        ":mpmd_repeated_output_copy",
        ":mpmd_parameter_replication",
        ":mpmd_unused_param_output_remover",
        ":mpmd_hoist_loop_convert",
        ":mpmd_hoist_shard_map_reduce",
        ":mpmd_reorder_shard_map_transpose",
        ":mpmd_shard_map_loop_reduce",
        ":mpmd_simple_loop_increment_coloring",
        ":mpmd_loop_unroll",
        ":mpmd_insert_reshard",
        ":mpmd_buffer_scheduling_name",
        ":iota_sharding_sanitizer",
        ":scalar_argument",
    ],
    alwayslink = True,
)

cc_library(
    name = "mpmd_input_output_buffer_alias",
    srcs = ["mpmd_input_output_buffer_alias.cc"],
    hdrs = ["mpmd_input_output_buffer_alias.h"],
    deps = [
        "@com_google_absl//absl/status",
        "//xla/service:hlo_pass",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        ":legate_sharding",
    ],
)

cc_library(
    name = "legate_pjrt_executable",
    srcs = ["legate_pjrt_executable.cc"],
    hdrs = ["legate_pjrt_executable.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@tsl//tsl/platform:fingerprint",
        "//xla/pjrt:pjrt_client",
        "//xla/service:computation_placer",
        "//xla/pjrt/legate:legate_computation",
        "//xla/pjrt:pjrt_stream_executor_client",
        "//xla/service/gpu:gpu_executable",
        ":hlo_partition",
        ":cuda_utils",
        ":legate_utils",
        ":legate_pjrt_buffer",
        ":zuku_execute_context",
        ":loop_scheduler",
        "@zuku",
    ],
    alwayslink = True,
)

cc_library(
    name = "legate_pjrt_buffer",
    srcs = ["legate_pjrt_buffer.cc"],
    hdrs = ["legate_pjrt_buffer.h"],
    deps = [
        "//xla/pjrt:pjrt_stream_executor_client",
        "//xla/pjrt/cpu:tracked_tfrt_cpu_device_buffer",
        "//xla/pjrt:tfrt_cpu_pjrt_client",
        "//xla/service/cpu:cpu_event",
        ":legate_buffer_action",
        ":legate_utils",
        ":zuku_execute_context",
        "@zuku",
    ],
    alwayslink = True,
)

cc_library(
    name = "legate_buffer_action",
    srcs = ["legate_buffer_action.cc"],
    hdrs = ["legate_buffer_action.h"],
    deps = [
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_stream_executor_client",
        ":cuda_utils",
        ":json_utils",
    ],
    alwayslink = True,
)

cc_library(
    name = "allocator",
    hdrs = ["allocator.h"],
    deps = [
      "@zuku",
      "@realm",
    ],
)

cc_library(
    name = "executable_cache",
    srcs = ["executable_cache.cc"],
    hdrs = ["executable_cache.h"],
    deps = [
        "@zuku",
        "@realm",
        ":legate_utils",
        ":legate_computation",
    ],
)

cc_library(
  name = "hlo_executor",
  hdrs = ["hlo_executor.h"],
  srcs = ["hlo_executor.cc"],
  deps = [
    "@zuku",
    "@realm",
    ":allocator",
    ":legate_computation",
    ":scalar_argument",
  ]
)

cc_library(
    name = "hlo_loader",
    hdrs = ["hlo_loader.h"],
    srcs = ["hlo_loader.cc"],
    deps = [
      "@zuku",
      "@realm",
      ":legate_computation",
      ":allocator",
      ":executable_cache",
    ]
)

cc_library(
    name = "store_handle_fwd",
    hdrs = ["store_handle_fwd.h"],
    deps = [
      "@zuku",
      "@realm",
      ":allocator",
    ],
)

cc_library(
    name = "store_handle",
    hdrs = ["store_handle.h"],
    deps = [
      "@zuku",
      "@realm",
      ":store_handle_fwd",
    ],
)

cc_library(
    name = "legate_device_assignment",
    hdrs = ["legate_device_assignment.h"],
    srcs = ["legate_device_assignment.cc"],
    deps = [
        "@zuku",
        "@realm",
        "//xla/service:computation_placer",
    ],
)

cc_library(
    name = "scalar_argument",
    hdrs = ["scalar_argument.h"],
)

cc_library(
    name = "zuku_execute_context",
    hdrs = ["zuku_execute_context.h"],
    deps = [
        "@zuku",
        "@realm",
        ":scalar_argument",
        ":legate_buffer_action",
        ":allocator",
        ":store_handle_fwd",
    ],
)

cc_library(
    name = "zuku_execute_context_impl",
    hdrs = ["zuku_execute_context_impl.h"],
    srcs = ["zuku_execute_context_impl.cc"],
    deps = [
        "@zuku",
        "@realm",
        ":zuku_execute_context",
        ":legate_computation",
        ":legate_utils",
        ":hlo_executor",
        ":hlo_loader",
        ":store_handle",
    ],
)

cc_library(
    name = "legate_pjrt_client",
    srcs = ["legate_pjrt_client.cc", "legate_pjrt_executable.cc", "legate_pjrt_buffer.cc"],
    hdrs = ["legate_pjrt_client.h", "legate_pjrt_executable.h", "legate_pjrt_buffer.h"],
    deps = [
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_stream_executor_client",
        "//xla/client:local_client",
        ":legate_buffer_action",
        ":legate_utils",
        ":legate_computation",
        ":zuku_execute_context",
        ":mpmd_partition",
        "//xla/pjrt/gpu:se_gpu_pjrt_client",
        "//xla/pjrt:transpose",
        "@zuku",
    ],
    alwayslink = True,
)

cc_library(
    name = "legate_c_api_client",
    srcs = ["legate_c_api_client.cc"],
    deps = [
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_stream_executor_client",
        "//xla/client:local_client",
        ":legate_pjrt_client",
        ":zuku_execute_context_impl",
        "//xla/pjrt/gpu:se_gpu_pjrt_client",
        "//xla/pjrt/c:pjrt_c_api_hdrs",
        "//xla/pjrt/c:pjrt_c_api_helpers",
        "//xla/pjrt/c:pjrt_c_api_wrapper_impl",
        "//xla/backends/cpu/collectives:gloo_collectives",
        "//xla/backends/cpu/collectives:gloo_kv_store",
        "//xla/python:custom_partition_callback",
        "//xla/backends/profiler/plugin:plugin_tracer_impl",
        "//xla/pjrt/c:pjrt_c_api_custom_partitioner_extension_hdrs",
        "//xla/pjrt/c:pjrt_c_api_gpu_extension_hdrs",
        "//xla/pjrt/c:pjrt_c_api_ffi_extension_hdrs",
        "//xla/pjrt/c:pjrt_c_api_ffi_internal",
        "//xla/ffi",
        "//xla/ffi:ffi_api",
        "//xla/ffi/api:c_api",
        "@gloo//:transport_tcp",
        "@zuku",
    ],
    alwayslink = True,
)

cuda_library(
    name = "cuda_utils",
    srcs = ["cuda_utils.cu.cc"],
    hdrs = ["cuda_utils.h"],
    copts = select({
        "@local_config_cuda//cuda:using_nvcc": [
            "-nvcc_options",
            "ptxas-options=-v",
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = [
        "//xla/tsl/cuda",  # buildcleaner: keep
        "//xla/tsl/cuda:cudart",  # buildcleaner: keep
        "//xla/stream_executor/gpu:gpu_stream",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
        "@zuku",
    ],
    alwayslink = True
)

tsl_pybind_extension(
    name = "legate_xla_compiler",
    srcs = ["xla_compiler.cc"],
    deps = [
        "@nanobind",
        "//third_party/python_runtime:headers",
        "//xla:util",
        "//xla/pjrt:status_casters",
        "//xla/pjrt/c:pjrt_c_api_gpu_extension_hdrs",
        "//xla/pjrt/c:pjrt_c_api_hdrs",
        "//xla/pjrt/c:pjrt_c_api_helpers",
        "//xla/ffi/api:c_api",
    ]
)

tsl_pybind_extension(
    name = "liblegate_xla_client",
    additional_exported_symbols = [
        "GetLegatePjrtApi",
        "CompileHloModuleFromFile",
        "LegateShutdown",
        "ClearMetadataNameTasks",
        "SetEnableMetadataNameTasks",
        "EnableFastPath",
        "EnableTaskFusion",
        "EnableOnlyFuseLoopTasks",
        "EnableLegateRecomputation",
        "RegisterMetadataNameTask",
        "ReplicateParametersSmallerThanNumElements",
        "SetHostOffloadMinReuseDistance",
        "SetHostOffloadMinSize",
        "SetStartupConfig",
    ],
    features = ["-use_header_modules"],
    srcs = [ "dummy.cc" ],
    copts = [
        "-fno-strict-aliasing",
        "-fexceptions",
    ],
    deps = [
        ":cuda_utils",
        ":legate_computation",
        ":legate_utils",
        ":legate_pjrt_client",
        ":legate_c_api_client",
        ":zuku_execute_context_impl",
        "//xla:xla_data_proto_cc",
        "//xla/pjrt/gpu:se_gpu_pjrt_client",
        "//xla/service/cpu:cpu_executable",
        "//xla/service:cpu_plugin",
        "//xla/service:buffer_assignment",
        "//xla/tools:hlo_module_loader",
        "//xla/pjrt:pjrt_api",
        "//xla/pjrt:tfrt_cpu_pjrt_client",
        "//xla/service/gpu:gpu_executable",
        "//xla/service/gpu/model:gpu_hlo_cost_analysis",
        "//xla/service:gpu_plugin",
        "//xla/pjrt/gpu:nccl_id_store",
        "//xla/tsl/cuda",  # buildcleaner: keep
        "//xla/tsl/cuda:cudart",  # buildcleaner: keep
        "//xla/stream_executor/cuda:cublas_plugin",
        "//xla/stream_executor/cuda:cuda_platform",
        "//xla/stream_executor/cuda:cudnn_plugin",
        "//xla/stream_executor/cuda:cufft_plugin",
        "//xla/service/gpu:gpu_executable_run_options",
        "@tsl//tsl/platform:bfloat16",
        "//xla:xla_data_proto_cc_impl",
        "//xla/python:config",
        "//xla/python:custom_call_sharding",
        "//xla/python:dlpack",
        "//xla/python:guard_lib",
        "//xla/python:jax_jit",
        "//xla/python:logging",
        "//xla/python:mlir",
        "//xla/python:nb_absl_flat_hash_map",
        "//xla/python:nb_absl_span",
        "//xla/python:nb_class_ptr",
        "//xla/python:ops",
        "//xla/python:pjit",
        "//xla/python:pmap_lib",
        "//xla/python:pprof_profile_builder",
        "//xla/python:profiler",
        "//xla/python:py_client",
        "//xla/python:python_ref_manager",
        "//xla/python:pytree",
        "//xla/python:refine_polymorphic_shapes",
        "//xla/python:sdy",
        "//xla/python:traceback",
        "//xla/python:types",
        "//xla/python:util",
        "//xla/python:weakref_lru_cache",
        "//xla/python:xla_compiler",
        # placeholder for index annotation deps
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/log:initialize",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@nanobind",
        "@local_config_python//:python_headers",  # buildcleaner: keep
        "//xla:literal",
        "//xla:shape_util",
        "//xla:types",
        "//xla:util",
        "//xla/backends/cpu/collectives:cpu_collectives",
        "//xla/ffi:ffi_api",
        "//xla/pjrt:exceptions",
        "//xla/pjrt:mlir_to_hlo",
        "//xla/pjrt:pjrt_c_api_client",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_common",
        "//xla/pjrt:pjrt_compiler",
        "//xla/pjrt:pjrt_executable",
        "//xla/pjrt:pjrt_layout",
        "//xla/pjrt:status_casters",
        "//xla/pjrt/c:pjrt_c_api_hdrs",
        "//xla/pjrt/distributed",
        "//xla/pjrt/distributed:client",
        "//xla/pjrt/distributed:key_value_store_interface",
        "//xla/pjrt/distributed:protocol_proto_cc",
        "//xla/pjrt/distributed:service",
        "//xla/pjrt/plugin/xla_cpu:cpu_client_options",
        "//xla/pjrt/plugin/xla_cpu:xla_cpu_pjrt_client",
        "//xla/python/ifrt",
        "//xla/python/ifrt:plugin_program",
        "//xla/python/ifrt:plugin_program_serdes",
        "//xla/python/ifrt_proxy/client:py_module",
        "//xla/python/pjrt_ifrt",
        "//xla/python/pjrt_ifrt:pjrt_attribute_map_util",
        "//xla/python/pjrt_ifrt:xla_ifrt",
        "//xla/tsl/concurrency:ref_count",
        "//xla/tsl/distributed_runtime/preemption:preemption_sync_manager",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:status",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform/cloud:gcs_file_system",
        "//xla/tsl/python/lib/core:numpy",
        "@tsl//tsl/platform",
    ],
)

cc_library(
    name = "legate_mock_legate_xla",
    srcs = ["legate_mock_legate_xla.cc"],
    hdrs = ["legate_mock_legate_xla.h"],
    deps = [
        ":legate_utils",
        ":legate_computation",
        ":zuku_execute_context",
    ],
    testonly = True,
)

cc_library(
    name = "legate_test_base",
    srcs = ["legate_test_base.cc"],
    hdrs = ["legate_test_base.h"],
    deps = [
        ":legate_utils",
        ":legate_pjrt_client",
        ":legate_mock_legate_xla",
        ":hlo_partition",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/tests:hlo_test_base",
    ],
    testonly = True,
)

cc_library(
    name = "mpmd_test_base",
    srcs = ["mpmd_test_base.cc"],
    hdrs = ["mpmd_test_base.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/tests:hlo_test_base",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/builder:xla_computation",
        ":hlo_partition",
        ":mpmd_instruction",
        ":mpmd_partition",
        ":mpmd_store",
        ":json_utils",
        "@zuku",
    ],
    testonly = True
)

xla_cc_test(
    name = "legate_pjrt_executable_test",
    srcs = ["legate_pjrt_executable_test.cc"],
    deps = [
        ":legate_mock_legate_xla",
        ":legate_pjrt_client",
        ":hlo_partition",
        ":mpmd_test_base",
        ":legate_test_base",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/hlo/parser:hlo_parser",
        "@tsl//tsl/platform:test_main",
        "//xla/service:cpu_plugin",
        "//xla/service:gpu_plugin",
    ],
)

xla_cc_test(
    name = "legate_pjrt_client_test",
    srcs = ["legate_pjrt_client_test.cc"],
    deps = [
        ":legate_mock_legate_xla",
        ":legate_pjrt_client",
        ":hlo_partition",
        ":legate_test_base",
        ":mpmd_test_base",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_matchers",
        "@tsl//tsl/platform:test_main",
        "//xla/service:cpu_plugin",
        "//xla/service:gpu_plugin",
    ],
)

xla_cc_test(
    name = "mpmd_partition_test",
    srcs = ["mpmd_partition_test.cc"],
    deps = [
        ":mpmd_partition",
        ":mpmd_test_base",
        ":legate_mock_legate_xla",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/tests:hlo_test_base",
        "@tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "mpmd_coloring_test",
    srcs = ["mpmd_coloring_test.cc"],
    deps = [
        ":mpmd_coloring",
        ":hlo_partition",
        ":mpmd_test_base",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_matchers",
        "@tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "mpmd_logical_sharding_propagation_test",
    srcs = ["mpmd_logical_sharding_propagation_test.cc"],
    deps = [
        ":hlo_partition",
        ":mpmd_logical_sharding_propagation",
        ":mpmd_test_base",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_matchers",
        "@tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "mpmd_logical_to_gspmd_sharding_test",
    srcs = ["mpmd_logical_to_gspmd_sharding_test.cc"],
    deps = [
        ":hlo_partition",
        ":mpmd_coloring",
        ":mpmd_computation_grouper",
        ":mpmd_computation_inliner",
        ":mpmd_logical_to_gspmd_sharding",
        ":mpmd_test_base",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_matchers",
        "@tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "mpmd_computation_grouper_test",
    srcs = ["mpmd_computation_grouper_test.cc"],
    deps = [
        ":mpmd_computation_grouper",
        ":mpmd_computation_inliner",
        ":mpmd_coloring",
        ":hlo_partition",
        ":mpmd_test_base",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_matchers",
        "@tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "mpmd_computation_fusion_test",
    srcs = ["mpmd_computation_fusion_test.cc"],
    deps = [
        ":mpmd_computation_fusion",
        ":mpmd_coloring",
        ":hlo_partition",
        ":mpmd_test_base",
        "//xla/hlo/ir:hlo",
        "@tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "mpmd_hoist_loop_convert_test",
    srcs = ["mpmd_hoist_loop_convert_test.cc"],
    deps = [
        ":mpmd_hoist_loop_convert",
        ":hlo_partition",
        ":mpmd_test_base",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/hlo/transforms/simplifiers:hlo_dce",
        "@tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "mpmd_buffer_scheduling_name_test",
    srcs = ["mpmd_buffer_scheduling_name_test.cc"],
    deps = [
        ":mpmd_buffer_scheduling_name",
        ":hlo_partition",
        ":mpmd_test_base",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_matchers",
        "@tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "mpmd_loop_unroll_test",
    srcs = ["mpmd_loop_unroll_test.cc"],
    deps = [
        ":mpmd_loop_unroll",
        ":hlo_partition",
        ":mpmd_test_base",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_matchers",
        "@tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "mpmd_insert_reshard_test",
    srcs = ["mpmd_insert_reshard_test.cc"],
    deps = [
        ":mpmd_insert_reshard",
        ":hlo_partition",
        ":mpmd_test_base",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_matchers",
        "@tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "mpmd_reorder_shard_map_transpose_test",
    srcs = ["mpmd_reorder_shard_map_transpose_test.cc"],
    deps = [
        ":mpmd_reorder_shard_map_transpose",
        ":mpmd_hoist_shard_map_reduce",
        ":hlo_partition",
        ":mpmd_test_base",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_matchers",
        "@tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "mpmd_hoist_shard_map_reduce_test",
    srcs = ["mpmd_hoist_shard_map_reduce_test.cc"],
    deps = [
        ":mpmd_hoist_shard_map_reduce",
        ":hlo_partition",
        ":mpmd_test_base",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_matchers",
        "@tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "mpmd_shard_map_loop_reduce_test",
    srcs = ["mpmd_shard_map_loop_reduce_test.cc"],
    deps = [
        ":mpmd_shard_map_loop_reduce",
        ":mpmd_test_base",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_matchers",
        "@tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "mpmd_unused_param_output_remover_test",
    srcs = ["mpmd_unused_param_output_remover_test.cc"],
    deps = [
        ":mpmd_unused_param_output_remover",
        ":hlo_partition",
        ":mpmd_test_base",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_matchers",
        "@tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "mpmd_unused_loop_output_remover_test",
    srcs = ["mpmd_unused_loop_output_remover_test.cc"],
    deps = [
        ":mpmd_unused_loop_output_remover",
        ":hlo_partition",
        ":mpmd_test_base",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_matchers",
        "@tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "mpmd_microbatch_loop_canonicalizer_test",
    srcs = ["mpmd_microbatch_loop_canonicalizer_test.cc"],
    deps = [
        ":mpmd_microbatch_loop_canonicalizer",
        ":hlo_partition",
        ":mpmd_test_base",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_matchers",
        "@tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "mpmd_utils_test",
    srcs = ["mpmd_utils_test.cc"],
    deps = [
        ":hlo_partition",
        ":mpmd_utils",
        ":mpmd_test_base",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_matchers",
        "@tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "mpmd_cross_task_barrier_remover_test",
    srcs = ["mpmd_cross_task_barrier_remover_test.cc"],
    deps = [
        ":mpmd_cross_task_barrier_remover",
        ":mpmd_test_base",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_matchers",
        "@tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "mpmd_sharding_propagation_test",
    srcs = ["mpmd_sharding_propagation_test.cc"],
    deps = [
        ":mpmd_sharding_propagation",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/service/gpu/transforms:reduce_scatter_creator",
        "@tsl//tsl/platform:test_main",
        "@zuku",
        ":hlo_partition",
        ":mpmd_test_base",
        ":mpmd_coloring",
        ":mpmd_logical_to_gspmd_sharding",
        ":mpmd_logical_sharding_propagation",
        ":mpmd_computation_grouper",
    ],
)

xla_cc_test(
    name = "mpmd_cut_size_minimizer_test",
    srcs = ["mpmd_cut_size_minimizer_test.cc"],
    deps = [
        ":mpmd_cut_size_minimizer",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_matchers",
        "@tsl//tsl/platform:test_main",
        "@zuku",
        ":hlo_partition",
        ":mpmd_test_base",
    ],
)

xla_cc_test(
    name = "mpmd_instruction_delay_recolor_test",
    srcs = ["mpmd_instruction_delay_recolor_test.cc"],
    deps = [
        ":mpmd_instruction_delay_recolor",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_matchers",
        "@tsl//tsl/platform:test_main",
        "@zuku",
        ":hlo_partition",
        ":mpmd_test_base",
    ],
)

xla_cc_test(
    name = "mpmd_argument_recompute_test",
    srcs = ["mpmd_argument_recompute_test.cc"],
    deps = [
        ":mpmd_argument_recompute",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_matchers",
        "@tsl//tsl/platform:test_main",
        "@zuku",
        ":hlo_partition",
        ":mpmd_test_base",
    ],
)

xla_cc_test(
    name = "mpmd_simple_loop_increment_coloring_test",
    srcs = ["mpmd_simple_loop_increment_coloring_test.cc"],
    deps = [
        ":mpmd_simple_loop_increment_coloring",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_matchers",
        "@tsl//tsl/platform:test_main",
        "@zuku",
        ":hlo_partition",
        ":mpmd_test_base",
    ],
)

xla_cc_test(
    name = "mpmd_unpack_optimization_barrier_test",
    srcs = ["mpmd_unpack_optimization_barrier_test.cc"],
    deps = [
        ":mpmd_unpack_optimization_barrier",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_matchers",
        "@tsl//tsl/platform:test_main",
        ":mpmd_test_base",
    ],
)

xla_cc_test(
    name = "mpmd_repack_optimization_barrier_test",
    srcs = ["mpmd_repack_optimization_barrier_test.cc"],
    deps = [
        ":mpmd_repack_optimization_barrier",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_matchers",
        "@tsl//tsl/platform:test_main",
        ":mpmd_test_base",
    ],
)

xla_cc_test(
    name = "mpmd_all_reduce_hoister_test",
    srcs = ["mpmd_all_reduce_hoister_test.cc"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/service/gpu/transforms:reduce_scatter_creator",
        "@tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "legate_sharding_test",
    srcs = ["legate_sharding_test.cc"],
    deps = [
        ":legate_sharding",
        "@tsl//tsl/platform:test_main",
        "//xla:xla_data_proto_cc",
    ],
)

xla_cc_test(
    name = "loop_scheduler_test",
    srcs = ["loop_scheduler_test.cc"],
    deps = [
        ":loop_scheduler",
        ":legate_mock_legate_xla",
        "//xla/hlo/parser:hlo_parser",
        "//xla/tests:hlo_test_base",
        "@tsl//tsl/platform:test_main",
    ],
)

xla_cc_test(
    name = "legate_buffer_action_test",
    srcs = ["legate_buffer_action_test.cc"],
    deps = [
        ":legate_buffer_action",
        "//xla/pjrt/gpu:se_gpu_pjrt_client",
        "@xla//xla/tsl/platform:test",
        "@tsl//tsl/platform:test_main",
        "//xla/tsl/cuda",  # buildcleaner: keep
        "//xla/tsl/cuda:cudart",  # buildcleaner: keep
        "//xla/service:gpu_plugin",
    ],
)
