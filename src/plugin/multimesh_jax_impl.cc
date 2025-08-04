/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#define PYBIND11_DETAILED_ERROR_MESSAGES
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include <string>
#include <tuple>

#include "zuku/init.h"
#include "pycallback_types.h"

namespace {

template <typename Struct, typename... Args, std::size_t... indices>
Struct tuple_to_struct(std::tuple<Args...> t, std::index_sequence<indices...>) {
  return {std::get<indices>(t)...};
}

template <typename Struct, typename... Args>
Struct tuple_to_struct(std::tuple<Args...> t) {
  return tuple_to_struct<Struct, Args...>(t,
                                          std::index_sequence_for<Args...>{});
}

}  // namespace

extern "C" void SetStartupConfig(zuku::RealmConfig cfg);

extern "C" void CompileHloModuleFromFile(const std::string& hlo_file,
                                         const std::string& platform_name,
                                         int replica_count, int num_partitions,
                                         bool erase_sharding, bool autoshard,
                                         std::optional<int64_t> device_mem);

extern "C" void ZukuShutdown();

extern "C" void SetEnableMetadataNameTasks(bool flag);

extern "C" void EnableFastPath(bool enable);

extern "C" void ClearMetadataNameTasks();

extern "C" void PushMetadataNameTaskContext();

extern "C" void PopMetadataNameTaskContext();

extern "C" void EnableMultiMeshRecomputation(bool enable);

extern "C" void ReplicateParametersSmallerThanNumElements(int64_t num_elements);

extern "C" void RecomputeArgumentsIfCostLessThan(int64_t cost);

extern "C" void SetHostOffloadMinReuseDistance(int64_t reuse_distance);

extern "C" void SetHostOffloadMinSize(int64_t size);

namespace py = pybind11;

template <class To, class From>
typename std::enable_if<sizeof(To) == sizeof(From) &&
                            std::is_trivially_copyable<From>::value &&
                            std::is_trivially_copyable<To>::value,
                        To>::type
bit_cast(const From& src) noexcept {
  static_assert(std::is_trivially_constructible<To>::value,
                "This implementation additionally requires destination type to "
                "be trivially constructible");

  To dst;
  memcpy(&dst, &src, sizeof(To));
  return dst;
}

template <typename T>
pybind11::bytes PackDescriptor(const T& descriptor) {
  return pybind11::bytes(PackDescriptorAsString(descriptor));
}

template <typename T>
pybind11::capsule EncapsulateFunction(T* fn) {
  return pybind11::capsule(bit_cast<void*>(fn), "xla._CUSTOM_CALL_TARGET");
}

void no_op_entrypoint() {}

struct LogicalAxis {
  std::string logical_name;
  std::string device_name;
};

PYBIND11_MODULE(multimesh_jax_impl, m) {
  m.def("shutdown", []() {
    py::gil_scoped_release release;
    ZukuShutdown();
  });
  m.def("no_op_custom_call",
        []() { return EncapsulateFunction(no_op_entrypoint); });
  m.def("enable_metadata_name_tasks",
        [](bool flag) { SetEnableMetadataNameTasks(flag); });
  m.def(
      "set_startup_config",
      [](std::optional<int> cpus, std::optional<int> gpus,
         std::optional<int64_t> fbmem, std::optional<int64_t> sysmem,
         std::optional<int64_t> zcmem, std::optional<std::string> network,
         bool kthreads, std::vector<std::string> argv, bool profile) {
        SetStartupConfig({.cpus = cpus,
                          .gpus = gpus,
                          .sysmem = sysmem,
                          .fbmem = fbmem,
                          .zcmem = zcmem,
                          .network = network.value_or("default"),
                          .kthreads = kthreads,
                          .profile = profile,
                          .argv = argv});
      },
      py::arg("cpus") = py::none(), py::arg("gpus") = py::none(),
      py::arg("fbmem") = py::none(), py::arg("sysmem") = py::none(),
      py::arg("zcmem") = py::none(), py::arg("network") = py::none(),
      py::arg("kthreads") = false, py::arg("argv") = std::vector<std::string>{},
      py::arg("profile") = false);
  m.def(
      "register_task_factory",
      [](std::string task_regex,
         std::function<multimesh::TaskOptionsTuple(const std::string&, bool)>
             cpp_callback) {
        auto wrapped_callback = [=](const std::string& name,
                                    bool backprop) -> multimesh::TaskOptions {
          auto tuple = cpp_callback(name, backprop);
          return tuple_to_struct<multimesh::TaskOptions>(std::move(tuple));
        };
        RegisterMetadataNameTask(task_regex, wrapped_callback);
      },
      py::arg("task_regex"), py::arg("callback"));
  m.def("pop_task_context", []() { PopMetadataNameTaskContext(); });
  m.def("push_task_context", []() { PushMetadataNameTaskContext(); });
  m.def("clear_tasks", []() { ClearMetadataNameTasks(); });
  m.def("clear_tasks", []() { ClearMetadataNameTasks(); });
  m.def(
      "compile_hlo_module",
      [](std::string path, std::string platform, int replica_count,
         int num_partitions, bool erase_sharding, bool autoshard,
         int device_mem_gb) {
        py::gil_scoped_release release;
        std::optional<int64_t> device_mem;
        if (device_mem_gb != 0) {
          device_mem = int64_t(device_mem_gb) * 1000000000ULL;
        }
        CompileHloModuleFromFile(path, platform, replica_count, num_partitions,
                                 erase_sharding, autoshard, device_mem);
      },
      py::arg("path"), py::arg("platform") = "gpu",
      py::arg("replica_count") = 1, py::arg("num_partitions") = 1,
      py::arg("erase_sharding") = false, py::arg("autoshard") = true,
      py::arg("device_mem_gb") = 0);
  m.def(
      "enable_fast_path", [](bool enable) { EnableFastPath(enable); },
      py::arg("enable"));
  m.def(
      "replicate_parameters_smaller_than_num_elements",
      [](int64_t num_elements) {
        ReplicateParametersSmallerThanNumElements(num_elements);
      },
      py::arg("num_elements"));
  m.def(
      "recompute_from_arguments_if_cost_less_than",
      [](int64_t cost) { ReplicateParametersSmallerThanNumElements(cost); },
      py::arg("cost"));
  m.def(
      "enable_recomputation",
      [](bool enable) { EnableMultiMeshRecomputation(enable); },
      py::arg("enable"));
}
