#include <iostream>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <string>

#include "zuku/init.h"

extern "C" void SetStartupConfig(zuku::RealmConfig cfg);

extern "C" void CompileHloModuleFromFile(const std::string &hlo_file,
                                         const std::string &platform_name,
                                         int replica_count, int num_partitions,
                                         bool erase_sharding, bool autoshard,
                                         std::optional<int64_t> device_mem);

extern "C" void LegateShutdown();

extern "C" void RegisterImplicitTask(
    std::string matcher, std::vector<int64_t> devices,
    std::vector<int64_t> dims, std::vector<std::string> axes,
    std::vector<std::pair<std::string, std::string>> logical_axes,
    int64_t fusion_color, std::optional<int64_t> loop_submesh_size,
    bool loop_submesh_reverse);

extern "C" void SetEnableImplicitTasks(bool flag);

extern "C" void RegisterImplicitTaskWithFactory(
    std::string matcher,
    std::function<std::vector<int64_t>(const std::string &task)> device_factory,
    std::vector<int64_t> dims, std::vector<std::string> axes,
    std::vector<std::pair<std::string, std::string>> logical_axes,
    int64_t fusion_color);

extern "C" void UnregisterImplicitTask(std::string matcher);

extern "C" void EnableFastPath(bool enable);

extern "C" void EnableOnlyFuseLoopTasks(bool enable);

extern "C" void EnableTaskFusion(bool enable);

extern "C" void ClearImplicitTasks();

extern "C" void EnableLegateRecomputation(bool enable);

extern "C" void ReplicateParametersSmallerThanNumElements(int64_t num_elements);

extern "C" void RecomputeArgumentsIfCostLessThan(int64_t cost);

extern "C" void SetStoreCacheMinParallelism(int64_t parallelism);

extern "C" void SetHostOffloadMinReuseDistance(int64_t reuse_distance);

extern "C" void SetHostOffloadMinSize(int64_t size);

namespace py = pybind11;

template <class To, class From>
typename std::enable_if<sizeof(To) == sizeof(From) &&
                            std::is_trivially_copyable<From>::value &&
                            std::is_trivially_copyable<To>::value,
                        To>::type
bit_cast(const From &src) noexcept {
  static_assert(std::is_trivially_constructible<To>::value,
                "This implementation additionally requires destination type to "
                "be trivially constructible");

  To dst;
  memcpy(&dst, &src, sizeof(To));
  return dst;
}

template <typename T> pybind11::bytes PackDescriptor(const T &descriptor) {
  return pybind11::bytes(PackDescriptorAsString(descriptor));
}

template <typename T> pybind11::capsule EncapsulateFunction(T *fn) {
  return pybind11::capsule(bit_cast<void *>(fn), "xla._CUSTOM_CALL_TARGET");
}

void no_op_entrypoint() {}

struct LogicalAxis {
  std::string logical_name;
  std::string device_name;
};

PYBIND11_MODULE(legate_jax_impl, m) {
  m.def("shutdown", []() {
    py::gil_scoped_release release;
    LegateShutdown();
  });
  m.def("no_op_custom_call",
        []() { return EncapsulateFunction(no_op_entrypoint); });
  m.def("enable_implicit_tasks",
        [](bool flag) { SetEnableImplicitTasks(flag); });
  m.def(
      "_register_task",
      [](py::str task_regex, py::list py_devices, py::list py_device_dims,
         py::list py_device_axes, py::list py_logical_axes,
         int64_t fusion_color, std::optional<int64_t> loop_submesh_size,
         bool loop_submesh_reverse) {
        RegisterImplicitTask(
            task_regex.cast<std::string>(),
            py_devices.cast<std::vector<int64_t>>(),
            py_device_dims.cast<std::vector<int64_t>>(),
            py_device_axes.cast<std::vector<std::string>>(),
            py_logical_axes
                .cast<std::vector<std::pair<std::string, std::string>>>(),
            fusion_color, loop_submesh_size, loop_submesh_reverse);
      },
      py::arg("task_regex"), py::arg("devices"), py::arg("dims"),
      py::arg("device_axes"), py::arg("logical_axes"),
      py::arg("fusion_color") = 0, py::arg("loop_submesh_size") = py::none(),
      py::arg("loop_submesh_reverse") = false);
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
      "_register_task_factory",
      [](py::str task_regex,
         std::function<std::vector<int64_t>(const std::string &)>
             device_callback,
         py::list py_device_dims, py::list py_device_axes,
         py::list py_logical_axes, int64_t fusion_color) {
        RegisterImplicitTaskWithFactory(
            task_regex.cast<std::string>(), device_callback,
            py_device_dims.cast<std::vector<int64_t>>(),
            py_device_axes.cast<std::vector<std::string>>(),
            py_logical_axes
                .cast<std::vector<std::pair<std::string, std::string>>>(),
            fusion_color);
      },
      py::arg("task_regex"), py::arg("device_callback"), py::arg("dims"),
      py::arg("device_axes"), py::arg("logical_axes"),
      py::arg("fusion_color") = 0);
  m.def("unregister_task", [](py::str task_regex) {
    UnregisterImplicitTask(task_regex.cast<std::string>());
  });
  m.def("clear_tasks", []() { ClearImplicitTasks(); });
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
      "enable_only_fuse_loop_tasks",
      [](bool enable) { EnableOnlyFuseLoopTasks(enable); }, py::arg("enable"));
  m.def(
      "enable_task_fusion", [](bool enable) { EnableTaskFusion(enable); },
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
      "set_store_cache_min_parallelism",
      [](int64_t parallelism) { SetStoreCacheMinParallelism(parallelism); },
      py::arg("parallelism"));
  m.def(
      "enable_recomputation",
      [](bool enable) { EnableLegateRecomputation(enable); },
      py::arg("enable"));
  m.def(
      "set_host_offload_min_reuse_distance",
      [](int64_t reuse_distance) {
        SetHostOffloadMinReuseDistance(reuse_distance);
      },
      py::arg("reuse_distance"));
  m.def(
      "set_host_offload_min_size",
      [](int64_t size) { SetHostOffloadMinSize(size); }, py::arg("size"));
}
