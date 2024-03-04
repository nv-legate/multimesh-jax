#include <iostream>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

#include "legate_to_xla.h"

extern "C" void RegisterImplicitTask(
    std::string matcher, std::vector<int64_t> devices,
    std::vector<int64_t> dims, std::vector<std::string> axes,
    std::vector<std::pair<std::string, std::string>> logical_axes);

extern "C" void SetEnableImplicitTasks(bool flag);

extern "C" void RegisterImplicitTaskWithFactory(
    std::string matcher,
    std::function<std::vector<int64_t>(const std::string &task)> device_factory,
    std::vector<int64_t> dims, std::vector<std::string> axes,
    std::vector<std::pair<std::string, std::string>> logical_axes);

extern "C" void UnregisterImplicitTask(std::string matcher);

extern "C" void EnableFastPath();

extern "C" void DisableFastPath();

extern "C" void ClearImplicitTasks();

namespace py = pybind11;

extern "C" void LegateShutdown();

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
  m.def("shutdown", []() { LegateShutdown(); });
  m.def("no_op_custom_call",
        []() { return EncapsulateFunction(no_op_entrypoint); });
  m.def("enable_implicit_tasks", [] { SetEnableImplicitTasks(true); });
  m.def("disable_implicit_tasks", [] { SetEnableImplicitTasks(false); });
  m.def(
      "register_task",
      [](py::str task_regex, py::list py_devices, py::list py_device_dims,
         py::list py_device_axes, py::list py_logical_axes) {
        RegisterImplicitTask(
            task_regex.cast<std::string>(),
            py_devices.cast<std::vector<int64_t>>(),
            py_device_dims.cast<std::vector<int64_t>>(),
            py_device_axes.cast<std::vector<std::string>>(),
            py_logical_axes
                .cast<std::vector<std::pair<std::string, std::string>>>());
      },
      py::arg("task_regex"), py::arg("devices"), py::arg("dims"),
      py::arg("device_axes"), py::arg("logical_axes"));
  m.def(
      "register_task_factory",
      [](py::str task_regex,
         std::function<std::vector<int64_t>(const std::string &)>
             device_callback,
         py::list py_device_dims, py::list py_device_axes,
         py::list py_logical_axes) {
        RegisterImplicitTaskWithFactory(
            task_regex.cast<std::string>(), device_callback,
            py_device_dims.cast<std::vector<int64_t>>(),
            py_device_axes.cast<std::vector<std::string>>(),
            py_logical_axes
                .cast<std::vector<std::pair<std::string, std::string>>>());
      },
      py::arg("task_regex"), py::arg("device_callback"), py::arg("dims"),
      py::arg("device_axes"), py::arg("logical_axes"));
  m.def("unregister_task", [](py::str task_regex) {
    UnregisterImplicitTask(task_regex.cast<std::string>());
  });
  m.def("clear_tasks", []() { ClearImplicitTasks(); });
  m.def(
      "compile_hlo_module",
      [](std::string path, std::string platform, int replica_count,
         int num_partitions, bool erase_sharding, int device_mem_gb) {
        legate_xla::CompileConfig config{.replica_count = replica_count,
                                         .num_partitions = num_partitions,
                                         .print_stats = true,
                                         .erase_sharding = erase_sharding};

        std::optional<int64_t> device_mem;
        if (device_mem_gb != 0) {
          device_mem = int64_t(device_mem_gb) * 1000000000ULL;
        }
        CompileHloModuleFromFile(path, platform, replica_count, num_partitions,
                                 erase_sharding, device_mem);
      },
      py::arg("path"), py::arg("platform") = "gpu",
      py::arg("replica_count") = 1, py::arg("num_partitions") = 1,
      py::arg("erase_sharding") = false, py::arg("device_mem_gb") = 0);
  m.def(
      "enable_fast_path",
      [](bool enable) {
        if (enable) {
          EnableFastPath();
        } else {
          DisableFastPath();
        }
      },
      py::arg("enable"));
}
