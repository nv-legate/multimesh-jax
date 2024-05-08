#include <iostream>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

#include "legate_to_xla.h"

extern "C" void RegisterImplicitTask(
    std::string matcher, std::vector<int64_t> devices,
    std::vector<int64_t> dims, std::vector<std::string> axes,
    std::vector<std::pair<std::string, std::string>> logical_axes,
    int64_t fusion_color);

extern "C" void SetEnableImplicitTasks(bool flag);

extern "C" void RegisterImplicitTaskWithFactory(
    std::string matcher,
    std::function<std::vector<int64_t>(const std::string &task)> device_factory,
    std::vector<int64_t> dims, std::vector<std::string> axes,
    std::vector<std::pair<std::string, std::string>> logical_axes,
    int64_t fusion_color);

extern "C" void UnregisterImplicitTask(std::string matcher);

extern "C" void EnableFastPath();

extern "C" void DisableFastPath();

extern "C" void ClearImplicitTasks();

extern "C" void EnableLegateTracing();

extern "C" void DisableLegateTracing();

extern "C" void EnableLegateRecomputation();

extern "C" void DisableLegateRecomputation();

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
  m.def("shutdown", []() {
    py::gil_scoped_release release;
    LegateShutdown();
  });
  m.def("no_op_custom_call",
        []() { return EncapsulateFunction(no_op_entrypoint); });
  m.def("enable_implicit_tasks", [] { SetEnableImplicitTasks(true); });
  m.def("disable_implicit_tasks", [] { SetEnableImplicitTasks(false); });
  m.def(
      "register_task",
      [](py::str task_regex, py::list py_devices, py::list py_device_dims,
         py::list py_device_axes, py::list py_logical_axes,
         int64_t fusion_color) {
        RegisterImplicitTask(
            task_regex.cast<std::string>(),
            py_devices.cast<std::vector<int64_t>>(),
            py_device_dims.cast<std::vector<int64_t>>(),
            py_device_axes.cast<std::vector<std::string>>(),
            py_logical_axes
                .cast<std::vector<std::pair<std::string, std::string>>>(),
            fusion_color);
      },
      py::arg("task_regex"), py::arg("devices"), py::arg("dims"),
      py::arg("device_axes"), py::arg("logical_axes"),
      py::arg("fusion_color") = 0);
  m.def(
      "register_task_factory",
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
      "enable_fast_path",
      [](bool enable) {
        if (enable) {
          EnableFastPath();
        } else {
          DisableFastPath();
        }
      },
      py::arg("enable"));
  m.def(
      "enable_tracing",
      [](bool enable) {
        if (enable) {
          EnableLegateTracing();
        } else {
          DisableLegateTracing();
        }
      },
      py::arg("enable"));
  m.def(
      "enable_recomputation",
      [](bool enable) {
        if (enable) {
          EnableLegateRecomputation();
        } else {
          DisableLegateRecomputation();
        }
      },
      py::arg("enable"));
}
