/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/legate/python_callback.h"

#include "Python.h"
#include "xla/pjrt/legate/mpmd_instruction.h"

namespace xla {

namespace {
PyObject* PyTasksToList(
    const std::vector<std::vector<HloInstruction*>>& tasks) {
  PyObject* list = PyList_New(tasks.size());
  for (int i = 0; i < tasks.size(); i++) {
    PyObject* task_list = PyList_New(tasks[i].size());
    for (int j = 0; j < tasks[i].size(); j++) {
      PyList_SetItem(task_list, j, PyLong_FromVoidPtr(tasks[i][j]));
    }
    PyList_SetItem(list, i, task_list);
  }
  return list;
}

std::vector<HloInstruction*> PyListToSchedule(PyObject* list) {
  std::vector<HloInstruction*> schedule;
  schedule.reserve(PyList_Size(list));
  for (int i = 0; i < PyList_Size(list); i++) {
    // task is a pointer.
    PyObject* task = PyList_GetItem(list, i);
    HloInstruction* hlo_inst =
        static_cast<HloInstruction*>(PyLong_AsVoidPtr(task));
    schedule.push_back(hlo_inst);
  }
  return schedule;
}
}  // namespace

absl::StatusOr<std::vector<HloInstruction*>> CallCustomPythonCallback(
    const HloPartition& partition, const LoopConfig& config,
    const std::vector<std::vector<HloInstruction*>>& tasks) {
  bool python_initialized = Py_IsInitialized();
  if (!python_initialized) {
    // We could already be running inside a Python interpreter (inception!!)
    // so we don't need to initialize it again.
    Py_Initialize();
  }

  PyGILState_STATE state = PyGILState_Ensure();  // Acquire the GIL incase
  // we're running inside a larger Python interpreter.
  // https://github.com/python/cpython/issues/89076
  PyObject* module_handle = PyModule_New("embedded_scheduling_callback");

  PyObject* global_dict = PyModule_GetDict(module_handle);
  PyObject* local_dict = PyDict_New();

  PyObject* tasks_list = PyTasksToList(tasks);
  PyDict_SetItemString(local_dict, "tasks", tasks_list);

  // Set useful values from config to the local dict
  PyObject* config_dict = PyDict_New();
  PyDict_SetItemString(config_dict, "num_iterations",
                       Py_BuildValue("L", config.num_iterations));
  PyDict_SetItemString(config_dict, "unique_id",
                       Py_BuildValue("L", config.unique_id));
  PyDict_SetItemString(config_dict, "microbatch_size",
                       Py_BuildValue("i", config.microbatch_size));
  PyDict_SetItemString(config_dict, "microbatch_dim",
                       Py_BuildValue("i", config.microbatch_dim));
  PyDict_SetItemString(config_dict, "slice_dim",
                       Py_BuildValue("i", config.slice_dim));
  if (config.interleave.has_value())
    PyDict_SetItemString(config_dict, "interleave",
                         Py_BuildValue("i", config.interleave.value()));
  if (config.num_stages.has_value())
    PyDict_SetItemString(config_dict, "num_stages",
                         Py_BuildValue("i", config.num_stages.value()));
  if (config.unrolling.has_value())
    PyDict_SetItemString(config_dict, "unrolling",
                         Py_BuildValue("i", config.unrolling.value()));

  // Set useful values computed from the partition to the local dict
  int64_t total_num_devices = partition.TotalDevices();
  PyDict_SetItemString(config_dict, "total_num_devices",
                       Py_BuildValue("L", total_num_devices));
  absl::flat_hash_map<zuku::DeviceList, int64_t> unique_groups;
  int num_pipeline_tasks = 0;
  for (auto&& task : tasks[0]) {
    auto color = Color(task);
    auto devices = partition.DevicesForColor(*color);
    if (devices.size() != total_num_devices) {
      // don't count tasks over the global set of devices
      unique_groups[devices] += 1;
      ++num_pipeline_tasks;
    }
  }
  PyDict_SetItemString(config_dict, "num_pipeline_tasks",
                       Py_BuildValue("i", num_pipeline_tasks));
  PyDict_SetItemString(
      config_dict, "num_groups",
      Py_BuildValue("i", std::max<int>(1, unique_groups.size())));

  PyDict_SetItemString(local_dict, "config", config_dict);

  PyRun_String(config.custom_callback->c_str(), Py_file_input, global_dict,
               local_dict);
  PyObject* result = PyDict_GetItemString(local_dict, "schedule");

  std::vector<HloInstruction*> schedule = PyListToSchedule(result);

  PyGILState_Release(state);
  if (!python_initialized) {
    Py_Finalize();
  }

  return schedule;
}

}  // namespace xla