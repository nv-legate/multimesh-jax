/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef XLA_PJRT_LEGATE_LEGATE_DEVICE_ASSIGNMENT_H_
#define XLA_PJRT_LEGATE_LEGATE_DEVICE_ASSIGNMENT_H_

#include <cstdint>

#include "src/zuku/mesh.h"
#include "xla/service/computation_placer.h"

namespace xla {

struct LegateDeviceAssignmentConfig {
  int64_t local_device_id = 0;
  int64_t global_device_id = 0;
  int64_t replica_count = 1;
  int64_t num_partitions = 1;
};

class LegateDeviceAssignment {
 public:
  explicit LegateDeviceAssignment(const LegateDeviceAssignmentConfig& config,
                                  zuku::DeviceList devices)
      : local_device_id_(config.local_device_id),
        global_device_id_(config.global_device_id),
        replica_count_(config.replica_count),
        num_partitions_(config.num_partitions),
        devices_(std::move(devices)) {}

  int64_t operator()(int replica, int partition) {
    return devices_[partition * replica_count_ + replica];
  }

  int64_t GlobalDeviceId() const { return global_device_id_; }

  int64_t LocalDeviceId() const { return local_device_id_; }

  int64_t ReplicaCount() const { return replica_count_; }

  int64_t NumPartitions() const { return num_partitions_; }

  int64_t GlobalDeviceId(int replica, int partition) const {
    return devices_[partition * replica_count_ + replica];
  }

 private:
  int64_t local_device_id_;
  int64_t global_device_id_;
  int64_t replica_count_;
  int64_t num_partitions_;
  zuku::DeviceList devices_;
};

absl::StatusOr<xla::DeviceAssignment> LegateToXlaDeviceAssignment(
    const LegateDeviceAssignment& da);

}  // namespace xla

#endif  // XLA_PJRT_LEGATE_LEGATE_DEVICE_ASSIGNMENT_H_
