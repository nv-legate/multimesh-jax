#include "xla/pjrt/legate/legate_device_assignment.h"

namespace xla {

absl::StatusOr<DeviceAssignment> LegateToXlaDeviceAssignment(
    const LegateDeviceAssignment& da) {
  xla::DeviceAssignment xla_device_assignment(da.ReplicaCount(),
                                              da.NumPartitions());
  int idx = 0;
  for (int r = 0; r < da.ReplicaCount(); ++r) {
    for (int c = 0; c < da.NumPartitions(); ++c, ++idx) {
      xla_device_assignment(r, c) = da.GlobalDeviceId(r, c);
    }
  }
  return std::move(xla_device_assignment);
}

}  // namespace xla