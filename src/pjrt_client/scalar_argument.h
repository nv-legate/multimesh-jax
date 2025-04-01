#ifndef XLA_PJRT_LEGATE_SCALAR_ARGUMENT_H_
#define XLA_PJRT_LEGATE_SCALAR_ARGUMENT_H_

#include <cstdint>
#include <variant>

namespace xla {

struct ScalarArgument {
  using ValueVariant =
      std::variant<float, double, int32_t, int64_t, uint32_t, uint64_t>;
  ValueVariant value;
  int64_t parameter_number;
};

}  // namespace xla

#endif  // XLA_PJRT_LEGATE_LEGATE_DEVICE_ASSIGNMENT_H_