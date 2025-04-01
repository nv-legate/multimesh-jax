#ifndef XLA_PJRT_LEGATE_MPMD_STORE_H_
#define XLA_PJRT_LEGATE_MPMD_STORE_H_

#include <string>

#include "src/zuku/shape.h"
#include "xla/hlo/ir/hlo_sharding.h"

namespace xla {

struct Store {
  enum class Type { PARAM, ROOT, TEMP };
  Type type;
  int64_t index{-1};  // minus 1 for fuzzing
  std::string name;
  bool scalar{false};
  zuku::ShardedShape sharded_shape;
  HloSharding mpmd_sharding;
  Shape shape;
};

inline std::ostream& operator<<(std::ostream& os, Store::Type type) {
  switch (type) {
    case Store::Type::PARAM:
      os << "PARAM";
      break;
    case Store::Type::ROOT:
      os << "ROOT";
      break;
    case Store::Type::TEMP:
      os << "TEMP";
      break;
  }
  return os;
}

}  // namespace xla

#endif  // XLA_PJRT_LEGATE_MPMD_STORE_H_