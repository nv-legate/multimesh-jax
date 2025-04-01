#ifndef _XLA_PJRT_LEGATE_STORE_HANDLE_H
#define _XLA_PJRT_LEGATE_STORE_HANDLE_H

#include "src/zuku/tiled_array.h"
#include "xla/pjrt/legate/store_handle_fwd.h"

namespace xla {

struct StoreHandleImpl;

struct StoreHandleImpl {
  zuku::Store<zuku::ShardedArray> array;
  std::string name;
  ~StoreHandleImpl();
};

}  // namespace xla

#endif