#ifndef _XLA_PJRT_LEGATE_STORE_HANDLE_FWD_H
#define _XLA_PJRT_LEGATE_STORE_HANDLE_FWD_H

#include <memory>

namespace xla {

struct StoreHandleImpl;

struct StoreHandle {
  // the default constructor has to be defined with the impl struct
  ~StoreHandle();
  std::shared_ptr<StoreHandleImpl> impl{nullptr};
  int64_t unique_id{-1};
};

}  // namespace xla

#endif