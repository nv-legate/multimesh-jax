#include "xla_task.h"

#include "cuda.h"
#include "legate_xla_common.h"
#include "task_utils.h"

namespace legate_xla {

void CopyDeviceToDevice(const void *src, zuku::ShardedArray& array)
{
  //TODO: fill this in
  throw std::runtime_error("CopyDeviceToDevice: unimplemented");
}

void ApplyStoreBufferAction(int64_t local_device_id, BufferAction* action, zuku::ShardedArray& array){
  //TODO: implement this
  throw std::runtime_error("StoreBufferAction: unimplemented");
}

} // namespace legate_xla
