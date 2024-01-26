#include "xla_task.h"

#include "cuda.h"
#include "legate_xla_common.h"
#include "task_utils.h"
#include <core/utilities/dispatch.h>

namespace legate_xla {

namespace {

struct get_write_only_buffer_fn {
  template <legate::Type::Code TYPE_CODE, int32_t DIM>
  BufferAllocation operator()(legate::PhysicalStore &store) {
    using VAL = legate::type_of<TYPE_CODE>;
    auto shape = store.shape<DIM>();
    auto acc = store.write_accessor<VAL, DIM>();
    void *buffer = static_cast<void *>(acc.ptr(shape));
    size_t size =
        sizeof(legate::type_of<TYPE_CODE>) * store.domain().get_volume();
    return BufferAllocation{.buffer = buffer, .size = size};
  }
};

struct init_buffer_fn {
  template <legate::Type::Code TYPE_CODE, int32_t DIM>
  void operator()(legate::PhysicalStore &store) {
    using VAL = legate::type_of<TYPE_CODE>;
    auto shape = store.shape<DIM>();
    auto acc = store.write_accessor<VAL, DIM>();
    VAL *buffer = acc.ptr(shape);
    size_t size = store.domain().get_volume();
    cudaMemset(buffer, 0, size * sizeof(VAL));
  }
};

} // namespace

Legion::Logger log_xla("legate.xla");

/*static*/ legate::TaskRegistrar &Registry::get_registrar() {
  static legate::TaskRegistrar registrar;
  return registrar;
}

/*static*/ void
XLACopyDeviceToDevice::gpu_variant(legate::TaskContext context) {
  auto output_store = context.outputs()[0].data();
  auto output_buffer =
      legate::double_dispatch(output_store.dim(), output_store.code(),
                              get_write_only_buffer_fn{}, output_store);

  const void *src = reinterpret_cast<void *>(
      context.scalar(ScalarSourcePointer).value<uint64_t>());
  uint64_t src_size = context.scalar(ScalarSourceSize).value<uint64_t>();

  log_xla.debug() << "CopyDeviceToDevice from " << src << " -> "
                  << output_buffer.buffer << " of size " << src_size;

  TaskWaiter *waiter = reinterpret_cast<TaskWaiter *>(
      context.scalar(ScalarTaskWaiter).value<uint64_t>());

  if (src_size != 0) {
    if (output_buffer.size != src_size) {
      std::cerr << "Device-to-device copy has mismatched sizes "
                << output_buffer.size << " != " << src_size << std::endl;
      abort();
    }
    cudaMemcpy(output_buffer.buffer, src, src_size, cudaMemcpyDeviceToDevice);
  }

  log_xla.debug() << "Done copying, signal waiter";
  waiter->Signal();
}

/*static*/ void XLAInitZeroTask::gpu_variant(legate::TaskContext context) {
  auto output_store = context.outputs()[0].data();
  auto output_alloc =
      legate::double_dispatch(output_store.dim(), output_store.code(),
                              get_write_only_buffer_fn{}, output_store);

  log_xla.debug() << "XLAInitZeroTask: initializing  " << output_alloc.size
                  << " bytes to store of dimension " << output_store.dim();

  cudaMemset(output_alloc.buffer, 0, output_alloc.size);
}

void XLAStoreBufferActionTask::cpu_variant(legate::TaskContext context) {
  run_task(context);
}

void XLAStoreBufferActionTask::gpu_variant(legate::TaskContext context) {
  run_task(context);
}

void XLAStoreBufferActionTask::run_task(legate::TaskContext context) {

  auto output_store = context.outputs()[0].data();
  auto output_alloc =
      legate::double_dispatch(output_store.dim(), output_store.code(),
                              get_write_only_buffer_fn{}, output_store);
  auto cfg = get_task_config(context);
  log_xla.debug() << "XLAStoreBufferActionTask: running  on "
                  << cfg.my_device_id;
  auto *action = reinterpret_cast<BufferAction *>(
      context.scalar(ScalarAction + cfg.local_device_id).value<uint64_t>());

  action->Act(output_alloc.buffer, cfg.local_device_id);
  bool blocking = context.scalar(ScalarIsBlocking).value<bool>();
  if (blocking) {
    TaskWaiter *waiter = reinterpret_cast<TaskWaiter *>(
        context.scalar(ScalarTaskWaiter).value<uint64_t>());
    log_xla.debug() << "XLAStoreBufferActionTask: cleared on "
                    << cfg.my_device_id;
    waiter->Signal();
  }
}

void XLASetScalarTask::cpu_variant(legate::TaskContext context) {
  int32_t scalar = context.scalar(0).value<int32_t>();
  auto output_store = context.outputs()[0].data();
  auto output_alloc =
      legate::double_dispatch(output_store.dim(), output_store.code(),
                              get_write_only_buffer_fn{}, output_store);
  int32_t *dst = static_cast<int32_t *>(output_alloc.buffer);
  *dst = scalar;
}

void XLASetScalarTask::gpu_variant(legate::TaskContext context) {
  int32_t scalar = context.scalar(0).value<int32_t>();
  auto output_store = context.outputs()[0].data();
  auto output_alloc =
      legate::double_dispatch(output_store.dim(), output_store.code(),
                              get_write_only_buffer_fn{}, output_store);
  cudaMemcpy(output_alloc.buffer, &scalar, sizeof(int32_t),
             cudaMemcpyHostToDevice);
}

void XLAMaterializeTask::cpu_variant(legate::TaskContext context) {}

void XLAMaterializeTask::gpu_variant(legate::TaskContext context) {}

namespace // unnamed
{
static void __attribute__((constructor)) register_tasks(void) {
  XLACopyDeviceToDevice::register_variants();
  XLAInitZeroTask::register_variants();
  XLAStoreBufferActionTask::register_variants();
  XLASetScalarTask::register_variants();
  XLAMaterializeTask::register_variants();
}
} // namespace

} // namespace legate_xla
