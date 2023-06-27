#include "xla_task.h"

#include "legate_xla_common.h"

namespace legate_xla {

namespace {

struct get_write_only_buffer_fn {
  template <legate::Type::Code TYPE_CODE, int32_t DIM>
  BufferAllocation operator()(legate::Store &store) {
    using VAL = legate::legate_type_of<TYPE_CODE>;
    auto shape = store.shape<DIM>();
    auto acc = store.write_accessor<VAL, DIM>();
    void *buffer = static_cast<void *>(acc.ptr(shape));
    size_t size =
        sizeof(legate::legate_type_of<TYPE_CODE>) * store.domain().get_volume();
    return BufferAllocation{.buffer = buffer, .size = size};
  }
};

struct init_buffer_fn {
  template <legate::Type::Code TYPE_CODE, int32_t DIM>
  void operator()(legate::Store &store) {
    using VAL = legate::legate_type_of<TYPE_CODE>;
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

/*static*/ void XLAInitFromHostTask::gpu_variant(legate::TaskContext &context) {
  auto &scalars = context.scalars();
  auto bytes = scalars[0].value<uint64_t>();
  auto input_ptr = reinterpret_cast<void *>(scalars[1].value<uint64_t>());

  auto has_on_done_function = scalars[2].value<bool>();
  std::function<void()> *on_done_function;
  if (has_on_done_function) {
    on_done_function =
        reinterpret_cast<std::function<void()> *>(scalars[3].value<uint64_t>());
  }

  auto &output_store = context.outputs()[0];
  log_xla.debug() << "XLAInitFromHostTask: copying " << bytes
                  << " bytes to store of dimension " << output_store.dim();
  auto output_ptr =
      legate::double_dispatch(output_store.dim(), output_store.code(),
                              get_write_only_buffer_fn{}, output_store);

  cudaMemcpy(output_ptr.buffer, input_ptr, bytes, cudaMemcpyHostToDevice);

  if (has_on_done_function) {
    try {
      (*on_done_function)();
    } catch (const std::exception &e) {
      log_xla.error() << "Standard exception caught during on_done callback "
                         "excecution, message '"
                      << e.what() << "'";
    } catch (...) {
      log_xla.error() << "Exception caught during on_done callback excecution";
    }
    delete on_done_function;
  }
}

/*static*/ void XLAInitZeroTask::gpu_variant(legate::TaskContext &context) {
  auto &output_store = context.outputs()[0];
  auto output_alloc =
      legate::double_dispatch(output_store.dim(), output_store.code(),
                              get_write_only_buffer_fn{}, output_store);

  log_xla.debug() << "XLAInitZeroTask: initializing  " << output_alloc.size
                  << " bytes to store of dimension " << output_store.dim();

  cudaMemset(output_alloc.buffer, 0, output_alloc.size);
}

namespace // unnamed
{
static void __attribute__((constructor)) register_tasks(void) {
  XLAInitFromHostTask::register_variants();
  XLAInitZeroTask::register_variants();
}
} // namespace

} // namespace legate_xla
