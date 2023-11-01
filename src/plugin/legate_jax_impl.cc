#include <pybind11/pybind11.h>

namespace legate_xla {
void StopLegate();
}
extern "C" void ShutdownLegateClient();

template <class To, class From>
typename std::enable_if<sizeof(To) == sizeof(From) &&
                            std::is_trivially_copyable<From>::value &&
                            std::is_trivially_copyable<To>::value,
                        To>::type
bit_cast(const From &src) noexcept {
  static_assert(std::is_trivially_constructible<To>::value,
                "This implementation additionally requires destination type to "
                "be trivially constructible");

  To dst;
  memcpy(&dst, &src, sizeof(To));
  return dst;
}

template <typename T> pybind11::bytes PackDescriptor(const T &descriptor) {
  return pybind11::bytes(PackDescriptorAsString(descriptor));
}

template <typename T> pybind11::capsule EncapsulateFunction(T *fn) {
  return pybind11::capsule(bit_cast<void *>(fn), "xla._CUSTOM_CALL_TARGET");
}

void no_op_entrypoint() {}

static void AtExit() { legate_xla::StopLegate(); }

PYBIND11_MODULE(legate_jax_impl, m) {
  m.def("shutdown", []() -> void {
    // Make sure to clear all handles held by Legate
    // so that nothing gets deleted during program cleanup
    ShutdownLegateClient();
    // Delay shutting down Legate until atexit
    // so that all Legate shutdown occurs after PyFinalize
    atexit(AtExit);
  });
  m.def("no_op_custom_call",
        []() { return EncapsulateFunction(no_op_entrypoint); });
}