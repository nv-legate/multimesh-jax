#include <pybind11/pybind11.h>

namespace legate_xla {
void StopLegate();
}
extern "C" void ShutdownLegateClient();

PYBIND11_MODULE(legate_jax_impl, m) {
  m.def("shutdown", []() -> void {
    legate_xla::StopLegate();
    ShutdownLegateClient();
  });
}