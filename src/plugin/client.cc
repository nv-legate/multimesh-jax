#include "legate_to_xla.h"
#include "xla_to_legate.h"

static void Shutdown() {
  // Make sure to clear all handles held by Legate
  // so that nothing gets deleted during program cleanup
  ShutdownLegateClient();
  legate_xla::StopLegate();
}

extern "C" {

struct PJRT_Api;

const PJRT_Api *GetPjrtApi() {
  legate_xla::StartLegate();
  atexit(Shutdown);
  return GetLegatePjrtApi();
}
}
