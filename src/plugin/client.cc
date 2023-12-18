#include "legate_to_xla.h"
#include "xla_to_legate.h"

extern "C" {

struct PJRT_Api;

const PJRT_Api *GetPjrtApi() {
  return GetLegatePjrtApi();
}

}
