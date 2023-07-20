#include "legate_to_xla.h"

extern "C" {

struct PJRT_Api;
const PJRT_Api *GetPjrtApi() { return GetLegatePjrtApi(); }
}
