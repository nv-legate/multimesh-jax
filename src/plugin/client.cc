#include "zuku/init.h"

extern "C" {

struct PJRT_Api;

PJRT_Api *GetLegatePjrtApi();

const PJRT_Api *GetPjrtApi() { return GetLegatePjrtApi(); }
}
