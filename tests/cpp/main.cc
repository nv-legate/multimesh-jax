#include "legate.h"
#include "legate_xla_c.h"
#include "xla_to_legate.h"
#include <gtest/gtest.h>

class SingleGpuEnvironment : public ::testing::Environment {
public:
  SingleGpuEnvironment(int argc, char **argv) {}

  void SetUp() override {
    ASSERT_EQ(legate::start(0, nullptr), 0);

    legate_xla_perform_registration();
  }

  void TearDown() override { EXPECT_EQ(legate::finish(), 0); }
};

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new SingleGpuEnvironment(argc, argv));

  return RUN_ALL_TESTS();
}

// Needed to satisfy XLA symbols from the runtime library
struct PJRT_Api;
extern "C" const PJRT_Api *GetLegatePjrtApi() { return nullptr; }

extern "C" void ShutdownLegateClient() {}