#include "../../src/legate_xla_c.h"
#include "legate.h"
#include "xla_to_legate.h"
#include <gtest/gtest.h>

class SingleGpuEnvironment : public ::testing::Environment {
public:
  SingleGpuEnvironment(int argc, char **argv) {
    for (int i = 0; i < argc; i++)
      argv_.push_back(argv[i]);
    for (const auto &arg : extra_args)
      argv_.push_back((char *)arg.data());
    argv_.push_back(nullptr);
  }

  void SetUp() override {
    EXPECT_EQ(legate::start(argv_.size() - 1, argv_.data()), 0);

    legate_xla_perform_registration();
  }
  void TearDown() override { EXPECT_EQ(legate::finish(), 0); }

private:
  std::vector<std::string> extra_args = {"-ll:gpu", "1"};
  std::vector<char *> argv_;
};

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new SingleGpuEnvironment(argc, argv));

  return RUN_ALL_TESTS();
}
