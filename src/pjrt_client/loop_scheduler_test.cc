/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/multimesh/loop_scheduler.h"

#include "gmock/gmock.h"
#include "tsl/platform/regexp.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/pjrt/multimesh/hlo_partition.h"
#include "xla/pjrt/multimesh/mpmd_instruction.h"
#include "xla/util.h"

namespace xla {
namespace {

using ::testing::Each;
using ::testing::Eq;

static constexpr absl::string_view kFwdPrefix = "fwd-";
static constexpr absl::string_view kBwdPrefix = "bwd-";
static constexpr absl::string_view kLastLayerPrefix = "last-";

static constexpr absl::string_view kSimpleHlo = R"(
dummy {
  ROOT constant = f32[] constant(0.0)
}

ENTRY main.15 {
  ROOT Arg_0.1 = f32[8]{0} parameter(0), sharding={replicated}
} // main.15
)";

struct TestConfig {
  int num_layers;
  int num_iterations;
  bool fuse_last_layer{false};
  LoopConfig::Schedule schedule{LoopConfig::Schedule::kFillDrain};
  int interleave{1};
  bool add_embeddings_and_logits{false};
  bool loop_submesh_embeddings_logits{false};
};

struct TestSetup {
  std::vector<std::vector<HloInstruction*>> tasks;
  std::shared_ptr<LoopConfig> loop_config;
  std::unique_ptr<HloModule> module;
  HloPartition partition;
  int num_layers{};
};

HloComputation* GetDummyComputation(HloModule& module) {
  for (auto* comp : module.computations()) {
    if (comp->name() == "dummy") {
      return comp;
    }
  }
  return nullptr;
}

HloInstruction* CreateTask(HloPartition& partition, HloModule& module,
                           std::string name) {
  Shape shape{PrimitiveType::F32, {}, {}, {}};
  HloInstruction* dummy = module.entry_computation()->AddInstruction(
      HloInstruction::CreateCall(shape, {}, GetDummyComputation(module)));
  dummy->SetAndSanitizeName(name);
  return dummy;
}

TestSetup CreateTest(TestConfig config) {
  auto loop_config = std::make_shared<LoopConfig>(
      LoopConfig{.num_iterations = config.num_iterations,
                 .unique_id = 0,
                 .schedule = config.schedule,
                 .interleave = config.interleave,
                 .num_stages = config.num_layers});

  const int num_fwd_bwd_layers = [=] {
    int num_layers = config.num_layers;
    if (config.fuse_last_layer) {
      num_layers--;
    }
    if (config.add_embeddings_and_logits) {
      num_layers += 2;
    }
    return num_layers;
  }();
  const int total_num_layers =
      config.fuse_last_layer ? num_fwd_bwd_layers + 1 : num_fwd_bwd_layers;

  auto module = *ParseAndReturnUnverifiedModule(kSimpleHlo);

  const int num_groups = config.num_layers / config.interleave;
  auto partition = HloPartition::Create(
      module.get(), {{.start = 0, .num_devices = num_groups}});

  std::vector<std::vector<HloInstruction*>> tasks(config.num_iterations);
  for (auto& iter : tasks) {
    iter.reserve(2 * total_num_layers);
  }

  auto add_task = [&](std::string name, int group) {
    zuku::DeviceList devices{{.start = group, .num_devices = 1}};
    auto color = partition->FindOrAllocateColor(name, devices, nullptr);
    for (int64_t iter = 0; iter < config.num_iterations; ++iter) {
      std::string full_name = absl::StrCat(name, "_iter_", iter);
      tasks[iter].push_back(
          CreateTask(*partition, *module, std::move(full_name)));
      AssignColor(tasks[iter].back(), *color);
    }
  };

  const int last_layer = [=] {
    if (config.fuse_last_layer) {
      return num_fwd_bwd_layers;
    }
    return num_fwd_bwd_layers - 1;
  }();

  auto get_group = [=](int layer) {
    if (config.add_embeddings_and_logits) {
      if (layer == 0) {
        return 0;
      }
      if (layer == last_layer) {
        return num_groups - 1;
      }
      return (layer - 1) % num_groups;
    }
    return layer % num_groups;
  };

  for (int i = 0; i < num_fwd_bwd_layers; ++i) {
    add_task(absl::StrCat(kFwdPrefix, i), get_group(i));
  }

  if (config.fuse_last_layer) {
    add_task(absl::StrCat(kLastLayerPrefix, num_fwd_bwd_layers),
             get_group(num_fwd_bwd_layers));
  }

  for (int i = num_fwd_bwd_layers - 1; i >= 0; --i) {
    add_task(absl::StrCat(kBwdPrefix, i), get_group(i));
  }

  return TestSetup{.loop_config = std::move(loop_config),
                   .module = std::move(module),
                   .tasks = std::move(tasks),
                   .partition = *std::move(partition),
                   .num_layers = total_num_layers};
}

auto GetLayerInfo(absl::string_view name) {
  std::string type;
  std::string number_str;
  std::string iter_str;
  bool match = re2::RE2::PartialMatch(name, "([a-z]+-)(\\d+)_iter_(\\d+)",
                                      &type, &number_str, &iter_str);
  int number;
  int iter;
  if (match) {
    match = absl::SimpleAtoi(number_str, &number);
    match = absl::SimpleAtoi(iter_str, &iter);
  }
  return std::make_tuple(match, type, number, iter);
}

TEST(LoopSchedulerTest, FillDrain) {
  static constexpr int kNumLayers = 4;
  static constexpr int kNumIters = 8;

  TestSetup test = CreateTest({.num_layers = kNumLayers,
                               .num_iterations = kNumIters,
                               .fuse_last_layer = false,
                               .schedule = LoopConfig::Schedule::kFillDrain});

  TF_ASSERT_OK_AND_ASSIGN(
      auto scheduled_tasks,
      ScheduleLoops(test.partition, *test.loop_config, test.tasks));

  int num_fwd_scheduled = 0;
  int num_bwd_scheduled = 0;

  absl::flat_hash_map<int, int> max_fwd_iter_visited_for_layer;
  absl::flat_hash_map<int, int> max_fwd_layer_visited_for_iter;
  absl::flat_hash_map<int, int> max_bwd_iter_visited_for_layer;

  // All the forward tasks should come before the backward tasks
  for (auto&& scheduled_task : scheduled_tasks[0]) {
    auto [matched, type, layer_num, iter] =
        GetLayerInfo(scheduled_task->name());

    if (type == kFwdPrefix) {
      EXPECT_LE(max_fwd_iter_visited_for_layer[layer_num], iter);
      EXPECT_LE(max_fwd_layer_visited_for_iter[iter], layer_num);
      max_fwd_iter_visited_for_layer[layer_num] =
          std::max(max_fwd_iter_visited_for_layer[layer_num], iter);
      max_fwd_layer_visited_for_iter[iter] =
          std::max(max_fwd_layer_visited_for_iter[iter], layer_num);
    } else if (type == kBwdPrefix) {
      EXPECT_LE(max_bwd_iter_visited_for_layer[layer_num], iter);
      max_bwd_iter_visited_for_layer[layer_num] =
          std::max(max_bwd_iter_visited_for_layer[layer_num], iter);
    }

    if (absl::StrContains(scheduled_task->name(), kFwdPrefix)) {
      EXPECT_EQ(num_bwd_scheduled, 0);
      ++num_fwd_scheduled;
    } else {
      EXPECT_EQ(num_fwd_scheduled, kNumLayers * kNumIters);
      ++num_bwd_scheduled;
    }
  }
}

void Run1F1BTest(TestConfig config,
                 LoopConfig::Schedule schedule = LoopConfig::Schedule::k1F1B) {
  config.schedule = schedule;
  TestSetup test = CreateTest(config);

  absl::flat_hash_map<int, int> max_fwd_iter_visited_for_layer;
  absl::flat_hash_map<int, int> max_fwd_layer_visited_for_iter;
  absl::flat_hash_map<int, int> max_bwd_iter_visited_for_layer;

  TF_ASSERT_OK_AND_ASSIGN(
      auto scheduled_tasks,
      ScheduleLoops(test.partition, *test.loop_config, test.tasks));
  std::vector<int> num_fwd_tasks_done(test.num_layers, 0);
  std::vector<int> num_bwd_tasks_done(test.num_layers, 0);
  for (auto&& scheduled_task : scheduled_tasks[0]) {
    VLOG(3) << scheduled_task->name();
    auto [matched, type, layer_num, iter] =
        GetLayerInfo(scheduled_task->name());

    if (type == kFwdPrefix) {
      EXPECT_LE(max_fwd_iter_visited_for_layer[layer_num], iter);
      EXPECT_LE(max_fwd_layer_visited_for_iter[iter], layer_num);
      max_fwd_iter_visited_for_layer[layer_num] =
          std::max(max_fwd_iter_visited_for_layer[layer_num], iter);
      max_fwd_layer_visited_for_iter[iter] =
          std::max(max_fwd_layer_visited_for_iter[iter], layer_num);
    } else if (type == kBwdPrefix) {
      EXPECT_LE(max_bwd_iter_visited_for_layer[layer_num], iter);
      max_bwd_iter_visited_for_layer[layer_num] =
          std::max(max_bwd_iter_visited_for_layer[layer_num], iter);
    }

    ASSERT_TRUE(matched);
    if (type == kFwdPrefix) {
      num_fwd_tasks_done[layer_num]++;
    } else if (type == kBwdPrefix) {
      num_bwd_tasks_done[layer_num]++;
    } else if (type == kLastLayerPrefix) {
      num_fwd_tasks_done[layer_num]++;
      num_bwd_tasks_done[layer_num]++;
    }

    ::testing::ScopedTrace scope{__FILE__, __LINE__, scheduled_task->name()};

    for (int prev_layer = 0; prev_layer < layer_num; ++prev_layer) {
      // previous layers should have run at least as many fwd tasks
      // if dependency order is valid
      EXPECT_LE(num_fwd_tasks_done[layer_num], num_fwd_tasks_done[prev_layer]);
    }
    // there must be at least as many bwd layers as fwd layers
    EXPECT_GE(num_fwd_tasks_done[layer_num], num_bwd_tasks_done[layer_num]);
  }

  EXPECT_THAT(num_fwd_tasks_done, Each(Eq(config.num_iterations)));
  EXPECT_THAT(num_bwd_tasks_done, Each(Eq(config.num_iterations)));
}

TEST(LoopSchedulerTest, EvenWavefront) {
  Run1F1BTest({.num_layers = 4, .num_iterations = 8, .fuse_last_layer = false},
              LoopConfig::Schedule::kWavefront);
}

TEST(LoopSchedulerTest, EvenWavefrontEmbeddings) {
  Run1F1BTest({.num_layers = 4,
               .num_iterations = 8,
               .fuse_last_layer = false,
               .add_embeddings_and_logits = true},
              LoopConfig::Schedule::kWavefront);
}

TEST(LoopSchedulerTest, EvenWavefrontInterleaved) {
  Run1F1BTest({.num_layers = 4,
               .num_iterations = 8,
               .fuse_last_layer = false,
               .interleave = 2},
              LoopConfig::Schedule::kWavefront);
}

TEST(LoopSchedulerTest, EvenWavefrontFewIters) {
  Run1F1BTest({.num_layers = 8, .num_iterations = 4, .fuse_last_layer = false},
              LoopConfig::Schedule::kWavefront);
}

TEST(LoopSchedulerTest, FusedLastLayerWavefront) {
  Run1F1BTest({.num_layers = 4, .num_iterations = 8, .fuse_last_layer = true},
              LoopConfig::Schedule::kWavefront);
}

TEST(LoopSchedulerTest, FusedLastLayerWavefrontMinPipeline) {
  Run1F1BTest({.num_layers = 4, .num_iterations = 4, .fuse_last_layer = true},
              LoopConfig::Schedule::kWavefront);
}

TEST(LoopSchedulerTest, FusedLastLayerWavefrontLarge) {
  Run1F1BTest({.num_layers = 8, .num_iterations = 24, .fuse_last_layer = true},
              LoopConfig::Schedule::kWavefront);
}

TEST(LoopSchedulerTest, FusedLastLayerWavefrontLargeEmbeddings) {
  Run1F1BTest({.num_layers = 8,
               .num_iterations = 24,
               .fuse_last_layer = true,
               .add_embeddings_and_logits = true},
              LoopConfig::Schedule::kWavefront);
}

TEST(LoopSchedulerTest, FusedLastLayerWavefrontLargeInterleaved2) {
  Run1F1BTest({.num_layers = 8,
               .num_iterations = 24,
               .fuse_last_layer = true,
               .interleave = 2},
              LoopConfig::Schedule::kWavefront);
}

TEST(LoopSchedulerTest, FusedLastLayerWavefrontLargeInterleaved4) {
  Run1F1BTest({.num_layers = 8,
               .num_iterations = 24,
               .fuse_last_layer = true,
               .interleave = 4},
              LoopConfig::Schedule::kWavefront);
}

TEST(LoopSchedulerTest, FusedLastLayerWavefrontLargeInterleaved2Embeddings) {
  Run1F1BTest({.num_layers = 8,
               .num_iterations = 24,
               .fuse_last_layer = true,
               .interleave = 2,
               .add_embeddings_and_logits = true},
              LoopConfig::Schedule::kWavefront);
}

TEST(LoopSchedulerTest, FusedLastLayerWavefrontLargeInterleaved4Embeddings) {
  Run1F1BTest({.num_layers = 8,
               .num_iterations = 24,
               .fuse_last_layer = true,
               .interleave = 4,
               .add_embeddings_and_logits = true},
              LoopConfig::Schedule::kWavefront);
}

TEST(LoopSchedulerTest,
     FusedLastLayerWavefrontLargeInterleaved2EmbeddingsDynamicSlice) {
  Run1F1BTest({.num_layers = 8,
               .num_iterations = 24,
               .fuse_last_layer = true,
               .interleave = 2,
               .add_embeddings_and_logits = true,
               .loop_submesh_embeddings_logits = true},
              LoopConfig::Schedule::kWavefront);
}

TEST(LoopSchedulerTest, PrefetchWavefrontEmbeddings) {
  Run1F1BTest({.num_layers = 4,
               .num_iterations = 8,
               .fuse_last_layer = true,
               .add_embeddings_and_logits = true,
               .loop_submesh_embeddings_logits = true},
              LoopConfig::Schedule::kPrefetchWavefront);
}

}  // namespace
}  // namespace xla
