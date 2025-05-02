/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) The 2022 OpenXLA Authors.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "xla/pjrt/multimesh/mm_computation.h"

#include <chrono>

#include "xla/hlo/ir/dfs_hlo_visitor.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/pjrt/cpu/cpu_client.h"
#include "xla/pjrt/multimesh/cuda_utils.h"
#include "xla/pjrt/multimesh/mm_utils.h"
#include "xla/pjrt/pjrt_stream_executor_client.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/cpu_executable.h"
#include "xla/service/dump.h"
#include "xla/service/gpu/gpu_executable.h"
#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "xla/service/maybe_owning_device_memory.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/platform.h"
#include "xla/util.h"

using namespace std::chrono_literals;

namespace xla {

namespace {

constexpr int kMaxFloatsToPrint = 4;
constexpr int kCompileStatsVlogLevel = 1;

bool TupledArgs(const HloModule& hlo_module) {
  HloComputation* computation = hlo_module.entry_computation();
  return computation->parameter_instructions().size() == 1 &&
         computation->parameter_instruction(0)->shape().IsTuple();
}

static absl::StatusOr<Compiler::TargetConfig> GetConfigWithDeviceMem(
    int64_t device_mem, se::StreamExecutor* se) {
  Compiler::TargetConfig config{se};
  auto config_proto = config.ToProto();
  config_proto.mutable_gpu_device_info()->set_device_memory_size(device_mem);
  return Compiler::TargetConfig{config_proto};
}

std::optional<std::string> _MemcpyHtoDAsync(Backend* backend, void* dst,
                                            const void* src, size_t size,
                                            int64_t local_device_id) {
  auto status = [&] {
    TF_ASSIGN_OR_RETURN(se::Stream * stream,
                        GetCachedStream(backend, local_device_id));
    stream_executor::DeviceMemoryBase gpu_mem{dst, size};
    return stream->Memcpy(&gpu_mem, src, size);
  }();
  if (!status.ok()) {
    return std::string(status.message());
  }
  return std::nullopt;
}

std::optional<std::string> _MemcpyDtoDAsync(Backend* backend, void* dst,
                                            const void* src, size_t size,
                                            bool cpu, int64_t local_device_id) {
  if (cpu) {
    memcpy(dst, src, size);
    return std::nullopt;
  }

  auto status = [&] {
    TF_ASSIGN_OR_RETURN(se::Stream * stream,
                        GetCachedStream(backend, local_device_id));
    stream_executor::DeviceMemoryBase dst_mem{dst, size};
    stream_executor::DeviceMemoryBase src_mem{const_cast<void*>(src), size};
    return stream->MemcpyD2D(&dst_mem, src_mem, size);
  }();
  if (!status.ok()) {
    return std::string(status.message());
  }
  return std::nullopt;
}

}  // namespace

void MultiMeshXla::CostAnalysis(int vlog) {
  HloCostAnalysis::ShapeSizeFunction shape_function = [](const Shape& shape) {
    constexpr size_t pointer_size = 8;
    if (shape.is_static() || shape.IsTuple()) {
      return ShapeUtil::ByteSizeOf(shape, pointer_size);
    }
    // Each dynamic dimension size is represented as a S32.
    int64_t metadata_size = sizeof(int32_t) * shape.dimensions_size();
    return ShapeUtil::ByteSizeOf(shape, pointer_size) + metadata_size;
  };
  HloCostAnalysis::Options options{.shape_size = std::move(shape_function)};

  gpu::GpuHloCostAnalysis cost_analysis{options};
  auto status =
      executable_->module().entry_computation()->Accept(&cost_analysis);

  if (!status.ok()) {
    VLOG(vlog) << "Cost analysis failed to run: " << status.message();
    return;
  }

  VLOG(vlog) << "  Cost Summary for Module " << executable_->module().name()
             << "\n    Flops                 = " << cost_analysis.flop_count()
             << "\n    Bytes                 = "
             << cost_analysis.bytes_accessed()
             << "\n    Transcendentals       = "
             << cost_analysis.transcendental_count()
             << "\n    Operational Intensity = "
             << cost_analysis.flop_count() / cost_analysis.bytes_accessed();
}

absl::StatusOr<std::shared_ptr<HloModule>> MultiMeshXla::GetHloModule() const {
  if (!executable_) {
    return tsl::errors::InvalidArgument("MultiMesh executable is null");
  }
  auto* gpu_exe = dynamic_cast<gpu::GpuExecutable*>(executable_.get());
  if (!gpu_exe) {
    return tsl::errors::InvalidArgument("executable is not a GPU executable");
  }
  return gpu_exe->shared_module();
}

absl::StatusOr<std::vector<std::shared_ptr<HloModule>>>
MultiMeshXla::GetHloModules() const {
  TF_ASSIGN_OR_RETURN(auto module, GetHloModule());
  return std::vector<std::shared_ptr<HloModule>>{std::move(module)};
}

int64_t MultiMeshXla::TempRequired() const {
  CHECK(temp_required_.has_value());
  return *temp_required_;
}

void MultiMeshXla::MemorySummary(int vlog) {
  if (!executable_) {
    VLOG(vlog) << "Executable is null in MemorySummary" << std::endl;
    return;
  }

  auto* gpu_exe = dynamic_cast<gpu::GpuExecutable*>(executable_.get());
  cpu::CpuExecutable* cpu_exe = nullptr;
  if (!gpu_exe) {
    cpu_exe = dynamic_cast<cpu::CpuExecutable*>(executable_.get());
    if (!cpu_exe) {
      VLOG(vlog) << "Executable is not a GPU or CPU executable" << std::endl;
      return;
    }
  }

  absl::Span<const BufferAllocation> buffer_allocations =
      cpu_exe ? cpu_exe->buffer_assignment().Allocations()
              : gpu_exe->GetAllocations();

  std::stringstream ss;

  int64_t totals[] = {0, 0, 0, 0};
  int64_t counts[] = {0, 0, 0, 0};
  for (const BufferAllocation& alloc : buffer_allocations) {
    if (alloc.is_entry_computation_parameter()) {
      totals[0] += alloc.size();
      counts[0]++;
    } else if (alloc.maybe_live_out()) {
      totals[1] += alloc.size();
      counts[1]++;
    } else if (alloc.IsPreallocatedTempBuffer()) {
      totals[2] += alloc.size();
      counts[2]++;
    } else if (alloc.is_constant()) {
      totals[3] += alloc.size();
      counts[3]++;
    }
  }

  int64_t total_mem = totals[0] + totals[1] + totals[2] + totals[3];

  ss << "  Buffer Summary for Module " << executable_->module().name() << "\n"
     << "    Input   : " << totals[0] / 1e9 << "GB\n"
     << "      No. input buffers    = " << counts[0] << "\n"
     << "    Output  : " << totals[1] / 1e9 << "GB\n"
     << "      No. output buffers   = " << counts[1] << "\n"
     << "    Temp    : " << totals[2] / 1e9 << "GB\n"
     << "      No. temp buffers     = " << counts[2] << "\n"
     << "    Constant: " << totals[3] / 1e9 << "GB\n"
     << "      No. constant buffers = " << counts[3] << "\n"
     << "    Total   : " << total_mem / 1e9 << "GB\n"
     << "   " << buffer_allocations.size() << " total buffers" << std::endl;

  VLOG(vlog) << ss.str();
}

std::optional<int64_t> MultiMeshXla::OutputAlias(int64_t root_index) const {
  auto iter = output_input_alias_.find(root_index);
  if (iter == output_input_alias_.end()) {
    return std::nullopt;
  }
  return iter->second;
}

absl::Status MultiMeshXla::SetupOutputInputAlias() {
  const HloModule& hlo_module = *input_module_;
  const HloInputOutputAliasConfig& alias_config =
      hlo_module.input_output_alias_config();
  if (VLOG_IS_ON(5)) {
    VLOG(5) << hlo_module.ToString();
  }
  HloComputation* computation = hlo_module.entry_computation();
  bool tupled_args = TupledArgs(hlo_module);
  int number_of_parameters = [&]() -> int {
    if (tupled_args) {
      CHECK_EQ(computation->num_parameters(), 1);
      const Shape& input_tuple_shape =
          computation->parameter_instruction(0)->shape();
      CHECK(input_tuple_shape.IsTuple());
      return input_tuple_shape.tuple_shapes_size();
    } else {
      return computation->num_parameters();
    }
  }();

  VLOG(5) << hlo_module.name() << " has alias config "
          << alias_config.ToShortString();

  TF_RETURN_IF_ERROR(alias_config.ForEachAliasWithStatus(
      [&](const ShapeIndex& output_index,
          const HloInputOutputAliasConfig::Alias& alias) {
        if (tupled_args) {
          if (alias.parameter_number != 0) {
            return InvalidArgument(
                "Unexpected parameter number %d in alias config with tupled "
                "inputs",
                alias.parameter_number);
          }
          const ShapeIndex& index = alias.parameter_index;
          if (!index.empty()) {
            int parameter_number = index.data()[0];
            if (parameter_number >= number_of_parameters) {
              return InvalidArgument(
                  "Unexpected parameter index %s in alias config with tupled "
                  "inputs and %d parameters",
                  index.ToString(), number_of_parameters);
            }
            VLOG(5) << "Parameter " << parameter_number << " donated to "
                    << output_index << " on " << input_module_->name();
            int64_t root_index =
                output_index.empty() ? 0 : output_index.data()[0];
            output_input_alias_[root_index] = parameter_number;
          }
        } else {
          if (alias.parameter_number >= number_of_parameters) {
            return InvalidArgument(
                "parameter number %d exceeds no. of parameters",
                alias.parameter_number);
          }
          VLOG(5) << "Parameter " << alias.parameter_number << " donated to "
                  << output_index << " on " << input_module_->name();
          int64_t root_index =
              output_index.empty() ? 0 : output_index.data()[0];
          output_input_alias_[root_index] = alias.parameter_number;
        }
        return absl::OkStatus();
      }));
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<MultiMeshXla>> MultiMeshXla::Create(
    std::unique_ptr<HloModule> module, Shape root_shape, Backend* backend,
    PjRtClient* client) {
  const HloSharding* first_sharding = nullptr;

  size_t launch_size =
      module->config().replica_count() * module->config().num_partitions();

  auto global_result_shape =
      module->entry_computation()->root_instruction()->shape();
  std::vector<Shape> global_shapes;
  if (global_result_shape.IsTuple()) {
    global_shapes.reserve(global_result_shape.tuple_shapes_size());
    for (const auto& shape : global_result_shape.tuple_shapes()) {
      global_shapes.push_back(shape);
    }
  } else {
    global_shapes = {global_result_shape};
  }

  auto computation = std::unique_ptr<MultiMeshXla>(new MultiMeshXla(
      std::move(module), backend, client, launch_size, std::move(root_shape)));
  TF_RETURN_IF_ERROR(computation->SetupOutputInputAlias());
  return std::move(computation);
}

const HloModule& MultiMeshXla::optimized_module() const {
  if (executable_) {
    return executable_->module();
  }
  return *opt_module_;
}

absl::Status MultiMeshXla::Compile(uint64_t run_id,
                                   const MultiMeshCompileConfig& config) {
  TF_ASSIGN_OR_RETURN(auto* se,
                      backend_->stream_executor(config.stream_executor_index));
  TF_ASSIGN_OR_RETURN(auto* stream,
                      GetCachedStream(se, config.stream_executor_index));
  StreamWrapper stream_wrapper(run_id, config.stream_executor_index, stream,
                               xla::DeviceAssignment{}, backend_,
                               config.allocator);

  auto global_result_shape = input_module_->result_shape();

  if (VLOG_IS_ON(5)) {
    VLOG(5) << "Compiling " << input_module_->ToString();
  }

  DumpHloModuleIfEnabled(*input_module_, "before_optimizations");

  if (config.run_hlo_passes && config.run_backend) {
    auto* stream_client = dynamic_cast<PjRtStreamExecutorClient*>(client_);
    tsl::thread::ThreadPool* thread_pool =
        stream_client ? stream_client->thread_pool() : nullptr;

    Compiler::CompileOptions options{
        .device_allocator = stream_wrapper.MemoryAllocator(),
        .thread_pool = thread_pool,
    };

    if (config.device_mem.has_value()) {
      TF_ASSIGN_OR_RETURN(options.target_config,
                          GetConfigWithDeviceMem(*config.device_mem, se));
    }

    auto module_group =
        std::make_unique<HloModuleGroup>(std::move(opt_module_));

    if (config.device_mem.has_value()) {
      TF_ASSIGN_OR_RETURN(options.target_config,
                          GetConfigWithDeviceMem(*config.device_mem, se));
    }

    TF_ASSIGN_OR_RETURN(auto executables,
                        backend_->compiler()->Compile(std::move(module_group),
                                                      {{se}}, options));
    executable_ = std::move(executables[0]);
  } else if (config.run_backend) {
    TF_ASSIGN_OR_RETURN(executable_, backend_->compiler()->RunBackend(
                                         std::move(opt_module_), se,
                                         stream_wrapper.MemoryAllocator()));
  } else {
    Compiler::CompileOptions options{
        .device_allocator = stream_wrapper.MemoryAllocator(),
        .thread_pool = nullptr,
    };
    TF_ASSIGN_OR_RETURN(opt_module_, backend_->compiler()->RunHloPasses(
                                         std::move(opt_module_), se, options));
    return absl::OkStatus();
  }

  // if the optimized module does not contain any collectives
  // then we can mark this as not concurrent
  concurrent_ = [&] {
    for (auto* computation : this->optimized_module().computations()) {
      for (auto* instruction : computation->instructions()) {
        switch (instruction->opcode()) {
          case HloOpcode::kCollectivePermute:
          case HloOpcode::kCollectivePermuteStart:
          case HloOpcode::kAllGather:
          case HloOpcode::kAllGatherStart:
          case HloOpcode::kReduceScatter:
          case HloOpcode::kAllReduce:
          case HloOpcode::kAllReduceStart:
          case HloOpcode::kAllToAll:
          case HloOpcode::kCollectiveBroadcast:
            return true;
          default:
            break;
        }
      }
    }
    return false;
  }();

  auto buffer_allocations = [&]() -> absl::Span<const BufferAllocation> {
    auto* gpu_exe = dynamic_cast<gpu::GpuExecutable*>(executable_.get());
    if (gpu_exe) {
      return gpu_exe->GetAllocations();
    }

    auto* cpu_exe = dynamic_cast<cpu::CpuExecutable*>(executable_.get());
    CHECK(cpu_exe != nullptr);
    return cpu_exe->buffer_assignment().Allocations();
  }();

  int64_t temp = 0;
  for (const BufferAllocation& alloc : buffer_allocations) {
    if (alloc.IsPreallocatedTempBuffer()) {
      temp += alloc.size();
    }
  }
  temp_required_ = temp;

  if (VLOG_IS_ON(kCompileStatsVlogLevel)) {
    MemorySummary(kCompileStatsVlogLevel);
    CostAnalysis(kCompileStatsVlogLevel);
  }

  return absl::OkStatus();
}

MultiMeshCompiler::MultiMeshCompiler(std::unique_ptr<MultiMeshXla> comp)
    : MultiMeshStream(comp->mutable_backend()), computation_(std::move(comp)) {}

absl::StatusOr<std::unique_ptr<MultiMeshCompiler>> MultiMeshCompiler::Create(
    std::unique_ptr<HloModule> module, Shape root_shape, Backend* backend,
    PjRtClient* client) {
  TF_ASSIGN_OR_RETURN(
      auto comp, MultiMeshXla::Create(std::move(module), std::move(root_shape),
                                      backend, client));
  return std::unique_ptr<MultiMeshCompiler>(
      new MultiMeshCompiler(std::move(comp)));
}

void MultiMeshCompiler::Compile(uint64_t run_id,
                                const MultiMeshCompileConfig& config) {
  absl::Status status;
  try {
    status = computation_->Compile(run_id, config);
  } catch (const std::exception& e) {
    LOG(ERROR) << "Standard exception caught during 'Compile', message '"
               << e.what() << "'";
  } catch (...) {
    LOG(ERROR) << "Exception caught during 'Compile'";
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);
    compilation_finished_ = true;
  }
  if (!status.ok()) {
    LOG(ERROR) << "MultiMeshCompiler::Compile of " << Name()
               << " failed: " << status.message();
    std::cerr << "Compile failed for " << Name() << ": " << status.message()
              << std::endl;
    abort();
  }

  condition_.notify_all();
}

std::unique_ptr<MultiMeshExecutable> MultiMeshCompiler::MakeExecutable() {
  Block();
  return std::make_unique<MultiMeshExecutable>(computation_);
}

void MultiMeshCompiler::Block() {
  if (!compilation_finished_) {
    std::unique_lock<std::mutex> lock(mutex_);
    while (!condition_.wait_for(lock, 1000ms,
                                [&] { return compilation_finished_; })) {
      int process = this->computation_->client()->process_index();
      LOG(ERROR) << "MultiMeshCompiler::Block: "
                 << " rank " << process << " still waiting for compile of '"
                 << Name() << "'";
    }
  }
}

namespace se = stream_executor;

class TaskTempMemoryAllocator {
 public:
  TaskTempMemoryAllocator(absl::Span<const BufferAllocation> allocations,
                          TaskMemoryAllocator* allocator,
                          bool tupled_args = false, size_t num_args = 0,
                          bool tupled_outputs = false, size_t num_outputs = 0)
      : allocator_(allocator), position_(0) {
    total_size_ = 0;
    if (tupled_args) {
      size_t alloc_size = PaddedSize(num_args * sizeof(void*));
      total_size_ += alloc_size;
    }
    if (tupled_outputs) {
      size_t alloc_size = PaddedSize(num_outputs * sizeof(void*));
      total_size_ += alloc_size;
    }
    for (const BufferAllocation& alloc : allocations) {
      if (alloc.IsPreallocatedTempBuffer()) {
        // Round to an increment of padding
        total_size_ += PaddedSize(alloc.size());
      }
    }
    buffer_ = (char*)allocator->Allocate(total_size_);
  }

  ~TaskTempMemoryAllocator() { allocator_->Free(buffer_, total_size_); }

  void* Next(size_t alloc_size) {
    void* buf = buffer_ + position_;
    position_ += PaddedSize(alloc_size);
    return buf;
  }

 private:
  static size_t PaddedSize(size_t size) { return (size + padding) & ~padding; }

  // The -1 enables bit twiddling
  static constexpr size_t padding = 4096 - 1;

  size_t position_;
  size_t total_size_;
  char* buffer_;
  TaskMemoryAllocator* allocator_;
};

std::optional<std::string> MultiMeshStream::MemcpyHtoDAsync(
    void* dst, const void* src, size_t size, int64_t local_device_id) const {
  return _MemcpyHtoDAsync(backend_, dst, src, size, local_device_id);
}

std::optional<std::string> MultiMeshStream::MemcpyDtoDAsync(
    void* dst, const void* src, size_t size, bool cpu,
    int64_t local_device_id) const {
  return _MemcpyDtoDAsync(backend_, dst, src, size, cpu, local_device_id);
}

absl::Status RunCpuExecutable(
    uint64_t run_id, cpu::CpuExecutable* exe, xla::Backend* backend,
    const std::vector<se::DeviceMemoryBase>& inputs,
    const std::vector<se::DeviceMemoryBase>& outputs,
    TaskMemoryAllocator* allocator,
    const MultiMeshDeviceAssignment& device_assignment, bool tupled_args,
    PjRtClient* client) {
  VLOG(1) << "Running CPU executable " << exe->module().name() << " on "
          << device_assignment.GlobalDeviceId();

  auto* cpu_client = dynamic_cast<TfrtCpuClient*>(client);
  if (!cpu_client) {
    return InternalStrCat("client is not a TfrtCpuClient in RunCpuExecutable");
  }

  std::vector<stream_executor::OwningDeviceMemory> temp_memories;

  bool tupled_outputs =
      exe->module().entry_computation()->root_instruction()->shape().IsTuple();
  TaskTempMemoryAllocator temp_allocator(exe->buffer_assignment().Allocations(),
                                         allocator, tupled_args, inputs.size(),
                                         tupled_outputs, outputs.size());

  auto* root = exe->module().entry_computation()->root_instruction();
  absl::flat_hash_map<int64_t, int> output_buffer_indices;
  if (root->shape().IsTuple()) {
    int index = 0;
    for (auto* operand : root->operands()) {
      output_buffer_indices[operand->unique_id()] = index++;
    }
  } else {
    output_buffer_indices[root->unique_id()] = 0;
  }

  int result_index = -1;
  size_t num_allocs = exe->buffer_assignment().Allocations().size();

  std::vector<MaybeOwningDeviceMemory> buffers;

  auto get_output_index = [&](const BufferAllocation& alloc) {
    int output_index = -1;
    for (const auto& [value, offset] : alloc.assigned_buffers()) {
      for (const auto& position : value->positions()) {
        auto iter =
            output_buffer_indices.find(position.instruction->unique_id());
        if (iter != output_buffer_indices.end()) {
          return iter->second;
        }
      }
    }
    return -1;
  };

  auto print_buffer = [](cpu::CpuExecutable* exe, const char* type, int index,
                         void* buffer, size_t size) {
    std::stringstream sstr;
    float* data = (float*)buffer;
    int num_floats = size / sizeof(float);
    sstr << exe->module().name() << "." << exe->module().unique_id()
         << " data for " << type << " " << index << " of size " << size
         << " = { ";

    auto num_to_print = std::min(num_floats, kMaxFloatsToPrint);
    for (int i = 0; i < num_to_print; ++i) {
      sstr << data[i] << ", ";
    }
    if (num_to_print < num_floats) {
      sstr << "...";
    }
    double norm = 0;
    for (int i = 0; i < num_floats; ++i) {
      norm += data[i] * data[i];
    }
    sstr << "}, norm^2=" << norm << ", buffer=" << buffer;
    VLOG(10) << sstr.str();
  };

  const auto& constants = exe->constants();

  buffers.reserve(exe->buffer_assignment().Allocations().size());
  for (const BufferAllocation& alloc : exe->buffer_assignment().Allocations()) {
    VLOG(5) << "Have CPU allocation " << alloc.ToString();
    if (alloc.is_entry_computation_parameter()) {
      // zero-size allocations get a non-zero size in MultiMesh
      if (alloc.size() != 0 &&
          inputs[alloc.parameter_number()].size() != alloc.size()) {
        if (VLOG_IS_ON(3)) {
          for (const auto& alloc : exe->buffer_assignment().Allocations()) {
            VLOG(3) << alloc.ToString();
          }
        }
        return InvalidArgumentStrCat(
            "input ", alloc.parameter_number(), " on ", exe->module().name(),
            ".", exe->module().unique_id(),
            " has mismatched sizes. XLA expected ", alloc.size(),
            ", but MultiMesh has ", inputs[alloc.parameter_number()].size());
      }
      buffers.emplace_back(inputs[alloc.parameter_number()]);
      if (alloc.maybe_live_out()) {  // donated parameter
        int output_index = get_output_index(alloc);
        if (output_index == -1) {
          return InternalStrCat(
              "buffer allocation could not be matched against output index on ",
              exe->module().name(), ": ", alloc.ToString());
        }
        VLOG(3) << exe->module().name() << " output " << output_index
                << " aliases buffer allocation for parameter "
                << alloc.parameter_number();
      }
      if (VLOG_IS_ON(10)) {
        print_buffer(exe, "input", alloc.parameter_number(),
                     buffers[alloc.index()].AsDeviceMemoryBase().opaque(),
                     alloc.size());
      }
    } else if (alloc.maybe_live_out()) {
      if (alloc.is_tuple()) {
        result_index = buffers.size();
        size_t size = outputs.size() * sizeof(void*);
        void* output_tuple = temp_allocator.Next(size);
        buffers.emplace_back(se::DeviceMemoryBase{output_tuple, size});
      } else {
        if (result_index == -1) {
          result_index = buffers.size();
        }

        int output_index = get_output_index(alloc);

        if (output_index == -1) {
          if (VLOG_IS_ON(3)) {
            VLOG(3) << exe->module()
                           .entry_computation()
                           ->root_instruction()
                           ->ToString();
          }
          return InternalStrCat(
              "buffer allocation could not be matched against output index on ",
              exe->module().name(), ": ", alloc.ToString());
        }

        if (alloc.size() != 0 && outputs[output_index].size() != alloc.size()) {
          if (VLOG_IS_ON(3)) {
            for (const auto& alloc : exe->buffer_assignment().Allocations()) {
              VLOG(3) << alloc.ToString();
            }
          }
          return InvalidArgumentStrCat(
              "output ", output_index, " on ", exe->module().name(), ".",
              exe->module().unique_id(), " has mismatched sizes. XLA expected ",
              alloc.size(), ", but MultiMesh has ",
              outputs[output_index].size());
        }
        buffers.emplace_back(outputs[output_index]);
      }
    } else if (alloc.IsPreallocatedTempBuffer()) {
      buffers.emplace_back(se::DeviceMemoryBase{
          temp_allocator.Next(alloc.size()), uint64_t(alloc.size())});
    } else if (alloc.is_constant() && alloc.index() < constants.size()) {
      buffers.emplace_back(constants[alloc.index()].AsDeviceMemoryBase());
    } else if (alloc.is_constant() || alloc.is_thread_local()) {
      buffers.emplace_back(se::DeviceMemoryBase{nullptr, uint64_t(0)});
    } else {
      return InvalidArgumentStrCat(
          "Got buffer allocation that is not an input, output, or temp: ",
          alloc.ToString());
    }
  }

  ExecutableRunOptions run_options;
  run_options.set_run_id(RunId(run_id));
  run_options.set_device_ordinal(device_assignment.GlobalDeviceId());
  run_options.set_intra_op_thread_pool(cpu_client->eigen_intraop_device());

  // Need to keep device_assignment alive until execution completes.
  TF_ASSIGN_OR_RETURN(DeviceAssignment xla_device_assignment,
                      MultiMeshToXlaDeviceAssignment(device_assignment));
  run_options.set_device_assignment(&xla_device_assignment);

  if (exe->has_compute_function()) {
    TF_RETURN_IF_ERROR(exe->ExecuteComputeFunction(&run_options, buffers));
  } else if (exe->has_thunks()) {
    TF_RETURN_IF_ERROR(exe->ExecuteThunks(&run_options, buffers));
  } else {
    return Internal("No compute function or thunks found.");
  }
  return absl::OkStatus();
}

absl::Status RunGpuExecutable(
    zuku::Stream* zs, uint64_t run_id, gpu::GpuExecutable* exe,
    xla::Backend* backend, const std::vector<se::DeviceMemoryBase>& inputs,
    const std::vector<se::DeviceMemoryBase>& outputs,
    TaskMemoryAllocator* allocator,
    const MultiMeshDeviceAssignment& device_assignment, int64_t num_local,
    bool tupled_args, bool blocking) {
  bool tupled_outputs = false;
  for (const BufferAllocation& alloc : exe->GetAllocations()) {
    if (alloc.maybe_live_out() && alloc.is_tuple()) {
      tupled_outputs = true;
      break;
    }
  }

  std::vector<se::DeviceMemoryBase> buffers;
  TaskTempMemoryAllocator temp_allocator(exe->GetAllocations(), allocator,
                                         tupled_args, inputs.size(),
                                         tupled_outputs, outputs.size());

  auto device_id = device_assignment.LocalDeviceId();
  TF_ASSIGN_OR_RETURN(se::StreamExecutor * executor,
                      backend->stream_executor(device_id));

  TF_ASSIGN_OR_RETURN(auto* stream, GetCachedStream(executor, device_id));

  TF_ASSIGN_OR_RETURN(DeviceAssignment xla_device_assignment,
                      MultiMeshToXlaDeviceAssignment(device_assignment));
  // At this point there should be no more Legion blocking operations so it is
  // safe to allocate a monotonically increasing run_id
  auto stream_wrapper = std::make_unique<StreamWrapper>(
      run_id, device_assignment.GlobalDeviceId(), stream,
      std::move(xla_device_assignment), backend, allocator);
  stream_wrapper->RunOptions()->mutable_run_options()->set_local_device_count(
      num_local);

  TF_ASSIGN_OR_RETURN(auto* globals,
                      exe->ResolveConstantGlobals(stream_wrapper->XlaStream()));

  std::map<int, int> output_buffer_reorder;
  for (const auto& pair : exe->GetOutputInfo()) {
    const gpu::GpuExecutable::OutputInfo& info = pair.second;
    const ShapeIndex& shape_index = pair.first;
    if (shape_index.empty()) {
      output_buffer_reorder[info.allocation_index] = 0;
    } else {
      output_buffer_reorder[info.allocation_index] = pair.first.front();
    }
  }

  void* input_tuple_bufs_device = nullptr;
  void* output_tuple_bufs_device = nullptr;
  for (const BufferAllocation& alloc : exe->GetAllocations()) {
    if (alloc.is_thread_local()) {
      buffers.emplace_back();
    } else if (alloc.is_entry_computation_parameter()) {
      if (tupled_args && alloc.param_shape_index().empty()) {
        // this is the top-level tuple that will contain all the parameters
        size_t size = inputs.size() * sizeof(void*);
        input_tuple_bufs_device = temp_allocator.Next(size);
        buffers.emplace_back(input_tuple_bufs_device, size);
      } else {
        int input_index = 0;
        if (alloc.param_shape_index().size() == 1) {
          input_index = alloc.param_shape_index()[0];
        } else if (alloc.param_shape_index().size() > 1) {
          return tsl::errors::InvalidArgument(
              "Do yet not support arg tuples with nested shapes");
        } else {
          input_index = alloc.parameter_number();
        }
        if (input_index >= inputs.size()) {
          return tsl::errors::InvalidArgument(
              "Too few inputs given to GPU executable");
        }
        if (inputs[input_index].size() != alloc.size() && alloc.size() > 0) {
          return tsl::errors::InvalidArgument(absl::StrCat(
              "Mismatched size on input ", input_index, ": XLA expected ",
              alloc.size(), " but MultiMesh has ", inputs[input_index].size(),
              "\n", alloc.ToString()));
        }
        buffers.emplace_back(inputs[input_index]);
      }
    } else if (alloc.maybe_live_out()) {
      if (alloc.is_tuple()) {
        size_t size = inputs.size() * sizeof(void*);
        output_tuple_bufs_device = temp_allocator.Next(size);
        buffers.emplace_back(output_tuple_bufs_device, size);
      } else {
        // we need to figure out which output this actually corresponds to
        auto iter = output_buffer_reorder.find(alloc.index());
        if (iter == output_buffer_reorder.end()) {
          return tsl::errors::InvalidArgument(
              "Buffer allocation missing from GpuExecutable::OutputInfo");
        }
        int output_index = iter->second;
        if (output_index >= outputs.size()) {
          return tsl::errors::InvalidArgument(absl::StrCat(
              "Too few outputs given to GPU executable. Executable has output "
              "index ",
              output_index, " but only ", outputs.size(),
              " outputs were given"));
        }
        if (outputs[output_index].size() != alloc.size() && alloc.size() > 0) {
          return tsl::errors::InvalidArgument(absl::StrCat(
              "Mismatched size on output ", output_index, ": XLA expected ",
              alloc.size(), " but MultiMesh has ", outputs[output_index].size(),
              "\n", alloc.ToString()));
        }
        buffers.emplace_back(outputs[output_index]);
      }
    } else if (alloc.IsPreallocatedTempBuffer()) {
      void* buf = temp_allocator.Next(alloc.size());
      buffers.emplace_back(buf, alloc.size());
    } else if (alloc.is_constant()) {
      auto iter = globals->find(alloc.index());
      if (iter == globals->end()) {
        std::cerr
            << "Constant allocation could not be resolved in executable: " +
                   alloc.ToString()
            << std::endl;
        buffers.emplace_back();
      } else {
        buffers.push_back(iter->second);
      }
    } else {
      return tsl::errors::InvalidArgument(
          "Got buffer allocation that is not an input, output, or temp: " +
          alloc.ToString());
    }
  }

  if (tupled_args) {
    std::vector<const void*> tuple_bufs_host(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
      tuple_bufs_host[i] = inputs[i].opaque();
    }
    _MemcpyHtoDAsync(backend, input_tuple_bufs_device, tuple_bufs_host.data(),
                     inputs.size() * sizeof(void*), device_id);
  }
  if (tupled_outputs) {
    std::vector<const void*> tuple_bufs_host(outputs.size());
    for (size_t i = 0; i < outputs.size(); ++i) {
      tuple_bufs_host[i] = outputs[i].opaque();
    }
    _MemcpyHtoDAsync(backend, output_tuple_bufs_device, tuple_bufs_host.data(),
                     outputs.size() * sizeof(void*), device_id);
  }

  gpu::BufferAllocations buffer_allocations{
      buffers,
      /*device_ordinal=*/(int)device_assignment.LocalDeviceId(),
      stream_wrapper->MemoryAllocator()};

  // Lock the GPU with a shared lock so that we don't interfere with autotuning
  // that may be running during JIT compilation while allowing multiple XLA
  // computations to use the same GPU simultaneously.
  // gpu::NonAtomicallyUpgradeableRWLock gpu_lock(&gpu::GetGpuMutex(executor));

  // This must block until the execution is completed because any eager/temp
  // buffers will be cleaned up after returning from the function.
  // If this does not block, then pending kernels may operate on
  // deallocated memory.
  TF_RETURN_IF_ERROR(
      exe->ExecuteThunks(buffer_allocations, stream_wrapper->RunOptions()));

  return cuda_utils::RecordCuEvent(zs, stream);
}

MultiMeshExecutable::MultiMeshExecutable(
    const std::shared_ptr<MultiMeshXla>& computation)
    : computation_(computation) {}

std::optional<std::string> MultiMeshExecutable::Execute(
    zuku::Stream* zs, uint64_t run_id,
    const std::vector<se::DeviceMemoryBase>& inputs,
    const std::vector<se::DeviceMemoryBase>& outputs,
    TaskMemoryAllocator* allocator,
    const MultiMeshDeviceAssignment& device_assignment, int64_t num_local,
    Platform platform, bool blocking) const {
  if (!computation_->mutable_executable()) {
    return "executable is null";
  }

  if (platform == CPU) {
    auto* cpu_exe =
        dynamic_cast<cpu::CpuExecutable*>(computation_->mutable_executable());
    if (!cpu_exe) {
      return "Executable is not a cpu::CpuExecutable";
    }

    auto status =
        RunCpuExecutable(run_id, cpu_exe, computation_->mutable_backend(),
                         inputs, outputs, allocator, device_assignment,
                         /*tupled_args=*/false, computation_->client());

    // nothing to record on CPU
    zs->Record(nullptr);

    if (!status.ok()) {
      return absl::StrCat(
          "Failed running executable ", cpu_exe->module().name(),
          cpu_exe->module().unique_id(), ": ", status.message());
    }
  } else {
    auto* gpu_exe =
        dynamic_cast<gpu::GpuExecutable*>(computation_->mutable_executable());
    if (!gpu_exe) {
      return "Executable is not a gpu::GpuExecutable";
    }

    VLOG(1) << "Running GPU executable " << gpu_exe->module().name();

    auto status =
        RunGpuExecutable(zs, run_id, gpu_exe, computation_->mutable_backend(),
                         inputs, outputs, allocator, device_assignment,
                         num_local, TupledArgs(gpu_exe->module()), blocking);

    if (!status.ok()) {
      return absl::StrCat(
          "Failed running executable ", gpu_exe->module().name(),
          gpu_exe->module().unique_id(), ": ", status.message());
    }
  }

  return std::nullopt;
}

std::optional<std::string> MultiMeshExecutable::MemcpyHtoDAsync(
    void* dst, const void* src, size_t size, int64_t local_device_id) const {
  return _MemcpyHtoDAsync(computation_->mutable_backend(), dst, src, size,
                          local_device_id);
}

std::optional<std::string> MultiMeshExecutable::MemcpyDtoDAsync(
    void* dst, const void* src, size_t size, bool cpu,
    int64_t local_device_id) const {
  return _MemcpyDtoDAsync(computation_->mutable_backend(), dst, src, size, cpu,
                          local_device_id);
}

}  // namespace xla
