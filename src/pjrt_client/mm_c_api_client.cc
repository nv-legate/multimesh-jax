/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) The 2022 OpenXLA Authors.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <memory>
#include <optional>

#include "gloo/transport/tcp/attr.h"
#include "gloo/transport/tcp/device.h"
#include "xla/backends/cpu/collectives/gloo_collectives.h"
#include "xla/backends/cpu/collectives/gloo_kv_store.h"
#include "xla/backends/profiler/plugin/plugin_tracer_impl.h"
#include "xla/backends/profiler/plugin/profiler_c_api.h"
#include "xla/backends/profiler/plugin/profiler_error.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_custom_partitioner_extension.h"
#include "xla/pjrt/c/pjrt_c_api_ffi_extension.h"
#include "xla/pjrt/c/pjrt_c_api_ffi_internal.h"
#include "xla/pjrt/c/pjrt_c_api_gpu_extension.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/c/pjrt_c_api_profiler_extension.h"
#include "xla/pjrt/c/pjrt_c_api_wrapper_impl.h"
#include "xla/pjrt/cpu/cpu_client.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_client.h"
#include "xla/pjrt/multimesh/mm_pjrt_client.h"
#include "xla/pjrt/multimesh/mm_utils.h"
#include "xla/pjrt/multimesh/zuku_execute_context_impl.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_stream_executor_client.h"
#include "xla/python/custom_partition_callback.h"
#include "xla/service/backend.h"
#include "xla/service/custom_call_target_registry.h"
#include "xla/service/platform_util.h"
#include "xla/util.h"

namespace xla {

namespace {

constexpr absl::string_view kNodeIdName = "node_id";
constexpr absl::string_view kNumNodesName = "num_nodes";
constexpr char kPluginName[] = "multimesh";
constexpr char kCudaPluginName[] = "CUDA";
constexpr std::array kPluginsToRegister = {kPluginName, kCudaPluginName};
zuku::RealmConfig startup_config;

}  // namespace

// Implements plugin client initialization from 'pjrt_plugin_device_client.h'
absl::StatusOr<std::unique_ptr<PjRtClient>> GetTfrtPluginDeviceClient(
    PJRT_Client_Create_Args* args) {
  std::optional<std::string> platform_name;

  auto kv_store = pjrt::ToCppKeyValueStore(
      args->kv_get_callback, args->kv_get_user_arg, args->kv_try_get_callback,
      args->kv_try_get_user_arg, args->kv_put_callback, args->kv_put_user_arg);

  for (const auto& custom_call : {"Microbatch"}) {
    void* fxn = xla::CustomCallTargetRegistry::Global()->Lookup(custom_call,
                                                                kPluginName);
    if (!fxn) {
      return InternalStrCat(custom_call, " custom call not registered");
    }
    xla::CustomCallTargetRegistry::Global()->Register(custom_call, fxn,
                                                      kCudaPluginName);
  }

  int64_t node_id = 0, num_nodes = 1;
  for (int i = 0; i < args->num_options; ++i) {
    if (args->create_options[i].name == kNodeIdName) {
      node_id = args->create_options[i].int64_value;
    } else if (args->create_options[i].name == kNumNodesName) {
      num_nodes = args->create_options[i].int64_value;
    }
  }

  auto execute_context = ZukuExecuteContextImpl::Create(startup_config);

  std::set<int> allowed_devices = execute_context->GetLocalDevices();
  bool is_gpu = execute_context->IsGpu();
  if (is_gpu) {
    xla::GpuClientOptions options;
    options.allocator_config = {.memory_fraction = 0.02, .preallocate = false};
    options.node_id = node_id;
    options.num_nodes = num_nodes;
    options.allowed_devices = std::move(allowed_devices);
    options.platform_name = platform_name;
    options.kv_store = kv_store;

    TF_ASSIGN_OR_RETURN(auto pjrt_client, GetStreamExecutorGpuClient(options));

    auto* gpu_client =
        dynamic_cast<PjRtStreamExecutorClient*>(pjrt_client.get());
    if (!gpu_client) {
      return InternalStrCat(
          "underlying client is not a PjRtStreamExecutorClient");
    }

    TF_RETURN_IF_ERROR(InitDistributedRuntimeParams(
        num_nodes, node_id, gpu_client->addressable_device_count(), kv_store));

    return std::unique_ptr<PjRtClient>(std::make_unique<MultiMeshClient>(
        std::move(pjrt_client), gpu_client->client()->mutable_backend(),
        std::move(execute_context)));
  }

  xla::CpuClientOptions options;
  options.asynchronous = false;
  options.process_id = node_id;
  options.cpu_device_count = allowed_devices.size();

  if (num_nodes > 1) {
    auto gloo_kv_store = std::make_unique<cpu::GlooKeyValueStore>(kv_store);
    auto tcp_attrs = gloo::transport::tcp::attr();
    auto tcp_device = gloo::transport::tcp::CreateDevice(tcp_attrs);
    options.collectives = std::make_shared<cpu::GlooCollectives>(
        std::move(gloo_kv_store), std::move(tcp_device));
  }

  TF_ASSIGN_OR_RETURN(auto pjrt_client, GetTfrtCpuClient(options));

  TF_ASSIGN_OR_RETURN(se::Platform * platform,
                      PlatformUtil::GetPlatform("cpu"));

  // allow 2 threads per GPU/CPU used for computation
  auto num_threads = allowed_devices.size() * 2;
  TF_ASSIGN_OR_RETURN(std::unique_ptr<Backend> backend,
                      Backend::CreateBackend(
                          BackendOptions()
                              .set_platform(platform)
                              .set_intra_op_parallelism_threads(num_threads)));

  TF_RETURN_IF_ERROR(InitDistributedRuntimeParams(
      /*num_procs=*/1, /*node_id=*/0, pjrt_client->addressable_device_count(),
      nullptr));
  return std::unique_ptr<PjRtClient>(std::make_unique<MultiMeshClient>(
      std::move(pjrt_client), backend.release(), std::move(execute_context)));
}

}  // namespace xla

namespace mm_plugin {

PJRT_Error* PJRT_Client_Create(PJRT_Client_Create_Args* args) {
  PJRT_RETURN_IF_ERROR(pjrt::ActualStructSizeIsGreaterOrEqual(
      "PJRT_Client_Create_Args", PJRT_Client_Create_Args_STRUCT_SIZE,
      args->struct_size));

  PJRT_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtClient> client,
                        xla::GetTfrtPluginDeviceClient(args));
  args->client = pjrt::CreateWrapperClient(std::move(client));
  return nullptr;
}

PJRT_Error* PJRT_CpuDeviceTopology_Create(
    PJRT_TopologyDescription_Create_Args* args) {
  return new PJRT_Error{tsl::errors::Unimplemented(
      "Topology not supported for MultiMesh compilation.")};
}

extern "C" PJRT_Error* PJRT_ExecuteContext_Create(
    PJRT_ExecuteContext_Create_Args* args) {
  PJRT_RETURN_IF_ERROR(pjrt::ActualStructSizeIsGreaterOrEqual(
      "PJRT_ExecuteContext_Create_Args",
      PJRT_ExecuteContext_Create_Args_STRUCT_SIZE, args->struct_size));
  auto execute_context = std::make_unique<xla::ExecuteContext>();
  args->context = pjrt::CreateWrapperExecuteContext(std::move(execute_context));
  return nullptr;
}

PJRT_Error* PJRT_Init(PJRT_Plugin_Initialize_Args* args) { return nullptr; }

PJRT_Error* PJRT_MultiMesh_Register_Custom_Call(
    PJRT_Gpu_Register_Custom_Call_Args* args) {
  PJRT_RETURN_IF_ERROR(pjrt::ActualStructSizeIsGreaterOrEqual(
      "PJRT_Gpu_Register_Custom_Call_Args",
      PJRT_Gpu_Register_Custom_Call_Args_STRUCT_SIZE, args->struct_size));
  std::string function_name(args->function_name, args->function_name_size);
  switch (args->api_version) {
    case 0: {
      for (auto&& name : xla::kPluginsToRegister) {
        xla::CustomCallTargetRegistry::Global()->Register(
            function_name, args->handler_execute, name);
      }
      return nullptr;
    }

    case 1: {
      for (auto&& name : xla::kPluginsToRegister) {
        xla::ffi::Ffi::RegisterStaticHandler(
            xla::ffi::GetXlaFfiApi(), function_name, name,
            XLA_FFI_Handler_Bundle{
                reinterpret_cast<XLA_FFI_Handler*>(args->handler_instantiate),
                reinterpret_cast<XLA_FFI_Handler*>(args->handler_prepare),
                reinterpret_cast<XLA_FFI_Handler*>(args->handler_initialize),
                reinterpret_cast<XLA_FFI_Handler*>(args->handler_execute)});
      }
      return nullptr;
    }
    default:
      return new PJRT_Error{absl::UnimplementedError(
          absl::StrFormat("API version %d not supported for PJRT GPU plugin. "
                          "Supported versions are 0 and 1.",
                          args->api_version))};
  }
}

PLUGIN_Profiler_Api profiler_api{
    /*struct_size=*/PLUGIN_Profiler_Api_STRUCT_SIZE,
    /*priv=*/nullptr,
    /*error_destroy=*/xla::profiler::PLUGIN_Profiler_Error_Destroy,
    /*error_message=*/xla::profiler::PLUGIN_Profiler_Error_Message,
    /*error_get_code=*/xla::profiler::PLUGIN_Profiler_Error_GetCode,
    /*create=*/xla::profiler::PLUGIN_Profiler_Create,
    /*destroy=*/xla::profiler::PLUGIN_Profiler_Destroy,
    /*start=*/xla::profiler::PLUGIN_Profiler_Start,
    /*stop=*/xla::profiler::PLUGIN_Profiler_Stop,
    /*collect_data=*/xla::profiler::PLUGIN_Profiler_CollectData,
};

PJRT_Profiler_Extension profiler_extension{
    /*struct_size=*/PJRT_Profiler_Extension_STRUCT_SIZE,
    /*type=*/PJRT_Extension_Type::PJRT_Extension_Type_Profiler,
    /*next=*/nullptr,
    /*profiler_api=*/&profiler_api,
};

PJRT_Error* PJRT_Register_Custom_Partitioner(
    PJRT_Register_Custom_Partitioner_Args* args) {
  PJRT_RETURN_IF_ERROR(pjrt::ActualStructSizeIsGreaterOrEqual(
      "PJRT_Register_Custom_Partitioner_Args",
      PJRT_Register_Custom_Partitioner_Args_STRUCT_SIZE, args->struct_size));
  std::string name(args->name, args->name_size);
  RegisterCustomCallPartitioner(
      name, jax::CreateCApiCustomCallPartitioner(args->callbacks));
  return nullptr;
}

PJRT_Custom_Partitioner_Extension custom_partitioner{
    /*struct_size=*/PJRT_Gpu_Custom_Call_STRUCT_SIZE,
    /*type=*/PJRT_Extension_Type::PJRT_Extension_Type_Custom_Partitioner,
    /*next=*/reinterpret_cast<PJRT_Extension_Base*>(&profiler_extension),
    /*register_custom_partitioner=*/PJRT_Register_Custom_Partitioner,
};

}  // namespace mm_plugin

extern "C" const PJRT_Api* GetMultiMeshPjRtApi() {
  static PJRT_Gpu_Custom_Call custom_call{
      /*struct_size=*/PJRT_Gpu_Custom_Call_STRUCT_SIZE,
      /*type=*/PJRT_Extension_Type::PJRT_Extension_Type_Gpu_Custom_Call,
      /*next=*/
      reinterpret_cast<PJRT_Extension_Base*>(&mm_plugin::custom_partitioner),
      /*custom_call=*/mm_plugin::PJRT_MultiMesh_Register_Custom_Call,
  };
  static PJRT_Layouts_Extension layouts_extension =
      pjrt::CreateLayoutsExtension(
          reinterpret_cast<PJRT_Extension_Base*>(&custom_call));

  static PJRT_FFI_Extension ffi_extension = pjrt::CreateFfiExtension(
      reinterpret_cast<PJRT_Extension_Base*>(&layouts_extension));

  static const PJRT_Api pjrt_api = pjrt::CreatePjrtApi(
      mm_plugin::PJRT_Client_Create, mm_plugin::PJRT_ExecuteContext_Create,
      mm_plugin::PJRT_CpuDeviceTopology_Create, mm_plugin::PJRT_Init,
      reinterpret_cast<PJRT_Extension_Base*>(&ffi_extension));

  return &pjrt_api;
}

extern "C" void SetStartupConfig(zuku::RealmConfig cfg) {
  xla::startup_config = cfg;
}

extern "C" void NoOpCustomCall() {}

XLA_CPU_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("Microbatch", NoOpCustomCall);
XLA_CPU_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("MicrobatchInit", NoOpCustomCall);
XLA_CPU_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("MicrobatchSlice", NoOpCustomCall);

XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("Microbatch", NoOpCustomCall,
                                         xla::kCudaPluginName);
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("Microbatch", NoOpCustomCall,
                                         xla::kPluginName);
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("MicrobatchSlice", NoOpCustomCall,
                                         xla::kCudaPluginName);
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("MicrobatchSlice", NoOpCustomCall,
                                         xla::kPluginName);
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("MicrobatchInit", NoOpCustomCall,
                                         xla::kCudaPluginName);
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("MicrobatchInit", NoOpCustomCall,
                                         xla::kPluginName);
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("Marking", NoOpCustomCall,
                                         xla::kPluginName);
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("Marking", NoOpCustomCall,
                                         xla::kCudaPluginName);
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("MultiMeshTask", NoOpCustomCall,
                                         xla::kPluginName);
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("MultiMeshTask", NoOpCustomCall,
                                         xla::kCudaPluginName);
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("AutoSharding", NoOpCustomCall,
                                         xla::kPluginName);
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("AutoSharding", NoOpCustomCall,
                                         xla::kCudaPluginName);
