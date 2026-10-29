// Copyright (c) 2025-2026 Winyunq. All rights reserved.

#define LITERT_LM_IMPLEMENTATION
#include "runtime/engine/litert_lm_api.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <dxgi1_2.h>
#elif defined(__ANDROID__)
#include <dlfcn.h>
#include <vulkan/vulkan.h>
#endif

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/components/logits_processor/constrained_decoding/llg_constraint_config.h"
#include "runtime/conversation/conversation.h"
#include "runtime/engine/engine_factory.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/util/logging.h"
#include "runtime/util/scoped_file.h"

using litert::lm::AdvancedSettings;
using litert::lm::Backend;
using litert::lm::BenchmarkInfo;
using litert::lm::Conversation;
using litert::lm::ConversationConfig;
using litert::lm::CpuConfig;
using litert::lm::Engine;
using litert::lm::EngineFactory;
using litert::lm::EngineSettings;
using litert::lm::GetBackendString;
using litert::lm::GpuConfig;
using litert::lm::JsonPreface;
using litert::lm::LlgConstraintType;
using litert::lm::LlGuidanceConfig;
using litert::lm::LlGuidanceConstraintArg;
using litert::lm::Message;
using litert::lm::ModelAssets;
using litert::lm::OptionalArgs;
using litert::lm::SessionConfig;
using litert::lm::ScopedFile;
namespace proto = litert::lm::proto;
using OrderedJson = nlohmann::ordered_json;

OrderedJson NormalizeToolDeclarations(const OrderedJson& tools) {
  OrderedJson normalized = OrderedJson::array();
  if (!tools.is_array()) return normalized;
  for (const auto& tool : tools) {
    if (tool.is_object() && tool.contains("function") &&
        tool["function"].is_object()) {
      normalized.push_back(tool["function"]);
    } else {
      normalized.push_back(tool);
    }
  }
  return normalized;
}

struct LiteRtLm_RequestObject;

struct LiteRtLm_ConversationDefinition {
  OrderedJson messages = OrderedJson::array();
  OrderedJson tools = OrderedJson::array();
  OrderedJson extra_context = OrderedJson::object();
  std::string tools_json;
  std::string system_message;
  std::string initial_messages_json;
  std::string extra_context_json;
  std::string tool_call_channel = "tool_call";
  std::string lora_path;
  std::string audio_lora_path;
  uint32_t flags = LITERT_LM_CONVERSATION_RETURN_ERROR_ON_PARSE_FAILURE;
  uint32_t default_max_output_tokens = 0;
};

#if defined(_WIN32)
// The native Engine may retain function pointers into both GPU sidecars.  This
// owner is intentionally declared before LiteRtLm_EngineObject::engine below,
// so the Engine is destroyed before the DLL references are released.
struct WindowsGpuSidecars {
  HMODULE accelerator = nullptr;
  HMODULE sampler = nullptr;

  WindowsGpuSidecars() = default;
  WindowsGpuSidecars(const WindowsGpuSidecars&) = delete;
  WindowsGpuSidecars& operator=(const WindowsGpuSidecars&) = delete;

  WindowsGpuSidecars(WindowsGpuSidecars&& other) noexcept
      : accelerator(std::exchange(other.accelerator, nullptr)),
        sampler(std::exchange(other.sampler, nullptr)) {}

  WindowsGpuSidecars& operator=(WindowsGpuSidecars&& other) noexcept {
    if (this == &other) return *this;
    Reset();
    accelerator = std::exchange(other.accelerator, nullptr);
    sampler = std::exchange(other.sampler, nullptr);
    return *this;
  }

  ~WindowsGpuSidecars() { Reset(); }

  void Reset() noexcept {
    // Release in reverse load order.  FreeLibrary is safe here because the
    // owning native Engine has already been destroyed.
    if (sampler != nullptr) {
      FreeLibrary(sampler);
      sampler = nullptr;
    }
    if (accelerator != nullptr) {
      FreeLibrary(accelerator);
      accelerator = nullptr;
    }
  }
};
#endif

#if defined(__ANDROID__)
// Android GPU sampler plug-ins call the public LiteRT C API, but they are
// loaded lazily and do not carry a DT_NEEDED entry for libLiteRt.so. Keep the
// matching runtime globally visible for the complete native Engine lifetime so
// dlopen() can resolve those symbols when a Conversation creates its sampler.
struct AndroidGpuSidecars {
  void* litert_runtime = nullptr;

  AndroidGpuSidecars() = default;
  AndroidGpuSidecars(const AndroidGpuSidecars&) = delete;
  AndroidGpuSidecars& operator=(const AndroidGpuSidecars&) = delete;

  AndroidGpuSidecars(AndroidGpuSidecars&& other) noexcept
      : litert_runtime(std::exchange(other.litert_runtime, nullptr)) {}

  AndroidGpuSidecars& operator=(AndroidGpuSidecars&& other) noexcept {
    if (this == &other) return *this;
    Reset();
    litert_runtime = std::exchange(other.litert_runtime, nullptr);
    return *this;
  }

  ~AndroidGpuSidecars() { Reset(); }

  void Reset() noexcept {
    if (litert_runtime != nullptr) {
      dlclose(litert_runtime);
      litert_runtime = nullptr;
    }
  }
};
#endif

struct LiteRtLm_EngineObject {
  uint64_t magic = 0;
#if defined(_WIN32)
  WindowsGpuSidecars gpu_sidecars;
#elif defined(__ANDROID__)
  AndroidGpuSidecars gpu_sidecars;
#endif
  std::unique_ptr<Engine> engine;
  LiteRtLm_Backend configured_backend = LITERT_LM_BACKEND_GPU;
  Backend resolved_backend = Backend::GPU;
  Backend sampler_backend = Backend::GPU;
  uint32_t flags = 0;
  uint32_t max_context_tokens = 0;
  LiteRtLm_Sampling applied_sampling{};
  std::string backend_description;
  std::mutex operation_mutex;
  std::atomic<uint32_t> conversation_count{0};
  std::atomic<uint32_t> request_count{0};
  std::atomic<LiteRtLm_RequestObject*> active_request{nullptr};
};

struct LiteRtLm_ConversationObject {
  uint64_t magic = 0;
  LiteRtLm_EngineObject* owner = nullptr;
  std::mutex native_mutex;
  std::unique_ptr<Conversation> conversation;
  // Sampling is a native Conversation/session property. ABI 1.2 request
  // overrides therefore rebuild only this Conversation while the Engine/model
  // remains resident.
  LiteRtLm_Sampling native_sampling{};
  LiteRtLm_ConversationDefinition definition;
  // Dynamic history already materialized into the current Conversation's
  // preface after a sampling switch. Keep it separate from definition.messages
  // so ResetConversation still means "return to the original preface".
  OrderedJson replayed_messages = OrderedJson::array();
  std::atomic<uint64_t> generation{1};
  std::atomic<uint32_t> request_count{0};
  std::atomic<LiteRtLm_RequestObject*> active_request{nullptr};
  std::atomic<bool> needs_reset{false};
};

struct LiteRtLm_RequestObject {
  uint64_t magic = 0;
  LiteRtLm_EngineObject* engine = nullptr;
  LiteRtLm_ConversationObject* conversation = nullptr;
  Conversation* native_conversation = nullptr;
  LiteRtLm_EventCallback callback = nullptr;
  void* user_data = nullptr;
  std::string task_group_id;
  uint32_t max_output_tokens = 0;
  uint64_t conversation_generation = 0;
  LiteRtLm_Sampling applied_sampling{};

  std::recursive_mutex callback_mutex;
  std::mutex metrics_mutex;
  std::mutex mutex;
  std::condition_variable condition;
  LiteRtLm_RequestState state = LITERT_LM_REQUEST_CREATED;
  LiteRtLm_FinishReason finish_reason = LITERT_LM_FINISH_NONE;
  LiteRtLm_Status status = LITERT_LM_STATUS_OK;
  LiteRtLm_ErrorPhase error_phase = LITERT_LM_PHASE_NONE;
  int32_t native_code = 0;
  bool retryable = false;
  bool terminal = false;
  bool submit_in_progress = true;
  bool callback_in_progress = false;
  bool terminal_event_dispatched = false;
  bool native_wait_complete = false;
  bool metrics_finalized = false;
  bool cancel_requested = false;
  bool used_constraint = false;
  std::atomic<uint32_t> native_callbacks_in_flight{0};
  uint64_t sequence = 0;
  uint64_t callback_count = 0;

  std::string text;
  std::string response_json;
  std::string error_message;

  bool benchmark_baseline_valid = false;
  uint64_t prefill_turn_baseline = 0;
  uint64_t decode_turn_baseline = 0;
  uint64_t valid_metrics = 0;
  uint64_t input_tokens = 0;
  uint64_t output_tokens = 0;
  uint64_t context_tokens = 0;
  uint64_t prefill_time_us = 0;
  uint64_t decode_time_us = 0;
  uint64_t time_to_first_token_us = 0;
  uint64_t total_time_us = 0;
  double output_tokens_per_second = 0.0;

  std::chrono::steady_clock::time_point start_time;
  std::chrono::steady_clock::time_point first_output_time;
  std::chrono::steady_clock::time_point end_time;
  bool first_output_seen = false;
};

namespace {

constexpr uint64_t kEngineMagic = 0x4C4D454E47494E45ull;       // LMENGINE
constexpr uint64_t kConversationMagic = 0x4C4D434F4E565253ull; // LMCONVRS
constexpr uint64_t kRequestMagic = 0x4C4D524551554553ull;      // LMREQUES
constexpr uint32_t kKnownEngineFlags =
    LITERT_LM_ENGINE_ENABLE_METRICS |
    LITERT_LM_ENGINE_DISABLE_PARALLEL_FILE_LOADING;
constexpr uint32_t kKnownConversationFlags =
    LITERT_LM_CONVERSATION_ENABLE_CONSTRAINED_DECODING |
    LITERT_LM_CONVERSATION_PREFILL_PREFACE |
    LITERT_LM_CONVERSATION_FILTER_CHANNEL_CONTENT_FROM_KV |
    LITERT_LM_CONVERSATION_RETURN_ERROR_ON_PARSE_FAILURE |
    LITERT_LM_CONVERSATION_RETURN_ERROR_ON_MAX_TOKENS |
    LITERT_LM_CONVERSATION_ENABLE_THINKING |
    LITERT_LM_CONVERSATION_STREAM_TOOL_CALLS |
    LITERT_LM_CONVERSATION_DISABLE_PROMPT_TEMPLATE;
constexpr uint32_t kKnownChatFlags =
    LITERT_LM_CHAT_OVERRIDE_SAMPLING | LITERT_LM_CHAT_OVERRIDE_TOOLS |
    LITERT_LM_CHAT_OVERRIDE_EXTRA_CONTEXT;
constexpr int32_t kMaximumTopK = 1024;

constexpr char kBuildId[] =
    "LiteRT-LM stable table ABI 1.4; explicit backend; multi-conversation; "
    "per-request sampling; request lifecycle; tokenizer; " __DATE__ " "
    __TIME__;

std::atomic<uint64_t> g_request_id{1};
std::atomic<uint64_t> g_seed_sequence{1};
thread_local LiteRtLm_RequestObject* g_callback_request = nullptr;
#if defined(_WIN32)
// GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS) uses this address
// to locate the wrapper DLL even when the process working directory differs.
int g_windows_wrapper_module_anchor = 0;
#endif

#define LITERT_LM_FIELD_END(type, field) \
  (offsetof(type, field) + sizeof(((type*)0)->field))

#define LITERT_LM_HAS_FIELD(value, type, field) \
  ((value) != nullptr && \
   (value)->struct_size >= LITERT_LM_FIELD_END(type, field))

LiteRtLm_StringView ViewOf(const std::string& value) {
  LiteRtLm_StringView view;
  view.data = value.empty() ? nullptr : value.data();
  view.size = static_cast<uint64_t>(value.size());
  return view;
}

LiteRtLm_StringView ViewOf(const char* value, size_t size) {
  LiteRtLm_StringView view;
  view.data = size == 0 ? nullptr : value;
  view.size = static_cast<uint64_t>(size);
  return view;
}

bool IsViewValid(const LiteRtLm_StringView& view) {
  return (view.data != nullptr || view.size == 0) &&
         view.size <= static_cast<uint64_t>(std::numeric_limits<size_t>::max());
}

std::string CopyView(const LiteRtLm_StringView& view) {
  if (view.size == 0) return {};
  return std::string(view.data, static_cast<size_t>(view.size));
}

void ClearError(LiteRtLm_ErrorInfo* error) {
  if (error == nullptr) return;
  uint32_t capacity = error->struct_size;
  if (capacity < sizeof(LiteRtLm_ErrorInfo)) return;
  std::memset(error, 0, sizeof(LiteRtLm_ErrorInfo));
  error->struct_size = static_cast<uint32_t>(sizeof(LiteRtLm_ErrorInfo));
  error->code = LITERT_LM_STATUS_OK;
}

LiteRtLm_Status SetError(LiteRtLm_ErrorInfo* error, LiteRtLm_Status code,
                         LiteRtLm_ErrorPhase phase, std::string_view message,
                         int32_t native_code = 0, bool retryable = false) {
  if (error != nullptr &&
      error->struct_size >= sizeof(LiteRtLm_ErrorInfo)) {
    LiteRtLm_ErrorInfo value{};
    value.struct_size = static_cast<uint32_t>(sizeof(value));
    value.code = code;
    value.phase = phase;
    value.native_code = native_code;
    value.retryable = retryable ? 1u : 0u;
    const size_t count = std::min(message.size(), sizeof(value.message) - 1);
    if (count != 0) std::memcpy(value.message, message.data(), count);
    value.message[count] = '\0';
    std::memcpy(error, &value, sizeof(LiteRtLm_ErrorInfo));
  }
  return code;
}

LiteRtLm_Status CopyStringToCaller(const std::string& value, char* output,
                                   uint64_t* inout_size,
                                   LiteRtLm_ErrorPhase phase,
                                   LiteRtLm_ErrorInfo* error) {
  if (inout_size == nullptr) {
    return SetError(error, LITERT_LM_STATUS_INVALID_ARGUMENT, phase,
                    "The buffer size pointer is null.");
  }
  if (value.size() == (std::numeric_limits<size_t>::max)()) {
    return SetError(error, LITERT_LM_STATUS_RESOURCE_EXHAUSTED, phase,
                    "The result is too large to terminate safely.");
  }
  const uint64_t required = static_cast<uint64_t>(value.size()) + 1u;
  const uint64_t capacity = *inout_size;
  *inout_size = required;
  if (output == nullptr) return LITERT_LM_STATUS_OK;
  if (capacity < required) {
    return SetError(error, LITERT_LM_STATUS_RESOURCE_EXHAUSTED, phase,
                    "The caller-provided buffer is too small.");
  }
  if (!value.empty()) std::memcpy(output, value.data(), value.size());
  output[value.size()] = '\0';
  return LITERT_LM_STATUS_OK;
}

#if defined(_WIN32)
bool GetWindowsModulePath(HMODULE module, std::wstring* out_path,
                          DWORD* out_error) {
  if (out_path == nullptr || out_error == nullptr) return false;
  out_path->clear();
  *out_error = ERROR_SUCCESS;

  std::wstring buffer(512, L'\0');
  constexpr size_t kMaximumWindowsPathLength = 32768;
  for (;;) {
    SetLastError(ERROR_SUCCESS);
    DWORD copied = GetModuleFileNameW(
        module, buffer.data(), static_cast<DWORD>(buffer.size()));
    if (copied == 0) {
      *out_error = GetLastError();
      return false;
    }
    if (copied < buffer.size()) {
      buffer.resize(copied);
      *out_path = std::move(buffer);
      return true;
    }
    if (buffer.size() >= kMaximumWindowsPathLength) {
      *out_error = ERROR_INSUFFICIENT_BUFFER;
      return false;
    }
    buffer.resize(std::min(buffer.size() * 2,
                           kMaximumWindowsPathLength));
  }
}

std::string WindowsPathForError(std::wstring_view path) {
  if (path.empty()) return {};
  int required = WideCharToMultiByte(
      CP_UTF8, WC_ERR_INVALID_CHARS, path.data(),
      static_cast<int>(path.size()), nullptr, 0, nullptr, nullptr);
  if (required <= 0) return "<path could not be encoded as UTF-8>";
  std::string result(static_cast<size_t>(required), '\0');
  int written = WideCharToMultiByte(
      CP_UTF8, WC_ERR_INVALID_CHARS, path.data(),
      static_cast<int>(path.size()), result.data(), required, nullptr, nullptr);
  if (written != required) return "<path could not be encoded as UTF-8>";
  return result;
}

LiteRtLm_Status FindWindowsHardwareGpu(std::string* out_device_name,
                                       LiteRtLm_ErrorInfo* error) {
  if (out_device_name == nullptr) {
    return SetError(error, LITERT_LM_STATUS_INTERNAL_ERROR,
                    LITERT_LM_PHASE_ENGINE_CREATE,
                    "Internal hardware GPU result is null.");
  }
  out_device_name->clear();

  HMODULE dxgi_module = LoadLibraryExW(
      L"dxgi.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
  if (dxgi_module == nullptr) {
    DWORD native_code = GetLastError();
    return SetError(error, LITERT_LM_STATUS_BACKEND_ERROR,
                    LITERT_LM_PHASE_ENGINE_CREATE,
                    "Strict GPU initialization could not load system "
                    "dxgi.dll to verify a hardware adapter.",
                    static_cast<int32_t>(native_code));
  }

  using CreateDxgiFactory1Function = HRESULT(WINAPI*)(REFIID, void**);
  auto create_factory = reinterpret_cast<CreateDxgiFactory1Function>(
      GetProcAddress(dxgi_module, "CreateDXGIFactory1"));
  if (create_factory == nullptr) {
    DWORD native_code = GetLastError();
    FreeLibrary(dxgi_module);
    return SetError(error, LITERT_LM_STATUS_BACKEND_ERROR,
                    LITERT_LM_PHASE_ENGINE_CREATE,
                    "System dxgi.dll does not expose CreateDXGIFactory1.",
                    static_cast<int32_t>(native_code));
  }

  IDXGIFactory1* factory = nullptr;
  HRESULT hr = create_factory(__uuidof(IDXGIFactory1),
                              reinterpret_cast<void**>(&factory));
  if (FAILED(hr) || factory == nullptr) {
    FreeLibrary(dxgi_module);
    return SetError(error, LITERT_LM_STATUS_BACKEND_ERROR,
                    LITERT_LM_PHASE_ENGINE_CREATE,
                    "Strict GPU initialization could not create a DXGI "
                    "factory to verify a hardware adapter.",
                    static_cast<int32_t>(hr));
  }

  std::wstring best_name;
  SIZE_T best_dedicated_memory = 0;
  bool found_hardware = false;
  for (UINT index = 0;; ++index) {
    IDXGIAdapter1* adapter = nullptr;
    hr = factory->EnumAdapters1(index, &adapter);
    if (hr == DXGI_ERROR_NOT_FOUND) break;
    if (FAILED(hr) || adapter == nullptr) continue;

    DXGI_ADAPTER_DESC1 description{};
    const HRESULT description_hr = adapter->GetDesc1(&description);
    adapter->Release();
    if (FAILED(description_hr)) continue;
    if ((description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0) continue;

    const std::wstring_view name(description.Description);
    if (name.find(L"Microsoft Basic Render Driver") !=
        std::wstring_view::npos) {
      continue;
    }

    // Prefer a discrete adapter when present, but accept a real integrated
    // GPU. The exact name is passed to WebGPU so WARP/software adapters cannot
    // satisfy the strict request.
    if (!found_hardware ||
        description.DedicatedVideoMemory > best_dedicated_memory) {
      found_hardware = true;
      best_dedicated_memory = description.DedicatedVideoMemory;
      best_name.assign(name);
    }
  }

  factory->Release();
  FreeLibrary(dxgi_module);

  if (!found_hardware || best_name.empty()) {
    return SetError(
        error, LITERT_LM_STATUS_BACKEND_ERROR,
        LITERT_LM_PHASE_ENGINE_CREATE,
        "Strict GPU initialization found no hardware DXGI adapter. "
        "CPU/WARP/software adapters are not permitted.");
  }

  *out_device_name = WindowsPathForError(best_name);
  if (out_device_name->empty() ||
      *out_device_name == "<path could not be encoded as UTF-8>") {
    return SetError(error, LITERT_LM_STATUS_BACKEND_ERROR,
                    LITERT_LM_PHASE_ENGINE_CREATE,
                    "The selected hardware GPU name could not be encoded "
                    "for WebGPU adapter selection.");
  }
  return LITERT_LM_STATUS_OK;
}

LiteRtLm_Status LoadRequiredWindowsGpuSidecars(
    WindowsGpuSidecars* out_sidecars, LiteRtLm_ErrorInfo* error) {
  if (out_sidecars == nullptr) {
    return SetError(error, LITERT_LM_STATUS_INTERNAL_ERROR,
                    LITERT_LM_PHASE_ENGINE_CREATE,
                    "Internal GPU sidecar owner is null.");
  }

  HMODULE wrapper_module = nullptr;
  if (!GetModuleHandleExW(
          GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
              GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
          reinterpret_cast<LPCWSTR>(&g_windows_wrapper_module_anchor),
          &wrapper_module)) {
    DWORD native_code = GetLastError();
    return SetError(
        error, LITERT_LM_STATUS_BACKEND_ERROR,
        LITERT_LM_PHASE_ENGINE_CREATE,
        "Strict GPU initialization could not locate the LiteRT-LM wrapper "
        "module (GetModuleHandleExW failed).",
        static_cast<int32_t>(native_code));
  }

  std::wstring wrapper_path;
  DWORD native_code = ERROR_SUCCESS;
  if (!GetWindowsModulePath(wrapper_module, &wrapper_path, &native_code)) {
    return SetError(
        error, LITERT_LM_STATUS_BACKEND_ERROR,
        LITERT_LM_PHASE_ENGINE_CREATE,
        "Strict GPU initialization could not resolve the LiteRT-LM wrapper "
        "DLL path (GetModuleFileNameW failed).",
        static_cast<int32_t>(native_code));
  }
  const size_t separator = wrapper_path.find_last_of(L"\\/");
  if (separator == std::wstring::npos) {
    return SetError(error, LITERT_LM_STATUS_BACKEND_ERROR,
                    LITERT_LM_PHASE_ENGINE_CREATE,
                    "The LiteRT-LM wrapper DLL path has no directory.",
                    static_cast<int32_t>(ERROR_BAD_PATHNAME));
  }
  const std::wstring wrapper_directory =
      wrapper_path.substr(0, separator + 1);

  WindowsGpuSidecars loaded;
  auto load_one = [&](const wchar_t* file_name,
                      HMODULE* destination) -> LiteRtLm_Status {
    const std::wstring absolute_path = wrapper_directory + file_name;
    DWORD attributes = GetFileAttributesW(absolute_path.c_str());
    if (attributes == INVALID_FILE_ATTRIBUTES) {
      DWORD load_error = GetLastError();
      return SetError(
          error, LITERT_LM_STATUS_BACKEND_ERROR,
          LITERT_LM_PHASE_ENGINE_CREATE,
          std::string("Required strict-GPU sidecar is missing or unreadable: ") +
              WindowsPathForError(absolute_path),
          static_cast<int32_t>(load_error));
    }
    if ((attributes & FILE_ATTRIBUTE_DIRECTORY) != 0) {
      return SetError(
          error, LITERT_LM_STATUS_BACKEND_ERROR,
          LITERT_LM_PHASE_ENGINE_CREATE,
          std::string("Required strict-GPU sidecar path is a directory: ") +
              WindowsPathForError(absolute_path),
          static_cast<int32_t>(ERROR_BAD_FORMAT));
    }

    HMODULE module = LoadLibraryExW(
        absolute_path.c_str(), nullptr,
        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR |
            LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (module == nullptr) {
      DWORD load_error = GetLastError();
      return SetError(
          error, LITERT_LM_STATUS_BACKEND_ERROR,
          LITERT_LM_PHASE_ENGINE_CREATE,
          std::string("Failed to load required strict-GPU sidecar and its ") +
              "dependencies: " + WindowsPathForError(absolute_path),
          static_cast<int32_t>(load_error));
    }

    // Windows may return an already-loaded DLL with the same basename even
    // when LoadLibraryExW receives an absolute path.  Reject that ambiguity:
    // strict GPU must use the sidecar deployed beside this wrapper.
    std::wstring loaded_path;
    DWORD path_error = ERROR_SUCCESS;
    if (!GetWindowsModulePath(module, &loaded_path, &path_error)) {
      FreeLibrary(module);
      return SetError(
          error, LITERT_LM_STATUS_BACKEND_ERROR,
          LITERT_LM_PHASE_ENGINE_CREATE,
          "Loaded a strict-GPU sidecar but could not verify its module path.",
          static_cast<int32_t>(path_error));
    }
    if (CompareStringOrdinal(absolute_path.c_str(), -1, loaded_path.c_str(),
                             -1, TRUE) != CSTR_EQUAL) {
      FreeLibrary(module);
      return SetError(
          error, LITERT_LM_STATUS_BACKEND_ERROR,
          LITERT_LM_PHASE_ENGINE_CREATE,
          std::string("A same-named GPU DLL is already loaded from an ") +
              "unexpected location; expected " +
              WindowsPathForError(absolute_path) + "; loaded " +
              WindowsPathForError(loaded_path),
          static_cast<int32_t>(ERROR_ALREADY_EXISTS));
    }

    *destination = module;
    return LITERT_LM_STATUS_OK;
  };

  LiteRtLm_Status status =
      load_one(L"libLiteRtWebGpuAccelerator.dll", &loaded.accelerator);
  if (status != LITERT_LM_STATUS_OK) return status;
  status = load_one(L"libLiteRtTopKWebGpuSampler.dll", &loaded.sampler);
  if (status != LITERT_LM_STATUS_OK) return status;

  *out_sidecars = std::move(loaded);
  return LITERT_LM_STATUS_OK;
}
#endif

#if defined(__ANDROID__)
LiteRtLm_Status LoadRequiredAndroidGpuSidecars(
    AndroidGpuSidecars* out_sidecars, LiteRtLm_ErrorInfo* error) {
  if (out_sidecars == nullptr) {
    return SetError(error, LITERT_LM_STATUS_INTERNAL_ERROR,
                    LITERT_LM_PHASE_ENGINE_CREATE,
                    "Internal Android GPU sidecar owner is null.");
  }

  dlerror();
  void* litert_runtime = dlopen("libLiteRt.so", RTLD_NOW | RTLD_GLOBAL);
  if (litert_runtime == nullptr) {
    const char* native_error = dlerror();
    return SetError(
        error, LITERT_LM_STATUS_BACKEND_ERROR,
        LITERT_LM_PHASE_ENGINE_CREATE,
        std::string("Strict GPU initialization could not globally load ") +
            "libLiteRt.so for Android sampler plug-ins: " +
            (native_error != nullptr ? native_error : "unknown dlopen error"));
  }

  dlerror();
  void* create_environment =
      dlsym(litert_runtime, "LiteRtCreateEnvironment");
  const char* symbol_error = dlerror();
  if (create_environment == nullptr || symbol_error != nullptr) {
    const std::string message =
        std::string("Android libLiteRt.so does not export the sampler-required ") +
        "LiteRtCreateEnvironment symbol: " +
        (symbol_error != nullptr ? symbol_error : "symbol address is null");
    dlclose(litert_runtime);
    return SetError(error, LITERT_LM_STATUS_BACKEND_ERROR,
                    LITERT_LM_PHASE_ENGINE_CREATE, message);
  }

  AndroidGpuSidecars loaded;
  loaded.litert_runtime = litert_runtime;
  *out_sidecars = std::move(loaded);
  return LITERT_LM_STATUS_OK;
}

LiteRtLm_Status FindAndroidHardwareGpu(std::string* out_device_name,
                                       LiteRtLm_ErrorInfo* error) {
  if (out_device_name == nullptr) {
    return SetError(error, LITERT_LM_STATUS_INTERNAL_ERROR,
                    LITERT_LM_PHASE_ENGINE_CREATE,
                    "Internal hardware GPU result is null.");
  }
  out_device_name->clear();

  void* vulkan_library = dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
  if (vulkan_library == nullptr) {
    return SetError(error, LITERT_LM_STATUS_BACKEND_ERROR,
                    LITERT_LM_PHASE_ENGINE_CREATE,
                    "Strict GPU initialization could not load the Android "
                    "Vulkan loader. CPU/software fallback is disabled.");
  }

  auto get_instance_proc_addr =
      reinterpret_cast<PFN_vkGetInstanceProcAddr>(
          dlsym(vulkan_library, "vkGetInstanceProcAddr"));
  if (get_instance_proc_addr == nullptr) {
    dlclose(vulkan_library);
    return SetError(error, LITERT_LM_STATUS_BACKEND_ERROR,
                    LITERT_LM_PHASE_ENGINE_CREATE,
                    "The Android Vulkan loader does not expose "
                    "vkGetInstanceProcAddr.");
  }

  auto create_instance = reinterpret_cast<PFN_vkCreateInstance>(
      get_instance_proc_addr(VK_NULL_HANDLE, "vkCreateInstance"));
  if (create_instance == nullptr) {
    dlclose(vulkan_library);
    return SetError(error, LITERT_LM_STATUS_BACKEND_ERROR,
                    LITERT_LM_PHASE_ENGINE_CREATE,
                    "The Android Vulkan loader cannot create an instance.");
  }

  VkApplicationInfo application_info{};
  application_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  application_info.pApplicationName = "LiteRT-LM strict GPU verification";
  application_info.applicationVersion = 1;
  application_info.pEngineName = "LiteRT-LM";
  application_info.engineVersion = 1;
  application_info.apiVersion = VK_API_VERSION_1_0;

  VkInstanceCreateInfo instance_info{};
  instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  instance_info.pApplicationInfo = &application_info;

  VkInstance instance = VK_NULL_HANDLE;
  VkResult result = create_instance(&instance_info, nullptr, &instance);
  if (result != VK_SUCCESS || instance == VK_NULL_HANDLE) {
    dlclose(vulkan_library);
    return SetError(error, LITERT_LM_STATUS_BACKEND_ERROR,
                    LITERT_LM_PHASE_ENGINE_CREATE,
                    "Strict GPU initialization could not create an Android "
                    "Vulkan instance.",
                    static_cast<int32_t>(result));
  }

  auto destroy_instance = reinterpret_cast<PFN_vkDestroyInstance>(
      get_instance_proc_addr(instance, "vkDestroyInstance"));
  auto enumerate_devices = reinterpret_cast<PFN_vkEnumeratePhysicalDevices>(
      get_instance_proc_addr(instance, "vkEnumeratePhysicalDevices"));
  auto get_properties = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(
      get_instance_proc_addr(instance, "vkGetPhysicalDeviceProperties"));
  if (destroy_instance == nullptr || enumerate_devices == nullptr ||
      get_properties == nullptr) {
    if (destroy_instance != nullptr) destroy_instance(instance, nullptr);
    dlclose(vulkan_library);
    return SetError(error, LITERT_LM_STATUS_BACKEND_ERROR,
                    LITERT_LM_PHASE_ENGINE_CREATE,
                    "The Android Vulkan loader lacks physical-device "
                    "inspection functions.");
  }

  uint32_t device_count = 0;
  result = enumerate_devices(instance, &device_count, nullptr);
  std::vector<VkPhysicalDevice> devices;
  if (result == VK_SUCCESS && device_count != 0) {
    devices.resize(device_count);
    result = enumerate_devices(instance, &device_count, devices.data());
    if (result == VK_SUCCESS || result == VK_INCOMPLETE) {
      devices.resize(device_count);
    } else {
      devices.clear();
    }
  }

  VkPhysicalDeviceProperties best_properties{};
  bool found_hardware = false;
  for (VkPhysicalDevice device : devices) {
    VkPhysicalDeviceProperties properties{};
    get_properties(device, &properties);
    const bool is_hardware =
        properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ||
        properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
    if (!is_hardware || properties.deviceName[0] == '\0') continue;
    if (!found_hardware ||
        properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
      best_properties = properties;
      found_hardware = true;
    }
    if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) break;
  }

  destroy_instance(instance, nullptr);
  dlclose(vulkan_library);

  if (!found_hardware) {
    return SetError(
        error, LITERT_LM_STATUS_BACKEND_ERROR,
        LITERT_LM_PHASE_ENGINE_CREATE,
        "Strict GPU initialization found no discrete or integrated Android "
        "Vulkan device. CPU/SwiftShader/software adapters are not permitted.");
  }

  *out_device_name = best_properties.deviceName;
  return LITERT_LM_STATUS_OK;
}
#endif

LiteRtLm_Status MapStatus(const absl::Status& status,
                          LiteRtLm_ErrorPhase phase) {
  if (status.ok()) return LITERT_LM_STATUS_OK;
  switch (status.code()) {
    case absl::StatusCode::kInvalidArgument:
      return LITERT_LM_STATUS_INVALID_ARGUMENT;
    case absl::StatusCode::kFailedPrecondition:
      return LITERT_LM_STATUS_INVALID_STATE;
    case absl::StatusCode::kUnimplemented:
      return LITERT_LM_STATUS_NOT_SUPPORTED;
    case absl::StatusCode::kAlreadyExists:
    case absl::StatusCode::kAborted:
      return LITERT_LM_STATUS_BUSY;
    case absl::StatusCode::kDeadlineExceeded:
      return LITERT_LM_STATUS_TIMEOUT;
    case absl::StatusCode::kCancelled:
      return LITERT_LM_STATUS_CANCELLED;
    case absl::StatusCode::kNotFound:
    case absl::StatusCode::kDataLoss:
      return phase == LITERT_LM_PHASE_ENGINE_CREATE
                 ? LITERT_LM_STATUS_MODEL_ERROR
                 : LITERT_LM_STATUS_IO_ERROR;
    case absl::StatusCode::kResourceExhausted:
      return LITERT_LM_STATUS_RESOURCE_EXHAUSTED;
    case absl::StatusCode::kUnavailable:
      return LITERT_LM_STATUS_BACKEND_ERROR;
    default:
      return LITERT_LM_STATUS_INTERNAL_ERROR;
  }
}

LiteRtLm_Status SetAbslError(LiteRtLm_ErrorInfo* error,
                             LiteRtLm_ErrorPhase phase,
                             const absl::Status& status) {
  return SetError(error, MapStatus(status, phase), phase, status.ToString(),
                  static_cast<int32_t>(status.raw_code()),
                  absl::IsUnavailable(status) ||
                      absl::IsResourceExhausted(status));
}

template <typename Function>
LiteRtLm_Status Boundary(LiteRtLm_ErrorPhase phase, LiteRtLm_ErrorInfo* error,
                         Function&& function) noexcept {
  ClearError(error);
  try {
    return function();
  } catch (const std::bad_alloc&) {
    return SetError(error, LITERT_LM_STATUS_RESOURCE_EXHAUSTED, phase,
                    "Memory allocation failed.");
  } catch (const std::exception& exception) {
    return SetError(error, LITERT_LM_STATUS_INTERNAL_ERROR, phase,
                    exception.what());
  } catch (...) {
    return SetError(error, LITERT_LM_STATUS_INTERNAL_ERROR, phase,
                    "Unhandled non-standard exception.");
  }
}

bool IsEngineValid(LiteRtLm_Engine engine) {
  return engine != nullptr && engine->magic == kEngineMagic &&
         engine->engine != nullptr;
}

bool IsConversationValid(LiteRtLm_Conversation conversation) {
  return conversation != nullptr && conversation->magic == kConversationMagic &&
         IsEngineValid(conversation->owner);
}

bool IsRequestValid(LiteRtLm_Request request) {
  return request != nullptr && request->magic == kRequestMagic &&
         IsEngineValid(request->engine) &&
         request->conversation != nullptr;
}

Backend ToNativeBackend(LiteRtLm_Backend backend) {
  return backend == LITERT_LM_BACKEND_CPU ? Backend::CPU : Backend::GPU;
}

LiteRtLm_Backend ToPublicBackend(Backend backend) {
  return backend == Backend::CPU ? LITERT_LM_BACKEND_CPU
                                 : LITERT_LM_BACKEND_GPU;
}

int32_t GenerateSeed() {
  uint64_t value = static_cast<uint64_t>(
      std::chrono::high_resolution_clock::now().time_since_epoch().count());
  value ^= g_seed_sequence.fetch_add(1, std::memory_order_relaxed) *
           0x9E3779B97F4A7C15ull;
  value ^= reinterpret_cast<uintptr_t>(&value);
  value ^= value >> 30;
  value *= 0xBF58476D1CE4E5B9ull;
  value ^= value >> 27;
  value *= 0x94D049BB133111EBull;
  value ^= value >> 31;
  int32_t seed = static_cast<int32_t>(value & 0x7FFFFFFFu);
  return seed == 0 ? 1 : seed;
}

LiteRtLm_Status ValidateSampling(const LiteRtLm_Sampling& sampling,
                                 LiteRtLm_ErrorPhase phase,
                                 LiteRtLm_ErrorInfo* error) {
  if (sampling.struct_size <
      LITERT_LM_FIELD_END(LiteRtLm_Sampling, seed)) {
    return SetError(error, LITERT_LM_STATUS_INVALID_ARGUMENT,
                    phase, "Sampling.struct_size is too small.");
  }
  if (!std::isfinite(sampling.temperature) || sampling.temperature < 0.0f) {
    return SetError(error, LITERT_LM_STATUS_INVALID_ARGUMENT,
                    phase,
                    "Sampling.temperature must be finite and >= 0.");
  }
  if (!std::isfinite(sampling.top_p) || sampling.top_p < 0.0f ||
      sampling.top_p > 1.0f) {
    return SetError(error, LITERT_LM_STATUS_INVALID_ARGUMENT,
                    phase,
                    "Sampling.top_p must be finite and in [0, 1].");
  }
  if (sampling.top_k <= 0 || sampling.top_k > kMaximumTopK) {
    return SetError(error, LITERT_LM_STATUS_INVALID_ARGUMENT,
                    phase,
                    "Sampling.top_k must be in [1, 1024].");
  }
  if (sampling.seed_mode != LITERT_LM_SEED_RANDOM &&
      sampling.seed_mode != LITERT_LM_SEED_FIXED) {
    return SetError(error, LITERT_LM_STATUS_INVALID_ARGUMENT,
                    phase,
                    "Sampling.seed_mode is invalid.");
  }
  return LITERT_LM_STATUS_OK;
}

LiteRtLm_Sampling ResolveSampling(const LiteRtLm_Sampling& requested) {
  LiteRtLm_Sampling applied = requested;
  applied.struct_size = static_cast<uint32_t>(sizeof(applied));
  if (requested.seed_mode == LITERT_LM_SEED_RANDOM) {
    applied.seed = GenerateSeed();
  }
  // RequestInfo contains an immutable, directly reproducible snapshot.
  applied.seed_mode = LITERT_LM_SEED_FIXED;
  return applied;
}

bool SamplingEquals(const LiteRtLm_Sampling& left,
                    const LiteRtLm_Sampling& right) {
  return left.temperature == right.temperature && left.top_p == right.top_p &&
         left.top_k == right.top_k && left.seed_mode == right.seed_mode &&
         left.seed == right.seed;
}

void ApplySampling(const LiteRtLm_Sampling& sampling,
                   SessionConfig& session_config) {
  proto::SamplerParameters& params = session_config.GetMutableSamplerParams();
  params.set_type(proto::SamplerParameters::TOP_P);
  params.set_k(sampling.top_k);
  params.set_p(sampling.top_p);
  params.set_temperature(sampling.temperature);
  params.set_seed(sampling.seed);
}

LiteRtLm_Status ParseConversationDefinition(
    const LiteRtLm_ConversationConfig* config,
    LiteRtLm_ConversationDefinition* out_definition,
    LiteRtLm_ErrorInfo* error) {
  if (out_definition == nullptr) {
    return SetError(error, LITERT_LM_STATUS_INVALID_ARGUMENT,
                    LITERT_LM_PHASE_CONVERSATION_CREATE,
                    "Conversation definition output is null.");
  }

  LiteRtLm_ConversationConfig defaults = LiteRtLm_DefaultConversationConfig();
  if (config == nullptr) config = &defaults;
  if (config->struct_size <
          LITERT_LM_FIELD_END(LiteRtLm_ConversationConfig, tools_json) ||
      !IsViewValid(config->tools_json)) {
    return SetError(error, LITERT_LM_STATUS_INVALID_ARGUMENT,
                    LITERT_LM_PHASE_CONVERSATION_CREATE,
                    "ConversationConfig is invalid.");
  }

  LiteRtLm_ConversationDefinition definition;
  definition.tools_json = CopyView(config->tools_json);
  if (!definition.tools_json.empty()) {
    OrderedJson parsed_tools =
        OrderedJson::parse(definition.tools_json, nullptr, false);
    if (parsed_tools.is_discarded() || !parsed_tools.is_array()) {
      return SetError(error, LITERT_LM_STATUS_PARSE_ERROR,
                      LITERT_LM_PHASE_CONVERSATION_CREATE,
                      "tools_json must be a valid JSON array.");
    }
    definition.tools = NormalizeToolDeclarations(parsed_tools);
  }

  if (LITERT_LM_HAS_FIELD(config, LiteRtLm_ConversationConfig, flags)) {
    definition.flags = config->flags;
  }
  if ((definition.flags & ~kKnownConversationFlags) != 0) {
    return SetError(error, LITERT_LM_STATUS_INVALID_ARGUMENT,
                    LITERT_LM_PHASE_CONVERSATION_CREATE,
                    "ConversationConfig.flags contains unsupported bits.");
  }
  if (LITERT_LM_HAS_FIELD(config, LiteRtLm_ConversationConfig,
                          default_max_output_tokens)) {
    definition.default_max_output_tokens = config->default_max_output_tokens;
    if (definition.default_max_output_tokens >
        static_cast<uint32_t>(std::numeric_limits<int>::max())) {
      return SetError(error, LITERT_LM_STATUS_INVALID_ARGUMENT,
                      LITERT_LM_PHASE_CONVERSATION_CREATE,
                      "default_max_output_tokens exceeds the native integer range.");
    }
  }

  auto copy_optional_view = [&](LiteRtLm_StringView view,
                                std::string* output,
                                const char* field_name) -> LiteRtLm_Status {
    if (!IsViewValid(view)) {
      return SetError(error, LITERT_LM_STATUS_INVALID_ARGUMENT,
                      LITERT_LM_PHASE_CONVERSATION_CREATE,
                      std::string(field_name) + " is not a valid string view.");
    }
    *output = CopyView(view);
    return LITERT_LM_STATUS_OK;
  };

  if (LITERT_LM_HAS_FIELD(config, LiteRtLm_ConversationConfig,
                          system_message)) {
    LiteRtLm_Status status = copy_optional_view(
        config->system_message, &definition.system_message, "system_message");
    if (status != LITERT_LM_STATUS_OK) return status;
  }
  if (!definition.system_message.empty()) {
    OrderedJson content = OrderedJson::parse(
        definition.system_message, nullptr, false);
    if (content.is_discarded()) content = definition.system_message;
    OrderedJson message = OrderedJson::object();
    message["role"] = "system";
    message["content"] = std::move(content);
    definition.messages.push_back(std::move(message));
  }

  if (LITERT_LM_HAS_FIELD(config, LiteRtLm_ConversationConfig,
                          initial_messages_json)) {
    LiteRtLm_Status status = copy_optional_view(
        config->initial_messages_json, &definition.initial_messages_json,
        "initial_messages_json");
    if (status != LITERT_LM_STATUS_OK) return status;
  }
  if (!definition.initial_messages_json.empty()) {
    OrderedJson initial = OrderedJson::parse(
        definition.initial_messages_json, nullptr, false);
    if (initial.is_discarded() || !initial.is_array()) {
      return SetError(error, LITERT_LM_STATUS_PARSE_ERROR,
                      LITERT_LM_PHASE_CONVERSATION_CREATE,
                      "initial_messages_json must be a valid JSON array.");
    }
    for (auto& message : initial) {
      if (!message.is_object()) {
        return SetError(error, LITERT_LM_STATUS_PARSE_ERROR,
                        LITERT_LM_PHASE_CONVERSATION_CREATE,
                        "Every initial message must be a JSON object.");
      }
      definition.messages.push_back(std::move(message));
    }
  }

  if (LITERT_LM_HAS_FIELD(config, LiteRtLm_ConversationConfig,
                          extra_context_json)) {
    LiteRtLm_Status status = copy_optional_view(
        config->extra_context_json, &definition.extra_context_json,
        "extra_context_json");
    if (status != LITERT_LM_STATUS_OK) return status;
  }
  if (!definition.extra_context_json.empty()) {
    definition.extra_context = OrderedJson::parse(
        definition.extra_context_json, nullptr, false);
    if (definition.extra_context.is_discarded() ||
        !definition.extra_context.is_object()) {
      return SetError(error, LITERT_LM_STATUS_PARSE_ERROR,
                      LITERT_LM_PHASE_CONVERSATION_CREATE,
                      "extra_context_json must be a valid JSON object.");
    }
  }

  if (LITERT_LM_HAS_FIELD(config, LiteRtLm_ConversationConfig,
                          tool_call_channel)) {
    std::string channel;
    LiteRtLm_Status status = copy_optional_view(
        config->tool_call_channel, &channel, "tool_call_channel");
    if (status != LITERT_LM_STATUS_OK) return status;
    if (!channel.empty()) definition.tool_call_channel = std::move(channel);
  }
  if (LITERT_LM_HAS_FIELD(config, LiteRtLm_ConversationConfig, lora_path)) {
    LiteRtLm_Status status = copy_optional_view(
        config->lora_path, &definition.lora_path, "lora_path");
    if (status != LITERT_LM_STATUS_OK) return status;
  }
  if (LITERT_LM_HAS_FIELD(config, LiteRtLm_ConversationConfig,
                          audio_lora_path)) {
    LiteRtLm_Status status = copy_optional_view(
        config->audio_lora_path, &definition.audio_lora_path,
        "audio_lora_path");
    if (status != LITERT_LM_STATUS_OK) return status;
  }

  *out_definition = std::move(definition);
  return LITERT_LM_STATUS_OK;
}

absl::StatusOr<std::unique_ptr<Conversation>> BuildConversation(
    LiteRtLm_EngineObject& engine,
    const LiteRtLm_ConversationDefinition& definition,
    const LiteRtLm_Sampling& sampling,
    const OrderedJson* replayed_messages = nullptr) {
  SessionConfig session_config = SessionConfig::CreateDefault();
  ApplySampling(sampling, session_config);
  session_config.SetSamplerBackend(engine.sampler_backend);
  session_config.SetVisionModalityEnabled(false);
  session_config.SetAudioModalityEnabled(false);
  if (definition.default_max_output_tokens != 0) {
    session_config.SetMaxOutputTokens(
        static_cast<int>(definition.default_max_output_tokens));
  }
  if ((definition.flags &
       LITERT_LM_CONVERSATION_DISABLE_PROMPT_TEMPLATE) != 0) {
    session_config.SetApplyPromptTemplateInSession(false);
  }
  if (!definition.lora_path.empty()) {
    auto file = ScopedFile::Open(definition.lora_path);
    if (!file.ok()) return file.status();
    session_config.SetScopedLoraFile(
        std::make_shared<ScopedFile>(std::move(*file)));
  }
  if (!definition.audio_lora_path.empty()) {
    auto file = ScopedFile::Open(definition.audio_lora_path);
    if (!file.ok()) return file.status();
    session_config.SetAudioScopedLoraFile(
        std::make_shared<ScopedFile>(std::move(*file)));
  }

  JsonPreface preface;
  preface.messages = definition.messages;
  if (replayed_messages != nullptr && replayed_messages->is_array()) {
    for (const auto& message : *replayed_messages) {
      preface.messages.push_back(message);
    }
  }
  preface.tools = definition.tools;
  preface.extra_context = definition.extra_context;

  ConversationConfig::Builder builder;
  builder.SetSessionConfig(session_config)
      .SetPreface(preface)
      .SetConstraintProviderConfig(LlGuidanceConfig())
      .SetEnableConstrainedDecoding(
          (definition.flags &
           LITERT_LM_CONVERSATION_ENABLE_CONSTRAINED_DECODING) != 0 ||
          !definition.tools.empty())
      .SetPrefillPrefaceOnInit(
          (definition.flags & LITERT_LM_CONVERSATION_PREFILL_PREFACE) != 0)
      .SetFilterChannelContentFromKvCache(
          (definition.flags &
           LITERT_LM_CONVERSATION_FILTER_CHANNEL_CONTENT_FROM_KV) != 0)
      .SetReturnErrorOnParseFailure(
          (definition.flags &
           LITERT_LM_CONVERSATION_RETURN_ERROR_ON_PARSE_FAILURE) != 0)
      .SetReturnErrorOnMaxTokensReached(
          (definition.flags &
           LITERT_LM_CONVERSATION_RETURN_ERROR_ON_MAX_TOKENS) != 0)
      .SetEnableThinking(
          (definition.flags & LITERT_LM_CONVERSATION_ENABLE_THINKING) != 0)
      .SetStreamToolCalls(
          (definition.flags &
           LITERT_LM_CONVERSATION_STREAM_TOOL_CALLS) != 0,
          definition.tool_call_channel);

  auto config = builder.Build(*engine.engine);
  if (!config.ok()) return config.status();
  return Conversation::Create(*engine.engine, *config);
}

OrderedJson CompleteConversationHistory(
    const LiteRtLm_ConversationObject& conversation) {
  OrderedJson history = OrderedJson::array();
  if (conversation.definition.messages.is_array()) {
    for (const auto& message : conversation.definition.messages) {
      history.push_back(message);
    }
  }
  if (conversation.replayed_messages.is_array()) {
    for (const auto& message : conversation.replayed_messages) {
      history.push_back(message);
    }
  }
  if (conversation.conversation != nullptr) {
    for (const auto& message : conversation.conversation->GetHistory()) {
      history.push_back(message);
    }
  }
  return history;
}

OrderedJson ReplayableDynamicHistory(
    const LiteRtLm_ConversationObject& conversation) {
  OrderedJson history = OrderedJson::array();
  if (conversation.replayed_messages.is_array()) {
    for (const auto& message : conversation.replayed_messages) {
      history.push_back(message);
    }
  }
  if (conversation.conversation != nullptr) {
    for (const auto& message : conversation.conversation->GetHistory()) {
      history.push_back(message);
    }
  }
  return history;
}

std::string ExtractText(const OrderedJson& value) {
  std::string output;
  auto append = [&output](const OrderedJson& item) {
    if (item.is_string()) {
      output += item.get_ref<const std::string&>();
    } else if (item.is_object()) {
      auto text = item.find("text");
      if (text != item.end() && text->is_string()) {
        output += text->get_ref<const std::string&>();
      }
    }
  };

  if (!value.is_object()) return output;
  auto content = value.find("content");
  if (content != value.end()) {
    if (content->is_array()) {
      for (const auto& item : *content) append(item);
    } else {
      append(*content);
    }
  }
  return output;
}

struct FinalMessage {
  std::string text;
  std::string json;
  bool has_tool_call = false;
};

FinalMessage GetFinalMessage(Conversation* conversation) {
  FinalMessage final_message;
  if (conversation == nullptr) return final_message;
  conversation->AccessHistory(
      [&final_message](const std::vector<Message>& history) {
        for (auto item = history.rbegin(); item != history.rend(); ++item) {
          if (!item->is_object()) continue;
          auto role = item->find("role");
          if (role == item->end() || !role->is_string() ||
              role->get_ref<const std::string&>() != "assistant") {
            continue;
          }
          final_message.text = ExtractText(*item);
          final_message.json = item->dump();
          auto tool_calls = item->find("tool_calls");
          final_message.has_tool_call =
              tool_calls != item->end() && !tool_calls->is_null() &&
              !tool_calls->empty();
          break;
        }
      });
  return final_message;
}

uint64_t ToMicroseconds(std::chrono::steady_clock::duration duration) {
  auto value = std::chrono::duration_cast<std::chrono::microseconds>(duration)
                   .count();
  return value <= 0 ? 0u : static_cast<uint64_t>(value);
}

void CaptureBenchmarkBaseline(LiteRtLm_RequestObject& request) {
  if ((request.engine->flags & LITERT_LM_ENGINE_ENABLE_METRICS) == 0 ||
      request.native_conversation == nullptr) {
    return;
  }
  auto info = request.native_conversation->GetBenchmarkInfo();
  if (!info.ok()) return;
  request.benchmark_baseline_valid = true;
  request.prefill_turn_baseline = info->GetTotalPrefillTurns();
  request.decode_turn_baseline = info->GetTotalDecodeTurns();
}

void RefreshMetrics(LiteRtLm_RequestObject& request) {
  std::lock_guard<std::mutex> metrics_lock(request.metrics_mutex);
  {
    std::lock_guard<std::mutex> lock(request.mutex);
    if (request.metrics_finalized) return;
  }
  uint64_t valid = LITERT_LM_METRIC_TOTAL_TIME;
  uint64_t input_tokens = 0;
  uint64_t output_tokens = 0;
  uint64_t prefill_us = 0;
  uint64_t decode_us = 0;
  uint64_t context_tokens = 0;

  if (request.native_conversation != nullptr) {
    auto token_count = request.native_conversation->GetTokenCount();
    if (token_count.ok() && *token_count >= 0) {
      context_tokens = static_cast<uint64_t>(*token_count);
      valid |= LITERT_LM_METRIC_CONTEXT_TOKENS;
    }

    if (request.benchmark_baseline_valid) {
      auto info = request.native_conversation->GetBenchmarkInfo();
      if (info.ok()) {
        bool prefill_complete = true;
        for (uint64_t index = request.prefill_turn_baseline;
             index < info->GetTotalPrefillTurns(); ++index) {
          auto turn = info->GetPrefillTurn(static_cast<int>(index));
          if (!turn.ok()) {
            prefill_complete = false;
            break;
          }
          input_tokens += turn->num_tokens;
          int64_t duration = absl::ToInt64Microseconds(turn->duration);
          if (duration > 0) prefill_us += static_cast<uint64_t>(duration);
        }
        bool decode_complete = true;
        for (uint64_t index = request.decode_turn_baseline;
             index < info->GetTotalDecodeTurns(); ++index) {
          auto turn = info->GetDecodeTurn(static_cast<int>(index));
          if (!turn.ok()) {
            decode_complete = false;
            break;
          }
          output_tokens += turn->num_tokens;
          int64_t duration = absl::ToInt64Microseconds(turn->duration);
          if (duration > 0) decode_us += static_cast<uint64_t>(duration);
        }
        if (prefill_complete) {
          valid |= LITERT_LM_METRIC_INPUT_TOKENS |
                   LITERT_LM_METRIC_PREFILL_TIME;
        } else {
          input_tokens = 0;
          prefill_us = 0;
        }
        if (decode_complete) {
          valid |= LITERT_LM_METRIC_OUTPUT_TOKENS |
                   LITERT_LM_METRIC_DECODE_TIME;
          if (decode_us != 0) valid |= LITERT_LM_METRIC_OUTPUT_RATE;
        } else {
          output_tokens = 0;
          decode_us = 0;
        }
      }
    }
  }

  std::lock_guard<std::mutex> lock(request.mutex);
  request.valid_metrics |= valid;
  request.input_tokens = input_tokens;
  request.output_tokens = output_tokens;
  request.context_tokens = context_tokens;
  request.prefill_time_us = prefill_us;
  request.decode_time_us = decode_us;
  request.output_tokens_per_second =
      decode_us == 0
          ? 0.0
          : static_cast<double>(output_tokens) * 1000000.0 /
                static_cast<double>(decode_us);
  request.metrics_finalized = true;
}

void RefreshMetricsNoThrow(LiteRtLm_RequestObject& request) noexcept {
  try {
    RefreshMetrics(request);
  } catch (...) {
    // Metrics are diagnostic. They must never alter request completion.
  }
}

void ReleaseActiveRequest(LiteRtLm_RequestObject& request) {
  LiteRtLm_RequestObject* expected = &request;
  request.conversation->active_request.compare_exchange_strong(
      expected, nullptr, std::memory_order_acq_rel);
  expected = &request;
  request.engine->active_request.compare_exchange_strong(
      expected, nullptr, std::memory_order_acq_rel);
}

void MaybeReleaseActiveRequest(LiteRtLm_RequestObject& request) {
  bool release = false;
  {
    std::lock_guard<std::mutex> lock(request.mutex);
    release = request.terminal && !request.callback_in_progress &&
              !request.submit_in_progress;
  }
  if (release) ReleaseActiveRequest(request);
}

void FinishRequest(LiteRtLm_RequestObject& request, LiteRtLm_Status status,
                   LiteRtLm_FinishReason reason,
                   std::string_view error_message,
                   LiteRtLm_ErrorPhase error_phase = LITERT_LM_PHASE_NONE,
                   int32_t native_code = 0,
                   bool retryable = false) noexcept {
  std::lock_guard<std::recursive_mutex> callback_lock(request.callback_mutex);

  {
    std::lock_guard<std::mutex> lock(request.mutex);
    if (request.terminal) return;
  }

  FinalMessage final_message;
  std::string stable_error;
  try {
    if (status == LITERT_LM_STATUS_OK) {
      final_message = GetFinalMessage(request.native_conversation);
    } else {
      stable_error.assign(error_message.data(), error_message.size());
    }
  } catch (...) {
    status = LITERT_LM_STATUS_INTERNAL_ERROR;
    reason = LITERT_LM_FINISH_ERROR;
    error_phase = LITERT_LM_PHASE_DECODE;
    native_code = 0;
    retryable = false;
    try {
      stable_error = "Failed while preparing the terminal result.";
    } catch (...) {
      stable_error.clear();
    }
  }

  LiteRtLm_EventCallback callback = nullptr;
  void* user_data = nullptr;
  uint64_t sequence = 0;
  {
    std::lock_guard<std::mutex> lock(request.mutex);
    if (request.terminal) return;
    if (request.cancel_requested) {
      status = LITERT_LM_STATUS_CANCELLED;
      reason = LITERT_LM_FINISH_CANCELLED;
      error_phase = LITERT_LM_PHASE_CANCEL;
      native_code = 0;
      retryable = true;
      if (stable_error.empty()) {
        try {
          stable_error = "Request cancelled.";
        } catch (...) {
        }
      }
    }
    request.status = status;
    request.error_phase =
        status == LITERT_LM_STATUS_OK ? LITERT_LM_PHASE_NONE : error_phase;
    request.native_code =
        status == LITERT_LM_STATUS_OK ? 0 : native_code;
    request.retryable =
        status == LITERT_LM_STATUS_OK ? false : retryable;
    request.finish_reason = reason;
    request.state = status == LITERT_LM_STATUS_OK
                        ? LITERT_LM_REQUEST_SUCCEEDED
                        : (status == LITERT_LM_STATUS_CANCELLED
                               ? LITERT_LM_REQUEST_CANCELLED
                               : LITERT_LM_REQUEST_FAILED);
    request.end_time = std::chrono::steady_clock::now();
    request.total_time_us =
        ToMicroseconds(request.end_time - request.start_time);
    request.valid_metrics |= LITERT_LM_METRIC_TOTAL_TIME;
    if (!request.first_output_seen &&
        (!final_message.text.empty() || !final_message.json.empty())) {
      request.first_output_seen = true;
      request.first_output_time = request.end_time;
    }
    if (request.first_output_seen) {
      request.time_to_first_token_us =
          ToMicroseconds(request.first_output_time - request.start_time);
      request.valid_metrics |= LITERT_LM_METRIC_TTFT;
    }
    if (status == LITERT_LM_STATUS_OK) {
      if (!final_message.text.empty()) request.text.swap(final_message.text);
      request.response_json.swap(final_message.json);
      if (final_message.has_tool_call) {
        request.finish_reason = LITERT_LM_FINISH_TOOL_CALL;
      }
    } else {
      request.error_message.swap(stable_error);
    }
    request.terminal = true;
    callback = request.callback;
    user_data = request.user_data;
    sequence = ++request.sequence;
    request.terminal_event_dispatched = true;
    if (callback != nullptr) {
      request.callback_in_progress = true;
      ++request.callback_count;
    }
  }

  if (callback != nullptr) {
    LiteRtLm_Event event{};
    event.struct_size = static_cast<uint32_t>(sizeof(event));
    event.kind = LITERT_LM_EVENT_COMPLETE;
    event.status = request.status;
    event.request = &request;
    event.sequence = sequence;
    event.response_json = ViewOf(request.response_json);
    event.error_message = ViewOf(request.error_message);
    LiteRtLm_RequestObject* previous = g_callback_request;
    g_callback_request = &request;
    try {
      callback(&event, user_data);
    } catch (...) {
      // A foreign callback must never unwind through the runtime.
    }
    g_callback_request = previous;
  }

  {
    std::lock_guard<std::mutex> lock(request.mutex);
    request.callback_in_progress = false;
  }
  if (request.status != LITERT_LM_STATUS_OK || request.used_constraint) {
    request.conversation->needs_reset.store(true, std::memory_order_release);
  }
  MaybeReleaseActiveRequest(request);
  request.condition.notify_all();
}

void HandleNativeMessage(LiteRtLm_RequestObject* request,
                         absl::StatusOr<Message> message) noexcept {
  if (request == nullptr || request->magic != kRequestMagic) return;
  struct NativeCallbackGuard {
    explicit NativeCallbackGuard(LiteRtLm_RequestObject& value)
        : request(value) {
      request.native_callbacks_in_flight.fetch_add(1,
                                                   std::memory_order_acq_rel);
    }
    ~NativeCallbackGuard() {
      {
        std::lock_guard<std::mutex> lock(request.mutex);
        request.native_callbacks_in_flight.fetch_sub(
            1, std::memory_order_acq_rel);
      }
      request.condition.notify_all();
    }
    LiteRtLm_RequestObject& request;
  } native_callback_guard(*request);
  std::lock_guard<std::recursive_mutex> callback_lock(request->callback_mutex);
  try {
    if (!message.ok()) {
      LiteRtLm_Status status = MapStatus(message.status(),
                                         LITERT_LM_PHASE_DECODE);
      FinishRequest(*request, status,
                    status == LITERT_LM_STATUS_CANCELLED
                        ? LITERT_LM_FINISH_CANCELLED
                        : LITERT_LM_FINISH_ERROR,
                    message.status().ToString(), LITERT_LM_PHASE_DECODE,
                    static_cast<int32_t>(message.status().raw_code()),
                    absl::IsUnavailable(message.status()) ||
                        absl::IsResourceExhausted(message.status()));
      return;
    }

    if (message->empty()) {
      FinishRequest(*request, LITERT_LM_STATUS_OK,
                    LITERT_LM_FINISH_COMPLETE, {});
      return;
    }

    std::string delta = ExtractText(*message);
    std::string json = message->dump();
    LiteRtLm_EventKind event_kind = LITERT_LM_EVENT_TEXT_DELTA;
    if (message->is_object()) {
      auto tool_calls = message->find("tool_calls");
      auto channels = message->find("channels");
      if (tool_calls != message->end() && !tool_calls->is_null() &&
          !tool_calls->empty()) {
        event_kind = LITERT_LM_EVENT_TOOL_CALL_DELTA;
      } else if (channels != message->end() && !channels->is_null() &&
                 !channels->empty()) {
        event_kind = LITERT_LM_EVENT_CHANNEL_DELTA;
      }
    }
    LiteRtLm_EventCallback callback = nullptr;
    void* user_data = nullptr;
    uint64_t sequence = 0;
    {
      std::lock_guard<std::mutex> lock(request->mutex);
      if (request->terminal || request->cancel_requested) return;
      if (!delta.empty()) request->text += delta;
      if (!request->first_output_seen && (!delta.empty() || !json.empty())) {
        request->first_output_seen = true;
        request->first_output_time = std::chrono::steady_clock::now();
      }
      callback = request->callback;
      user_data = request->user_data;
      sequence = ++request->sequence;
      if (callback != nullptr) {
        request->callback_in_progress = true;
        ++request->callback_count;
      }
    }

    bool callback_failed = false;
    if (callback != nullptr) {
      LiteRtLm_Event event{};
      event.struct_size = static_cast<uint32_t>(sizeof(event));
      event.kind = event_kind;
      event.status = LITERT_LM_STATUS_OK;
      event.request = request;
      event.sequence = sequence;
      event.text_delta = ViewOf(delta);
      event.response_json = ViewOf(json);
      LiteRtLm_RequestObject* previous = g_callback_request;
      g_callback_request = request;
      try {
        callback(&event, user_data);
      } catch (...) {
        callback_failed = true;
      }
      g_callback_request = previous;
      {
        std::lock_guard<std::mutex> lock(request->mutex);
        request->callback_in_progress = false;
      }
      request->condition.notify_all();
    }

    if (callback_failed) {
      request->conversation->needs_reset.store(true, std::memory_order_release);
      request->native_conversation->CancelGroup(request->task_group_id);
      FinishRequest(*request, LITERT_LM_STATUS_INTERNAL_ERROR,
                    LITERT_LM_FINISH_ERROR,
                    "The client stream callback threw an exception.",
                    LITERT_LM_PHASE_CALLBACK);
    }
  } catch (...) {
    FinishRequest(*request, LITERT_LM_STATUS_INTERNAL_ERROR,
                   LITERT_LM_FINISH_ERROR,
                   "Native callback processing failed.",
                   LITERT_LM_PHASE_CALLBACK);
  }
}

struct SubmitReservation {
  LiteRtLm_RequestObject* request = nullptr;
  bool engine_count = false;
  bool conversation_count = false;
  bool engine_active = false;
  bool conversation_active = false;
  bool committed = false;

  ~SubmitReservation() {
    if (committed || request == nullptr) return;
    if (conversation_active || engine_active) ReleaseActiveRequest(*request);
    if (conversation_count) {
      request->conversation->request_count.fetch_sub(
          1, std::memory_order_acq_rel);
    }
    if (engine_count) {
      request->engine->request_count.fetch_sub(1, std::memory_order_acq_rel);
    }
    request->magic = 0;
  }
};

LiteRtLm_Status ApiCreateEngine(const LiteRtLm_EngineConfig* config,
                                LiteRtLm_Engine* out_engine,
                                LiteRtLm_ErrorInfo* error) {
  return Boundary(LITERT_LM_PHASE_ENGINE_CREATE, error,
                  [&]() -> LiteRtLm_Status {
    if (out_engine == nullptr) {
      return SetError(error, LITERT_LM_STATUS_INVALID_ARGUMENT,
                      LITERT_LM_PHASE_ENGINE_CREATE,
                      "out_engine must not be null.");
    }
    *out_engine = nullptr;
    if (config == nullptr ||
        config->struct_size <
            LITERT_LM_FIELD_END(LiteRtLm_EngineConfig, sampling)) {
      return SetError(error, LITERT_LM_STATUS_INVALID_ARGUMENT,
                      LITERT_LM_PHASE_ENGINE_CREATE,
                      "EngineConfig is null or has an invalid struct_size.");
    }
    if (!IsViewValid(config->model_path) || config->model_path.size == 0) {
      return SetError(error, LITERT_LM_STATUS_INVALID_ARGUMENT,
                      LITERT_LM_PHASE_ENGINE_CREATE,
                      "EngineConfig.model_path must be a non-empty UTF-8 string.");
    }
    if (config->backend != LITERT_LM_BACKEND_CPU &&
        config->backend != LITERT_LM_BACKEND_GPU) {
      return SetError(error, LITERT_LM_STATUS_INVALID_ARGUMENT,
                      LITERT_LM_PHASE_ENGINE_CREATE,
                      "EngineConfig.backend must explicitly be CPU or GPU.");
    }
    if ((config->flags & (LITERT_LM_ENGINE_ENABLE_VISION |
                          LITERT_LM_ENGINE_ENABLE_AUDIO)) != 0) {
      return SetError(
          error, LITERT_LM_STATUS_NOT_SUPPORTED,
          LITERT_LM_PHASE_ENGINE_CREATE,
          "The stable wrapper currently supports text only. Vision and audio "
          "are not exposed until their complete submodel pipelines can prove "
          "strict GPU acceleration.");
    }
    if ((config->flags & ~kKnownEngineFlags) != 0) {
      return SetError(error, LITERT_LM_STATUS_INVALID_ARGUMENT,
                      LITERT_LM_PHASE_ENGINE_CREATE,
                      "EngineConfig.flags contains unsupported bits.");
    }
    LiteRtLm_Status sampling_status = ValidateSampling(
        config->sampling, LITERT_LM_PHASE_ENGINE_CREATE, error);
    if (sampling_status != LITERT_LM_STATUS_OK) return sampling_status;

    const bool has_num_threads =
        LITERT_LM_HAS_FIELD(config, LiteRtLm_EngineConfig, num_threads);
    const bool has_prefill_chunk =
        LITERT_LM_HAS_FIELD(config, LiteRtLm_EngineConfig,
                            prefill_chunk_size);
    const bool has_cache_dir =
        LITERT_LM_HAS_FIELD(config, LiteRtLm_EngineConfig, cache_dir);
    const bool has_dispatch_dir =
        LITERT_LM_HAS_FIELD(config, LiteRtLm_EngineConfig,
                            dispatch_library_dir);
    if ((has_cache_dir && !IsViewValid(config->cache_dir)) ||
        (has_dispatch_dir && !IsViewValid(config->dispatch_library_dir))) {
      return SetError(error, LITERT_LM_STATUS_INVALID_ARGUMENT,
                      LITERT_LM_PHASE_ENGINE_CREATE,
                      "EngineConfig contains an invalid optional string view.");
    }
    if (has_num_threads &&
        config->num_threads > static_cast<uint32_t>(
                                  std::numeric_limits<int32_t>::max())) {
      return SetError(error, LITERT_LM_STATUS_INVALID_ARGUMENT,
                      LITERT_LM_PHASE_ENGINE_CREATE,
                      "num_threads exceeds the native integer range.");
    }
    if (has_prefill_chunk && config->prefill_chunk_size < 0) {
      return SetError(error, LITERT_LM_STATUS_INVALID_ARGUMENT,
                      LITERT_LM_PHASE_ENGINE_CREATE,
                      "prefill_chunk_size must be zero or positive.");
    }

    const std::string model_path = CopyView(config->model_path);
    const std::string cache_dir =
        has_cache_dir ? CopyView(config->cache_dir) : std::string();
    const std::string dispatch_library_dir =
        has_dispatch_dir ? CopyView(config->dispatch_library_dir)
                         : std::string();
    Backend backend = ToNativeBackend(config->backend);
    LiteRtLm_Sampling applied_sampling = ResolveSampling(config->sampling);

#if defined(_WIN32) || defined(__ANDROID__)
    std::string hardware_gpu_name;
#endif
#if defined(_WIN32)
    // Keep these references alive for the entire native Engine lifetime.  In
    // particular, do not depend on the host process working directory or PATH
    // to discover GPU plug-ins.  CPU engines intentionally skip this step.
    WindowsGpuSidecars gpu_sidecars;
    if (backend == Backend::GPU) {
      LiteRtLm_Status adapter_status =
          FindWindowsHardwareGpu(&hardware_gpu_name, error);
      if (adapter_status != LITERT_LM_STATUS_OK) return adapter_status;
      LiteRtLm_Status sidecar_status =
          LoadRequiredWindowsGpuSidecars(&gpu_sidecars, error);
      if (sidecar_status != LITERT_LM_STATUS_OK) return sidecar_status;
    }
#elif defined(__ANDROID__)
    AndroidGpuSidecars gpu_sidecars;
    if (backend == Backend::GPU) {
      LiteRtLm_Status adapter_status =
          FindAndroidHardwareGpu(&hardware_gpu_name, error);
      if (adapter_status != LITERT_LM_STATUS_OK) return adapter_status;
      LiteRtLm_Status sidecar_status =
          LoadRequiredAndroidGpuSidecars(&gpu_sidecars, error);
      if (sidecar_status != LITERT_LM_STATUS_OK) return sidecar_status;
    }
#endif

    auto assets = ModelAssets::Create(model_path);
    if (!assets.ok()) {
      return SetAbslError(error, LITERT_LM_PHASE_ENGINE_CREATE,
                          assets.status());
    }
    auto settings = EngineSettings::CreateDefault(
        std::move(*assets), backend, /*vision_backend=*/std::nullopt,
        /*audio_backend=*/std::nullopt, backend);
    if (!settings.ok()) {
      return SetAbslError(error, LITERT_LM_PHASE_ENGINE_CREATE,
                          settings.status());
    }

    auto& main_settings = settings->GetMutableMainExecutorSettings();
    if (!cache_dir.empty()) main_settings.SetCacheDir(cache_dir);
    if (!dispatch_library_dir.empty()) {
      main_settings.SetLitertDispatchLibDir(dispatch_library_dir);
    }
    if ((config->flags &
         LITERT_LM_ENGINE_DISABLE_PARALLEL_FILE_LOADING) != 0) {
      settings->SetParallelFileSectionLoading(false);
    }
    if (config->max_context_tokens != 0) {
      if (config->max_context_tokens >
          static_cast<uint32_t>(std::numeric_limits<int>::max())) {
        return SetError(error, LITERT_LM_STATUS_INVALID_ARGUMENT,
                        LITERT_LM_PHASE_ENGINE_CREATE,
                        "max_context_tokens exceeds the native integer range.");
      }
      main_settings.SetMaxNumTokens(
          static_cast<int>(config->max_context_tokens));
    }

    if (backend == Backend::CPU &&
        ((has_num_threads && config->num_threads != 0) ||
         (has_prefill_chunk && config->prefill_chunk_size != 0))) {
      auto cpu_config = main_settings.MutableBackendConfig<CpuConfig>();
      if (!cpu_config.ok()) {
        return SetAbslError(error, LITERT_LM_PHASE_ENGINE_CREATE,
                            cpu_config.status());
      }
      CpuConfig value = *cpu_config;
      if (config->num_threads != 0) {
        value.number_of_threads = config->num_threads;
      }
      if (config->prefill_chunk_size != 0) {
        value.prefill_chunk_size = config->prefill_chunk_size;
      }
      main_settings.SetBackendConfig(value);
    }

    if (backend == Backend::GPU) {
      auto gpu_config = main_settings.MutableBackendConfig<GpuConfig>();
      if (!gpu_config.ok()) {
        return SetAbslError(error, LITERT_LM_PHASE_ENGINE_CREATE,
                            gpu_config.status());
      }
      GpuConfig value = *gpu_config;
      value.max_top_k = static_cast<uint32_t>(applied_sampling.top_k);
      value.require_full_acceleration = true;
      main_settings.SetBackendConfig(value);

#if defined(_WIN32) || defined(__ANDROID__)
      AdvancedSettings advanced =
          main_settings.GetAdvancedSettings().value_or(AdvancedSettings());
      advanced.preferred_device_substr = hardware_gpu_name;
      main_settings.SetAdvancedSettings(advanced);
#endif
    }

    if ((config->flags & LITERT_LM_ENGINE_ENABLE_METRICS) != 0) {
      AdvancedSettings advanced =
          main_settings.GetAdvancedSettings().value_or(AdvancedSettings());
      advanced.is_benchmark = true;
      main_settings.SetAdvancedSettings(advanced);
      settings->GetMutableBenchmarkParams() = proto::BenchmarkParams();
    }

    auto native_engine = EngineFactory::CreateDefault(std::move(*settings));
    if (!native_engine.ok()) {
      LiteRtLm_Status code =
          MapStatus(native_engine.status(), LITERT_LM_PHASE_ENGINE_CREATE);
      return SetError(error, code, LITERT_LM_PHASE_ENGINE_CREATE,
                      native_engine.status().ToString(),
                      static_cast<int32_t>(native_engine.status().raw_code()),
                      absl::IsUnavailable(native_engine.status()) ||
                          absl::IsResourceExhausted(native_engine.status()));
    }

    Backend resolved = (*native_engine)
                           ->GetEngineSettings()
                           .GetMainExecutorSettings()
                           .GetBackend();
    Backend sampler = (*native_engine)
                          ->GetEngineSettings()
                          .GetMainExecutorSettings()
                          .GetSamplerBackend();
    if (resolved != backend) {
      return SetError(
          error, LITERT_LM_STATUS_BACKEND_ERROR,
          LITERT_LM_PHASE_ENGINE_CREATE,
          "The live Engine resolved a backend different from the explicitly "
          "requested backend; fallback is not permitted.");
    }
    if (sampler != backend) {
      return SetError(
          error, LITERT_LM_STATUS_BACKEND_ERROR,
          LITERT_LM_PHASE_ENGINE_CREATE,
          "The live Engine configured a sampler backend different from the "
          "explicitly requested backend; sampler fallback is not permitted.");
    }

    auto handle = std::unique_ptr<LiteRtLm_EngineObject>(
        new (std::nothrow) LiteRtLm_EngineObject());
    if (!handle) {
      return SetError(error, LITERT_LM_STATUS_RESOURCE_EXHAUSTED,
                      LITERT_LM_PHASE_ENGINE_CREATE,
                      "Failed to allocate the Engine handle.");
    }
    handle->magic = kEngineMagic;
#if defined(_WIN32)
    handle->gpu_sidecars = std::move(gpu_sidecars);
#elif defined(__ANDROID__)
    handle->gpu_sidecars = std::move(gpu_sidecars);
#endif
    handle->engine = std::move(*native_engine);
    handle->configured_backend = config->backend;
    handle->resolved_backend = resolved;
    handle->sampler_backend = sampler;
    handle->flags = config->flags;
    handle->max_context_tokens = config->max_context_tokens;
    handle->applied_sampling = applied_sampling;
    handle->backend_description =
        std::string("live Engine configured=") +
        std::string(GetBackendString(resolved)) + "; sampler=" +
        std::string(GetBackendString(sampler)) +
        (resolved == Backend::GPU
             ? "; strict text pipeline verified fully GPU accelerated"
             : "; explicit CPU pipeline");
#if defined(_WIN32) || defined(__ANDROID__)
    if (resolved == Backend::GPU) {
      handle->backend_description +=
          std::string("; hardware adapter=") + hardware_gpu_name;
    }
#endif
    *out_engine = handle.release();
    return LITERT_LM_STATUS_OK;
  });
}

LiteRtLm_Status ApiGetEngineInfo(LiteRtLm_Engine engine,
                                 LiteRtLm_EngineInfo* out_info,
                                 LiteRtLm_ErrorInfo* error) {
  return Boundary(LITERT_LM_PHASE_API, error, [&]() -> LiteRtLm_Status {
    if (!IsEngineValid(engine)) {
      return SetError(error, LITERT_LM_STATUS_INVALID_ARGUMENT,
                      LITERT_LM_PHASE_API, "Engine handle is invalid.");
    }
    if (out_info == nullptr ||
        out_info->struct_size <
            LITERT_LM_FIELD_END(LiteRtLm_EngineInfo, backend_description)) {
      return SetError(error, LITERT_LM_STATUS_INVALID_ARGUMENT,
                      LITERT_LM_PHASE_API,
                      "EngineInfo is null or has an invalid struct_size.");
    }
    uint32_t capacity = out_info->struct_size;
    LiteRtLm_EngineInfo value{};
    value.struct_size = static_cast<uint32_t>(sizeof(value));
    value.configured_backend = engine->configured_backend;
    value.resolved_backend = ToPublicBackend(engine->resolved_backend);
    value.configured_sampler_backend =
        ToPublicBackend(engine->sampler_backend);
#if defined(_WIN32) || defined(__ANDROID__)
    value.backend_evidence =
        engine->resolved_backend == Backend::GPU
            ? LITERT_LM_BACKEND_EVIDENCE_FULLY_ACCELERATED
            : LITERT_LM_BACKEND_EVIDENCE_ENGINE_CONFIGURED;
#else
    value.backend_evidence = LITERT_LM_BACKEND_EVIDENCE_ENGINE_CONFIGURED;
#endif
    value.enabled_modalities = LITERT_LM_MODALITY_TEXT;
    value.max_context_tokens = engine->max_context_tokens;
    value.conversation_count =
        engine->conversation_count.load(std::memory_order_acquire);
    value.active_request_count =
        engine->active_request.load(std::memory_order_acquire) == nullptr ? 0u
                                                                          : 1u;
    value.retained_request_count =
        engine->request_count.load(std::memory_order_acquire);
    value.applied_sampling = engine->applied_sampling;
    value.backend_description = ViewOf(engine->backend_description);
    std::memcpy(out_info, &value,
                std::min<size_t>(capacity, sizeof(LiteRtLm_EngineInfo)));
    return LITERT_LM_STATUS_OK;
  });
}

LiteRtLm_Status ApiDestroyEngine(LiteRtLm_Engine engine,
                                 LiteRtLm_ErrorInfo* error) {
  return Boundary(LITERT_LM_PHASE_DESTROY, error,
                  [&]() -> LiteRtLm_Status {
    if (!IsEngineValid(engine)) {
      return SetError(error, LITERT_LM_STATUS_INVALID_ARGUMENT,
                      LITERT_LM_PHASE_DESTROY, "Engine handle is invalid.");
    }
    if (engine->active_request.load(std::memory_order_acquire) != nullptr ||
        engine->request_count.load(std::memory_order_acquire) != 0 ||
        engine->conversation_count.load(std::memory_order_acquire) != 0) {
      return SetError(error, LITERT_LM_STATUS_BUSY,
                      LITERT_LM_PHASE_DESTROY,
                      "Destroy all Requests and Conversations before the Engine.");
    }
    engine->magic = 0;
    delete engine;
    return LITERT_LM_STATUS_OK;
  });
}

LiteRtLm_Status ApiCreateConversation(
    LiteRtLm_Engine engine, const LiteRtLm_ConversationConfig* config,
    LiteRtLm_Conversation* out_conversation, LiteRtLm_ErrorInfo* error) {
  return Boundary(LITERT_LM_PHASE_CONVERSATION_CREATE, error,
                  [&]() -> LiteRtLm_Status {
    if (!IsEngineValid(engine)) {
      return SetError(error, LITERT_LM_STATUS_INVALID_ARGUMENT,
                      LITERT_LM_PHASE_CONVERSATION_CREATE,
                      "Engine handle is invalid.");
    }
    if (g_callback_request != nullptr &&
        g_callback_request->engine == engine) {
      return SetError(error, LITERT_LM_STATUS_INVALID_STATE,
                      LITERT_LM_PHASE_CALLBACK,
                      "Cannot create a Conversation reentrantly from its Engine callback.");
    }
    std::lock_guard<std::mutex> operation_lock(engine->operation_mutex);
    if (out_conversation == nullptr) {
      return SetError(error, LITERT_LM_STATUS_INVALID_ARGUMENT,
                      LITERT_LM_PHASE_CONVERSATION_CREATE,
                      "out_conversation must not be null.");
    }
    *out_conversation = nullptr;
    if (engine->request_count.load(std::memory_order_acquire) != 0) {
      return SetError(error, LITERT_LM_STATUS_BUSY,
                      LITERT_LM_PHASE_CONVERSATION_CREATE,
                      "Create Conversations only while the Engine is idle.");
    }

    LiteRtLm_ConversationDefinition definition;
    LiteRtLm_Status definition_status =
        ParseConversationDefinition(config, &definition, error);
    if (definition_status != LITERT_LM_STATUS_OK) return definition_status;

    auto native =
        BuildConversation(*engine, definition, engine->applied_sampling);
    if (!native.ok()) {
      return SetAbslError(error, LITERT_LM_PHASE_CONVERSATION_CREATE,
                          native.status());
    }
    auto handle = std::unique_ptr<LiteRtLm_ConversationObject>(
        new (std::nothrow) LiteRtLm_ConversationObject());
    if (!handle) {
      return SetError(error, LITERT_LM_STATUS_RESOURCE_EXHAUSTED,
                      LITERT_LM_PHASE_CONVERSATION_CREATE,
                      "Failed to allocate the Conversation handle.");
    }
    handle->magic = kConversationMagic;
    handle->owner = engine;
    handle->conversation = std::move(*native);
    handle->native_sampling = engine->applied_sampling;
    handle->definition = std::move(definition);
    engine->conversation_count.fetch_add(1, std::memory_order_acq_rel);
    *out_conversation = handle.release();
    return LITERT_LM_STATUS_OK;
  });
}

LiteRtLm_Status ApiResetConversation(LiteRtLm_Conversation conversation,
                                     LiteRtLm_ErrorInfo* error) {
  return Boundary(LITERT_LM_PHASE_CONTEXT_PREPARE, error,
                  [&]() -> LiteRtLm_Status {
    if (!IsConversationValid(conversation)) {
      return SetError(error, LITERT_LM_STATUS_INVALID_ARGUMENT,
                      LITERT_LM_PHASE_CONTEXT_PREPARE,
                      "Conversation handle is invalid.");
    }
    if (g_callback_request != nullptr &&
        g_callback_request->engine == conversation->owner) {
      return SetError(error, LITERT_LM_STATUS_INVALID_STATE,
                      LITERT_LM_PHASE_CALLBACK,
                      "Cannot reset a Conversation from its Engine callback.");
    }
    std::lock_guard<std::mutex> operation_lock(
        conversation->owner->operation_mutex);
    if (conversation->owner->request_count.load(std::memory_order_acquire) != 0 ||
        conversation->active_request.load(std::memory_order_acquire) != nullptr ||
        conversation->request_count.load(std::memory_order_acquire) != 0) {
      return SetError(error, LITERT_LM_STATUS_BUSY,
                      LITERT_LM_PHASE_CONTEXT_PREPARE,
                      "Destroy the current Request before resetting its Conversation.");
    }
    {
      std::lock_guard<std::mutex> lock(conversation->native_mutex);
      conversation->conversation.reset();
      conversation->needs_reset.store(true, std::memory_order_release);
    }
    auto replacement = BuildConversation(
        *conversation->owner, conversation->definition,
        conversation->owner->applied_sampling);
    if (!replacement.ok()) {
      return SetAbslError(error, LITERT_LM_PHASE_CONTEXT_PREPARE,
                          replacement.status());
    }
    {
      std::lock_guard<std::mutex> lock(conversation->native_mutex);
      conversation->conversation = std::move(*replacement);
      conversation->native_sampling = conversation->owner->applied_sampling;
      conversation->replayed_messages = OrderedJson::array();
      conversation->generation.fetch_add(1, std::memory_order_acq_rel);
      conversation->needs_reset.store(false, std::memory_order_release);
    }
    return LITERT_LM_STATUS_OK;
  });
}

LiteRtLm_Status ApiDestroyConversation(LiteRtLm_Conversation conversation,
                                       LiteRtLm_ErrorInfo* error) {
  return Boundary(LITERT_LM_PHASE_DESTROY, error,
                  [&]() -> LiteRtLm_Status {
    if (!IsConversationValid(conversation)) {
      return SetError(error, LITERT_LM_STATUS_INVALID_ARGUMENT,
                      LITERT_LM_PHASE_DESTROY,
                      "Conversation handle is invalid.");
    }
    if (g_callback_request != nullptr &&
        g_callback_request->engine == conversation->owner) {
      return SetError(error, LITERT_LM_STATUS_INVALID_STATE,
                      LITERT_LM_PHASE_CALLBACK,
                      "Cannot destroy a Conversation from its Engine callback.");
    }
    std::lock_guard<std::mutex> operation_lock(
        conversation->owner->operation_mutex);
    if (conversation->active_request.load(std::memory_order_acquire) != nullptr ||
        conversation->request_count.load(std::memory_order_acquire) != 0) {
      return SetError(error, LITERT_LM_STATUS_BUSY,
                      LITERT_LM_PHASE_DESTROY,
                      "Destroy every Request before its Conversation.");
    }
    LiteRtLm_EngineObject* owner = conversation->owner;
    conversation->magic = 0;
    delete conversation;
    owner->conversation_count.fetch_sub(1, std::memory_order_acq_rel);
    return LITERT_LM_STATUS_OK;
  });
}

LiteRtLm_Status ApiCloneConversation(
    LiteRtLm_Conversation source, LiteRtLm_Conversation* out_conversation,
    LiteRtLm_ErrorInfo* error) {
  return Boundary(LITERT_LM_PHASE_CONVERSATION_CREATE, error,
                  [&]() -> LiteRtLm_Status {
    if (!IsConversationValid(source) || out_conversation == nullptr) {
      return SetError(error, LITERT_LM_STATUS_INVALID_ARGUMENT,
                      LITERT_LM_PHASE_CONVERSATION_CREATE,
                      "Source Conversation or output pointer is invalid.");
    }
    *out_conversation = nullptr;
    if (g_callback_request != nullptr &&
        g_callback_request->engine == source->owner) {
      return SetError(error, LITERT_LM_STATUS_INVALID_STATE,
                      LITERT_LM_PHASE_CALLBACK,
                      "Cannot clone a Conversation from its Engine callback.");
    }
    std::lock_guard<std::mutex> operation_lock(source->owner->operation_mutex);
    if (source->owner->request_count.load(std::memory_order_acquire) != 0 ||
        source->request_count.load(std::memory_order_acquire) != 0 ||
        source->needs_reset.load(std::memory_order_acquire)) {
      return SetError(error, LITERT_LM_STATUS_BUSY,
                      LITERT_LM_PHASE_CONVERSATION_CREATE,
                      "Clone only an idle, reusable Conversation.");
    }

    absl::StatusOr<std::unique_ptr<Conversation>> cloned;
    {
      std::lock_guard<std::mutex> lock(source->native_mutex);
      if (source->conversation == nullptr) {
        return SetError(error, LITERT_LM_STATUS_INVALID_STATE,
                        LITERT_LM_PHASE_CONVERSATION_CREATE,
                        "Source Conversation has no native session.");
      }
      cloned = source->conversation->Clone();
    }
    if (!cloned.ok()) {
      return SetAbslError(error, LITERT_LM_PHASE_CONVERSATION_CREATE,
                          cloned.status());
    }

    auto handle = std::unique_ptr<LiteRtLm_ConversationObject>(
        new (std::nothrow) LiteRtLm_ConversationObject());
    if (!handle) {
      return SetError(error, LITERT_LM_STATUS_RESOURCE_EXHAUSTED,
                      LITERT_LM_PHASE_CONVERSATION_CREATE,
                      "Failed to allocate the cloned Conversation handle.");
    }
    handle->magic = kConversationMagic;
    handle->owner = source->owner;
    handle->conversation = std::move(*cloned);
    handle->native_sampling = source->native_sampling;
    handle->definition = source->definition;
    handle->replayed_messages = source->replayed_messages;
    handle->generation.store(
        source->generation.load(std::memory_order_acquire),
        std::memory_order_release);
    source->owner->conversation_count.fetch_add(1, std::memory_order_acq_rel);
    *out_conversation = handle.release();
    return LITERT_LM_STATUS_OK;
  });
}

LiteRtLm_Status ApiGetConversationInfo(
    LiteRtLm_Conversation conversation, LiteRtLm_ConversationInfo* out_info,
    LiteRtLm_ErrorInfo* error) {
  return Boundary(LITERT_LM_PHASE_API, error, [&]() -> LiteRtLm_Status {
    if (!IsConversationValid(conversation)) {
      return SetError(error, LITERT_LM_STATUS_INVALID_ARGUMENT,
                      LITERT_LM_PHASE_API,
                      "Conversation handle is invalid.");
    }
    if (out_info == nullptr ||
        out_info->struct_size <
            LITERT_LM_FIELD_END(LiteRtLm_ConversationInfo, tools_json)) {
      return SetError(error, LITERT_LM_STATUS_INVALID_ARGUMENT,
                      LITERT_LM_PHASE_API,
                      "ConversationInfo is null or has an invalid struct_size.");
    }
    std::lock_guard<std::mutex> operation_lock(
        conversation->owner->operation_mutex);
    if (conversation->owner->request_count.load(std::memory_order_acquire) !=
        0) {
      return SetError(error, LITERT_LM_STATUS_BUSY,
                      LITERT_LM_PHASE_API,
                      "Conversation info is stable only while the Engine is idle.");
    }

    uint64_t history_count = 0;
    uint64_t token_count = 0;
    {
      std::lock_guard<std::mutex> lock(conversation->native_mutex);
      if (conversation->conversation == nullptr) {
        return SetError(error, LITERT_LM_STATUS_INVALID_STATE,
                        LITERT_LM_PHASE_API,
                        "Conversation has no native session.");
      }
      OrderedJson history = CompleteConversationHistory(*conversation);
      history_count = static_cast<uint64_t>(history.size());
      auto native_token_count = conversation->conversation->GetTokenCount();
      if (!native_token_count.ok()) {
        return SetAbslError(error, LITERT_LM_PHASE_API,
                            native_token_count.status());
      }
      if (*native_token_count > 0) {
        token_count = static_cast<uint64_t>(*native_token_count);
      }
    }

    const uint32_t capacity = out_info->struct_size;
    LiteRtLm_ConversationInfo value{};
    value.struct_size = static_cast<uint32_t>(sizeof(value));
    value.flags = conversation->definition.flags;
    value.active_request_count =
        conversation->active_request.load(std::memory_order_acquire) == nullptr
            ? 0u
            : 1u;
    value.default_max_output_tokens =
        conversation->definition.default_max_output_tokens;
    value.generation =
        conversation->generation.load(std::memory_order_acquire);
    value.history_message_count = history_count;
    value.kv_token_count = token_count;
    value.tools_json = ViewOf(conversation->definition.tools_json);
    std::memcpy(out_info, &value,
                std::min<size_t>(capacity, sizeof(value)));
    return LITERT_LM_STATUS_OK;
  });
}

LiteRtLm_Status ApiCopyConversationHistoryJson(
    LiteRtLm_Conversation conversation, char* out_json, uint64_t* inout_size,
    LiteRtLm_ErrorInfo* error) {
  return Boundary(LITERT_LM_PHASE_API, error, [&]() -> LiteRtLm_Status {
    if (!IsConversationValid(conversation)) {
      return SetError(error, LITERT_LM_STATUS_INVALID_ARGUMENT,
                      LITERT_LM_PHASE_API,
                      "Conversation handle is invalid.");
    }
    std::lock_guard<std::mutex> operation_lock(
        conversation->owner->operation_mutex);
    if (conversation->owner->request_count.load(std::memory_order_acquire) !=
        0) {
      return SetError(error, LITERT_LM_STATUS_BUSY,
                      LITERT_LM_PHASE_API,
                      "Conversation history is stable only while the Engine is idle.");
    }
    std::string history;
    {
      std::lock_guard<std::mutex> lock(conversation->native_mutex);
      if (conversation->conversation == nullptr) {
        return SetError(error, LITERT_LM_STATUS_INVALID_STATE,
                        LITERT_LM_PHASE_API,
                        "Conversation has no native session.");
      }
      history = CompleteConversationHistory(*conversation).dump();
    }
    return CopyStringToCaller(history, out_json, inout_size,
                              LITERT_LM_PHASE_API, error);
  });
}

LiteRtLm_Status ApiSubmitChat(LiteRtLm_Conversation conversation,
                              const LiteRtLm_ChatRequest* chat,
                              LiteRtLm_Request* out_request,
                              LiteRtLm_ErrorInfo* error) {
  return Boundary(LITERT_LM_PHASE_CONTEXT_PREPARE, error,
                  [&]() -> LiteRtLm_Status {
    if (!IsConversationValid(conversation)) {
      return SetError(error, LITERT_LM_STATUS_INVALID_ARGUMENT,
                      LITERT_LM_PHASE_CONTEXT_PREPARE,
                      "Conversation handle is invalid.");
    }
    if (g_callback_request != nullptr &&
        g_callback_request->engine == conversation->owner) {
      return SetError(error, LITERT_LM_STATUS_INVALID_STATE,
                      LITERT_LM_PHASE_CALLBACK,
                      "Cannot submit a new Request from its Engine callback.");
    }
    std::lock_guard<std::mutex> operation_lock(
        conversation->owner->operation_mutex);
    if (out_request == nullptr) {
      return SetError(error, LITERT_LM_STATUS_INVALID_ARGUMENT,
                      LITERT_LM_PHASE_CONTEXT_PREPARE,
                      "out_request must not be null.");
    }
    *out_request = nullptr;
    if (chat == nullptr ||
        chat->struct_size <
            LITERT_LM_FIELD_END(LiteRtLm_ChatRequest, user_data) ||
        !IsViewValid(chat->messages_json) || chat->messages_json.size == 0 ||
        !IsViewValid(chat->constraint)) {
      return SetError(error, LITERT_LM_STATUS_INVALID_ARGUMENT,
                      LITERT_LM_PHASE_CONTEXT_PREPARE,
                      "ChatRequest is invalid or contains no messages.");
    }
    if (chat->context_mode != LITERT_LM_CONTEXT_CONTINUE &&
        chat->context_mode != LITERT_LM_CONTEXT_REPLACE) {
      return SetError(error, LITERT_LM_STATUS_INVALID_ARGUMENT,
                      LITERT_LM_PHASE_CONTEXT_PREPARE,
                      "ChatRequest.context_mode is invalid.");
    }
    if (chat->constraint_kind > LITERT_LM_CONSTRAINT_LARK) {
      return SetError(error, LITERT_LM_STATUS_INVALID_ARGUMENT,
                      LITERT_LM_PHASE_CONTEXT_PREPARE,
                      "ChatRequest.constraint_kind is invalid.");
    }
    if (chat->constraint_kind != LITERT_LM_CONSTRAINT_NONE &&
        chat->constraint.size == 0) {
      return SetError(error, LITERT_LM_STATUS_INVALID_ARGUMENT,
                      LITERT_LM_PHASE_CONTEXT_PREPARE,
                      "A non-empty constraint is required for constrained decoding.");
    }
    if (chat->max_output_tokens >
        static_cast<uint32_t>(std::numeric_limits<int>::max())) {
      return SetError(error, LITERT_LM_STATUS_INVALID_ARGUMENT,
                      LITERT_LM_PHASE_CONTEXT_PREPARE,
                      "max_output_tokens exceeds the native integer range.");
    }
    uint32_t chat_flags = 0;
    bool override_sampling = false;
    bool override_tools = false;
    std::string requested_tools_json;
    bool override_extra_context = false;
    std::string requested_extra_context_json;
    LiteRtLm_Sampling applied_sampling = conversation->owner->applied_sampling;
    if (LITERT_LM_HAS_FIELD(chat, LiteRtLm_ChatRequest, flags)) {
      chat_flags = chat->flags;
      if ((chat_flags & ~kKnownChatFlags) != 0) {
        return SetError(error, LITERT_LM_STATUS_INVALID_ARGUMENT,
                        LITERT_LM_PHASE_CONTEXT_PREPARE,
                        "ChatRequest contains unknown flags.");
      }
      override_sampling =
          (chat_flags & LITERT_LM_CHAT_OVERRIDE_SAMPLING) != 0;
      if (override_sampling) {
        if (!LITERT_LM_HAS_FIELD(chat, LiteRtLm_ChatRequest, sampling)) {
          return SetError(error, LITERT_LM_STATUS_INVALID_ARGUMENT,
                          LITERT_LM_PHASE_CONTEXT_PREPARE,
                          "ChatRequest sampling override is truncated.");
        }
        LiteRtLm_Status sampling_status = ValidateSampling(
            chat->sampling, LITERT_LM_PHASE_CONTEXT_PREPARE, error);
        if (sampling_status != LITERT_LM_STATUS_OK) return sampling_status;
        applied_sampling = ResolveSampling(chat->sampling);
      }
      override_tools = (chat_flags & LITERT_LM_CHAT_OVERRIDE_TOOLS) != 0;
      if (override_tools) {
        if (!LITERT_LM_HAS_FIELD(chat, LiteRtLm_ChatRequest, tools_json) ||
            !IsViewValid(chat->tools_json)) {
          return SetError(error, LITERT_LM_STATUS_INVALID_ARGUMENT,
                          LITERT_LM_PHASE_CONTEXT_PREPARE,
                          "ChatRequest tools override is truncated or invalid.");
        }
        requested_tools_json = CopyView(chat->tools_json);
        OrderedJson tools = OrderedJson::parse(
            requested_tools_json, nullptr, false);
        if (tools.is_discarded() || !tools.is_array()) {
          return SetError(error, LITERT_LM_STATUS_PARSE_ERROR,
                          LITERT_LM_PHASE_CONTEXT_PREPARE,
                          "ChatRequest.tools_json must be a valid JSON array.");
        }
      }
      override_extra_context =
          (chat_flags & LITERT_LM_CHAT_OVERRIDE_EXTRA_CONTEXT) != 0;
      if (override_extra_context) {
        if (!LITERT_LM_HAS_FIELD(chat, LiteRtLm_ChatRequest,
                                 extra_context_json) ||
            !IsViewValid(chat->extra_context_json)) {
          return SetError(
              error, LITERT_LM_STATUS_INVALID_ARGUMENT,
              LITERT_LM_PHASE_CONTEXT_PREPARE,
              "ChatRequest extra-context override is truncated or invalid.");
        }
        requested_extra_context_json = CopyView(chat->extra_context_json);
        OrderedJson extra_context = OrderedJson::parse(
            requested_extra_context_json, nullptr, false);
        if (extra_context.is_discarded() || !extra_context.is_object()) {
          return SetError(
              error, LITERT_LM_STATUS_PARSE_ERROR,
              LITERT_LM_PHASE_CONTEXT_PREPARE,
              "ChatRequest.extra_context_json must be a JSON object.");
        }
      }
    }
    if (chat->context_mode == LITERT_LM_CONTEXT_CONTINUE &&
        conversation->needs_reset.load(std::memory_order_acquire)) {
      return SetError(
          error, LITERT_LM_STATUS_INVALID_STATE,
          LITERT_LM_PHASE_CONTEXT_PREPARE,
          "The prior request failed or was cancelled. Reset or replace the "
          "Conversation before continuing.");
    }

    OrderedJson messages = OrderedJson::parse(
        CopyView(chat->messages_json), nullptr, false);
    if (messages.is_discarded() ||
        (!messages.is_object() && !messages.is_array()) ||
        (messages.is_array() && messages.empty())) {
      return SetError(error, LITERT_LM_STATUS_PARSE_ERROR,
                      LITERT_LM_PHASE_CONTEXT_PREPARE,
                      "messages_json must be a JSON object or non-empty array.");
    }
    if (messages.is_array()) {
      for (const auto& message : messages) {
        if (!message.is_object()) {
          return SetError(error, LITERT_LM_STATUS_PARSE_ERROR,
                          LITERT_LM_PHASE_CONTEXT_PREPARE,
                          "Every element of messages_json must be an object.");
        }
      }
    }

    auto request = std::unique_ptr<LiteRtLm_RequestObject>(
        new (std::nothrow) LiteRtLm_RequestObject());
    if (!request) {
      return SetError(error, LITERT_LM_STATUS_RESOURCE_EXHAUSTED,
                      LITERT_LM_PHASE_CONTEXT_PREPARE,
                      "Failed to allocate the Request handle.");
    }
    request->magic = kRequestMagic;
    request->engine = conversation->owner;
    request->conversation = conversation;
    request->callback = chat->callback;
    request->user_data = chat->user_data;
    request->max_output_tokens = chat->max_output_tokens;
    request->applied_sampling = applied_sampling;
    request->used_constraint =
        chat->constraint_kind != LITERT_LM_CONSTRAINT_NONE;
    request->task_group_id =
        "litertlm-request-" + std::to_string(
                                    g_request_id.fetch_add(
                                        1, std::memory_order_relaxed));
    request->text.reserve(4096);
    request->response_json.reserve(4096);
    request->error_message.reserve(512);

    OptionalArgs args;
    args.task_group_id = request->task_group_id;
    if (override_extra_context) {
      args.extra_context = OrderedJson::parse(requested_extra_context_json);
    }
    if (chat->max_output_tokens != 0) {
      args.max_output_tokens = static_cast<int>(chat->max_output_tokens);
    }
    if (chat->constraint_kind != LITERT_LM_CONSTRAINT_NONE) {
      LlGuidanceConstraintArg constraint;
      switch (chat->constraint_kind) {
        case LITERT_LM_CONSTRAINT_REGEX:
          constraint.constraint_type = LlgConstraintType::kRegex;
          break;
        case LITERT_LM_CONSTRAINT_JSON_SCHEMA:
          constraint.constraint_type = LlgConstraintType::kJsonSchema;
          break;
        case LITERT_LM_CONSTRAINT_LARK:
          constraint.constraint_type = LlgConstraintType::kLark;
          break;
        default:
          break;
      }
      constraint.constraint_string = CopyView(chat->constraint);
      args.decoding_constraint = std::move(constraint);
    }

    SubmitReservation reservation;
    reservation.request = request.get();
    uint32_t expected_count = 0;
    if (!conversation->owner->request_count.compare_exchange_strong(
            expected_count, 1, std::memory_order_acq_rel)) {
      return SetError(
          error, LITERT_LM_STATUS_BUSY,
          LITERT_LM_PHASE_CONTEXT_PREPARE,
          "Destroy the prior Request before submitting the next serial inference.");
    }
    reservation.engine_count = true;
    expected_count = 0;
    if (!conversation->request_count.compare_exchange_strong(
            expected_count, 1, std::memory_order_acq_rel)) {
      return SetError(error, LITERT_LM_STATUS_BUSY,
                      LITERT_LM_PHASE_CONTEXT_PREPARE,
                      "Destroy the prior Request before reusing this Conversation.");
    }
    reservation.conversation_count = true;
    LiteRtLm_RequestObject* expected = nullptr;
    if (!conversation->owner->active_request.compare_exchange_strong(
            expected, request.get(), std::memory_order_acq_rel)) {
      return SetError(error, LITERT_LM_STATUS_BUSY,
                      LITERT_LM_PHASE_CONTEXT_PREPARE,
                      "This Engine already has an active Request; inference is serial.");
    }
    reservation.engine_active = true;
    expected = nullptr;
    if (!conversation->active_request.compare_exchange_strong(
            expected, request.get(), std::memory_order_acq_rel)) {
      return SetError(error, LITERT_LM_STATUS_BUSY,
                      LITERT_LM_PHASE_CONTEXT_PREPARE,
                      "This Conversation already has an active Request.");
    }
    reservation.conversation_active = true;

    LiteRtLm_ConversationDefinition requested_definition =
        conversation->definition;
    if (override_tools) {
      requested_definition.tools_json = requested_tools_json;
      OrderedJson parsed_tools =
          OrderedJson::parse(requested_tools_json, nullptr, false);
      requested_definition.tools = NormalizeToolDeclarations(parsed_tools);
    }

    Conversation* target = nullptr;
    uint64_t generation = conversation->generation.load(std::memory_order_acquire);
    if (chat->context_mode == LITERT_LM_CONTEXT_REPLACE) {
      {
        std::lock_guard<std::mutex> lock(conversation->native_mutex);
        conversation->conversation.reset();
        conversation->replayed_messages = OrderedJson::array();
        conversation->needs_reset.store(true, std::memory_order_release);
      }
      auto replacement_or = BuildConversation(
          *conversation->owner, requested_definition, applied_sampling);
      if (!replacement_or.ok()) {
        return SetAbslError(error, LITERT_LM_PHASE_CONTEXT_PREPARE,
                            replacement_or.status());
      }
      generation += 1;
      {
        std::lock_guard<std::mutex> lock(conversation->native_mutex);
        conversation->conversation = std::move(*replacement_or);
        conversation->definition = std::move(requested_definition);
        conversation->native_sampling = applied_sampling;
        target = conversation->conversation.get();
        conversation->generation.store(generation, std::memory_order_release);
        conversation->needs_reset.store(false, std::memory_order_release);
      }
    } else {
      OrderedJson replayed = OrderedJson::array();
      bool rebuild_for_options = false;
      {
        std::lock_guard<std::mutex> lock(conversation->native_mutex);
        target = conversation->conversation.get();
        if (target == nullptr) {
          return SetError(
              error, LITERT_LM_STATUS_INVALID_STATE,
              LITERT_LM_PHASE_CONTEXT_PREPARE,
              "Conversation has no live native session; reset or replace it.");
        }
        if (!SamplingEquals(conversation->native_sampling,
                            applied_sampling) ||
            conversation->definition.tools_json !=
                requested_definition.tools_json) {
          replayed = ReplayableDynamicHistory(*conversation);
          conversation->replayed_messages = replayed;
          conversation->conversation.reset();
          conversation->needs_reset.store(true, std::memory_order_release);
          target = nullptr;
          rebuild_for_options = true;
        }
      }
      if (rebuild_for_options) {
        auto replacement_or = BuildConversation(
            *conversation->owner, requested_definition, applied_sampling,
            &replayed);
        if (!replacement_or.ok()) {
          return SetAbslError(error, LITERT_LM_PHASE_CONTEXT_PREPARE,
                              replacement_or.status());
        }
        generation += 1;
        std::lock_guard<std::mutex> lock(conversation->native_mutex);
        conversation->conversation = std::move(*replacement_or);
        conversation->definition = std::move(requested_definition);
        conversation->native_sampling = applied_sampling;
        conversation->replayed_messages = std::move(replayed);
        target = conversation->conversation.get();
        conversation->generation.store(generation, std::memory_order_release);
        conversation->needs_reset.store(false, std::memory_order_release);
      }
    }

    request->native_conversation = target;
    request->conversation_generation = generation;
    request->state = LITERT_LM_REQUEST_RUNNING;
    request->start_time = std::chrono::steady_clock::now();
    CaptureBenchmarkBaseline(*request);

    LiteRtLm_RequestObject* request_ptr = request.get();
    absl::Status send_status = target->SendMessageAsync(
        messages,
        [request_ptr](absl::StatusOr<Message> message) mutable {
          HandleNativeMessage(request_ptr, std::move(message));
        },
        std::move(args));
    if (!send_status.ok()) {
      conversation->needs_reset.store(true, std::memory_order_release);
      return SetAbslError(error, LITERT_LM_PHASE_PREFILL, send_status);
    }

    {
      std::lock_guard<std::mutex> lock(request->mutex);
      request->submit_in_progress = false;
    }
    reservation.committed = true;
    *out_request = request.release();
    MaybeReleaseActiveRequest(**out_request);
    return LITERT_LM_STATUS_OK;
  });
}

LiteRtLm_Status ApiGetRequestInfo(LiteRtLm_Request request,
                                  LiteRtLm_RequestInfo* out_info,
                                  LiteRtLm_ErrorInfo* error) {
  return Boundary(LITERT_LM_PHASE_API, error, [&]() -> LiteRtLm_Status {
    if (!IsRequestValid(request)) {
      return SetError(error, LITERT_LM_STATUS_INVALID_ARGUMENT,
                      LITERT_LM_PHASE_API, "Request handle is invalid.");
    }
    if (out_info == nullptr ||
        out_info->struct_size <
            LITERT_LM_FIELD_END(LiteRtLm_RequestInfo, error_message)) {
      return SetError(error, LITERT_LM_STATUS_INVALID_ARGUMENT,
                      LITERT_LM_PHASE_API,
                      "RequestInfo is null or has an invalid struct_size.");
    }
    uint32_t capacity = out_info->struct_size;
    LiteRtLm_RequestInfo value{};
    value.struct_size = static_cast<uint32_t>(sizeof(value));
    {
      std::lock_guard<std::mutex> lock(request->mutex);
      value.state = request->state;
      value.finish_reason = request->finish_reason;
      value.status = request->status;
      value.backend = ToPublicBackend(request->engine->resolved_backend);
      value.error_phase = request->error_phase;
      value.native_code = request->native_code;
      value.retryable = request->retryable ? 1u : 0u;
      value.valid_metrics = request->valid_metrics;
      value.input_tokens = request->input_tokens;
      value.output_tokens = request->output_tokens;
      value.context_tokens = request->context_tokens;
      value.prefill_time_us = request->prefill_time_us;
      value.decode_time_us = request->decode_time_us;
      value.time_to_first_token_us = request->time_to_first_token_us;
      value.total_time_us = request->total_time_us;
      value.output_tokens_per_second = request->output_tokens_per_second;
      value.conversation_generation = request->conversation_generation;
      value.callback_count = request->callback_count;
      value.applied_sampling = request->applied_sampling;
      if (request->terminal) {
        value.text = ViewOf(request->text);
        value.response_json = ViewOf(request->response_json);
        value.error_message = ViewOf(request->error_message);
      }
    }
    std::memcpy(out_info, &value,
                std::min<size_t>(capacity, sizeof(LiteRtLm_RequestInfo)));
    return LITERT_LM_STATUS_OK;
  });
}

LiteRtLm_Status ApiWaitRequest(LiteRtLm_Request request, uint32_t timeout_ms,
                               LiteRtLm_ErrorInfo* error) {
  return Boundary(LITERT_LM_PHASE_DECODE, error,
                  [&]() -> LiteRtLm_Status {
    if (!IsRequestValid(request)) {
      return SetError(error, LITERT_LM_STATUS_INVALID_ARGUMENT,
                      LITERT_LM_PHASE_DECODE, "Request handle is invalid.");
    }
    if (g_callback_request == request) {
      return SetError(
          error, LITERT_LM_STATUS_INVALID_STATE,
          LITERT_LM_PHASE_CALLBACK,
          "wait_request cannot be called from the same Request's callback.");
    }
    const auto start = std::chrono::steady_clock::now();
    const auto deadline = timeout_ms == LITERT_LM_WAIT_INFINITE
                              ? std::chrono::steady_clock::time_point::max()
                              : start + std::chrono::milliseconds(timeout_ms);

    bool native_wait_complete = false;
    {
      std::lock_guard<std::mutex> lock(request->mutex);
      native_wait_complete = request->native_wait_complete;
    }
    if (!native_wait_complete) {
      absl::Duration timeout =
          timeout_ms == LITERT_LM_WAIT_INFINITE
              ? absl::InfiniteDuration()
              : absl::Milliseconds(timeout_ms);
      absl::Status wait_status = request->engine->engine->WaitUntilDone(timeout);
      if (absl::IsDeadlineExceeded(wait_status)) {
        return SetError(error, LITERT_LM_STATUS_TIMEOUT,
                        LITERT_LM_PHASE_DECODE, wait_status.ToString(),
                        static_cast<int32_t>(wait_status.raw_code()), true);
      }
      if (!wait_status.ok()) {
        return SetAbslError(error, LITERT_LM_PHASE_DECODE, wait_status);
      }
      {
        std::lock_guard<std::mutex> lock(request->mutex);
        request->native_wait_complete = true;
      }
    }

    {
      std::unique_lock<std::mutex> lock(request->mutex);
      auto predicate = [&]() {
        return request->native_wait_complete && request->terminal &&
               !request->callback_in_progress &&
               request->native_callbacks_in_flight.load(
                   std::memory_order_acquire) == 0;
      };
      if (!predicate()) {
        if (timeout_ms == LITERT_LM_WAIT_INFINITE) {
          request->condition.wait(lock, predicate);
        } else if (!request->condition.wait_until(lock, deadline, predicate)) {
          return SetError(error, LITERT_LM_STATUS_TIMEOUT,
                          LITERT_LM_PHASE_CALLBACK,
                          "Timed out waiting for the terminal callback to return.",
                          0, true);
        }
      }
    }
    RefreshMetricsNoThrow(*request);
    return LITERT_LM_STATUS_OK;
  });
}

LiteRtLm_Status ApiCancelRequest(LiteRtLm_Request request,
                                 LiteRtLm_ErrorInfo* error) {
  return Boundary(LITERT_LM_PHASE_CANCEL, error,
                  [&]() -> LiteRtLm_Status {
    if (!IsRequestValid(request)) {
      return SetError(error, LITERT_LM_STATUS_INVALID_ARGUMENT,
                      LITERT_LM_PHASE_CANCEL, "Request handle is invalid.");
    }
    {
      std::lock_guard<std::mutex> lock(request->mutex);
      if (request->terminal) return LITERT_LM_STATUS_OK;
      request->cancel_requested = true;
    }
    request->conversation->needs_reset.store(true, std::memory_order_release);
    request->native_conversation->CancelGroup(request->task_group_id);
    return LITERT_LM_STATUS_OK;
  });
}

LiteRtLm_Status ApiDestroyRequest(LiteRtLm_Request request,
                                  LiteRtLm_ErrorInfo* error) {
  return Boundary(LITERT_LM_PHASE_DESTROY, error,
                  [&]() -> LiteRtLm_Status {
    if (!IsRequestValid(request)) {
      return SetError(error, LITERT_LM_STATUS_INVALID_ARGUMENT,
                      LITERT_LM_PHASE_DESTROY, "Request handle is invalid.");
    }
    {
      std::lock_guard<std::mutex> lock(request->mutex);
      if (request->submit_in_progress || !request->native_wait_complete ||
          !request->terminal ||
          request->callback_in_progress ||
          request->native_callbacks_in_flight.load(
              std::memory_order_acquire) != 0) {
        return SetError(error, LITERT_LM_STATUS_BUSY,
                        LITERT_LM_PHASE_DESTROY,
                        "Wait for native completion and callbacks before destroying the Request.");
      }
    }
    LiteRtLm_ConversationObject* conversation = request->conversation;
    LiteRtLm_EngineObject* engine = request->engine;
    request->magic = 0;
    delete request;
    conversation->request_count.fetch_sub(1, std::memory_order_acq_rel);
    engine->request_count.fetch_sub(1, std::memory_order_acq_rel);
    return LITERT_LM_STATUS_OK;
  });
}

LiteRtLm_Status ApiTokenize(LiteRtLm_Engine engine, LiteRtLm_StringView text,
                            int32_t* out_tokens, uint64_t* inout_count,
                            LiteRtLm_ErrorInfo* error) {
  return Boundary(LITERT_LM_PHASE_API, error, [&]() -> LiteRtLm_Status {
    if (!IsEngineValid(engine) || !IsViewValid(text) ||
        inout_count == nullptr) {
      return SetError(error, LITERT_LM_STATUS_INVALID_ARGUMENT,
                      LITERT_LM_PHASE_API,
                      "Engine, text, or token-count pointer is invalid.");
    }
    std::lock_guard<std::mutex> operation_lock(engine->operation_mutex);
    if (engine->request_count.load(std::memory_order_acquire) != 0) {
      return SetError(error, LITERT_LM_STATUS_BUSY,
                      LITERT_LM_PHASE_API,
                      "Tokenize only while the Engine is idle.");
    }
    const auto& tokenizer = engine->engine->GetTokenizer();
    auto ids = const_cast<litert::lm::Tokenizer&>(tokenizer)
                   .TextToTokenIds(CopyView(text));
    if (!ids.ok()) {
      return SetAbslError(error, LITERT_LM_PHASE_API, ids.status());
    }
    const uint64_t required = static_cast<uint64_t>(ids->size());
    const uint64_t capacity = *inout_count;
    *inout_count = required;
    if (out_tokens == nullptr) return LITERT_LM_STATUS_OK;
    if (capacity < required) {
      return SetError(error, LITERT_LM_STATUS_RESOURCE_EXHAUSTED,
                      LITERT_LM_PHASE_API,
                      "The caller-provided token buffer is too small.");
    }
    for (size_t index = 0; index < ids->size(); ++index) {
      out_tokens[index] = static_cast<int32_t>((*ids)[index]);
    }
    return LITERT_LM_STATUS_OK;
  });
}

LiteRtLm_Status ApiDetokenize(LiteRtLm_Engine engine, const int32_t* tokens,
                              uint64_t token_count, char* out_text,
                              uint64_t* inout_size,
                              LiteRtLm_ErrorInfo* error) {
  return Boundary(LITERT_LM_PHASE_API, error, [&]() -> LiteRtLm_Status {
    if (!IsEngineValid(engine) || inout_size == nullptr ||
        (tokens == nullptr && token_count != 0) ||
        token_count > static_cast<uint64_t>(
                          (std::numeric_limits<size_t>::max)())) {
      return SetError(error, LITERT_LM_STATUS_INVALID_ARGUMENT,
                      LITERT_LM_PHASE_API,
                      "Engine, tokens, or output-size pointer is invalid.");
    }
    std::lock_guard<std::mutex> operation_lock(engine->operation_mutex);
    if (engine->request_count.load(std::memory_order_acquire) != 0) {
      return SetError(error, LITERT_LM_STATUS_BUSY,
                      LITERT_LM_PHASE_API,
                      "Detokenize only while the Engine is idle.");
    }
    std::vector<int> ids;
    ids.reserve(static_cast<size_t>(token_count));
    for (uint64_t index = 0; index < token_count; ++index) {
      ids.push_back(static_cast<int>(tokens[index]));
    }
    const auto& tokenizer = engine->engine->GetTokenizer();
    auto text = const_cast<litert::lm::Tokenizer&>(tokenizer)
                    .TokenIdsToText(ids);
    if (!text.ok()) {
      return SetAbslError(error, LITERT_LM_PHASE_API, text.status());
    }
    return CopyStringToCaller(*text, out_text, inout_size,
                              LITERT_LM_PHASE_API, error);
  });
}

LiteRtLm_Status ApiSetLogLevel(int32_t minimum_level,
                               LiteRtLm_ErrorInfo* error) {
  return Boundary(LITERT_LM_PHASE_API, error, [&]() -> LiteRtLm_Status {
    if (!((minimum_level >= 0 && minimum_level <= 5) ||
          minimum_level == 1000)) {
      return SetError(error, LITERT_LM_STATUS_INVALID_ARGUMENT,
                      LITERT_LM_PHASE_API,
                      "Log level must be 0..5 or 1000 (silent).");
    }
    litert::lm::SetMinLogSeverity(
        static_cast<litert::lm::LogSeverity>(minimum_level));
    return LITERT_LM_STATUS_OK;
  });
}

}  // namespace

extern "C" LITERT_LM_EXPORT LiteRtLm_Status LITERT_LM_CALL LiteRtLm_GetApi(
    uint32_t requested_major, uint32_t out_api_size, LiteRtLm_Api* out_api) {
  if (requested_major != LITERT_LM_ABI_MAJOR) {
    return LITERT_LM_STATUS_ABI_MISMATCH;
  }
  constexpr size_t kRequiredSize =
      LITERT_LM_FIELD_END(LiteRtLm_Api, destroy_request);
  if (out_api == nullptr || out_api_size < kRequiredSize) {
    return LITERT_LM_STATUS_INVALID_ARGUMENT;
  }

  LiteRtLm_Api api{};
  api.struct_size = static_cast<uint32_t>(sizeof(api));
  api.abi_major = LITERT_LM_ABI_MAJOR;
  api.abi_minor = LITERT_LM_ABI_MINOR;
  api.compiled_backends =
      LITERT_LM_BACKEND_MASK_CPU | LITERT_LM_BACKEND_MASK_GPU;
  api.capabilities =
      LITERT_LM_CAP_TEXT | LITERT_LM_CAP_STREAMING | LITERT_LM_CAP_TOOLS |
      LITERT_LM_CAP_CONSTRAINTS | LITERT_LM_CAP_REQUEST_METRICS |
      LITERT_LM_CAP_CONTEXT_REPLACE |
      LITERT_LM_CAP_MULTI_CONVERSATION |
      LITERT_LM_CAP_CONVERSATION_CLONE |
      LITERT_LM_CAP_CONVERSATION_HISTORY | LITERT_LM_CAP_TOKENIZER |
      LITERT_LM_CAP_INITIAL_MESSAGES | LITERT_LM_CAP_THINKING |
      LITERT_LM_CAP_TOOL_CALL_STREAMING |
      LITERT_LM_CAP_PER_REQUEST_SAMPLING |
      LITERT_LM_CAP_PER_REQUEST_TOOLS |
      LITERT_LM_CAP_PER_REQUEST_EXTRA_CONTEXT;
  api.build_id = ViewOf(kBuildId, sizeof(kBuildId) - 1);
  api.create_engine = &ApiCreateEngine;
  api.get_engine_info = &ApiGetEngineInfo;
  api.destroy_engine = &ApiDestroyEngine;
  api.create_conversation = &ApiCreateConversation;
  api.reset_conversation = &ApiResetConversation;
  api.destroy_conversation = &ApiDestroyConversation;
  api.submit_chat = &ApiSubmitChat;
  api.get_request_info = &ApiGetRequestInfo;
  api.wait_request = &ApiWaitRequest;
  api.cancel_request = &ApiCancelRequest;
  api.destroy_request = &ApiDestroyRequest;
  api.clone_conversation = &ApiCloneConversation;
  api.get_conversation_info = &ApiGetConversationInfo;
  api.copy_conversation_history_json =
      &ApiCopyConversationHistoryJson;
  api.tokenize = &ApiTokenize;
  api.detokenize = &ApiDetokenize;
  api.set_log_level = &ApiSetLogLevel;

  const size_t bytes = std::min<size_t>(out_api_size, sizeof(api));
  std::memset(out_api, 0, bytes);
  std::memcpy(out_api, &api, bytes);
  return LITERT_LM_STATUS_OK;
}
