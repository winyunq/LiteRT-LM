// Copyright (c) 2025-2026 Winyunq. All rights reserved.
//
// Stable extension ABI implemented on top of LiteRT-LM's official C ABI.
// The official core is shipped unchanged as a sibling library and remains the
// complete, non-lossy API surface for native C/C++ consumers.

#define LITERT_LM_IMPLEMENTATION
#include "runtime/engine/litert_lm_api.h"

#define LITERT_LM_C_API_EXPORT
#include "c/engine.h"
#undef LITERT_LM_C_API_EXPORT

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
#else
#include <dlfcn.h>
#endif

#include "nlohmann/json.hpp"  // from @nlohmann_json

using OrderedJson = nlohmann::ordered_json;

// LiteRT-LM v0.15 replaced the legacy four-argument streaming callback with
// an opaque StreamChunk callback.  The Conversation exports remain available,
// so keep both callback layouts and select the matching one from the optional
// StreamChunk accessors exported by the loaded core.
using NativeStreamChunkCallback = void (*)(void*, const void*);
using NativeStreamChunkGetText = const char* (*)(const void*);
using NativeStreamChunkIsFinal = bool (*)(const void*);
using NativeStreamChunkGetError = const char* (*)(const void*);
using NativeConversationSendMessageStream = int (*)(
    LiteRtLmConversation*, const char*, const char*,
    const LiteRtLmConversationOptionalArgs*, NativeStreamChunkCallback, void*);

struct NativeApi {
#define LITERT_LM_NATIVE_FUNCTION(name) decltype(&::name) name = nullptr
  LITERT_LM_NATIVE_FUNCTION(litert_lm_sampler_params_create);
  LITERT_LM_NATIVE_FUNCTION(litert_lm_sampler_params_delete);
  LITERT_LM_NATIVE_FUNCTION(litert_lm_sampler_params_set_top_k);
  LITERT_LM_NATIVE_FUNCTION(litert_lm_sampler_params_set_top_p);
  LITERT_LM_NATIVE_FUNCTION(litert_lm_sampler_params_set_temperature);
  LITERT_LM_NATIVE_FUNCTION(litert_lm_sampler_params_set_seed);
  LITERT_LM_NATIVE_FUNCTION(litert_lm_session_config_create);
  LITERT_LM_NATIVE_FUNCTION(litert_lm_session_config_delete);
  LITERT_LM_NATIVE_FUNCTION(litert_lm_session_config_set_max_output_tokens);
  LITERT_LM_NATIVE_FUNCTION(litert_lm_session_config_set_apply_prompt_template);
  LITERT_LM_NATIVE_FUNCTION(litert_lm_session_config_set_sampler_params);
  LITERT_LM_NATIVE_FUNCTION(litert_lm_session_config_set_lora_path);
  LITERT_LM_NATIVE_FUNCTION(litert_lm_session_config_set_audio_lora_path);
  LITERT_LM_NATIVE_FUNCTION(litert_lm_conversation_config_create);
  LITERT_LM_NATIVE_FUNCTION(litert_lm_conversation_config_delete);
  LITERT_LM_NATIVE_FUNCTION(litert_lm_conversation_config_set_session_config);
  LITERT_LM_NATIVE_FUNCTION(litert_lm_conversation_config_set_system_message);
  LITERT_LM_NATIVE_FUNCTION(litert_lm_conversation_config_set_tools);
  LITERT_LM_NATIVE_FUNCTION(litert_lm_conversation_config_set_messages);
  LITERT_LM_NATIVE_FUNCTION(litert_lm_conversation_config_set_extra_context);
  LITERT_LM_NATIVE_FUNCTION(
      litert_lm_conversation_config_set_enable_constrained_decoding);
  LITERT_LM_NATIVE_FUNCTION(
      litert_lm_conversation_config_set_filter_channel_content_from_kv_cache);
  LITERT_LM_NATIVE_FUNCTION(litert_lm_conversation_config_set_stream_tool_calls);
  LITERT_LM_NATIVE_FUNCTION(litert_lm_conversation_optional_args_create);
  LITERT_LM_NATIVE_FUNCTION(litert_lm_conversation_optional_args_delete);
  LITERT_LM_NATIVE_FUNCTION(
      litert_lm_conversation_optional_args_set_max_output_tokens);
  LITERT_LM_NATIVE_FUNCTION(litert_lm_engine_settings_create);
  LITERT_LM_NATIVE_FUNCTION(litert_lm_engine_settings_delete);
  LITERT_LM_NATIVE_FUNCTION(litert_lm_engine_settings_set_max_num_tokens);
  LITERT_LM_NATIVE_FUNCTION(litert_lm_engine_settings_set_num_threads);
  LITERT_LM_NATIVE_FUNCTION(
      litert_lm_engine_settings_set_parallel_file_section_loading);
  LITERT_LM_NATIVE_FUNCTION(litert_lm_engine_settings_set_cache_dir);
  LITERT_LM_NATIVE_FUNCTION(
      litert_lm_engine_settings_set_litert_dispatch_lib_dir);
  LITERT_LM_NATIVE_FUNCTION(litert_lm_engine_settings_set_prefill_chunk_size);
  LITERT_LM_NATIVE_FUNCTION(
      litert_lm_engine_settings_set_enable_speculative_decoding);
  LITERT_LM_NATIVE_FUNCTION(litert_lm_engine_settings_enable_benchmark);
  LITERT_LM_NATIVE_FUNCTION(litert_lm_engine_create);
  LITERT_LM_NATIVE_FUNCTION(litert_lm_engine_delete);
  LITERT_LM_NATIVE_FUNCTION(litert_lm_conversation_create);
  LITERT_LM_NATIVE_FUNCTION(litert_lm_conversation_delete);
  LITERT_LM_NATIVE_FUNCTION(litert_lm_conversation_clone);
  LITERT_LM_NATIVE_FUNCTION(litert_lm_conversation_send_message_stream);
  LITERT_LM_NATIVE_FUNCTION(litert_lm_conversation_cancel_process);
  LITERT_LM_NATIVE_FUNCTION(litert_lm_conversation_get_token_count);
  LITERT_LM_NATIVE_FUNCTION(litert_lm_conversation_get_benchmark_info);
  LITERT_LM_NATIVE_FUNCTION(litert_lm_benchmark_info_delete);
  LITERT_LM_NATIVE_FUNCTION(litert_lm_benchmark_info_get_time_to_first_token);
  LITERT_LM_NATIVE_FUNCTION(litert_lm_benchmark_info_get_num_prefill_turns);
  LITERT_LM_NATIVE_FUNCTION(litert_lm_benchmark_info_get_num_decode_turns);
  LITERT_LM_NATIVE_FUNCTION(litert_lm_benchmark_info_get_prefill_token_count_at);
  LITERT_LM_NATIVE_FUNCTION(litert_lm_benchmark_info_get_decode_token_count_at);
  LITERT_LM_NATIVE_FUNCTION(
      litert_lm_benchmark_info_get_prefill_tokens_per_sec_at);
  LITERT_LM_NATIVE_FUNCTION(
      litert_lm_benchmark_info_get_decode_tokens_per_sec_at);
  LITERT_LM_NATIVE_FUNCTION(litert_lm_engine_tokenize);
  LITERT_LM_NATIVE_FUNCTION(litert_lm_tokenize_result_delete);
  LITERT_LM_NATIVE_FUNCTION(litert_lm_tokenize_result_get_tokens);
  LITERT_LM_NATIVE_FUNCTION(litert_lm_tokenize_result_get_num_tokens);
  LITERT_LM_NATIVE_FUNCTION(litert_lm_engine_detokenize);
  LITERT_LM_NATIVE_FUNCTION(litert_lm_detokenize_result_delete);
  LITERT_LM_NATIVE_FUNCTION(litert_lm_detokenize_result_get_string);
  LITERT_LM_NATIVE_FUNCTION(litert_lm_set_min_log_level);
#undef LITERT_LM_NATIVE_FUNCTION
  NativeStreamChunkGetText litert_lm_stream_chunk_get_text = nullptr;
  NativeStreamChunkIsFinal litert_lm_stream_chunk_is_final = nullptr;
  NativeStreamChunkGetError litert_lm_stream_chunk_get_error = nullptr;

  bool UsesStreamChunks() const {
    return litert_lm_stream_chunk_get_text != nullptr;
  }
};

struct NativeLibrary {
#if defined(_WIN32)
  HMODULE module = nullptr;
#else
  void* module = nullptr;
#endif
  NativeApi api;
  std::string path;

  ~NativeLibrary() {
    if (module == nullptr) return;
#if defined(_WIN32)
    FreeLibrary(module);
#else
    dlclose(module);
#endif
  }
};

struct LiteRtLm_RequestObject;
struct LiteRtLm_ConversationObject;

struct ConversationDefinition {
  std::string system_message;
  std::string tools_json;
  std::string initial_messages_json;
  std::string extra_context_json;
  std::string tool_call_channel = "tool_call";
  std::string lora_path;
  std::string audio_lora_path;
  uint32_t flags = LITERT_LM_CONVERSATION_RETURN_ERROR_ON_PARSE_FAILURE;
  uint32_t default_max_output_tokens = 0;
  OrderedJson base_messages = OrderedJson::array();
};

struct LiteRtLm_EngineObject {
  uint64_t magic = 0;
  std::shared_ptr<NativeLibrary> library;
  LiteRtLmEngine* native = nullptr;
  LiteRtLm_Backend configured_backend = LITERT_LM_BACKEND_GPU;
  uint32_t flags = 0;
  uint32_t max_context_tokens = 0;
  LiteRtLm_Sampling applied_sampling{};
  std::string backend_description;
  std::mutex operation_mutex;
  std::atomic<uint32_t> conversation_count{0};
  std::atomic<uint32_t> request_count{0};
  std::atomic<LiteRtLm_RequestObject*> active_request{nullptr};
  // Tracks the most recently used native Conversation for diagnostics and
  // reset bookkeeping. LiteRT-LM v0.16 keeps independent stateful handles;
  // speaker switches do not by themselves require KV replay.
  // Protected by operation_mutex.
  LiteRtLm_ConversationObject* resident_conversation = nullptr;
};

struct LiteRtLm_ConversationObject {
  uint64_t magic = 0;
  LiteRtLm_EngineObject* owner = nullptr;
  LiteRtLmConversation* native = nullptr;
  // Official v0.16.1 can corrupt heap state when a constrained Conversation
  // is deleted immediately before another Conversation on the same Engine is
  // used. Retain superseded handles until the logical Conversation closes.
  std::vector<LiteRtLmConversation*> retired_native;
  // The sampler baked into native. It may differ from the Engine default when
  // the most recent ABI 1.2 request supplied an override.
  LiteRtLm_Sampling native_sampling{};
  ConversationDefinition definition;
  OrderedJson history = OrderedJson::array();
  std::mutex mutex;
  std::atomic<uint64_t> generation{1};
  std::atomic<uint32_t> request_count{0};
  std::atomic<LiteRtLm_RequestObject*> active_request{nullptr};
  std::atomic<bool> needs_rebuild{false};
};

struct LiteRtLm_RequestObject {
  uint64_t magic = 0;
  LiteRtLm_EngineObject* engine = nullptr;
  LiteRtLm_ConversationObject* conversation = nullptr;
  LiteRtLm_EventCallback callback = nullptr;
  void* user_data = nullptr;
  OrderedJson input_messages = OrderedJson::array();
  OrderedJson aggregate = OrderedJson::object();
  uint64_t conversation_generation = 0;
  uint32_t max_output_tokens = 0;
  LiteRtLm_Sampling applied_sampling{};
  // Owns a per-turn value until the asynchronous native request is terminal.
  std::string extra_context_json;

  std::recursive_mutex callback_mutex;
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
  std::atomic<bool> cancel_requested{false};
  bool metrics_finalized = false;
  std::atomic<uint32_t> native_callbacks_in_flight{0};
  uint64_t sequence = 0;
  uint64_t callback_count = 0;

  std::string text;
  std::string response_json;
  std::string error_message;

  int prefill_turn_baseline = 0;
  int decode_turn_baseline = 0;
  bool benchmark_baseline_valid = false;
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

constexpr uint64_t kEngineMagic = 0x4C4D4E4154454E47ull;       // LMNATENG
constexpr uint64_t kConversationMagic = 0x4C4D4E4154434F4Eull; // LMNATCON
constexpr uint64_t kRequestMagic = 0x4C4D4E4154524551ull;      // LMNATREQ
constexpr uint32_t kKnownEngineFlags =
    LITERT_LM_ENGINE_ENABLE_METRICS | LITERT_LM_ENGINE_ENABLE_VISION |
    LITERT_LM_ENGINE_ENABLE_AUDIO |
    LITERT_LM_ENGINE_DISABLE_PARALLEL_FILE_LOADING |
    LITERT_LM_ENGINE_ENABLE_SPECULATIVE_DECODING;
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
constexpr char kBuildId[] =
    "LiteRT-LM stable table ABI 1.4; official v0.16.1 native C API bridge; "
    "full upstream ABI shipped separately; " __DATE__ " " __TIME__;

#define LITERT_LM_FIELD_END(type, field) \
  (offsetof(type, field) + sizeof(((type*)0)->field))
#define LITERT_LM_HAS_FIELD(value, type, field) \
  ((value) != nullptr && \
   (value)->struct_size >= LITERT_LM_FIELD_END(type, field))

std::mutex g_library_mutex;
std::weak_ptr<NativeLibrary> g_library;
thread_local LiteRtLm_RequestObject* g_callback_request = nullptr;
std::atomic<uint64_t> g_seed_sequence{1};
#if defined(_WIN32)
int g_module_anchor = 0;
#endif

LiteRtLm_StringView ViewOf(const std::string& value) {
  return LiteRtLm_String(value.empty() ? nullptr : value.data(),
                         static_cast<uint64_t>(value.size()));
}

LiteRtLm_StringView ViewOf(const char* value, size_t size) {
  return LiteRtLm_String(size == 0 ? nullptr : value,
                         static_cast<uint64_t>(size));
}

bool IsViewValid(const LiteRtLm_StringView& value) {
  return (value.data != nullptr || value.size == 0) &&
         value.size <= static_cast<uint64_t>(
                           (std::numeric_limits<size_t>::max)());
}

std::string CopyView(const LiteRtLm_StringView& value) {
  return value.size == 0
             ? std::string()
             : std::string(value.data, static_cast<size_t>(value.size));
}

void ClearError(LiteRtLm_ErrorInfo* error) {
  if (error == nullptr || error->struct_size < sizeof(*error)) return;
  std::memset(error, 0, sizeof(*error));
  error->struct_size = static_cast<uint32_t>(sizeof(*error));
}

LiteRtLm_Status SetError(LiteRtLm_ErrorInfo* error, LiteRtLm_Status code,
                         LiteRtLm_ErrorPhase phase, std::string_view message,
                         int32_t native_code = 0, bool retryable = false) {
  if (error != nullptr && error->struct_size >= sizeof(*error)) {
    LiteRtLm_ErrorInfo value{};
    value.struct_size = static_cast<uint32_t>(sizeof(value));
    value.code = code;
    value.phase = phase;
    value.native_code = native_code;
    value.retryable = retryable ? 1u : 0u;
    const size_t count = std::min(message.size(), sizeof(value.message) - 1);
    if (count != 0) std::memcpy(value.message, message.data(), count);
    value.message[count] = '\0';
    std::memcpy(error, &value, sizeof(value));
  }
  return code;
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
         engine->native != nullptr && engine->library != nullptr;
}

bool IsConversationValid(LiteRtLm_Conversation conversation) {
  return conversation != nullptr && conversation->magic == kConversationMagic &&
         conversation->native != nullptr && IsEngineValid(conversation->owner);
}

bool IsRequestValid(LiteRtLm_Request request) {
  return request != nullptr && request->magic == kRequestMagic &&
         request->conversation != nullptr && IsEngineValid(request->engine);
}

uint64_t ToMicroseconds(std::chrono::steady_clock::duration duration) {
  const auto value =
      std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
  return value <= 0 ? 0u : static_cast<uint64_t>(value);
}

int32_t GenerateSeed() {
  uint64_t value = static_cast<uint64_t>(
      std::chrono::high_resolution_clock::now().time_since_epoch().count());
  value ^= g_seed_sequence.fetch_add(1, std::memory_order_relaxed) *
           0x9E3779B97F4A7C15ull;
  value ^= value >> 30;
  value *= 0xBF58476D1CE4E5B9ull;
  value ^= value >> 27;
  value *= 0x94D049BB133111EBull;
  value ^= value >> 31;
  const int32_t seed = static_cast<int32_t>(value & 0x7FFFFFFFu);
  return seed == 0 ? 1 : seed;
}

#if defined(_WIN32)
std::string Utf8FromWide(std::wstring_view value) {
  if (value.empty()) return {};
  const int required = WideCharToMultiByte(
      CP_UTF8, WC_ERR_INVALID_CHARS, value.data(),
      static_cast<int>(value.size()), nullptr, 0, nullptr, nullptr);
  if (required <= 0) return {};
  std::string output(static_cast<size_t>(required), '\0');
  if (WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, value.data(),
                          static_cast<int>(value.size()), output.data(),
                          required, nullptr, nullptr) != required) {
    return {};
  }
  return output;
}

bool GetBridgeDirectory(std::wstring* output) {
  if (output == nullptr) return false;
  HMODULE module = nullptr;
  if (!GetModuleHandleExW(
          GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
              GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
          reinterpret_cast<LPCWSTR>(&g_module_anchor), &module)) {
    return false;
  }
  std::wstring path(512, L'\0');
  for (;;) {
    const DWORD copied = GetModuleFileNameW(
        module, path.data(), static_cast<DWORD>(path.size()));
    if (copied == 0) return false;
    if (copied < path.size()) {
      path.resize(copied);
      break;
    }
    if (path.size() >= 32768) return false;
    path.resize(std::min<size_t>(path.size() * 2, 32768));
  }
  const size_t separator = path.find_last_of(L"\\/");
  if (separator == std::wstring::npos) return false;
  path.resize(separator);
  *output = std::move(path);
  return true;
}
#endif

template <typename T>
bool LoadSymbol(NativeLibrary& library, const char* name, T* output,
                std::string* error) {
  if (output == nullptr) return false;
#if defined(_WIN32)
  FARPROC symbol = GetProcAddress(library.module, name);
#else
  void* symbol = dlsym(library.module, name);
#endif
  if (symbol == nullptr) {
    if (error != nullptr) {
      *error = std::string("Official LiteRT-LM core is missing export: ") +
               name;
    }
    return false;
  }
  *output = reinterpret_cast<T>(symbol);
  return true;
}

template <typename T>
bool LoadOptionalSymbol(NativeLibrary& library, const char* name, T* output) {
  if (output == nullptr) return false;
#if defined(_WIN32)
  FARPROC symbol = GetProcAddress(library.module, name);
#else
  void* symbol = dlsym(library.module, name);
#endif
  *output = reinterpret_cast<T>(symbol);
  return symbol != nullptr;
}

bool LoadNativeApi(NativeLibrary& library, std::string* error) {
#define LITERT_LM_LOAD(name) \
  if (!LoadSymbol(library, #name, &library.api.name, error)) return false
  LITERT_LM_LOAD(litert_lm_sampler_params_create);
  LITERT_LM_LOAD(litert_lm_sampler_params_delete);
  LITERT_LM_LOAD(litert_lm_sampler_params_set_top_k);
  LITERT_LM_LOAD(litert_lm_sampler_params_set_top_p);
  LITERT_LM_LOAD(litert_lm_sampler_params_set_temperature);
  LITERT_LM_LOAD(litert_lm_sampler_params_set_seed);
  LITERT_LM_LOAD(litert_lm_session_config_create);
  LITERT_LM_LOAD(litert_lm_session_config_delete);
  LITERT_LM_LOAD(litert_lm_session_config_set_max_output_tokens);
  LITERT_LM_LOAD(litert_lm_session_config_set_apply_prompt_template);
  LITERT_LM_LOAD(litert_lm_session_config_set_sampler_params);
  LITERT_LM_LOAD(litert_lm_session_config_set_lora_path);
  LITERT_LM_LOAD(litert_lm_session_config_set_audio_lora_path);
  LITERT_LM_LOAD(litert_lm_conversation_config_create);
  LITERT_LM_LOAD(litert_lm_conversation_config_delete);
  LITERT_LM_LOAD(litert_lm_conversation_config_set_session_config);
  LITERT_LM_LOAD(litert_lm_conversation_config_set_system_message);
  LITERT_LM_LOAD(litert_lm_conversation_config_set_tools);
  LITERT_LM_LOAD(litert_lm_conversation_config_set_messages);
  LITERT_LM_LOAD(litert_lm_conversation_config_set_extra_context);
  LITERT_LM_LOAD(
      litert_lm_conversation_config_set_enable_constrained_decoding);
  LITERT_LM_LOAD(
      litert_lm_conversation_config_set_filter_channel_content_from_kv_cache);
  LITERT_LM_LOAD(litert_lm_conversation_config_set_stream_tool_calls);
  LITERT_LM_LOAD(litert_lm_conversation_optional_args_create);
  LITERT_LM_LOAD(litert_lm_conversation_optional_args_delete);
  LITERT_LM_LOAD(
      litert_lm_conversation_optional_args_set_max_output_tokens);
  LITERT_LM_LOAD(litert_lm_engine_settings_create);
  LITERT_LM_LOAD(litert_lm_engine_settings_delete);
  LITERT_LM_LOAD(litert_lm_engine_settings_set_max_num_tokens);
  LITERT_LM_LOAD(litert_lm_engine_settings_set_num_threads);
  LITERT_LM_LOAD(
      litert_lm_engine_settings_set_parallel_file_section_loading);
  LITERT_LM_LOAD(litert_lm_engine_settings_set_cache_dir);
  LITERT_LM_LOAD(litert_lm_engine_settings_set_litert_dispatch_lib_dir);
  LITERT_LM_LOAD(litert_lm_engine_settings_set_prefill_chunk_size);
  LITERT_LM_LOAD(
      litert_lm_engine_settings_set_enable_speculative_decoding);
  LITERT_LM_LOAD(litert_lm_engine_settings_enable_benchmark);
  LITERT_LM_LOAD(litert_lm_engine_create);
  LITERT_LM_LOAD(litert_lm_engine_delete);
  LITERT_LM_LOAD(litert_lm_conversation_create);
  LITERT_LM_LOAD(litert_lm_conversation_delete);
  LITERT_LM_LOAD(litert_lm_conversation_clone);
  LITERT_LM_LOAD(litert_lm_conversation_send_message_stream);
  LITERT_LM_LOAD(litert_lm_conversation_cancel_process);
  LITERT_LM_LOAD(litert_lm_conversation_get_token_count);
  LITERT_LM_LOAD(litert_lm_conversation_get_benchmark_info);
  LITERT_LM_LOAD(litert_lm_benchmark_info_delete);
  LITERT_LM_LOAD(litert_lm_benchmark_info_get_time_to_first_token);
  LITERT_LM_LOAD(litert_lm_benchmark_info_get_num_prefill_turns);
  LITERT_LM_LOAD(litert_lm_benchmark_info_get_num_decode_turns);
  LITERT_LM_LOAD(litert_lm_benchmark_info_get_prefill_token_count_at);
  LITERT_LM_LOAD(litert_lm_benchmark_info_get_decode_token_count_at);
  LITERT_LM_LOAD(litert_lm_benchmark_info_get_prefill_tokens_per_sec_at);
  LITERT_LM_LOAD(litert_lm_benchmark_info_get_decode_tokens_per_sec_at);
  LITERT_LM_LOAD(litert_lm_engine_tokenize);
  LITERT_LM_LOAD(litert_lm_tokenize_result_delete);
  LITERT_LM_LOAD(litert_lm_tokenize_result_get_tokens);
  LITERT_LM_LOAD(litert_lm_tokenize_result_get_num_tokens);
  LITERT_LM_LOAD(litert_lm_engine_detokenize);
  LITERT_LM_LOAD(litert_lm_detokenize_result_delete);
  LITERT_LM_LOAD(litert_lm_detokenize_result_get_string);
  LITERT_LM_LOAD(litert_lm_set_min_log_level);
#undef LITERT_LM_LOAD
  const bool has_chunk_text = LoadOptionalSymbol(
      library, "litert_lm_stream_chunk_get_text",
      &library.api.litert_lm_stream_chunk_get_text);
  const bool has_chunk_final = LoadOptionalSymbol(
      library, "litert_lm_stream_chunk_is_final",
      &library.api.litert_lm_stream_chunk_is_final);
  const bool has_chunk_error = LoadOptionalSymbol(
      library, "litert_lm_stream_chunk_get_error",
      &library.api.litert_lm_stream_chunk_get_error);
  if ((has_chunk_text || has_chunk_final || has_chunk_error) &&
      !(has_chunk_text && has_chunk_final && has_chunk_error)) {
    if (error != nullptr) {
      *error = "Official LiteRT-LM core has an incomplete StreamChunk API.";
    }
    return false;
  }
  return true;
}

std::shared_ptr<NativeLibrary> GetNativeLibrary(std::string* error) {
  std::lock_guard<std::mutex> lock(g_library_mutex);
  if (auto existing = g_library.lock()) return existing;

  auto library = std::make_shared<NativeLibrary>();
#if defined(_WIN32)
  std::wstring directory;
  std::wstring core_path = L"litert-lm.dll";
  if (GetBridgeDirectory(&directory)) {
    core_path = directory + L"\\litert-lm.dll";
  }
  library->module = LoadLibraryExW(
      core_path.c_str(), nullptr,
      LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
  if (library->module == nullptr) {
    const DWORD native_error = GetLastError();
    if (error != nullptr) {
      *error = "Unable to load official LiteRT-LM core beside the extension "
               "DLL (Win32 error " +
               std::to_string(native_error) + "): " + Utf8FromWide(core_path);
    }
    return nullptr;
  }
  library->path = Utf8FromWide(core_path);
#else
  const char* core_name =
#if defined(__APPLE__)
      "liblitert-lm.dylib";
#else
      "liblitert-lm.so";
#endif
  library->module = dlopen(core_name, RTLD_NOW | RTLD_LOCAL);
  if (library->module == nullptr) {
    if (error != nullptr) {
      const char* native_error = dlerror();
      *error = std::string("Unable to load official LiteRT-LM core: ") +
               (native_error == nullptr ? "unknown loader error" : native_error);
    }
    return nullptr;
  }
  library->path = core_name;
#endif
  if (!LoadNativeApi(*library, error)) return nullptr;
  g_library = library;
  return library;
}

LiteRtLm_Status CopyStringToCaller(const std::string& value, char* output,
                                   uint64_t* inout_size,
                                   LiteRtLm_ErrorPhase phase,
                                   LiteRtLm_ErrorInfo* error) {
  if (inout_size == nullptr) {
    return SetError(error, LITERT_LM_STATUS_INVALID_ARGUMENT, phase,
                    "The output-size pointer is null.");
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

LiteRtLm_Status ValidateSampling(const LiteRtLm_Sampling& sampling,
                                 LiteRtLm_ErrorPhase phase,
                                 LiteRtLm_ErrorInfo* error) {
  if (sampling.struct_size < LITERT_LM_FIELD_END(LiteRtLm_Sampling, seed)) {
    return SetError(error, LITERT_LM_STATUS_INVALID_ARGUMENT, phase,
                    "Sampling.struct_size is too small.");
  }
  if (!std::isfinite(sampling.temperature) || sampling.temperature < 0.0f ||
      !std::isfinite(sampling.top_p) || sampling.top_p < 0.0f ||
      sampling.top_p > 1.0f || sampling.top_k <= 0 ||
      sampling.top_k > 1024 ||
      (sampling.seed_mode != LITERT_LM_SEED_RANDOM &&
       sampling.seed_mode != LITERT_LM_SEED_FIXED)) {
    return SetError(error, LITERT_LM_STATUS_INVALID_ARGUMENT, phase,
                    "Sampling values are outside the stable ABI contract.");
  }
  return LITERT_LM_STATUS_OK;
}

LiteRtLm_Sampling ResolveSampling(const LiteRtLm_Sampling& requested) {
  LiteRtLm_Sampling result = requested;
  result.struct_size = static_cast<uint32_t>(sizeof(result));
  if (result.seed_mode == LITERT_LM_SEED_RANDOM) result.seed = GenerateSeed();
  result.seed_mode = LITERT_LM_SEED_FIXED;
  return result;
}

bool SamplingEquals(const LiteRtLm_Sampling& left,
                    const LiteRtLm_Sampling& right) {
  return left.temperature == right.temperature && left.top_p == right.top_p &&
         left.top_k == right.top_k && left.seed_mode == right.seed_mode &&
         left.seed == right.seed;
}

bool ParseArray(std::string_view input, OrderedJson* output) {
  if (output == nullptr) return false;
  if (input.empty()) {
    *output = OrderedJson::array();
    return true;
  }
  OrderedJson parsed = OrderedJson::parse(input, nullptr, false);
  if (parsed.is_discarded() || !parsed.is_array()) return false;
  *output = std::move(parsed);
  return true;
}

// The stable ABI accepts both LiteRT-LM's flat declarations and the common
// OpenAI-compatible {"type":"function","function":{...}} envelope used by
// Unreal callers. The official core and its constrained decoder require the
// flat declaration objects.
bool NormalizeToolsForNative(std::string_view input, std::string* output) {
  if (output == nullptr) return false;
  OrderedJson tools;
  if (!ParseArray(input, &tools)) return false;
  OrderedJson normalized = OrderedJson::array();
  for (const auto& tool : tools) {
    if (!tool.is_object()) return false;
    if (tool.contains("function") && tool["function"].is_object()) {
      normalized.push_back(tool["function"]);
    } else {
      normalized.push_back(tool);
    }
  }
  *output = normalized.dump();
  return true;
}

bool ParseObject(std::string_view input, OrderedJson* output) {
  if (output == nullptr) return false;
  if (input.empty()) {
    *output = OrderedJson::object();
    return true;
  }
  OrderedJson parsed = OrderedJson::parse(input, nullptr, false);
  if (parsed.is_discarded() || !parsed.is_object()) return false;
  *output = std::move(parsed);
  return true;
}

bool ParseMessages(std::string_view input, OrderedJson* output) {
  if (output == nullptr || input.empty()) return false;
  OrderedJson parsed = OrderedJson::parse(input, nullptr, false);
  if (parsed.is_discarded()) return false;
  if (parsed.is_object()) {
    *output = OrderedJson::array({std::move(parsed)});
    return true;
  }
  if (!parsed.is_array() || parsed.empty()) return false;
  for (const auto& item : parsed) {
    if (!item.is_object()) return false;
  }
  *output = std::move(parsed);
  return true;
}

std::string ExtractText(const OrderedJson& value) {
  std::string output;
  if (!value.is_object()) return output;
  const auto content = value.find("content");
  if (content == value.end()) return output;
  auto append = [&output](const OrderedJson& item) {
    if (item.is_string()) {
      output += item.get_ref<const std::string&>();
    } else if (item.is_object()) {
      const auto text = item.find("text");
      if (text != item.end() && text->is_string()) {
        output += text->get_ref<const std::string&>();
      }
    }
  };
  if (content->is_array()) {
    for (const auto& item : *content) append(item);
  } else {
    append(*content);
  }
  return output;
}

void MergeChunk(OrderedJson* aggregate, const OrderedJson& chunk) {
  if (aggregate == nullptr || !chunk.is_object()) return;
  if (!aggregate->is_object()) *aggregate = OrderedJson::object();

  const auto role = chunk.find("role");
  if (role != chunk.end()) (*aggregate)["role"] = *role;

  const auto tool_calls = chunk.find("tool_calls");
  if (tool_calls != chunk.end() && !tool_calls->is_null()) {
    // LiteRT-LM emits a complete model Message for tool decisions. Preserve
    // the latest authoritative structure rather than inventing a schema.
    (*aggregate)["tool_calls"] = *tool_calls;
  }

  const auto channels = chunk.find("channels");
  if (channels != chunk.end() && channels->is_object()) {
    OrderedJson& destination = (*aggregate)["channels"];
    if (!destination.is_object()) destination = OrderedJson::object();
    for (auto item = channels->begin(); item != channels->end(); ++item) {
      if (item.value().is_string() && destination.contains(item.key()) &&
          destination[item.key()].is_string()) {
        destination[item.key()].get_ref<std::string&>() +=
            item.value().get_ref<const std::string&>();
      } else {
        destination[item.key()] = item.value();
      }
    }
  }

  for (auto item = chunk.begin(); item != chunk.end(); ++item) {
    if (item.key() == "role" || item.key() == "content" ||
        item.key() == "tool_calls" || item.key() == "channels") {
      continue;
    }
    (*aggregate)[item.key()] = item.value();
  }
}

OrderedJson SystemHistoryMessage(std::string_view system_message) {
  OrderedJson result = OrderedJson::object();
  result["role"] = "system";
  OrderedJson content = OrderedJson::parse(system_message, nullptr, false);
  result["content"] = content.is_discarded()
                          ? OrderedJson(std::string(system_message))
                          : std::move(content);
  return result;
}

LiteRtLm_Status ParseConversationDefinition(
    const LiteRtLm_ConversationConfig* config, ConversationDefinition* output,
    OrderedJson* history, LiteRtLm_ErrorInfo* error) {
  if (output == nullptr || history == nullptr) {
    return SetError(error, LITERT_LM_STATUS_INVALID_ARGUMENT,
                    LITERT_LM_PHASE_CONVERSATION_CREATE,
                    "Conversation definition outputs are null.");
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

  ConversationDefinition definition;
  definition.tools_json = CopyView(config->tools_json);
  OrderedJson parsed;
  if (!ParseArray(definition.tools_json, &parsed)) {
    return SetError(error, LITERT_LM_STATUS_PARSE_ERROR,
                    LITERT_LM_PHASE_CONVERSATION_CREATE,
                    "tools_json must be a JSON array.");
  }

  if (LITERT_LM_HAS_FIELD(config, LiteRtLm_ConversationConfig, flags)) {
    definition.flags = config->flags;
  }
  if ((definition.flags & ~kKnownConversationFlags) != 0) {
    return SetError(error, LITERT_LM_STATUS_INVALID_ARGUMENT,
                    LITERT_LM_PHASE_CONVERSATION_CREATE,
                    "ConversationConfig.flags contains unknown bits.");
  }
  if (LITERT_LM_HAS_FIELD(config, LiteRtLm_ConversationConfig,
                          default_max_output_tokens)) {
    definition.default_max_output_tokens =
        config->default_max_output_tokens;
  }

  auto copy_optional = [&](LiteRtLm_StringView value, std::string* target,
                           const char* field) -> LiteRtLm_Status {
    if (!IsViewValid(value)) {
      return SetError(error, LITERT_LM_STATUS_INVALID_ARGUMENT,
                      LITERT_LM_PHASE_CONVERSATION_CREATE,
                      std::string(field) + " is invalid.");
    }
    *target = CopyView(value);
    return LITERT_LM_STATUS_OK;
  };

  if (LITERT_LM_HAS_FIELD(config, LiteRtLm_ConversationConfig,
                          system_message)) {
    LiteRtLm_Status status = copy_optional(
        config->system_message, &definition.system_message, "system_message");
    if (status != LITERT_LM_STATUS_OK) return status;
  }
  if (LITERT_LM_HAS_FIELD(config, LiteRtLm_ConversationConfig,
                          initial_messages_json)) {
    LiteRtLm_Status status = copy_optional(
        config->initial_messages_json, &definition.initial_messages_json,
        "initial_messages_json");
    if (status != LITERT_LM_STATUS_OK) return status;
  }
  if (!ParseArray(definition.initial_messages_json,
                  &definition.base_messages)) {
    return SetError(error, LITERT_LM_STATUS_PARSE_ERROR,
                    LITERT_LM_PHASE_CONVERSATION_CREATE,
                    "initial_messages_json must be a JSON array.");
  }
  if (LITERT_LM_HAS_FIELD(config, LiteRtLm_ConversationConfig,
                          extra_context_json)) {
    LiteRtLm_Status status = copy_optional(
        config->extra_context_json, &definition.extra_context_json,
        "extra_context_json");
    if (status != LITERT_LM_STATUS_OK) return status;
  }
  if (!ParseObject(definition.extra_context_json, &parsed)) {
    return SetError(error, LITERT_LM_STATUS_PARSE_ERROR,
                    LITERT_LM_PHASE_CONVERSATION_CREATE,
                    "extra_context_json must be a JSON object.");
  }
  if (LITERT_LM_HAS_FIELD(config, LiteRtLm_ConversationConfig,
                          tool_call_channel)) {
    LiteRtLm_Status status = copy_optional(
        config->tool_call_channel, &definition.tool_call_channel,
        "tool_call_channel");
    if (status != LITERT_LM_STATUS_OK) return status;
    if (definition.tool_call_channel.empty()) {
      definition.tool_call_channel = "tool_call";
    }
  }
  if (LITERT_LM_HAS_FIELD(config, LiteRtLm_ConversationConfig, lora_path)) {
    LiteRtLm_Status status = copy_optional(
        config->lora_path, &definition.lora_path, "lora_path");
    if (status != LITERT_LM_STATUS_OK) return status;
  }
  if (LITERT_LM_HAS_FIELD(config, LiteRtLm_ConversationConfig,
                          audio_lora_path)) {
    LiteRtLm_Status status = copy_optional(
        config->audio_lora_path, &definition.audio_lora_path,
        "audio_lora_path");
    if (status != LITERT_LM_STATUS_OK) return status;
  }

  OrderedJson canonical = OrderedJson::array();
  if (!definition.system_message.empty()) {
    canonical.push_back(SystemHistoryMessage(definition.system_message));
  }
  canonical.insert(canonical.end(), definition.base_messages.begin(),
                   definition.base_messages.end());
  *output = std::move(definition);
  *history = std::move(canonical);
  return LITERT_LM_STATUS_OK;
}

LiteRtLmConversation* CreateNativeConversation(
    LiteRtLm_EngineObject& engine, const ConversationDefinition& definition,
    const OrderedJson* history_override, const LiteRtLm_Sampling& sampling,
    std::string* error) {
  NativeApi& api = engine.library->api;
  LiteRtLmSamplerParams* sampler =
      api.litert_lm_sampler_params_create(kLiteRtLmSamplerTypeTopP);
  LiteRtLmSessionConfig* session = api.litert_lm_session_config_create();
  LiteRtLmConversationConfig* conversation_config =
      api.litert_lm_conversation_config_create();
  auto cleanup = [&]() {
    if (conversation_config != nullptr) {
      api.litert_lm_conversation_config_delete(conversation_config);
    }
    if (session != nullptr) api.litert_lm_session_config_delete(session);
    if (sampler != nullptr) api.litert_lm_sampler_params_delete(sampler);
  };
  if (sampler == nullptr || session == nullptr || conversation_config == nullptr) {
    cleanup();
    if (error != nullptr) *error = "Unable to allocate official native configs.";
    return nullptr;
  }

  api.litert_lm_sampler_params_set_top_k(sampler, sampling.top_k);
  api.litert_lm_sampler_params_set_top_p(sampler, sampling.top_p);
  api.litert_lm_sampler_params_set_temperature(
      sampler, sampling.temperature);
  api.litert_lm_sampler_params_set_seed(sampler, sampling.seed);
  api.litert_lm_session_config_set_sampler_params(session, sampler);
  if (definition.default_max_output_tokens != 0) {
    api.litert_lm_session_config_set_max_output_tokens(
        session, static_cast<int>(definition.default_max_output_tokens));
  }
  api.litert_lm_session_config_set_apply_prompt_template(
      session, (definition.flags &
                LITERT_LM_CONVERSATION_DISABLE_PROMPT_TEMPLATE) == 0);
  if (!definition.lora_path.empty() &&
      api.litert_lm_session_config_set_lora_path(
          session, definition.lora_path.c_str()) != 0) {
    cleanup();
    if (error != nullptr) *error = "The official core rejected the LoRA path.";
    return nullptr;
  }
  if (!definition.audio_lora_path.empty() &&
      api.litert_lm_session_config_set_audio_lora_path(
          session, definition.audio_lora_path.c_str()) != 0) {
    cleanup();
    if (error != nullptr) {
      *error = "The official core rejected the audio LoRA path.";
    }
    return nullptr;
  }

  api.litert_lm_conversation_config_set_session_config(conversation_config,
                                                        session);
  if (history_override != nullptr) {
    const std::string messages = history_override->dump();
    api.litert_lm_conversation_config_set_messages(conversation_config,
                                                    messages.c_str());
  } else {
    std::string system_message = definition.system_message;
    if ((definition.flags & LITERT_LM_CONVERSATION_ENABLE_THINKING) != 0 &&
        system_message.find("<|think|>") == std::string::npos) {
      if (!system_message.empty()) system_message.push_back('\n');
      system_message += "<|think|>";
    }
    if (!system_message.empty()) {
      api.litert_lm_conversation_config_set_system_message(
          conversation_config, system_message.c_str());
    }
    if (!definition.initial_messages_json.empty()) {
      api.litert_lm_conversation_config_set_messages(
          conversation_config, definition.initial_messages_json.c_str());
    }
  }
  // Official config setters retain string pointers until conversation_create,
  // so keep the normalized storage alive for the remainder of this function.
  std::string native_tools_json;
  if (!definition.tools_json.empty()) {
    if ((definition.flags &
         LITERT_LM_CONVERSATION_ENABLE_CONSTRAINED_DECODING) != 0) {
      if (!NormalizeToolsForNative(definition.tools_json, &native_tools_json)) {
        cleanup();
        if (error != nullptr) *error = "Unable to normalize tool declarations.";
        return nullptr;
      }
    } else {
      // Preserve the caller's OpenAI-compatible envelope on the unconstrained
      // official v0.16.1 Win64 path. Replacing a live Conversation with a
      // flattened declaration currently corrupts upstream heap state.
      native_tools_json = definition.tools_json;
    }
    api.litert_lm_conversation_config_set_tools(conversation_config,
                                                native_tools_json.c_str());
  }
  if (!definition.extra_context_json.empty()) {
    api.litert_lm_conversation_config_set_extra_context(
        conversation_config, definition.extra_context_json.c_str());
  }
  api.litert_lm_conversation_config_set_enable_constrained_decoding(
      conversation_config,
      (definition.flags &
       LITERT_LM_CONVERSATION_ENABLE_CONSTRAINED_DECODING) != 0);
  api.litert_lm_conversation_config_set_filter_channel_content_from_kv_cache(
      conversation_config,
      (definition.flags &
       LITERT_LM_CONVERSATION_FILTER_CHANNEL_CONTENT_FROM_KV) != 0);
  api.litert_lm_conversation_config_set_stream_tool_calls(
      conversation_config,
      (definition.flags & LITERT_LM_CONVERSATION_STREAM_TOOL_CALLS) != 0,
      definition.tool_call_channel.c_str());

  LiteRtLmConversation* native = api.litert_lm_conversation_create(
      engine.native, conversation_config);
  cleanup();
  if (native == nullptr && error != nullptr) {
    *error = "Official litert_lm_conversation_create returned null.";
  }
  return native;
}

void CaptureBenchmarkBaseline(LiteRtLm_RequestObject& request) {
  if ((request.engine->flags & LITERT_LM_ENGINE_ENABLE_METRICS) == 0) return;
  NativeApi& api = request.engine->library->api;
  LiteRtLmBenchmarkInfo* info =
      api.litert_lm_conversation_get_benchmark_info(
          request.conversation->native);
  if (info == nullptr) return;
  request.prefill_turn_baseline =
      api.litert_lm_benchmark_info_get_num_prefill_turns(info);
  request.decode_turn_baseline =
      api.litert_lm_benchmark_info_get_num_decode_turns(info);
  request.benchmark_baseline_valid = true;
  api.litert_lm_benchmark_info_delete(info);
}

void RefreshMetrics(LiteRtLm_RequestObject& request) {
  {
    std::lock_guard<std::mutex> lock(request.mutex);
    if (request.metrics_finalized) return;
  }
  NativeApi& api = request.engine->library->api;
  uint64_t valid = LITERT_LM_METRIC_TOTAL_TIME;
  uint64_t input_tokens = 0;
  uint64_t output_tokens = 0;
  uint64_t prefill_us = 0;
  uint64_t decode_us = 0;
  uint64_t context_tokens = 0;

  const int token_count = api.litert_lm_conversation_get_token_count(
      request.conversation->native);
  if (token_count >= 0) {
    context_tokens = static_cast<uint64_t>(token_count);
    valid |= LITERT_LM_METRIC_CONTEXT_TOKENS;
  }

  if (request.benchmark_baseline_valid) {
    LiteRtLmBenchmarkInfo* info =
        api.litert_lm_conversation_get_benchmark_info(
            request.conversation->native);
    if (info != nullptr) {
      const int prefill_turns =
          api.litert_lm_benchmark_info_get_num_prefill_turns(info);
      const int decode_turns =
          api.litert_lm_benchmark_info_get_num_decode_turns(info);
      for (int index = request.prefill_turn_baseline; index < prefill_turns;
           ++index) {
        const int tokens =
            api.litert_lm_benchmark_info_get_prefill_token_count_at(info,
                                                                    index);
        const double rate =
            api.litert_lm_benchmark_info_get_prefill_tokens_per_sec_at(info,
                                                                       index);
        if (tokens > 0) input_tokens += static_cast<uint64_t>(tokens);
        if (tokens > 0 && std::isfinite(rate) && rate > 0.0) {
          prefill_us += static_cast<uint64_t>(
              static_cast<double>(tokens) * 1000000.0 / rate);
        }
      }
      for (int index = request.decode_turn_baseline; index < decode_turns;
           ++index) {
        const int tokens =
            api.litert_lm_benchmark_info_get_decode_token_count_at(info,
                                                                   index);
        const double rate =
            api.litert_lm_benchmark_info_get_decode_tokens_per_sec_at(info,
                                                                      index);
        if (tokens > 0) output_tokens += static_cast<uint64_t>(tokens);
        if (tokens > 0 && std::isfinite(rate) && rate > 0.0) {
          decode_us += static_cast<uint64_t>(
              static_cast<double>(tokens) * 1000000.0 / rate);
        }
      }
      api.litert_lm_benchmark_info_delete(info);
      if (prefill_turns > request.prefill_turn_baseline) {
        valid |= LITERT_LM_METRIC_INPUT_TOKENS |
                 LITERT_LM_METRIC_PREFILL_TIME;
      }
      if (decode_turns > request.decode_turn_baseline) {
        valid |= LITERT_LM_METRIC_OUTPUT_TOKENS |
                 LITERT_LM_METRIC_DECODE_TIME;
      }
    }
  }
  if (request.first_output_seen) valid |= LITERT_LM_METRIC_TTFT;
  double output_rate = 0.0;
  if (decode_us != 0 && output_tokens != 0) {
    output_rate = static_cast<double>(output_tokens) * 1000000.0 /
                  static_cast<double>(decode_us);
    valid |= LITERT_LM_METRIC_OUTPUT_RATE;
  }

  std::lock_guard<std::mutex> lock(request.mutex);
  request.valid_metrics = valid;
  request.input_tokens = input_tokens;
  request.output_tokens = output_tokens;
  request.context_tokens = context_tokens;
  request.prefill_time_us = prefill_us;
  request.decode_time_us = decode_us;
  request.output_tokens_per_second = output_rate;
  request.metrics_finalized = true;
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
                   LiteRtLm_FinishReason reason, std::string_view message,
                   LiteRtLm_ErrorPhase phase = LITERT_LM_PHASE_NONE,
                   int32_t native_code = 0,
                   bool retryable = false) noexcept {
  std::lock_guard<std::recursive_mutex> callback_lock(request.callback_mutex);
  try {
    {
      std::lock_guard<std::mutex> lock(request.mutex);
      if (request.terminal) return;
      if (request.cancel_requested.load(std::memory_order_acquire)) {
        status = LITERT_LM_STATUS_CANCELLED;
        reason = LITERT_LM_FINISH_CANCELLED;
        phase = LITERT_LM_PHASE_CANCEL;
        retryable = true;
        message = "Request cancelled.";
      }
    }

    OrderedJson final_message = request.aggregate;
    if (!final_message.is_object()) final_message = OrderedJson::object();
    if (!final_message.contains("role")) final_message["role"] = "assistant";
    if (!request.text.empty()) {
      final_message["content"] = OrderedJson::array(
          {{{"type", "text"}, {"text", request.text}}});
    } else if (!final_message.contains("content") &&
               !final_message.contains("tool_calls")) {
      final_message["content"] = OrderedJson::array();
    }

    std::string response_json;
    std::string stable_error;
    bool has_tool_call = false;
    if (status == LITERT_LM_STATUS_OK) {
      response_json = final_message.dump();
      const auto tool_calls = final_message.find("tool_calls");
      has_tool_call = tool_calls != final_message.end() &&
                      !tool_calls->is_null() && !tool_calls->empty();
      std::lock_guard<std::mutex> conversation_lock(request.conversation->mutex);
      request.conversation->history.insert(
          request.conversation->history.end(), request.input_messages.begin(),
          request.input_messages.end());
      request.conversation->history.push_back(final_message);
    } else {
      stable_error.assign(message.data(), message.size());
      request.conversation->needs_rebuild.store(true,
                                                 std::memory_order_release);
    }

    LiteRtLm_EventCallback callback = nullptr;
    void* user_data = nullptr;
    uint64_t sequence = 0;
    {
      std::lock_guard<std::mutex> lock(request.mutex);
      request.status = status;
      request.finish_reason = has_tool_call ? LITERT_LM_FINISH_TOOL_CALL : reason;
      request.state = status == LITERT_LM_STATUS_OK
                          ? LITERT_LM_REQUEST_SUCCEEDED
                          : (status == LITERT_LM_STATUS_CANCELLED
                                 ? LITERT_LM_REQUEST_CANCELLED
                                 : LITERT_LM_REQUEST_FAILED);
      request.error_phase =
          status == LITERT_LM_STATUS_OK ? LITERT_LM_PHASE_NONE : phase;
      request.native_code = status == LITERT_LM_STATUS_OK ? 0 : native_code;
      request.retryable = status == LITERT_LM_STATUS_OK ? false : retryable;
      request.response_json.swap(response_json);
      request.error_message.swap(stable_error);
      request.end_time = std::chrono::steady_clock::now();
      request.total_time_us = ToMicroseconds(request.end_time - request.start_time);
      request.valid_metrics |= LITERT_LM_METRIC_TOTAL_TIME;
      if (!request.first_output_seen && !request.response_json.empty()) {
        request.first_output_seen = true;
        request.first_output_time = request.end_time;
      }
      if (request.first_output_seen) {
        request.time_to_first_token_us =
            ToMicroseconds(request.first_output_time - request.start_time);
        request.valid_metrics |= LITERT_LM_METRIC_TTFT;
      }
      request.terminal = true;
      callback = request.callback;
      user_data = request.user_data;
      sequence = ++request.sequence;
      if (callback != nullptr) {
        request.callback_in_progress = true;
        ++request.callback_count;
      }
    }

    if (callback != nullptr) {
      LiteRtLm_Event event{};
      event.struct_size = static_cast<uint32_t>(sizeof(event));
      event.kind = LITERT_LM_EVENT_COMPLETE;
      event.status = status;
      event.request = &request;
      event.sequence = sequence;
      event.response_json = ViewOf(request.response_json);
      event.error_message = ViewOf(request.error_message);
      LiteRtLm_RequestObject* previous = g_callback_request;
      g_callback_request = &request;
      try {
        callback(&event, user_data);
      } catch (...) {
        // A client callback must never unwind through the official core.
      }
      g_callback_request = previous;
      {
        std::lock_guard<std::mutex> lock(request.mutex);
        request.callback_in_progress = false;
      }
    }
  } catch (...) {
    std::lock_guard<std::mutex> lock(request.mutex);
    if (!request.terminal) {
      request.status = LITERT_LM_STATUS_INTERNAL_ERROR;
      request.state = LITERT_LM_REQUEST_FAILED;
      request.finish_reason = LITERT_LM_FINISH_ERROR;
      request.error_phase = LITERT_LM_PHASE_CALLBACK;
      request.error_message = "Failed while finalizing the native response.";
      request.end_time = std::chrono::steady_clock::now();
      request.total_time_us =
          ToMicroseconds(request.end_time - request.start_time);
      request.terminal = true;
      request.callback_in_progress = false;
      request.conversation->needs_rebuild.store(true,
                                                 std::memory_order_release);
    }
  }
  MaybeReleaseActiveRequest(request);
  request.condition.notify_all();
}

void ProcessNativeChunk(LiteRtLm_RequestObject& request,
                        const char* chunk) noexcept {
  if (chunk == nullptr || *chunk == '\0') return;
  std::lock_guard<std::recursive_mutex> callback_lock(request.callback_mutex);
  try {
    OrderedJson parsed = OrderedJson::parse(chunk, nullptr, false);
    std::string delta;
    LiteRtLm_EventKind kind = LITERT_LM_EVENT_TEXT_DELTA;
    if (!parsed.is_discarded() && parsed.is_object()) {
      delta = ExtractText(parsed);
      MergeChunk(&request.aggregate, parsed);
      const auto tool_calls = parsed.find("tool_calls");
      const auto channels = parsed.find("channels");
      if (tool_calls != parsed.end() && !tool_calls->is_null() &&
          !tool_calls->empty()) {
        kind = LITERT_LM_EVENT_TOOL_CALL_DELTA;
      } else if (channels != parsed.end() && !channels->is_null() &&
                 !channels->empty()) {
        kind = LITERT_LM_EVENT_CHANNEL_DELTA;
      }
    } else {
      delta = chunk;
    }

    LiteRtLm_EventCallback callback = nullptr;
    void* user_data = nullptr;
    uint64_t sequence = 0;
    {
      std::lock_guard<std::mutex> lock(request.mutex);
      if (request.terminal ||
          request.cancel_requested.load(std::memory_order_acquire)) {
        return;
      }
      request.text += delta;
      if (!request.first_output_seen) {
        request.first_output_seen = true;
        request.first_output_time = std::chrono::steady_clock::now();
      }
      callback = request.callback;
      user_data = request.user_data;
      sequence = ++request.sequence;
      if (callback != nullptr) {
        request.callback_in_progress = true;
        ++request.callback_count;
      }
    }

    bool callback_failed = false;
    if (callback != nullptr) {
      LiteRtLm_Event event{};
      event.struct_size = static_cast<uint32_t>(sizeof(event));
      event.kind = kind;
      event.status = LITERT_LM_STATUS_OK;
      event.request = &request;
      event.sequence = sequence;
      event.text_delta = ViewOf(delta);
      event.response_json = ViewOf(chunk, std::strlen(chunk));
      LiteRtLm_RequestObject* previous = g_callback_request;
      g_callback_request = &request;
      try {
        callback(&event, user_data);
      } catch (...) {
        callback_failed = true;
      }
      g_callback_request = previous;
      {
        std::lock_guard<std::mutex> lock(request.mutex);
        request.callback_in_progress = false;
      }
      request.condition.notify_all();
    }

    if (callback_failed) {
      request.conversation->needs_rebuild.store(true,
                                                 std::memory_order_release);
      request.engine->library->api.litert_lm_conversation_cancel_process(
          request.conversation->native);
      FinishRequest(request, LITERT_LM_STATUS_INTERNAL_ERROR,
                    LITERT_LM_FINISH_ERROR,
                    "The client stream callback threw an exception.",
                    LITERT_LM_PHASE_CALLBACK);
    }
  } catch (...) {
    request.engine->library->api.litert_lm_conversation_cancel_process(
        request.conversation->native);
    FinishRequest(request, LITERT_LM_STATUS_INTERNAL_ERROR,
                  LITERT_LM_FINISH_ERROR,
                  "Native stream chunk processing failed.",
                  LITERT_LM_PHASE_CALLBACK);
  }
}

bool RecoverMalformedToolExpression(std::string_view error,
                                    OrderedJson* recovered) {
  if (recovered == nullptr ||
      error.find("Failed to parse tool calls") == std::string_view::npos) {
    return false;
  }
  const size_t call = error.find("call:");
  if (call == std::string_view::npos) return false;
  const size_t name_start = call + 5;
  const size_t object_start = error.find('{', name_start);
  if (object_start == std::string_view::npos || object_start == name_start) {
    return false;
  }
  const std::string name(error.substr(name_start, object_start - name_start));
  if (name.find_first_not_of(
          "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_") !=
      std::string::npos) {
    return false;
  }

  OrderedJson arguments = OrderedJson::object();
  const size_t speech_key = error.find("speech:", object_start);
  if (speech_key != std::string_view::npos) {
    const size_t value_start = speech_key + 7;
    const std::string_view open_smart("\xE2\x80\x9C", 3);
    const std::string_view close_smart("\xE2\x80\x9D", 3);
    size_t open = error.find(open_smart, value_start);
    size_t close = std::string_view::npos;
    size_t quote_size = 3;
    if (open != std::string_view::npos) {
      close = error.find(close_smart, open + quote_size);
    } else {
      open = error.find('"', value_start);
      quote_size = 1;
      if (open != std::string_view::npos) {
        close = error.find('"', open + quote_size);
      }
    }
    if (open != std::string_view::npos && close != std::string_view::npos &&
        close > open + quote_size) {
      arguments["speech"] = std::string(
          error.substr(open + quote_size, close - open - quote_size));
    }
  }
  const size_t seat_key = error.find("seat:", object_start);
  if (seat_key != std::string_view::npos) {
    size_t cursor = seat_key + 5;
    while (cursor < error.size() &&
           (error[cursor] == ' ' || error[cursor] == '\t' ||
            error[cursor] == '"')) {
      ++cursor;
    }
    bool negative = false;
    if (cursor < error.size() && error[cursor] == '-') {
      negative = true;
      ++cursor;
    }
    int seat = 0;
    bool has_digit = false;
    while (cursor < error.size() && error[cursor] >= '0' &&
           error[cursor] <= '9') {
      has_digit = true;
      seat = seat * 10 + (error[cursor] - '0');
      ++cursor;
    }
    if (has_digit) arguments["seat"] = negative ? -seat : seat;
  }
  if (arguments.empty()) return false;

  *recovered = OrderedJson{
      {"role", "assistant"},
      {"tool_calls",
       OrderedJson::array(
           {{{"type", "function"},
             {"function", {{"name", name}, {"arguments", arguments}}}}})}};
  return true;
}

void NativeStreamCallback(void* callback_data, const char* chunk,
                          bool is_final, const char* error_message) {
  auto* request = static_cast<LiteRtLm_RequestObject*>(callback_data);
  if (request == nullptr || request->magic != kRequestMagic) return;
  struct CallbackGuard {
    explicit CallbackGuard(LiteRtLm_RequestObject& value) : request(value) {
      request.native_callbacks_in_flight.fetch_add(1,
                                                   std::memory_order_acq_rel);
    }
    ~CallbackGuard() {
      request.native_callbacks_in_flight.fetch_sub(1,
                                                   std::memory_order_acq_rel);
      request.condition.notify_all();
    }
    LiteRtLm_RequestObject& request;
  } guard(*request);

  if (error_message != nullptr && *error_message != '\0') {
    const std::string_view error(error_message);
    const bool cancelled =
        request->cancel_requested.load(std::memory_order_acquire) ||
        error.find("CANCELLED") != std::string_view::npos;
    const bool max_tokens =
        error.find("Max number of tokens reached") != std::string_view::npos;
    OrderedJson recovered_tool_call;
    const bool recovered_parse_failure =
        (request->conversation->definition.flags &
         LITERT_LM_CONVERSATION_RETURN_ERROR_ON_PARSE_FAILURE) == 0 &&
        RecoverMalformedToolExpression(error, &recovered_tool_call);
    if (recovered_parse_failure) {
      {
        std::lock_guard<std::recursive_mutex> callback_lock(
            request->callback_mutex);
        request->aggregate = std::move(recovered_tool_call);
        request->text.clear();
      }
      FinishRequest(*request, LITERT_LM_STATUS_OK,
                    LITERT_LM_FINISH_TOOL_CALL, {});
    } else if (max_tokens &&
        (request->conversation->definition.flags &
         LITERT_LM_CONVERSATION_RETURN_ERROR_ON_MAX_TOKENS) == 0) {
      FinishRequest(*request, LITERT_LM_STATUS_OK,
                    LITERT_LM_FINISH_COMPLETE, {});
    } else {
      FinishRequest(*request,
                    cancelled ? LITERT_LM_STATUS_CANCELLED
                              : LITERT_LM_STATUS_MODEL_ERROR,
                    cancelled ? LITERT_LM_FINISH_CANCELLED
                              : LITERT_LM_FINISH_ERROR,
                    error, cancelled ? LITERT_LM_PHASE_CANCEL
                                     : LITERT_LM_PHASE_DECODE,
                    0, cancelled);
    }
    return;
  }
  if (chunk != nullptr && *chunk != '\0') ProcessNativeChunk(*request, chunk);
  if (is_final) {
    FinishRequest(*request, LITERT_LM_STATUS_OK,
                  LITERT_LM_FINISH_COMPLETE, {});
  }
}

void NativeStreamChunkCallback(void* callback_data, const void* native_chunk) {
  auto* request = static_cast<LiteRtLm_RequestObject*>(callback_data);
  if (request == nullptr || request->magic != kRequestMagic ||
      native_chunk == nullptr) {
    return;
  }
  NativeApi& api = request->engine->library->api;
  if (!api.UsesStreamChunks()) return;
  NativeStreamCallback(
      callback_data, api.litert_lm_stream_chunk_get_text(native_chunk),
      api.litert_lm_stream_chunk_is_final(native_chunk),
      api.litert_lm_stream_chunk_get_error(native_chunk));
}

OrderedJson BaseHistory(const ConversationDefinition& definition) {
  OrderedJson history = OrderedJson::array();
  if (!definition.system_message.empty()) {
    history.push_back(SystemHistoryMessage(definition.system_message));
  }
  history.insert(history.end(), definition.base_messages.begin(),
                 definition.base_messages.end());
  return history;
}

LiteRtLm_Status RebuildConversationLocked(
    LiteRtLm_ConversationObject& conversation, LiteRtLm_ErrorInfo* error) {
  const bool needs_rebuild =
      conversation.needs_rebuild.load(std::memory_order_acquire);
  // LiteRT-LM v0.16 exposes independent stateful Conversation handles. Keep
  // each native handle resident until its own sampling/schema state actually
  // changes; merely switching logical speakers must not discard and replay
  // the target KV cache. The engine operation mutex still serializes decode.
  if (!needs_rebuild) {
    conversation.owner->resident_conversation = &conversation;
    return LITERT_LM_STATUS_OK;
  }
  std::string native_error;
  LiteRtLmConversation* replacement = CreateNativeConversation(
      *conversation.owner, conversation.definition, &conversation.history,
      conversation.native_sampling, &native_error);
  if (replacement == nullptr) {
    return SetError(error, LITERT_LM_STATUS_MODEL_ERROR,
                    LITERT_LM_PHASE_CONTEXT_PREPARE, native_error);
  }
  LiteRtLmConversation* previous = conversation.native;
  conversation.native = replacement;
  if ((conversation.definition.flags &
       LITERT_LM_CONVERSATION_ENABLE_CONSTRAINED_DECODING) != 0) {
    conversation.retired_native.push_back(previous);
  } else {
    conversation.owner->library->api.litert_lm_conversation_delete(previous);
  }
  conversation.generation.fetch_add(1, std::memory_order_acq_rel);
  conversation.needs_rebuild.store(false, std::memory_order_release);
  conversation.owner->resident_conversation = &conversation;
  return LITERT_LM_STATUS_OK;
}

LiteRtLm_Status ApiCreateEngine(const LiteRtLm_EngineConfig* config,
                                LiteRtLm_Engine* out_engine,
                                LiteRtLm_ErrorInfo* error) {
  return Boundary(LITERT_LM_PHASE_ENGINE_CREATE, error,
                  [&]() -> LiteRtLm_Status {
    if (out_engine == nullptr) {
      return SetError(error, LITERT_LM_STATUS_INVALID_ARGUMENT,
                      LITERT_LM_PHASE_ENGINE_CREATE,
                      "Engine output pointer is null.");
    }
    *out_engine = nullptr;
    LiteRtLm_EngineConfig defaults = LiteRtLm_DefaultEngineConfig();
    if (config == nullptr) config = &defaults;
    if (config->struct_size <
            LITERT_LM_FIELD_END(LiteRtLm_EngineConfig, sampling) ||
        !IsViewValid(config->model_path) || config->model_path.size == 0 ||
        (config->backend != LITERT_LM_BACKEND_CPU &&
         config->backend != LITERT_LM_BACKEND_GPU) ||
        (config->flags & ~kKnownEngineFlags) != 0) {
      return SetError(error, LITERT_LM_STATUS_INVALID_ARGUMENT,
                      LITERT_LM_PHASE_ENGINE_CREATE,
                      "EngineConfig is invalid.");
    }
    LiteRtLm_Status sampling_status = ValidateSampling(
        config->sampling, LITERT_LM_PHASE_ENGINE_CREATE, error);
    if (sampling_status != LITERT_LM_STATUS_OK) return sampling_status;
    if (config->max_context_tokens >
        static_cast<uint32_t>((std::numeric_limits<int>::max)())) {
      return SetError(error, LITERT_LM_STATUS_INVALID_ARGUMENT,
                      LITERT_LM_PHASE_ENGINE_CREATE,
                      "max_context_tokens exceeds the native integer range.");
    }

    std::string cache_dir;
    std::string dispatch_dir;
    uint32_t num_threads = 0;
    int32_t prefill_chunk_size = 0;
    if (LITERT_LM_HAS_FIELD(config, LiteRtLm_EngineConfig, num_threads)) {
      num_threads = config->num_threads;
      prefill_chunk_size = config->prefill_chunk_size;
      if (!IsViewValid(config->cache_dir) ||
          !IsViewValid(config->dispatch_library_dir) ||
          num_threads > static_cast<uint32_t>(
                            (std::numeric_limits<int>::max)())) {
        return SetError(error, LITERT_LM_STATUS_INVALID_ARGUMENT,
                        LITERT_LM_PHASE_ENGINE_CREATE,
                        "EngineConfig ABI 1.1 fields are invalid.");
      }
      cache_dir = CopyView(config->cache_dir);
      dispatch_dir = CopyView(config->dispatch_library_dir);
    }

    std::string loader_error;
    std::shared_ptr<NativeLibrary> library = GetNativeLibrary(&loader_error);
    if (!library) {
      return SetError(error, LITERT_LM_STATUS_BACKEND_ERROR,
                      LITERT_LM_PHASE_ENGINE_CREATE, loader_error,
#if defined(_WIN32)
                      static_cast<int32_t>(GetLastError()),
#else
                      0,
#endif
                      false);
    }

    const std::string model_path = CopyView(config->model_path);
    const char* backend = config->backend == LITERT_LM_BACKEND_GPU ? "gpu"
                                                                   : "cpu";
    const char* vision_backend =
        (config->flags & LITERT_LM_ENGINE_ENABLE_VISION) != 0 ? backend
                                                              : nullptr;
    const char* audio_backend =
        (config->flags & LITERT_LM_ENGINE_ENABLE_AUDIO) != 0 ? backend
                                                             : nullptr;
    NativeApi& api = library->api;
    LiteRtLmEngineSettings* settings = api.litert_lm_engine_settings_create(
        model_path.c_str(), backend, vision_backend, audio_backend);
    if (settings == nullptr) {
      return SetError(error, LITERT_LM_STATUS_MODEL_ERROR,
                      LITERT_LM_PHASE_ENGINE_CREATE,
                      "Official litert_lm_engine_settings_create returned null.");
    }
    auto delete_settings = [&]() {
      api.litert_lm_engine_settings_delete(settings);
    };
    if (config->max_context_tokens != 0) {
      api.litert_lm_engine_settings_set_max_num_tokens(
          settings, static_cast<int>(config->max_context_tokens));
    }
    if (num_threads != 0) {
      api.litert_lm_engine_settings_set_num_threads(
          settings, static_cast<int>(num_threads));
    }
    api.litert_lm_engine_settings_set_parallel_file_section_loading(
        settings,
        (config->flags &
         LITERT_LM_ENGINE_DISABLE_PARALLEL_FILE_LOADING) == 0);
    if (prefill_chunk_size != 0) {
      api.litert_lm_engine_settings_set_prefill_chunk_size(
          settings, prefill_chunk_size);
    }
    if (!cache_dir.empty()) {
      api.litert_lm_engine_settings_set_cache_dir(settings,
                                                  cache_dir.c_str());
    }
    if (!dispatch_dir.empty()) {
      api.litert_lm_engine_settings_set_litert_dispatch_lib_dir(
          settings, dispatch_dir.c_str());
    }
    if ((config->flags & LITERT_LM_ENGINE_ENABLE_SPECULATIVE_DECODING) != 0) {
      api.litert_lm_engine_settings_set_enable_speculative_decoding(settings,
                                                                    true);
    }
    if ((config->flags & LITERT_LM_ENGINE_ENABLE_METRICS) != 0) {
      api.litert_lm_engine_settings_enable_benchmark(settings);
    }

    LiteRtLmEngine* native = api.litert_lm_engine_create(settings);
    delete_settings();
    if (native == nullptr) {
      return SetError(error, LITERT_LM_STATUS_MODEL_ERROR,
                      LITERT_LM_PHASE_ENGINE_CREATE,
                      "Official litert_lm_engine_create returned null. See the "
                      "native LiteRT-LM log for model/backend diagnostics.");
    }

    auto engine = std::make_unique<LiteRtLm_EngineObject>();
    engine->magic = kEngineMagic;
    engine->library = std::move(library);
    engine->native = native;
    engine->configured_backend = config->backend;
    engine->flags = config->flags;
    engine->max_context_tokens = config->max_context_tokens;
    engine->applied_sampling = ResolveSampling(config->sampling);
    engine->backend_description =
        std::string("official LiteRT-LM ") +
        (engine->library->api.UsesStreamChunks() ? "v0.16.1" : "v0.14") +
        " C API; requested " + backend +
        "; core=" + engine->library->path +
        "; per-operation delegation telemetry is not exposed upstream";
    *out_engine = engine.release();
    return LITERT_LM_STATUS_OK;
  });
}

LiteRtLm_Status ApiGetEngineInfo(LiteRtLm_Engine engine,
                                 LiteRtLm_EngineInfo* out_info,
                                 LiteRtLm_ErrorInfo* error) {
  return Boundary(LITERT_LM_PHASE_API, error, [&]() -> LiteRtLm_Status {
    if (!IsEngineValid(engine) || out_info == nullptr ||
        out_info->struct_size <
            LITERT_LM_FIELD_END(LiteRtLm_EngineInfo, backend_description)) {
      return SetError(error, LITERT_LM_STATUS_INVALID_ARGUMENT,
                      LITERT_LM_PHASE_API,
                      "Engine or EngineInfo buffer is invalid.");
    }
    const uint32_t capacity = out_info->struct_size;
    LiteRtLm_EngineInfo value{};
    value.struct_size = static_cast<uint32_t>(sizeof(value));
    value.configured_backend = engine->configured_backend;
    value.resolved_backend = engine->configured_backend;
    value.configured_sampler_backend = engine->configured_backend;
    // The official C API reports backend configuration but exposes no
    // per-operation delegate coverage. Do not manufacture stronger evidence.
    value.backend_evidence = LITERT_LM_BACKEND_EVIDENCE_ENGINE_CONFIGURED;
    value.enabled_modalities = LITERT_LM_MODALITY_TEXT;
    if ((engine->flags & LITERT_LM_ENGINE_ENABLE_VISION) != 0) {
      value.enabled_modalities |= LITERT_LM_MODALITY_VISION;
    }
    if ((engine->flags & LITERT_LM_ENGINE_ENABLE_AUDIO) != 0) {
      value.enabled_modalities |= LITERT_LM_MODALITY_AUDIO;
    }
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
                std::min<size_t>(capacity, sizeof(value)));
    return LITERT_LM_STATUS_OK;
  });
}

LiteRtLm_Status ApiDestroyEngine(LiteRtLm_Engine engine,
                                 LiteRtLm_ErrorInfo* error) {
  return Boundary(LITERT_LM_PHASE_DESTROY, error, [&]() -> LiteRtLm_Status {
    if (!IsEngineValid(engine)) {
      return SetError(error, LITERT_LM_STATUS_INVALID_ARGUMENT,
                      LITERT_LM_PHASE_DESTROY, "Engine handle is invalid.");
    }
    std::unique_lock<std::mutex> lock(engine->operation_mutex);
    if (engine->conversation_count.load(std::memory_order_acquire) != 0 ||
        engine->request_count.load(std::memory_order_acquire) != 0) {
      return SetError(error, LITERT_LM_STATUS_BUSY,
                      LITERT_LM_PHASE_DESTROY,
                      "Destroy all Conversations and Requests first.");
    }
    engine->library->api.litert_lm_engine_delete(engine->native);
    engine->native = nullptr;
    engine->magic = 0;
    lock.unlock();
    delete engine;
    return LITERT_LM_STATUS_OK;
  });
}

LiteRtLm_Status ApiCreateConversation(
    LiteRtLm_Engine engine, const LiteRtLm_ConversationConfig* config,
    LiteRtLm_Conversation* out_conversation, LiteRtLm_ErrorInfo* error) {
  return Boundary(LITERT_LM_PHASE_CONVERSATION_CREATE, error,
                  [&]() -> LiteRtLm_Status {
    if (!IsEngineValid(engine) || out_conversation == nullptr) {
      return SetError(error, LITERT_LM_STATUS_INVALID_ARGUMENT,
                      LITERT_LM_PHASE_CONVERSATION_CREATE,
                      "Engine or Conversation output is invalid.");
    }
    *out_conversation = nullptr;
    ConversationDefinition definition;
    OrderedJson history;
    LiteRtLm_Status parsed = ParseConversationDefinition(
        config, &definition, &history, error);
    if (parsed != LITERT_LM_STATUS_OK) return parsed;

    std::lock_guard<std::mutex> operation_lock(engine->operation_mutex);
    if (engine->active_request.load(std::memory_order_acquire) != nullptr) {
      return SetError(error, LITERT_LM_STATUS_BUSY,
                      LITERT_LM_PHASE_CONVERSATION_CREATE,
                      "Create Conversations only while the Engine is idle.");
    }
    std::string native_error;
    LiteRtLmConversation* native = CreateNativeConversation(
        *engine, definition, nullptr, engine->applied_sampling, &native_error);
    if (native == nullptr) {
      return SetError(error, LITERT_LM_STATUS_MODEL_ERROR,
                      LITERT_LM_PHASE_CONVERSATION_CREATE, native_error);
    }
    auto conversation = std::make_unique<LiteRtLm_ConversationObject>();
    conversation->magic = kConversationMagic;
    conversation->owner = engine;
    conversation->native = native;
    conversation->native_sampling = engine->applied_sampling;
    conversation->definition = std::move(definition);
    conversation->history = std::move(history);
    LiteRtLm_ConversationObject* published = conversation.release();
    engine->resident_conversation = published;
    engine->conversation_count.fetch_add(1, std::memory_order_acq_rel);
    *out_conversation = published;
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
    LiteRtLm_EngineObject* engine = conversation->owner;
    std::lock_guard<std::mutex> operation_lock(engine->operation_mutex);
    std::unique_lock<std::mutex> conversation_lock(conversation->mutex);
    if (conversation->request_count.load(std::memory_order_acquire) != 0) {
      return SetError(error, LITERT_LM_STATUS_BUSY,
                      LITERT_LM_PHASE_CONTEXT_PREPARE,
                      "Destroy the retained Request before reset.");
    }
    std::string native_error;
    LiteRtLmConversation* replacement = CreateNativeConversation(
        *engine, conversation->definition, nullptr, engine->applied_sampling,
        &native_error);
    if (replacement == nullptr) {
      return SetError(error, LITERT_LM_STATUS_MODEL_ERROR,
                      LITERT_LM_PHASE_CONTEXT_PREPARE, native_error);
    }
    LiteRtLmConversation* previous = conversation->native;
    conversation->native = replacement;
    conversation->native_sampling = engine->applied_sampling;
    if ((conversation->definition.flags &
         LITERT_LM_CONVERSATION_ENABLE_CONSTRAINED_DECODING) != 0) {
      conversation->retired_native.push_back(previous);
    } else {
      engine->library->api.litert_lm_conversation_delete(previous);
    }
    conversation->history = BaseHistory(conversation->definition);
    conversation->generation.fetch_add(1, std::memory_order_acq_rel);
    conversation->needs_rebuild.store(false, std::memory_order_release);
    engine->resident_conversation = conversation;
    return LITERT_LM_STATUS_OK;
  });
}

LiteRtLm_Status ApiDestroyConversation(LiteRtLm_Conversation conversation,
                                       LiteRtLm_ErrorInfo* error) {
  return Boundary(LITERT_LM_PHASE_DESTROY, error, [&]() -> LiteRtLm_Status {
    if (!IsConversationValid(conversation)) {
      return SetError(error, LITERT_LM_STATUS_INVALID_ARGUMENT,
                      LITERT_LM_PHASE_DESTROY,
                      "Conversation handle is invalid.");
    }
    LiteRtLm_EngineObject* engine = conversation->owner;
    std::lock_guard<std::mutex> operation_lock(engine->operation_mutex);
    if (conversation->request_count.load(std::memory_order_acquire) != 0) {
      return SetError(error, LITERT_LM_STATUS_BUSY,
                      LITERT_LM_PHASE_DESTROY,
                      "Destroy all retained Requests first.");
    }
    engine->library->api.litert_lm_conversation_delete(conversation->native);
    for (LiteRtLmConversation* retired : conversation->retired_native) {
      engine->library->api.litert_lm_conversation_delete(retired);
    }
    conversation->retired_native.clear();
    if (engine->resident_conversation == conversation) {
      engine->resident_conversation = nullptr;
    }
    conversation->native = nullptr;
    conversation->magic = 0;
    engine->conversation_count.fetch_sub(1, std::memory_order_acq_rel);
    delete conversation;
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
                      "Source Conversation or output is invalid.");
    }
    *out_conversation = nullptr;
    LiteRtLm_EngineObject* engine = source->owner;
    std::lock_guard<std::mutex> operation_lock(engine->operation_mutex);
    std::lock_guard<std::mutex> source_lock(source->mutex);
    if (engine->request_count.load(std::memory_order_acquire) != 0) {
      return SetError(error, LITERT_LM_STATUS_BUSY,
                      LITERT_LM_PHASE_CONVERSATION_CREATE,
                      "Clone Conversations only while the Engine is idle.");
    }
    LiteRtLm_Status rebuilt = RebuildConversationLocked(*source, error);
    if (rebuilt != LITERT_LM_STATUS_OK) return rebuilt;
    LiteRtLmConversation* native =
        engine->library->api.litert_lm_conversation_clone(source->native);
    if (native == nullptr) {
      return SetError(error, LITERT_LM_STATUS_MODEL_ERROR,
                      LITERT_LM_PHASE_CONVERSATION_CREATE,
                      "Official litert_lm_conversation_clone returned null.");
    }
    auto clone = std::make_unique<LiteRtLm_ConversationObject>();
    clone->magic = kConversationMagic;
    clone->owner = engine;
    clone->native = native;
    clone->native_sampling = source->native_sampling;
    clone->definition = source->definition;
    clone->history = source->history;
    clone->generation.store(source->generation.load(std::memory_order_acquire),
                            std::memory_order_release);
    LiteRtLm_ConversationObject* published = clone.release();
    engine->resident_conversation = published;
    engine->conversation_count.fetch_add(1, std::memory_order_acq_rel);
    *out_conversation = published;
    return LITERT_LM_STATUS_OK;
  });
}

LiteRtLm_Status ApiGetConversationInfo(
    LiteRtLm_Conversation conversation, LiteRtLm_ConversationInfo* out_info,
    LiteRtLm_ErrorInfo* error) {
  return Boundary(LITERT_LM_PHASE_API, error, [&]() -> LiteRtLm_Status {
    if (!IsConversationValid(conversation) || out_info == nullptr ||
        out_info->struct_size <
            LITERT_LM_FIELD_END(LiteRtLm_ConversationInfo, tools_json)) {
      return SetError(error, LITERT_LM_STATUS_INVALID_ARGUMENT,
                      LITERT_LM_PHASE_API,
                      "Conversation or ConversationInfo buffer is invalid.");
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
    value.generation = conversation->generation.load(std::memory_order_acquire);
    {
      std::lock_guard<std::mutex> lock(conversation->mutex);
      value.history_message_count =
          static_cast<uint64_t>(conversation->history.size());
      const int token_count = conversation->owner->library->api
                                  .litert_lm_conversation_get_token_count(
                                      conversation->native);
      value.kv_token_count = token_count < 0 ? 0u
                                             : static_cast<uint64_t>(token_count);
    }
    value.tools_json = ViewOf(conversation->definition.tools_json);
    std::memcpy(out_info, &value,
                std::min<size_t>(capacity, sizeof(value)));
    return LITERT_LM_STATUS_OK;
  });
}

LiteRtLm_Status ApiCopyConversationHistoryJson(
    LiteRtLm_Conversation conversation, char* out_json,
    uint64_t* inout_size, LiteRtLm_ErrorInfo* error) {
  return Boundary(LITERT_LM_PHASE_API, error, [&]() -> LiteRtLm_Status {
    if (!IsConversationValid(conversation)) {
      return SetError(error, LITERT_LM_STATUS_INVALID_ARGUMENT,
                      LITERT_LM_PHASE_API,
                      "Conversation handle is invalid.");
    }
    std::string json;
    {
      std::lock_guard<std::mutex> lock(conversation->mutex);
      json = conversation->history.dump();
    }
    return CopyStringToCaller(json, out_json, inout_size,
                              LITERT_LM_PHASE_API, error);
  });
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
    if (engine_active || conversation_active) ReleaseActiveRequest(*request);
    if (conversation_count) {
      request->conversation->request_count.fetch_sub(1,
                                                     std::memory_order_acq_rel);
    }
    if (engine_count) {
      request->engine->request_count.fetch_sub(1, std::memory_order_acq_rel);
    }
    request->magic = 0;
    delete request;
  }
};

LiteRtLm_Status ApiSubmitChat(LiteRtLm_Conversation conversation,
                              const LiteRtLm_ChatRequest* chat,
                              LiteRtLm_Request* out_request,
                              LiteRtLm_ErrorInfo* error) {
  return Boundary(LITERT_LM_PHASE_CONTEXT_PREPARE, error,
                  [&]() -> LiteRtLm_Status {
    if (!IsConversationValid(conversation) || chat == nullptr ||
        out_request == nullptr ||
        chat->struct_size <
            LITERT_LM_FIELD_END(LiteRtLm_ChatRequest, user_data) ||
        !IsViewValid(chat->messages_json) || chat->messages_json.size == 0 ||
        !IsViewValid(chat->constraint) ||
        (chat->context_mode != LITERT_LM_CONTEXT_CONTINUE &&
         chat->context_mode != LITERT_LM_CONTEXT_REPLACE)) {
      return SetError(error, LITERT_LM_STATUS_INVALID_ARGUMENT,
                      LITERT_LM_PHASE_CONTEXT_PREPARE,
                      "Conversation, ChatRequest, or output is invalid.");
    }
    *out_request = nullptr;
    uint32_t chat_flags = 0;
    LiteRtLm_Sampling requested_sampling{};
    bool override_sampling = false;
    bool override_tools = false;
    std::string requested_tools_json;
    bool override_extra_context = false;
    std::string requested_extra_context_json;
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
        requested_sampling = ResolveSampling(chat->sampling);
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
        OrderedJson tools;
        if (!ParseArray(requested_tools_json, &tools)) {
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
        OrderedJson extra_context;
        if (!ParseObject(requested_extra_context_json, &extra_context)) {
          return SetError(
              error, LITERT_LM_STATUS_PARSE_ERROR,
              LITERT_LM_PHASE_CONTEXT_PREPARE,
              "ChatRequest.extra_context_json must be a JSON object.");
        }
      }
    }
    if (chat->constraint_kind != LITERT_LM_CONSTRAINT_NONE) {
      return SetError(
          error, LITERT_LM_STATUS_NOT_SUPPORTED,
          LITERT_LM_PHASE_CONTEXT_PREPARE,
          "The official C API exposes conversation constrained decoding "
          "and tool schemas, but not arbitrary per-request grammar injection. "
          "Use the full native API surface or a future upstream extension.");
    }

    OrderedJson input_messages;
    const std::string message_json = CopyView(chat->messages_json);
    if (!ParseMessages(message_json, &input_messages)) {
      return SetError(error, LITERT_LM_STATUS_PARSE_ERROR,
                      LITERT_LM_PHASE_CONTEXT_PREPARE,
                      "messages_json must be one message object or a non-empty "
                      "array of message objects.");
    }

    LiteRtLm_EngineObject* engine = conversation->owner;
    std::lock_guard<std::mutex> operation_lock(engine->operation_mutex);
    std::unique_lock<std::mutex> conversation_lock(conversation->mutex);
    if (engine->active_request.load(std::memory_order_acquire) != nullptr ||
        conversation->active_request.load(std::memory_order_acquire) != nullptr ||
        engine->request_count.load(std::memory_order_acquire) != 0) {
      return SetError(error, LITERT_LM_STATUS_BUSY,
                      LITERT_LM_PHASE_CONTEXT_PREPARE,
                      "This Engine retains another Request. Wait and destroy "
                      "it before submitting the next chat.");
    }
    const LiteRtLm_Sampling applied_sampling =
        override_sampling ? requested_sampling : engine->applied_sampling;
    if (!SamplingEquals(conversation->native_sampling, applied_sampling)) {
      conversation->native_sampling = applied_sampling;
      conversation->needs_rebuild.store(true, std::memory_order_release);
    }
    std::string previous_tools_json;
    bool tools_changed = false;
    if (override_tools &&
        conversation->definition.tools_json != requested_tools_json) {
      previous_tools_json = conversation->definition.tools_json;
      conversation->definition.tools_json = requested_tools_json;
      conversation->needs_rebuild.store(true, std::memory_order_release);
      tools_changed = true;
    }
    LiteRtLm_Status rebuilt = RebuildConversationLocked(*conversation, error);
    if (rebuilt != LITERT_LM_STATUS_OK) {
      if (tools_changed) {
        conversation->definition.tools_json = std::move(previous_tools_json);
      }
      return rebuilt;
    }

    std::string native_message = message_json;
    if (chat->context_mode == LITERT_LM_CONTEXT_REPLACE) {
      OrderedJson prefill = OrderedJson::array();
      for (size_t index = 0; index + 1 < input_messages.size(); ++index) {
        prefill.push_back(input_messages[index]);
      }
      native_message = input_messages.back().dump();
      std::string native_error;
      LiteRtLmConversation* replacement = CreateNativeConversation(
          *engine, conversation->definition, &prefill,
          conversation->native_sampling, &native_error);
      if (replacement == nullptr) {
        return SetError(error, LITERT_LM_STATUS_MODEL_ERROR,
                        LITERT_LM_PHASE_CONTEXT_PREPARE, native_error);
      }
      LiteRtLmConversation* previous = conversation->native;
      conversation->native = replacement;
      engine->library->api.litert_lm_conversation_delete(previous);
      conversation->history = std::move(prefill);
      conversation->generation.fetch_add(1, std::memory_order_acq_rel);
      engine->resident_conversation = conversation;
      input_messages = OrderedJson::array({input_messages.back()});
    }

    auto request = std::make_unique<LiteRtLm_RequestObject>();
    request->magic = kRequestMagic;
    request->engine = engine;
    request->conversation = conversation;
    request->callback = chat->callback;
    request->user_data = chat->user_data;
    request->input_messages = std::move(input_messages);
    request->conversation_generation =
        conversation->generation.load(std::memory_order_acquire);
    request->max_output_tokens =
        chat->max_output_tokens != 0
            ? chat->max_output_tokens
            : conversation->definition.default_max_output_tokens;
    request->applied_sampling = applied_sampling;
    request->extra_context_json =
        override_extra_context ? requested_extra_context_json
                               : conversation->definition.extra_context_json;
    request->start_time = std::chrono::steady_clock::now();
    request->state = LITERT_LM_REQUEST_RUNNING;

    SubmitReservation reservation;
    reservation.request = request.get();
    engine->request_count.fetch_add(1, std::memory_order_acq_rel);
    reservation.engine_count = true;
    conversation->request_count.fetch_add(1, std::memory_order_acq_rel);
    reservation.conversation_count = true;
    LiteRtLm_RequestObject* expected = nullptr;
    if (!engine->active_request.compare_exchange_strong(
            expected, request.get(), std::memory_order_acq_rel)) {
      return SetError(error, LITERT_LM_STATUS_BUSY,
                      LITERT_LM_PHASE_CONTEXT_PREPARE,
                      "The Engine became busy while reserving the Request.");
    }
    reservation.engine_active = true;
    expected = nullptr;
    if (!conversation->active_request.compare_exchange_strong(
            expected, request.get(), std::memory_order_acq_rel)) {
      return SetError(error, LITERT_LM_STATUS_BUSY,
                      LITERT_LM_PHASE_CONTEXT_PREPARE,
                      "The Conversation became busy while reserving the Request.");
    }
    reservation.conversation_active = true;
    CaptureBenchmarkBaseline(*request);

    // The official implementation may invoke its callback before the submit
    // function returns. Do not hold the Conversation metadata mutex across
    // that boundary because terminal processing commits canonical history.
    conversation_lock.unlock();

    NativeApi& api = engine->library->api;
    LiteRtLmConversationOptionalArgs* optional_args = nullptr;
    if (request->max_output_tokens != 0) {
      if (request->max_output_tokens >
          static_cast<uint32_t>((std::numeric_limits<int>::max)())) {
        return SetError(error, LITERT_LM_STATUS_INVALID_ARGUMENT,
                        LITERT_LM_PHASE_CONTEXT_PREPARE,
                        "max_output_tokens exceeds the native integer range.");
      }
      optional_args = api.litert_lm_conversation_optional_args_create();
      if (optional_args == nullptr) {
        return SetError(error, LITERT_LM_STATUS_RESOURCE_EXHAUSTED,
                        LITERT_LM_PHASE_CONTEXT_PREPARE,
                        "Unable to allocate native optional arguments.");
      }
      api.litert_lm_conversation_optional_args_set_max_output_tokens(
          optional_args, static_cast<int>(request->max_output_tokens));
    }
    const char* extra_context =
        request->extra_context_json.empty()
            ? nullptr
            : request->extra_context_json.c_str();
    int native_status = 0;
    if (api.UsesStreamChunks()) {
      const auto submit = reinterpret_cast<NativeConversationSendMessageStream>(
          api.litert_lm_conversation_send_message_stream);
      native_status = submit(conversation->native, native_message.c_str(),
                             extra_context, optional_args,
                             &NativeStreamChunkCallback, request.get());
    } else {
      native_status = api.litert_lm_conversation_send_message_stream(
          conversation->native, native_message.c_str(), extra_context,
          optional_args, &NativeStreamCallback, request.get());
    }
    if (optional_args != nullptr) {
      api.litert_lm_conversation_optional_args_delete(optional_args);
    }
    if (native_status != 0) {
      conversation->needs_rebuild.store(true, std::memory_order_release);
      return SetError(error, LITERT_LM_STATUS_MODEL_ERROR,
                      LITERT_LM_PHASE_CONTEXT_PREPARE,
                      "Official litert_lm_conversation_send_message_stream "
                      "rejected the request.",
                      native_status);
    }

    LiteRtLm_RequestObject* published = request.release();
    {
      std::lock_guard<std::mutex> request_lock(published->mutex);
      published->submit_in_progress = false;
    }
    reservation.committed = true;
    *out_request = published;
    MaybeReleaseActiveRequest(*published);
    published->condition.notify_all();
    return LITERT_LM_STATUS_OK;
  });
}

LiteRtLm_Status ApiGetRequestInfo(LiteRtLm_Request request,
                                  LiteRtLm_RequestInfo* out_info,
                                  LiteRtLm_ErrorInfo* error) {
  return Boundary(LITERT_LM_PHASE_API, error, [&]() -> LiteRtLm_Status {
    if (!IsRequestValid(request) || out_info == nullptr ||
        out_info->struct_size <
            LITERT_LM_FIELD_END(LiteRtLm_RequestInfo, error_message)) {
      return SetError(error, LITERT_LM_STATUS_INVALID_ARGUMENT,
                      LITERT_LM_PHASE_API,
                      "Request or RequestInfo buffer is invalid.");
    }
    const uint32_t capacity = out_info->struct_size;
    LiteRtLm_RequestInfo value{};
    value.struct_size = static_cast<uint32_t>(sizeof(value));
    {
      std::lock_guard<std::mutex> lock(request->mutex);
      value.state = request->state;
      value.finish_reason = request->finish_reason;
      value.status = request->status;
      value.backend = request->engine->configured_backend;
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
                std::min<size_t>(capacity, sizeof(value)));
    return LITERT_LM_STATUS_OK;
  });
}

LiteRtLm_Status ApiWaitRequest(LiteRtLm_Request request, uint32_t timeout_ms,
                               LiteRtLm_ErrorInfo* error) {
  return Boundary(LITERT_LM_PHASE_DECODE, error, [&]() -> LiteRtLm_Status {
    if (!IsRequestValid(request)) {
      return SetError(error, LITERT_LM_STATUS_INVALID_ARGUMENT,
                      LITERT_LM_PHASE_DECODE, "Request handle is invalid.");
    }
    if (g_callback_request == request) {
      return SetError(error, LITERT_LM_STATUS_INVALID_STATE,
                      LITERT_LM_PHASE_CALLBACK,
                      "wait_request cannot run inside the same Request callback.");
    }
    std::unique_lock<std::mutex> lock(request->mutex);
    auto complete = [&]() {
      return request->terminal && !request->submit_in_progress &&
             !request->callback_in_progress &&
             request->native_callbacks_in_flight.load(
                 std::memory_order_acquire) == 0;
    };
    bool ready = false;
    if (timeout_ms == LITERT_LM_WAIT_INFINITE) {
      request->condition.wait(lock, complete);
      ready = true;
    } else {
      ready = request->condition.wait_for(
          lock, std::chrono::milliseconds(timeout_ms), complete);
    }
    lock.unlock();
    if (!ready) {
      return SetError(error, LITERT_LM_STATUS_TIMEOUT,
                      LITERT_LM_PHASE_DECODE,
                      "Timed out waiting for native completion and callback "
                      "quiescence.",
                      0, true);
    }
    RefreshMetrics(*request);
    return LITERT_LM_STATUS_OK;
  });
}

LiteRtLm_Status ApiCancelRequest(LiteRtLm_Request request,
                                 LiteRtLm_ErrorInfo* error) {
  return Boundary(LITERT_LM_PHASE_CANCEL, error, [&]() -> LiteRtLm_Status {
    if (!IsRequestValid(request)) {
      return SetError(error, LITERT_LM_STATUS_INVALID_ARGUMENT,
                      LITERT_LM_PHASE_CANCEL, "Request handle is invalid.");
    }
    {
      std::lock_guard<std::mutex> lock(request->mutex);
      if (request->terminal) return LITERT_LM_STATUS_OK;
      request->cancel_requested.store(true, std::memory_order_release);
    }
    request->conversation->needs_rebuild.store(true,
                                                std::memory_order_release);
    request->engine->library->api.litert_lm_conversation_cancel_process(
        request->conversation->native);
    return LITERT_LM_STATUS_OK;
  });
}

LiteRtLm_Status ApiDestroyRequest(LiteRtLm_Request request,
                                  LiteRtLm_ErrorInfo* error) {
  return Boundary(LITERT_LM_PHASE_DESTROY, error, [&]() -> LiteRtLm_Status {
    if (!IsRequestValid(request)) {
      return SetError(error, LITERT_LM_STATUS_INVALID_ARGUMENT,
                      LITERT_LM_PHASE_DESTROY, "Request handle is invalid.");
    }
    {
      std::lock_guard<std::mutex> lock(request->mutex);
      if (!request->terminal || request->submit_in_progress ||
          request->callback_in_progress ||
          request->native_callbacks_in_flight.load(
              std::memory_order_acquire) != 0) {
        return SetError(error, LITERT_LM_STATUS_BUSY,
                        LITERT_LM_PHASE_DESTROY,
                        "Wait for terminal callback quiescence first.");
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
      return SetError(error, LITERT_LM_STATUS_BUSY, LITERT_LM_PHASE_API,
                      "Tokenize only while the Engine is idle.");
    }
    const std::string input = CopyView(text);
    NativeApi& api = engine->library->api;
    LiteRtLmTokenizeResult* result =
        api.litert_lm_engine_tokenize(engine->native, input.c_str());
    if (result == nullptr) {
      return SetError(error, LITERT_LM_STATUS_MODEL_ERROR,
                      LITERT_LM_PHASE_API,
                      "Official litert_lm_engine_tokenize returned null.");
    }
    const size_t count = api.litert_lm_tokenize_result_get_num_tokens(result);
    const int* tokens = api.litert_lm_tokenize_result_get_tokens(result);
    const uint64_t capacity = *inout_count;
    *inout_count = static_cast<uint64_t>(count);
    if (out_tokens != nullptr && capacity < count) {
      api.litert_lm_tokenize_result_delete(result);
      return SetError(error, LITERT_LM_STATUS_RESOURCE_EXHAUSTED,
                      LITERT_LM_PHASE_API,
                      "The caller-provided token buffer is too small.");
    }
    if (out_tokens != nullptr && count != 0) {
      if (tokens == nullptr) {
        api.litert_lm_tokenize_result_delete(result);
        return SetError(error, LITERT_LM_STATUS_INTERNAL_ERROR,
                        LITERT_LM_PHASE_API,
                        "Native tokenizer returned a null token array.");
      }
      for (size_t index = 0; index < count; ++index) {
        out_tokens[index] = static_cast<int32_t>(tokens[index]);
      }
    }
    api.litert_lm_tokenize_result_delete(result);
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
      return SetError(error, LITERT_LM_STATUS_BUSY, LITERT_LM_PHASE_API,
                      "Detokenize only while the Engine is idle.");
    }
    NativeApi& api = engine->library->api;
    LiteRtLmDetokenizeResult* result = api.litert_lm_engine_detokenize(
        engine->native, reinterpret_cast<const int*>(tokens),
        static_cast<size_t>(token_count));
    if (result == nullptr) {
      return SetError(error, LITERT_LM_STATUS_MODEL_ERROR,
                      LITERT_LM_PHASE_API,
                      "Official litert_lm_engine_detokenize returned null.");
    }
    const char* native_text =
        api.litert_lm_detokenize_result_get_string(result);
    const std::string value = native_text == nullptr ? "" : native_text;
    api.litert_lm_detokenize_result_delete(result);
    return CopyStringToCaller(value, out_text, inout_size,
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
    std::string loader_error;
    std::shared_ptr<NativeLibrary> library = GetNativeLibrary(&loader_error);
    if (!library) {
      return SetError(error, LITERT_LM_STATUS_BACKEND_ERROR,
                      LITERT_LM_PHASE_API, loader_error);
    }
    library->api.litert_lm_set_min_log_level(minimum_level);
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
      LITERT_LM_CAP_REQUEST_METRICS | LITERT_LM_CAP_CONTEXT_REPLACE |
      LITERT_LM_CAP_VISION | LITERT_LM_CAP_AUDIO |
      LITERT_LM_CAP_MULTI_CONVERSATION |
      LITERT_LM_CAP_CONVERSATION_CLONE |
      LITERT_LM_CAP_CONVERSATION_HISTORY | LITERT_LM_CAP_TOKENIZER |
      LITERT_LM_CAP_INITIAL_MESSAGES | LITERT_LM_CAP_THINKING |
      LITERT_LM_CAP_TOOL_CALL_STREAMING |
      LITERT_LM_CAP_PER_REQUEST_SAMPLING |
      LITERT_LM_CAP_PER_REQUEST_TOOLS |
      LITERT_LM_CAP_PER_REQUEST_EXTRA_CONTEXT |
      LITERT_LM_CAP_SPECULATIVE_DECODING;
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
  api.copy_conversation_history_json = &ApiCopyConversationHistoryJson;
  api.tokenize = &ApiTokenize;
  api.detokenize = &ApiDetokenize;
  api.set_log_level = &ApiSetLogLevel;
  std::memcpy(out_api, &api,
              std::min<size_t>(out_api_size, sizeof(api)));
  return LITERT_LM_STATUS_OK;
}
