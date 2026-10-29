// Copyright (c) 2025-2026 Winyunq. All rights reserved.
//
// Stable C ABI for the LiteRT-LM shared library. LiteRtLm_GetApi is the sole
// entry point for the new ABI. C++ conveniences live in litert_lm.hpp and are
// compiled into the caller, so no STL or C++ ABI crosses the boundary.
#ifndef LITERT_LM_RUNTIME_ENGINE_LITERT_LM_API_H_
#define LITERT_LM_RUNTIME_ENGINE_LITERT_LM_API_H_

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define LITERT_LM_CALL __cdecl
#if defined(LITERT_LM_IMPLEMENTATION)
#define LITERT_LM_EXPORT
#else
#define LITERT_LM_EXPORT __declspec(dllimport)
#endif
#else
#define LITERT_LM_CALL
#define LITERT_LM_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define LITERT_LM_ABI_MAJOR 1u
#define LITERT_LM_ABI_MINOR 4u
#define LITERT_LM_WAIT_INFINITE UINT32_MAX

typedef int32_t LiteRtLm_Status;
enum {
  LITERT_LM_STATUS_OK = 0,
  LITERT_LM_STATUS_INVALID_ARGUMENT = 1,
  LITERT_LM_STATUS_INVALID_STATE = 2,
  LITERT_LM_STATUS_NOT_SUPPORTED = 3,
  LITERT_LM_STATUS_BUSY = 4,
  LITERT_LM_STATUS_NOT_READY = 5,
  LITERT_LM_STATUS_TIMEOUT = 6,
  LITERT_LM_STATUS_CANCELLED = 7,
  LITERT_LM_STATUS_IO_ERROR = 8,
  LITERT_LM_STATUS_MODEL_ERROR = 9,
  LITERT_LM_STATUS_BACKEND_ERROR = 10,
  LITERT_LM_STATUS_RESOURCE_EXHAUSTED = 11,
  LITERT_LM_STATUS_INTERNAL_ERROR = 12,
  LITERT_LM_STATUS_ABI_MISMATCH = 13,
  LITERT_LM_STATUS_PARSE_ERROR = 14
};

typedef uint32_t LiteRtLm_ErrorPhase;
enum {
  LITERT_LM_PHASE_NONE = 0,
  LITERT_LM_PHASE_API = 1,
  LITERT_LM_PHASE_ENGINE_CREATE = 2,
  LITERT_LM_PHASE_CONVERSATION_CREATE = 3,
  LITERT_LM_PHASE_CONTEXT_PREPARE = 4,
  LITERT_LM_PHASE_PREFILL = 5,
  LITERT_LM_PHASE_DECODE = 6,
  LITERT_LM_PHASE_CALLBACK = 7,
  LITERT_LM_PHASE_CANCEL = 8,
  LITERT_LM_PHASE_DESTROY = 9
};

typedef uint32_t LiteRtLm_Backend;
enum {
  // Backend selection is explicit and this wrapper never silently changes it.
  // ENGINE_CONFIGURED evidence is not per-operation delegation telemetry.
  LITERT_LM_BACKEND_CPU = 1,
  LITERT_LM_BACKEND_GPU = 2
};

enum {
  LITERT_LM_BACKEND_MASK_CPU = 1u << 0,
  LITERT_LM_BACKEND_MASK_GPU = 1u << 1
};

typedef uint32_t LiteRtLm_BackendEvidence;
enum {
  LITERT_LM_BACKEND_EVIDENCE_NONE = 0,
  // The live Engine was created with this resolved backend configuration.
  LITERT_LM_BACKEND_EVIDENCE_ENGINE_CONFIGURED = 1,
  // Reserved for runtimes which can prove every model operation is delegated.
  LITERT_LM_BACKEND_EVIDENCE_FULLY_ACCELERATED = 2
};

typedef uint32_t LiteRtLm_SeedMode;
enum {
  LITERT_LM_SEED_RANDOM = 0,
  LITERT_LM_SEED_FIXED = 1
};

typedef uint32_t LiteRtLm_ContextMode;
enum {
  // Append the supplied message object/array to the existing context.
  LITERT_LM_CONTEXT_CONTINUE = 0,
  // Release the old native session, rebuild the Conversation, then use the
  // supplied message object/array as the complete new context. This
  // deliberately avoids a double-KV memory peak. If rebuilding fails, the old
  // context is not retained.
  LITERT_LM_CONTEXT_REPLACE = 1
};

typedef uint32_t LiteRtLm_ConstraintKind;
enum {
  LITERT_LM_CONSTRAINT_NONE = 0,
  LITERT_LM_CONSTRAINT_REGEX = 1,
  LITERT_LM_CONSTRAINT_JSON_SCHEMA = 2,
  LITERT_LM_CONSTRAINT_LARK = 3
};

typedef uint32_t LiteRtLm_RequestState;
enum {
  LITERT_LM_REQUEST_CREATED = 0,
  LITERT_LM_REQUEST_RUNNING = 1,
  LITERT_LM_REQUEST_SUCCEEDED = 2,
  LITERT_LM_REQUEST_FAILED = 3,
  LITERT_LM_REQUEST_CANCELLED = 4
};

typedef uint32_t LiteRtLm_FinishReason;
enum {
  LITERT_LM_FINISH_NONE = 0,
  // The native Conversation completed. The current core does not preserve a
  // reliable distinction between a stop token and a max-token terminal state.
  LITERT_LM_FINISH_COMPLETE = 1,
  LITERT_LM_FINISH_TOOL_CALL = 2,
  LITERT_LM_FINISH_CANCELLED = 3,
  LITERT_LM_FINISH_ERROR = 4
};

typedef uint32_t LiteRtLm_EventKind;
enum {
  LITERT_LM_EVENT_TEXT_DELTA = 1,
  LITERT_LM_EVENT_COMPLETE = 2,
  // response_json contains the authoritative native delta for both events.
  LITERT_LM_EVENT_TOOL_CALL_DELTA = 3,
  LITERT_LM_EVENT_CHANNEL_DELTA = 4
};

enum {
  // Controls native benchmark collection. Wrapper wall-clock total/TTFT and
  // context-token metrics may still be available when this flag is disabled.
  LITERT_LM_ENGINE_ENABLE_METRICS = 1u << 0,
  LITERT_LM_ENGINE_ENABLE_VISION = 1u << 1,
  LITERT_LM_ENGINE_ENABLE_AUDIO = 1u << 2,
  LITERT_LM_ENGINE_DISABLE_PARALLEL_FILE_LOADING = 1u << 3,
  // Opt-in MTP/speculative decoding. Callers should first check
  // LITERT_LM_CAP_SPECULATIVE_DECODING; unsupported models may reject it.
  LITERT_LM_ENGINE_ENABLE_SPECULATIVE_DECODING = 1u << 4
};

enum {
  LITERT_LM_CONVERSATION_ENABLE_CONSTRAINED_DECODING = 1u << 0,
  LITERT_LM_CONVERSATION_PREFILL_PREFACE = 1u << 1,
  LITERT_LM_CONVERSATION_FILTER_CHANNEL_CONTENT_FROM_KV = 1u << 2,
  LITERT_LM_CONVERSATION_RETURN_ERROR_ON_PARSE_FAILURE = 1u << 3,
  LITERT_LM_CONVERSATION_RETURN_ERROR_ON_MAX_TOKENS = 1u << 4,
  LITERT_LM_CONVERSATION_ENABLE_THINKING = 1u << 5,
  LITERT_LM_CONVERSATION_STREAM_TOOL_CALLS = 1u << 6,
  LITERT_LM_CONVERSATION_DISABLE_PROMPT_TEMPLATE = 1u << 7
};

enum {
  // Rebuilds the logical Conversation with ChatRequest.sampling before this
  // request. Canonical history is replayed; no model or Engine is recreated.
  // Without this flag the Engine sampling configuration remains authoritative.
  LITERT_LM_CHAT_OVERRIDE_SAMPLING = 1u << 0,
  // ABI 1.3+: rebuilds only the target logical Conversation with this
  // request's tool declarations before inference. Canonical history and the
  // shared Engine/model are preserved. This lets applications expose exactly
  // the tool and argument schema that are legal for the current phase.
  LITERT_LM_CHAT_OVERRIDE_TOOLS = 1u << 1,
  // ABI 1.4+: supplies turn-specific prompt-template context without
  // rebuilding the Conversation. This is the native path for values such as
  // {"enable_thinking":false}.
  LITERT_LM_CHAT_OVERRIDE_EXTRA_CONTEXT = 1u << 2
};

enum {
  LITERT_LM_MODALITY_TEXT = 1u << 0,
  LITERT_LM_MODALITY_VISION = 1u << 1,
  LITERT_LM_MODALITY_AUDIO = 1u << 2
};

enum {
  LITERT_LM_CAP_TEXT = 1ull << 0,
  LITERT_LM_CAP_STREAMING = 1ull << 1,
  LITERT_LM_CAP_TOOLS = 1ull << 2,
  LITERT_LM_CAP_CONSTRAINTS = 1ull << 3,
  LITERT_LM_CAP_REQUEST_METRICS = 1ull << 4,
  LITERT_LM_CAP_CONTEXT_REPLACE = 1ull << 5,
  LITERT_LM_CAP_VISION = 1ull << 6,
  LITERT_LM_CAP_AUDIO = 1ull << 7,
  LITERT_LM_CAP_MULTI_CONVERSATION = 1ull << 8,
  LITERT_LM_CAP_CONVERSATION_CLONE = 1ull << 9,
  LITERT_LM_CAP_CONVERSATION_HISTORY = 1ull << 10,
  LITERT_LM_CAP_TOKENIZER = 1ull << 11,
  LITERT_LM_CAP_INITIAL_MESSAGES = 1ull << 12,
  LITERT_LM_CAP_THINKING = 1ull << 13,
  LITERT_LM_CAP_TOOL_CALL_STREAMING = 1ull << 14,
  LITERT_LM_CAP_PER_REQUEST_SAMPLING = 1ull << 15,
  LITERT_LM_CAP_SPECULATIVE_DECODING = 1ull << 16,
  LITERT_LM_CAP_PER_REQUEST_TOOLS = 1ull << 17,
  LITERT_LM_CAP_PER_REQUEST_EXTRA_CONTEXT = 1ull << 18
};

enum {
  LITERT_LM_METRIC_INPUT_TOKENS = 1ull << 0,
  LITERT_LM_METRIC_OUTPUT_TOKENS = 1ull << 1,
  LITERT_LM_METRIC_CONTEXT_TOKENS = 1ull << 2,
  LITERT_LM_METRIC_PREFILL_TIME = 1ull << 3,
  LITERT_LM_METRIC_DECODE_TIME = 1ull << 4,
  LITERT_LM_METRIC_TTFT = 1ull << 5,
  LITERT_LM_METRIC_TOTAL_TIME = 1ull << 6,
  LITERT_LM_METRIC_OUTPUT_RATE = 1ull << 7
};

typedef struct LiteRtLm_EngineObject* LiteRtLm_Engine;
typedef struct LiteRtLm_ConversationObject* LiteRtLm_Conversation;
typedef struct LiteRtLm_RequestObject* LiteRtLm_Request;

typedef struct LiteRtLm_StringView {
  const char* data;
  uint64_t size;
} LiteRtLm_StringView;

typedef struct LiteRtLm_Sampling {
  // This ABI value has a permanently fixed size. Future fields must consume
  // reserved slots because Sampling is embedded by value in other structures.
  uint32_t struct_size;
  float temperature;
  float top_p;
  int32_t top_k;
  LiteRtLm_SeedMode seed_mode;
  int32_t seed;
  uint32_t reserved[3];
} LiteRtLm_Sampling;

typedef struct LiteRtLm_ErrorInfo {
  // ErrorInfo also has a permanently fixed size. Set struct_size to sizeof
  // before every call. A smaller buffer is ignored rather than overwritten.
  uint32_t struct_size;
  LiteRtLm_Status code;
  LiteRtLm_ErrorPhase phase;
  int32_t native_code;
  uint32_t retryable;
  uint32_t reserved[3];
  char message[512];
} LiteRtLm_ErrorInfo;

typedef struct LiteRtLm_EngineConfig {
  uint32_t struct_size;
  LiteRtLm_StringView model_path;
  LiteRtLm_Backend backend;
  uint32_t flags;
  // Zero uses the model/runtime default. EngineInfo reports this configured
  // value; the current core does not expose the resolved model limit.
  uint32_t max_context_tokens;
  // SEED_RANDOM is resolved exactly once while creating the Engine. The
  // resulting fixed seed is reported by EngineInfo and RequestInfo.
  LiteRtLm_Sampling sampling;
  // ABI 1.0 tail. Kept in place so existing callers retain their layout.
  uint64_t reserved[4];

  // ABI 1.1 extensions. Zero/empty preserves the native runtime default.
  uint32_t num_threads;
  int32_t prefill_chunk_size;
  LiteRtLm_StringView cache_dir;
  LiteRtLm_StringView dispatch_library_dir;
  uint64_t extension_reserved[4];
} LiteRtLm_EngineConfig;

typedef struct LiteRtLm_ConversationConfig {
  uint32_t struct_size;
  // JSON array of tool declarations. Empty disables tools.
  LiteRtLm_StringView tools_json;
  // ABI 1.0 tail. Kept in place so existing callers retain their layout.
  uint64_t reserved[4];

  // ABI 1.1 extensions. system_message may be plain UTF-8 or JSON content.
  // initial_messages_json must be an array; extra_context_json an object.
  uint32_t flags;
  uint32_t default_max_output_tokens;
  LiteRtLm_StringView system_message;
  LiteRtLm_StringView initial_messages_json;
  LiteRtLm_StringView extra_context_json;
  LiteRtLm_StringView tool_call_channel;
  // Reserved now so LoRA can be implemented without another config layout.
  LiteRtLm_StringView lora_path;
  LiteRtLm_StringView audio_lora_path;
  uint64_t extension_reserved[4];
} LiteRtLm_ConversationConfig;

typedef struct LiteRtLm_EngineInfo {
  uint32_t struct_size;
  LiteRtLm_Backend configured_backend;
  LiteRtLm_Backend resolved_backend;
  // Configured sampler backend. The current core does not expose proof of a
  // possible sampler-only fallback, so this is not presented as telemetry.
  LiteRtLm_Backend configured_sampler_backend;
  LiteRtLm_BackendEvidence backend_evidence;
  uint32_t enabled_modalities;
  uint32_t max_context_tokens;
  uint32_t conversation_count;
  uint32_t active_request_count;
  // Terminal Requests also retain native diagnostic ownership until destroyed.
  uint32_t retained_request_count;
  uint32_t reserved_count;
  LiteRtLm_Sampling applied_sampling;
  // Diagnostic description owned by Engine; valid until DestroyEngine.
  LiteRtLm_StringView backend_description;
  uint64_t reserved[4];
} LiteRtLm_EngineInfo;

typedef struct LiteRtLm_ConversationInfo {
  uint32_t struct_size;
  uint32_t flags;
  uint32_t active_request_count;
  uint32_t default_max_output_tokens;
  uint64_t generation;
  uint64_t history_message_count;
  uint64_t kv_token_count;
  // Owned by the Conversation and valid until it is destroyed.
  LiteRtLm_StringView tools_json;
  uint64_t reserved[4];
} LiteRtLm_ConversationInfo;

typedef struct LiteRtLm_Event {
  uint32_t struct_size;
  LiteRtLm_EventKind kind;
  LiteRtLm_Status status;
  LiteRtLm_Request request;
  uint64_t sequence;
  // All views are valid only for the duration of the callback.
  LiteRtLm_StringView text_delta;
  LiteRtLm_StringView response_json;
  LiteRtLm_StringView error_message;
  uint64_t reserved[2];
} LiteRtLm_Event;

typedef void(LITERT_LM_CALL* LiteRtLm_EventCallback)(
    const LiteRtLm_Event* event, void* user_data);

typedef struct LiteRtLm_ChatRequest {
  uint32_t struct_size;
  // One JSON message object or a non-empty message array. Copied before return.
  LiteRtLm_StringView messages_json;
  LiteRtLm_ContextMode context_mode;
  // Zero uses the model/runtime default.
  uint32_t max_output_tokens;
  LiteRtLm_ConstraintKind constraint_kind;
  LiteRtLm_StringView constraint;
  LiteRtLm_EventCallback callback;
  void* user_data;
  // ABI 1.0 tail. Kept in place so existing callers retain their layout.
  uint64_t reserved[4];

  // ABI 1.2 extension. Set LITERT_LM_CHAT_OVERRIDE_SAMPLING to apply sampling
  // to this request. The bridge rebuilds only this logical Conversation and
  // replays its canonical history, preserving independent Agent memory.
  uint32_t flags;
  uint32_t extension_reserved0;
  LiteRtLm_Sampling sampling;
  uint64_t extension_reserved[4];

  // ABI 1.3 extension. Valid only with LITERT_LM_CHAT_OVERRIDE_TOOLS. The
  // value is a JSON array of tool declarations; an empty array disables tools
  // for this request. The override remains active for this logical
  // Conversation until another request replaces it.
  LiteRtLm_StringView tools_json;
  uint64_t tools_extension_reserved[3];

  // ABI 1.4 extension. Valid only with
  // LITERT_LM_CHAT_OVERRIDE_EXTRA_CONTEXT and must be a JSON object. The
  // value applies only to this turn and does not mutate Conversation config.
  LiteRtLm_StringView extra_context_json;
  uint64_t extra_context_extension_reserved[3];
} LiteRtLm_ChatRequest;

typedef struct LiteRtLm_RequestInfo {
  uint32_t struct_size;
  LiteRtLm_RequestState state;
  LiteRtLm_FinishReason finish_reason;
  LiteRtLm_Status status;
  LiteRtLm_Backend backend;
  // Terminal asynchronous error diagnostics. error_phase may be the nearest
  // observable wrapper phase when the native core cannot distinguish prefill
  // from decode precisely.
  LiteRtLm_ErrorPhase error_phase;
  int32_t native_code;
  uint32_t retryable;
  uint32_t error_reserved;
  uint64_t valid_metrics;
  uint64_t input_tokens;
  uint64_t output_tokens;
  uint64_t context_tokens;
  uint64_t prefill_time_us;
  uint64_t decode_time_us;
  uint64_t time_to_first_token_us;
  uint64_t total_time_us;
  double output_tokens_per_second;
  uint64_t conversation_generation;
  uint64_t callback_count;
  LiteRtLm_Sampling applied_sampling;
  // Terminal strings are immutable and valid until DestroyRequest.
  LiteRtLm_StringView text;
  LiteRtLm_StringView response_json;
  LiteRtLm_StringView error_message;
  uint64_t reserved[4];
} LiteRtLm_RequestInfo;

typedef struct LiteRtLm_Api {
  uint32_t struct_size;
  uint32_t abi_major;
  uint32_t abi_minor;
  // Library build support, not a probe of the current device.
  uint32_t compiled_backends;
  // Wrapper API surface, not a promise that every model supports each feature.
  uint64_t capabilities;
  LiteRtLm_StringView build_id;

  LiteRtLm_Status(LITERT_LM_CALL* create_engine)(
      const LiteRtLm_EngineConfig* config, LiteRtLm_Engine* out_engine,
      LiteRtLm_ErrorInfo* out_error);
  LiteRtLm_Status(LITERT_LM_CALL* get_engine_info)(
      LiteRtLm_Engine engine, LiteRtLm_EngineInfo* out_info,
      LiteRtLm_ErrorInfo* out_error);
  LiteRtLm_Status(LITERT_LM_CALL* destroy_engine)(
      LiteRtLm_Engine engine, LiteRtLm_ErrorInfo* out_error);

  // Multiple Conversations may coexist and own independent native history/KV.
  // This wrapper intentionally serializes inference per Engine.
  LiteRtLm_Status(LITERT_LM_CALL* create_conversation)(
      LiteRtLm_Engine engine, const LiteRtLm_ConversationConfig* config,
      LiteRtLm_Conversation* out_conversation, LiteRtLm_ErrorInfo* out_error);
  LiteRtLm_Status(LITERT_LM_CALL* reset_conversation)(
      LiteRtLm_Conversation conversation, LiteRtLm_ErrorInfo* out_error);
  LiteRtLm_Status(LITERT_LM_CALL* destroy_conversation)(
      LiteRtLm_Conversation conversation, LiteRtLm_ErrorInfo* out_error);

  LiteRtLm_Status(LITERT_LM_CALL* submit_chat)(
      LiteRtLm_Conversation conversation, const LiteRtLm_ChatRequest* chat,
      LiteRtLm_Request* out_request, LiteRtLm_ErrorInfo* out_error);
  LiteRtLm_Status(LITERT_LM_CALL* get_request_info)(
      LiteRtLm_Request request, LiteRtLm_RequestInfo* out_info,
      LiteRtLm_ErrorInfo* out_error);
  // OK means the request is terminal and every callback has returned. Calling
  // this from that same Request's callback returns INVALID_STATE.
  LiteRtLm_Status(LITERT_LM_CALL* wait_request)(
      LiteRtLm_Request request, uint32_t timeout_ms,
      LiteRtLm_ErrorInfo* out_error);
  // Idempotently requests cancellation and returns immediately. Call
  // wait_request afterwards to establish callback quiescence.
  LiteRtLm_Status(LITERT_LM_CALL* cancel_request)(
      LiteRtLm_Request request, LiteRtLm_ErrorInfo* out_error);
  LiteRtLm_Status(LITERT_LM_CALL* destroy_request)(
      LiteRtLm_Request request, LiteRtLm_ErrorInfo* out_error);

  // ABI 1.1 additions consume six of the eight ABI 1.0 pointer slots.
  LiteRtLm_Status(LITERT_LM_CALL* clone_conversation)(
      LiteRtLm_Conversation source, LiteRtLm_Conversation* out_conversation,
      LiteRtLm_ErrorInfo* out_error);
  LiteRtLm_Status(LITERT_LM_CALL* get_conversation_info)(
      LiteRtLm_Conversation conversation, LiteRtLm_ConversationInfo* out_info,
      LiteRtLm_ErrorInfo* out_error);
  // Buffer size includes the trailing NUL. Passing null out_json queries it.
  LiteRtLm_Status(LITERT_LM_CALL* copy_conversation_history_json)(
      LiteRtLm_Conversation conversation, char* out_json,
      uint64_t* inout_size, LiteRtLm_ErrorInfo* out_error);
  // Passing null out_tokens queries the required token count.
  LiteRtLm_Status(LITERT_LM_CALL* tokenize)(
      LiteRtLm_Engine engine, LiteRtLm_StringView text, int32_t* out_tokens,
      uint64_t* inout_count, LiteRtLm_ErrorInfo* out_error);
  // Buffer size includes the trailing NUL. Passing null out_text queries it.
  LiteRtLm_Status(LITERT_LM_CALL* detokenize)(
      LiteRtLm_Engine engine, const int32_t* tokens, uint64_t token_count,
      char* out_text, uint64_t* inout_size, LiteRtLm_ErrorInfo* out_error);
  LiteRtLm_Status(LITERT_LM_CALL* set_log_level)(
      int32_t minimum_level, LiteRtLm_ErrorInfo* out_error);

  void* reserved[2];
} LiteRtLm_Api;

// The sole stable export for the new ABI. out_api_size is the caller's buffer
// size, allowing later minor versions to append function pointers safely.
//
// Threading contract:
// - Calls on live handles may originate from different threads.
// - Destroying a handle concurrently with any other call using that handle (or
//   one of its children) is unsupported; the caller must serialize destruction.
// - Stream callbacks for one Request are serialized. In a callback, immediately
//   copy views and return. Only get_request_info and cancel_request are allowed
//   on the same Engine; do not Wait, Submit, create/reset/destroy a Conversation,
//   destroy the Request, block, or throw across the ABI boundary.
// - A single Engine permits many independent Conversations but one retained
//   Request at a time. Copy its result and DestroyRequest before submitting the
//   next serial inference.
LITERT_LM_EXPORT LiteRtLm_Status LITERT_LM_CALL LiteRtLm_GetApi(
    uint32_t requested_major, uint32_t out_api_size, LiteRtLm_Api* out_api);

// Header-only defaults; these do not add shared-library exports.
static inline LiteRtLm_StringView LiteRtLm_String(const char* data,
                                                   uint64_t size) {
  LiteRtLm_StringView value;
  value.data = data;
  value.size = size;
  return value;
}

static inline LiteRtLm_Sampling LiteRtLm_DefaultSampling(void) {
  LiteRtLm_Sampling value = {0};
  value.struct_size = (uint32_t)sizeof(LiteRtLm_Sampling);
  value.temperature = 1.0f;
  value.top_p = 0.95f;
  value.top_k = 64;
  value.seed_mode = LITERT_LM_SEED_RANDOM;
  return value;
}

static inline LiteRtLm_EngineConfig LiteRtLm_DefaultEngineConfig(void) {
  LiteRtLm_EngineConfig value = {0};
  value.struct_size = (uint32_t)sizeof(LiteRtLm_EngineConfig);
  value.backend = LITERT_LM_BACKEND_GPU;
  value.flags = LITERT_LM_ENGINE_ENABLE_METRICS;
  value.sampling = LiteRtLm_DefaultSampling();
  return value;
}

static inline LiteRtLm_ConversationConfig
LiteRtLm_DefaultConversationConfig(void) {
  LiteRtLm_ConversationConfig value = {0};
  value.struct_size = (uint32_t)sizeof(LiteRtLm_ConversationConfig);
  value.flags = LITERT_LM_CONVERSATION_RETURN_ERROR_ON_PARSE_FAILURE;
  return value;
}

static inline LiteRtLm_ChatRequest LiteRtLm_DefaultChatRequest(void) {
  LiteRtLm_ChatRequest value = {0};
  value.struct_size = (uint32_t)sizeof(LiteRtLm_ChatRequest);
  value.context_mode = LITERT_LM_CONTEXT_CONTINUE;
  value.constraint_kind = LITERT_LM_CONSTRAINT_NONE;
  value.sampling = LiteRtLm_DefaultSampling();
  return value;
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LITERT_LM_RUNTIME_ENGINE_LITERT_LM_API_H_
