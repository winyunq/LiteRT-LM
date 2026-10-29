// Copyright (c) 2025-2026 Winyunq. All rights reserved.
//
// Header-only C++ conveniences for litert_lm_api.h. API failures are returned
// as Status/Result values. Normal C++ allocation behavior remains controlled by
// the caller's runtime. No class or STL type is exported by the DLL/SO.
#ifndef LITERT_LM_RUNTIME_ENGINE_LITERT_LM_HPP_
#define LITERT_LM_RUNTIME_ENGINE_LITERT_LM_HPP_

#include "runtime/engine/litert_lm_api.h"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace litertlm {

class LiteRtLm;

enum class Backend : uint32_t {
  Cpu = LITERT_LM_BACKEND_CPU,
  Gpu = LITERT_LM_BACKEND_GPU,
};

enum class ContextMode : uint32_t {
  Continue = LITERT_LM_CONTEXT_CONTINUE,
  Replace = LITERT_LM_CONTEXT_REPLACE,
};

enum class ConstraintKind : uint32_t {
  None = LITERT_LM_CONSTRAINT_NONE,
  Regex = LITERT_LM_CONSTRAINT_REGEX,
  JsonSchema = LITERT_LM_CONSTRAINT_JSON_SCHEMA,
  Lark = LITERT_LM_CONSTRAINT_LARK,
};

class Status {
 public:
  Status() = default;
  Status(LiteRtLm_Status code, std::string message = {},
         LiteRtLm_ErrorPhase phase = LITERT_LM_PHASE_NONE,
         int32_t native_code = 0, bool retryable = false)
      : code_(code),
        phase_(phase),
        native_code_(native_code),
        retryable_(retryable),
        message_(std::move(message)) {}

  bool ok() const { return code_ == LITERT_LM_STATUS_OK; }
  explicit operator bool() const { return ok(); }
  LiteRtLm_Status code() const { return code_; }
  LiteRtLm_ErrorPhase phase() const { return phase_; }
  int32_t native_code() const { return native_code_; }
  bool retryable() const { return retryable_; }
  const std::string& message() const { return message_; }

 private:
  LiteRtLm_Status code_ = LITERT_LM_STATUS_OK;
  LiteRtLm_ErrorPhase phase_ = LITERT_LM_PHASE_NONE;
  int32_t native_code_ = 0;
  bool retryable_ = false;
  std::string message_;
};

template <typename T>
class Result {
 public:
  Result(T value) : value_(std::move(value)) {}
  Result(Status status) : status_(std::move(status)) {}

  bool ok() const { return status_.ok() && value_.has_value(); }
  explicit operator bool() const { return ok(); }
  const Status& status() const { return status_; }
  T& operator*() { return *value_; }
  const T& operator*() const { return *value_; }
  T* operator->() { return &*value_; }
  const T* operator->() const { return &*value_; }
  T&& TakeValue() { return std::move(*value_); }

 private:
  Status status_;
  std::optional<T> value_;
};

struct Sampling {
  float temperature = 1.0f;
  float top_p = 0.95f;
  int32_t top_k = 64;
  bool random_seed = true;
  int32_t seed = 0;

  static Sampling Creative() { return Sampling(); }

  // Low-entropy defaults for tool routing, game decisions, classification,
  // and other requests where schema fidelity matters more than prose variety.
  static Sampling Precise(int32_t seed_value = 0) {
    Sampling value;
    value.temperature = 0.1f;
    value.top_p = 0.8f;
    value.top_k = 20;
    value.random_seed = false;
    value.seed = seed_value;
    return value;
  }

  // Fixed-seed defaults for requests that use a thought channel before a
  // structured decision. Unlike Precise(), this leaves enough entropy for a
  // small model to finish its reasoning instead of falling into low-temperature
  // repetition.
  static Sampling ReasonedDecision(int32_t seed_value = 0) {
    Sampling value;
    value.temperature = 0.7f;
    value.top_p = 0.9f;
    value.top_k = 40;
    value.random_seed = false;
    value.seed = seed_value;
    return value;
  }

  // A fixed sampler seed. This does not promise bit-identical output across
  // different drivers, devices, or runtime versions.
  static Sampling FixedSeed(int32_t seed_value) {
    Sampling value;
    value.random_seed = false;
    value.seed = seed_value;
    return value;
  }
};

struct EngineOptions {
  std::string_view model_path;
  Backend backend = Backend::Gpu;
  Sampling sampling = Sampling::Creative();
  uint32_t max_context_tokens = 0;
  bool metrics = true;
  bool parallel_file_loading = true;
  bool speculative_decoding = false;
  uint32_t num_threads = 0;
  int32_t prefill_chunk_size = 0;
  std::string_view cache_dir;
  std::string_view dispatch_library_dir;
};

struct ConversationOptions {
  std::string_view system_prompt;
  std::string_view tools_json;
  std::string_view initial_messages_json;
  std::string_view extra_context_json;
  std::string_view tool_call_channel;
  std::string_view lora_path;
  std::string_view audio_lora_path;
  uint32_t flags = LITERT_LM_CONVERSATION_RETURN_ERROR_ON_PARSE_FAILURE;
  uint32_t default_max_output_tokens = 0;
};

struct ChatOptions {
  ContextMode context_mode = ContextMode::Continue;
  uint32_t max_output_tokens = 0;
  ConstraintKind constraint_kind = ConstraintKind::None;
  std::string_view constraint;
  // ABI 1.2+: when set, only this logical Conversation is rebuilt with the
  // requested sampler. Canonical history is preserved and replayed.
  std::optional<Sampling> sampling;
  // ABI 1.3+: when set, expose only these tool declarations for this request.
  // The Conversation is rebuilt with canonical history; the Engine/model is
  // retained. An empty JSON array intentionally disables tools.
  std::optional<std::string_view> tools_json;
  // ABI 1.4+: turn-specific JSON object merged into prompt-template context.
  // Unlike sampling/tools, this does not rebuild the Conversation.
  std::optional<std::string_view> extra_context_json;
  LiteRtLm_EventCallback callback = nullptr;
  void* user_data = nullptr;

  static ChatOptions ReasonedDecision(uint32_t token_budget = 1024,
                                      int32_t seed_value = 0) {
    ChatOptions value;
    value.max_output_tokens = token_budget;
    value.sampling = Sampling::ReasonedDecision(seed_value);
    return value;
  }
};

struct RequestResult {
  LiteRtLm_RequestState state = LITERT_LM_REQUEST_CREATED;
  LiteRtLm_FinishReason finish_reason = LITERT_LM_FINISH_NONE;
  LiteRtLm_Status status = LITERT_LM_STATUS_OK;
  Backend backend = Backend::Gpu;
  LiteRtLm_ErrorPhase error_phase = LITERT_LM_PHASE_NONE;
  int32_t native_code = 0;
  bool retryable = false;
  uint64_t valid_metrics = 0;
  uint64_t input_tokens = 0;
  uint64_t output_tokens = 0;
  uint64_t context_tokens = 0;
  uint64_t prefill_time_us = 0;
  uint64_t decode_time_us = 0;
  uint64_t time_to_first_token_us = 0;
  uint64_t total_time_us = 0;
  double output_tokens_per_second = 0.0;
  uint64_t conversation_generation = 0;
  uint64_t callback_count = 0;
  Sampling applied_sampling;
  std::string text;
  std::string response_json;
  std::string error_message;
};

struct EngineInfo {
  Backend configured_backend = Backend::Gpu;
  Backend resolved_backend = Backend::Gpu;
  Backend configured_sampler_backend = Backend::Gpu;
  LiteRtLm_BackendEvidence backend_evidence =
      LITERT_LM_BACKEND_EVIDENCE_NONE;
  uint32_t enabled_modalities = 0;
  uint32_t max_context_tokens = 0;
  uint32_t conversation_count = 0;
  uint32_t active_request_count = 0;
  uint32_t retained_request_count = 0;
  Sampling applied_sampling;
  std::string backend_description;
};

struct ConversationInfo {
  uint32_t flags = 0;
  uint32_t active_request_count = 0;
  uint32_t default_max_output_tokens = 0;
  uint64_t generation = 0;
  uint64_t history_message_count = 0;
  uint64_t kv_token_count = 0;
  std::string tools_json;
};

// The normal one-class entry point uses these flat options. All model-level
// fields are shared by contexts created with LiteRtLm::NewContext().
struct LiteRtLmOptions {
  std::string_view model_path;
  Backend backend = Backend::Gpu;
  Sampling sampling = Sampling::Creative();
  uint32_t max_context_tokens = 0;
  bool native_benchmark_metrics = true;
  bool parallel_file_loading = true;
  bool speculative_decoding = false;
  uint32_t num_threads = 0;
  int32_t prefill_chunk_size = 0;
  std::string_view cache_dir;
  std::string_view dispatch_library_dir;
  std::string_view system_prompt;
  std::string_view tools_json;
  std::string_view initial_messages_json;
  std::string_view extra_context_json;
  std::string_view tool_call_channel;
  std::string_view lora_path;
  std::string_view audio_lora_path;
  uint32_t conversation_flags =
      LITERT_LM_CONVERSATION_RETURN_ERROR_ON_PARSE_FAILURE;
  uint32_t default_max_output_tokens = 0;
};

struct ContextOptions {
  std::string_view system_prompt;
  std::string_view tools_json;
  std::string_view initial_messages_json;
  std::string_view extra_context_json;
  std::string_view tool_call_channel;
  std::string_view lora_path;
  std::string_view audio_lora_path;
  uint32_t flags = LITERT_LM_CONVERSATION_RETURN_ERROR_ON_PARSE_FAILURE;
  uint32_t default_max_output_tokens = 0;
};

struct AskOptions {
  uint32_t max_output_tokens = 0;
  ConstraintKind constraint_kind = ConstraintKind::None;
  std::string_view constraint;
  std::optional<Sampling> sampling;
  // ABI 1.3+: per-request tool declaration override. Use a single declaration
  // with enum-constrained arguments for deterministic phase routing while
  // keeping the model's reasoning channel enabled.
  std::optional<std::string_view> tools_json;
  // ABI 1.4+: turn-specific prompt-template context JSON object.
  std::optional<std::string_view> extra_context_json;
  // The callback runs while the shared model operation is active. Copy event
  // views and return promptly. It may call CancelCurrentInference, but must not
  // re-enter any other LiteRtLm facade method.
  LiteRtLm_EventCallback callback = nullptr;
  void* user_data = nullptr;
  // Optional header-only cancellation bridge. When it becomes true before
  // the native Request is published, the facade cancels immediately after
  // publication, closing the otherwise unavoidable submit/cancel race.
  const std::atomic_bool* cancel_requested = nullptr;
  std::chrono::milliseconds timeout = std::chrono::minutes(10);

  static AskOptions ReasonedDecision(uint32_t token_budget = 1024,
                                     int32_t seed_value = 0) {
    AskOptions value;
    value.max_output_tokens = token_budget;
    value.sampling = Sampling::ReasonedDecision(seed_value);
    return value;
  }
};

namespace detail {

struct RequestState;

inline LiteRtLm_StringView View(std::string_view value) {
  return LiteRtLm_String(value.data(), static_cast<uint64_t>(value.size()));
}

inline std::string Copy(LiteRtLm_StringView value) {
  return value.data == nullptr || value.size == 0
             ? std::string()
             : std::string(value.data, static_cast<size_t>(value.size));
}

inline LiteRtLm_ErrorInfo ErrorBuffer() {
  LiteRtLm_ErrorInfo error{};
  error.struct_size = static_cast<uint32_t>(sizeof(error));
  return error;
}

inline Status CallStatus(LiteRtLm_Status code,
                         const LiteRtLm_ErrorInfo& error) {
  if (code == LITERT_LM_STATUS_OK) return Status();
  return Status(code, error.message, error.phase, error.native_code,
                error.retryable != 0);
}

inline LiteRtLm_Sampling NativeSampling(const Sampling& sampling) {
  LiteRtLm_Sampling value = LiteRtLm_DefaultSampling();
  value.temperature = sampling.temperature;
  value.top_p = sampling.top_p;
  value.top_k = sampling.top_k;
  value.seed_mode = sampling.random_seed ? LITERT_LM_SEED_RANDOM
                                         : LITERT_LM_SEED_FIXED;
  value.seed = sampling.seed;
  return value;
}

inline Sampling CppSampling(const LiteRtLm_Sampling& sampling) {
  Sampling value;
  value.temperature = sampling.temperature;
  value.top_p = sampling.top_p;
  value.top_k = sampling.top_k;
  value.random_seed = sampling.seed_mode == LITERT_LM_SEED_RANDOM;
  value.seed = sampling.seed;
  return value;
}

inline std::string JsonRoleMessage(std::string_view role,
                                   std::string_view text) {
  static constexpr char kHex[] = "0123456789abcdef";
  std::string json;
  json.reserve(role.size() + text.size() + 40);
  auto append_escaped = [&](std::string_view input) {
    for (unsigned char value : input) {
      switch (value) {
        case '\"':
          json += "\\\"";
          break;
        case '\\':
          json += "\\\\";
          break;
        case '\b':
          json += "\\b";
          break;
        case '\f':
          json += "\\f";
          break;
        case '\n':
          json += "\\n";
          break;
        case '\r':
          json += "\\r";
          break;
        case '\t':
          json += "\\t";
          break;
        default:
          if (value < 0x20) {
            json += "\\u00";
            json += kHex[(value >> 4) & 0x0F];
            json += kHex[value & 0x0F];
          } else {
            json += static_cast<char>(value);
          }
          break;
      }
    }
  };

  json += "{\"role\":\"";
  append_escaped(role);
  json += "\",\"content\":\"";
  append_escaped(text);
  json += "\"}";
  return json;
}

inline std::string JsonUserMessage(std::string_view text) {
  return JsonRoleMessage("user", text);
}

// Small dependency-free validator/splitter for the C++ convenience layer. It
// preserves each message object verbatim and understands nested JSON values,
// strings, escapes, numbers, and literals. The native ABI performs its own
// authoritative parse again before inference.
class JsonParser {
 public:
  explicit JsonParser(std::string_view input)
      : cursor_(input.empty() ? kEmpty : input.data()),
        end_(cursor_ + input.size()) {}

  bool ParseMessages(bool allow_single_object, bool allow_empty_array,
                     std::vector<std::string>* output) {
    if (output == nullptr) return Fail("Output message list is null.");
    output->clear();
    SkipWhitespace();
    if (cursor_ == end_) return Fail("JSON message input is empty.");

    if (*cursor_ == '{') {
      if (!allow_single_object) {
        return Fail("Expected a JSON message array.");
      }
      const char* start = cursor_;
      if (!ParseObject(0)) return false;
      output->emplace_back(start, static_cast<size_t>(cursor_ - start));
    } else if (*cursor_ == '[') {
      ++cursor_;
      SkipWhitespace();
      if (cursor_ != end_ && *cursor_ == ']') {
        ++cursor_;
        if (!allow_empty_array) {
          return Fail("The JSON message array must not be empty.");
        }
      } else {
        while (true) {
          SkipWhitespace();
          if (cursor_ == end_ || *cursor_ != '{') {
            return Fail("Every message array element must be a JSON object.");
          }
          const char* start = cursor_;
          if (!ParseObject(0)) return false;
          output->emplace_back(start, static_cast<size_t>(cursor_ - start));
          SkipWhitespace();
          if (cursor_ == end_) return Fail("Unterminated JSON message array.");
          if (*cursor_ == ']') {
            ++cursor_;
            break;
          }
          if (*cursor_ != ',') {
            return Fail("Expected ',' or ']' in the JSON message array.");
          }
          ++cursor_;
        }
      }
    } else {
      return Fail("Expected a JSON message object or array.");
    }

    SkipWhitespace();
    if (cursor_ != end_) return Fail("Unexpected data after the JSON value.");
    return true;
  }

  const char* error() const { return error_; }

 private:
  static constexpr char kEmpty[] = "";
  static constexpr uint32_t kMaxDepth = 256;

  bool ParseValue(uint32_t depth) {
    if (depth > kMaxDepth) return Fail("JSON nesting is too deep.");
    SkipWhitespace();
    if (cursor_ == end_) return Fail("Unexpected end of JSON input.");
    switch (*cursor_) {
      case '{':
        return ParseObject(depth + 1);
      case '[':
        return ParseArray(depth + 1);
      case '\"':
        return ParseString();
      case 't':
        return ParseLiteral("true");
      case 'f':
        return ParseLiteral("false");
      case 'n':
        return ParseLiteral("null");
      default:
        return ParseNumber();
    }
  }

  bool ParseObject(uint32_t depth) {
    if (depth > kMaxDepth) return Fail("JSON nesting is too deep.");
    if (cursor_ == end_ || *cursor_ != '{') {
      return Fail("Expected a JSON object.");
    }
    ++cursor_;
    SkipWhitespace();
    if (cursor_ != end_ && *cursor_ == '}') {
      ++cursor_;
      return true;
    }
    while (true) {
      SkipWhitespace();
      if (!ParseString()) return false;
      SkipWhitespace();
      if (cursor_ == end_ || *cursor_ != ':') {
        return Fail("Expected ':' after a JSON object key.");
      }
      ++cursor_;
      if (!ParseValue(depth + 1)) return false;
      SkipWhitespace();
      if (cursor_ == end_) return Fail("Unterminated JSON object.");
      if (*cursor_ == '}') {
        ++cursor_;
        return true;
      }
      if (*cursor_ != ',') return Fail("Expected ',' or '}' in JSON object.");
      ++cursor_;
    }
  }

  bool ParseArray(uint32_t depth) {
    if (depth > kMaxDepth) return Fail("JSON nesting is too deep.");
    if (cursor_ == end_ || *cursor_ != '[') {
      return Fail("Expected a JSON array.");
    }
    ++cursor_;
    SkipWhitespace();
    if (cursor_ != end_ && *cursor_ == ']') {
      ++cursor_;
      return true;
    }
    while (true) {
      if (!ParseValue(depth + 1)) return false;
      SkipWhitespace();
      if (cursor_ == end_) return Fail("Unterminated JSON array.");
      if (*cursor_ == ']') {
        ++cursor_;
        return true;
      }
      if (*cursor_ != ',') return Fail("Expected ',' or ']' in JSON array.");
      ++cursor_;
    }
  }

  bool ParseString() {
    if (cursor_ == end_ || *cursor_ != '\"') {
      return Fail("Expected a JSON string.");
    }
    ++cursor_;
    while (cursor_ != end_) {
      unsigned char value = static_cast<unsigned char>(*cursor_++);
      if (value == '\"') return true;
      if (value < 0x20) return Fail("Unescaped control byte in JSON string.");
      if (value != '\\') continue;
      if (cursor_ == end_) return Fail("Unterminated JSON escape.");
      char escape = *cursor_++;
      if (escape == 'u') {
        for (int i = 0; i < 4; ++i) {
          if (cursor_ == end_ || !IsHex(*cursor_++)) {
            return Fail("Invalid JSON unicode escape.");
          }
        }
      } else if (escape != '\"' && escape != '\\' && escape != '/' &&
                 escape != 'b' && escape != 'f' && escape != 'n' &&
                 escape != 'r' && escape != 't') {
        return Fail("Invalid JSON escape.");
      }
    }
    return Fail("Unterminated JSON string.");
  }

  bool ParseNumber() {
    const char* start = cursor_;
    if (cursor_ != end_ && *cursor_ == '-') ++cursor_;
    if (cursor_ == end_) return Fail("Invalid JSON number.");
    if (*cursor_ == '0') {
      ++cursor_;
      if (cursor_ != end_ && IsDigit(*cursor_)) {
        return Fail("JSON numbers cannot have leading zeroes.");
      }
    } else if (IsDigitOneToNine(*cursor_)) {
      do {
        ++cursor_;
      } while (cursor_ != end_ && IsDigit(*cursor_));
    } else {
      return Fail("Invalid JSON value.");
    }
    if (cursor_ != end_ && *cursor_ == '.') {
      ++cursor_;
      if (cursor_ == end_ || !IsDigit(*cursor_)) {
        return Fail("Invalid JSON fractional number.");
      }
      while (cursor_ != end_ && IsDigit(*cursor_)) ++cursor_;
    }
    if (cursor_ != end_ && (*cursor_ == 'e' || *cursor_ == 'E')) {
      ++cursor_;
      if (cursor_ != end_ && (*cursor_ == '+' || *cursor_ == '-')) ++cursor_;
      if (cursor_ == end_ || !IsDigit(*cursor_)) {
        return Fail("Invalid JSON exponent.");
      }
      while (cursor_ != end_ && IsDigit(*cursor_)) ++cursor_;
    }
    return cursor_ != start;
  }

  bool ParseLiteral(const char* literal) {
    for (const char* item = literal; *item != '\0'; ++item) {
      if (cursor_ == end_ || *cursor_++ != *item) {
        return Fail("Invalid JSON literal.");
      }
    }
    return true;
  }

  void SkipWhitespace() {
    while (cursor_ != end_ &&
           (*cursor_ == ' ' || *cursor_ == '\t' || *cursor_ == '\r' ||
            *cursor_ == '\n')) {
      ++cursor_;
    }
  }

  static bool IsDigit(char value) { return value >= '0' && value <= '9'; }
  static bool IsDigitOneToNine(char value) {
    return value >= '1' && value <= '9';
  }
  static bool IsHex(char value) {
    return IsDigit(value) || (value >= 'a' && value <= 'f') ||
           (value >= 'A' && value <= 'F');
  }

  bool Fail(const char* message) {
    if (error_ == nullptr) error_ = message;
    return false;
  }

  const char* cursor_ = nullptr;
  const char* end_ = nullptr;
  const char* error_ = nullptr;
};

inline Status ParseMessages(std::string_view json, bool allow_single_object,
                            bool allow_empty_array,
                            std::vector<std::string>* output) {
  JsonParser parser(json);
  if (parser.ParseMessages(allow_single_object, allow_empty_array, output)) {
    return Status();
  }
  return Status(LITERT_LM_STATUS_PARSE_ERROR,
                parser.error() == nullptr ? "Invalid JSON messages."
                                          : parser.error(),
                LITERT_LM_PHASE_CONTEXT_PREPARE);
}

inline Status ParseSingleMessage(std::string_view json,
                                 std::string* output) {
  size_t first = json.find_first_not_of(" \t\r\n");
  if (first == std::string_view::npos || json[first] != '{') {
    return Status(LITERT_LM_STATUS_PARSE_ERROR,
                  "Expected one JSON message object.",
                  LITERT_LM_PHASE_CONTEXT_PREPARE);
  }
  std::vector<std::string> messages;
  Status parsed = ParseMessages(json, true, false, &messages);
  if (!parsed) return parsed;
  if (messages.size() != 1) {
    return Status(LITERT_LM_STATUS_PARSE_ERROR,
                  "Expected exactly one JSON message object.",
                  LITERT_LM_PHASE_CONTEXT_PREPARE);
  }
  if (output != nullptr) *output = std::move(messages.front());
  return Status();
}

inline std::string JoinMessages(const std::vector<std::string>& first,
                                const std::vector<std::string>& second = {}) {
  size_t size = 2;
  for (const auto& message : first) size += message.size() + 1;
  for (const auto& message : second) size += message.size() + 1;
  std::string json;
  json.reserve(size);
  json.push_back('[');
  bool needs_comma = false;
  auto append = [&](const std::vector<std::string>& messages) {
    for (const auto& message : messages) {
      if (needs_comma) json.push_back(',');
      json += message;
      needs_comma = true;
    }
  };
  append(first);
  append(second);
  json.push_back(']');
  return json;
}

struct EngineState {
  LiteRtLm_Api api{};
  LiteRtLm_Engine handle = nullptr;

  ~EngineState() {
    if (handle == nullptr) return;
    LiteRtLm_ErrorInfo error = ErrorBuffer();
    api.destroy_engine(handle, &error);
  }
};

struct ConversationState {
  std::shared_ptr<EngineState> engine;
  LiteRtLm_Conversation handle = nullptr;

  ~ConversationState() {
    if (handle == nullptr) return;
    LiteRtLm_ErrorInfo error = ErrorBuffer();
    engine->api.destroy_conversation(handle, &error);
  }
};

// Contexts share one loaded model/Engine, while each owns an independent
// logical Conversation and canonical history. Inference is serialized per
// Engine; the current Win64 runtime keeps one physical Conversation/KV
// resident and replays the target history when contexts switch.
struct SharedModelState {
  std::shared_ptr<EngineState> engine;
  uint64_t next_context_id = 1;
  LiteRtLm_Request active_request = nullptr;
  std::shared_ptr<RequestState> retained_request;
  std::mutex mutex;
  std::mutex cancel_mutex;
};

struct RequestState {
  std::shared_ptr<ConversationState> conversation;
  LiteRtLm_Request handle = nullptr;

  ~RequestState() {
    if (handle == nullptr) return;
    LiteRtLm_ErrorInfo error = ErrorBuffer();
    conversation->engine->api.cancel_request(handle, &error);
    error = ErrorBuffer();
    LiteRtLm_Status wait = conversation->engine->api.wait_request(
        handle, 30000, &error);
    if (wait != LITERT_LM_STATUS_OK) {
      // Safety beats a destructor freeze. The native handle hierarchy remains
      // alive (subsequent parent destroys return BUSY) and is intentionally
      // leaked for this abnormal process lifetime rather than risking UAF.
      return;
    }
    error = ErrorBuffer();
    conversation->engine->api.destroy_request(handle, &error);
  }
};

}  // namespace detail

class Request {
 public:
  Request() = default;
  Request(Request&&) noexcept = default;
  Request& operator=(Request&&) noexcept = default;
  Request(const Request&) = delete;
  Request& operator=(const Request&) = delete;

  bool valid() const { return state_ && state_->handle != nullptr; }

  Status Wait(uint32_t timeout_ms = LITERT_LM_WAIT_INFINITE) {
    if (!valid())
      return Status(LITERT_LM_STATUS_INVALID_STATE, "Request is empty.");
    LiteRtLm_ErrorInfo error = detail::ErrorBuffer();
    LiteRtLm_Status code = state_->conversation->engine->api.wait_request(
        state_->handle, timeout_ms, &error);
    return detail::CallStatus(code, error);
  }

  Status Wait(std::chrono::milliseconds timeout) {
    auto count = timeout.count();
    uint32_t milliseconds =
        count < 0
            ? LITERT_LM_WAIT_INFINITE
            : (count > static_cast<int64_t>(UINT32_MAX - 1)
                   ? UINT32_MAX - 1
                   : static_cast<uint32_t>(count));
    return Wait(milliseconds);
  }

  Status Cancel() {
    if (!valid())
      return Status(LITERT_LM_STATUS_INVALID_STATE, "Request is empty.");
    LiteRtLm_ErrorInfo error = detail::ErrorBuffer();
    LiteRtLm_Status code = state_->conversation->engine->api.cancel_request(
        state_->handle, &error);
    return detail::CallStatus(code, error);
  }

  // Explicitly releases the retained native result so the Engine can accept
  // its next serial Request. Wait must have completed first.
  Status Close() {
    if (!valid()) return Status();
    LiteRtLm_ErrorInfo error = detail::ErrorBuffer();
    LiteRtLm_Status code = state_->conversation->engine->api.destroy_request(
        state_->handle, &error);
    if (code == LITERT_LM_STATUS_OK) state_->handle = nullptr;
    return detail::CallStatus(code, error);
  }

  Result<RequestResult> GetResult() const {
    if (!valid())
      return Status(LITERT_LM_STATUS_INVALID_STATE, "Request is empty.");
    LiteRtLm_RequestInfo info{};
    info.struct_size = static_cast<uint32_t>(sizeof(info));
    LiteRtLm_ErrorInfo error = detail::ErrorBuffer();
    LiteRtLm_Status code = state_->conversation->engine->api.get_request_info(
        state_->handle, &info, &error);
    if (code != LITERT_LM_STATUS_OK) return detail::CallStatus(code, error);

    RequestResult result;
    result.state = info.state;
    result.finish_reason = info.finish_reason;
    result.status = info.status;
    result.backend = static_cast<Backend>(info.backend);
    result.error_phase = info.error_phase;
    result.native_code = info.native_code;
    result.retryable = info.retryable != 0;
    result.valid_metrics = info.valid_metrics;
    result.input_tokens = info.input_tokens;
    result.output_tokens = info.output_tokens;
    result.context_tokens = info.context_tokens;
    result.prefill_time_us = info.prefill_time_us;
    result.decode_time_us = info.decode_time_us;
    result.time_to_first_token_us = info.time_to_first_token_us;
    result.total_time_us = info.total_time_us;
    result.output_tokens_per_second = info.output_tokens_per_second;
    result.conversation_generation = info.conversation_generation;
    result.callback_count = info.callback_count;
    result.applied_sampling = detail::CppSampling(info.applied_sampling);
    result.text = detail::Copy(info.text);
    result.response_json = detail::Copy(info.response_json);
    result.error_message = detail::Copy(info.error_message);
    return result;
  }

  // Waits, copies the terminal diagnostics, and releases the retained native
  // Request. A failed/cancelled inference is still returned as a value so
  // callers can inspect partial text and metrics.
  Result<RequestResult> AwaitRaw(
      std::chrono::milliseconds timeout = std::chrono::minutes(10)) {
    Status wait = Wait(timeout);
    if (!wait) return wait;
    auto result = GetResult();
    if (!result) return result.status();
    Status close = Close();
    if (!close) return close;
    return result.TakeValue();
  }

  // Convenience form for normal chat code. Terminal inference failure or
  // cancellation is converted to a Status; use AwaitRaw for diagnostics.
  Result<RequestResult> Await(
      std::chrono::milliseconds timeout = std::chrono::minutes(10)) {
    auto result = AwaitRaw(timeout);
    if (!result) return result.status();
    if (result->status != LITERT_LM_STATUS_OK) {
      return Status(result->status, result->error_message,
                    result->error_phase, result->native_code,
                    result->retryable);
    }
    return result.TakeValue();
  }

 private:
  explicit Request(std::shared_ptr<detail::RequestState> state)
      : state_(std::move(state)) {}

  std::shared_ptr<detail::RequestState> state_;
  friend class Conversation;
  friend class LiteRtLm;
};

class Conversation {
 public:
  Conversation() = default;
  Conversation(Conversation&&) noexcept = default;
  Conversation& operator=(Conversation&&) noexcept = default;
  Conversation(const Conversation&) = delete;
  Conversation& operator=(const Conversation&) = delete;

  bool valid() const { return state_ && state_->handle != nullptr; }

  Status Reset() {
    if (!valid())
      return Status(LITERT_LM_STATUS_INVALID_STATE,
                    "Conversation is empty.");
    LiteRtLm_ErrorInfo error = detail::ErrorBuffer();
    LiteRtLm_Status code = state_->engine->api.reset_conversation(
        state_->handle, &error);
    return detail::CallStatus(code, error);
  }

  Result<Conversation> Clone() const {
    if (!valid()) {
      return Status(LITERT_LM_STATUS_INVALID_STATE,
                    "Conversation is empty.");
    }
    if (state_->engine->api.clone_conversation == nullptr) {
      return Status(LITERT_LM_STATUS_NOT_SUPPORTED,
                    "The loaded ABI does not support Conversation cloning.");
    }
    auto clone_state = std::make_shared<detail::ConversationState>();
    clone_state->engine = state_->engine;
    LiteRtLm_ErrorInfo error = detail::ErrorBuffer();
    LiteRtLm_Status code = state_->engine->api.clone_conversation(
        state_->handle, &clone_state->handle, &error);
    if (code != LITERT_LM_STATUS_OK) return detail::CallStatus(code, error);
    return Conversation(std::move(clone_state));
  }

  Result<ConversationInfo> GetInfo() const {
    if (!valid()) {
      return Status(LITERT_LM_STATUS_INVALID_STATE,
                    "Conversation is empty.");
    }
    if (state_->engine->api.get_conversation_info == nullptr) {
      return Status(LITERT_LM_STATUS_NOT_SUPPORTED,
                    "The loaded ABI does not expose ConversationInfo.");
    }
    LiteRtLm_ConversationInfo info{};
    info.struct_size = static_cast<uint32_t>(sizeof(info));
    LiteRtLm_ErrorInfo error = detail::ErrorBuffer();
    LiteRtLm_Status code = state_->engine->api.get_conversation_info(
        state_->handle, &info, &error);
    if (code != LITERT_LM_STATUS_OK) return detail::CallStatus(code, error);
    ConversationInfo result;
    result.flags = info.flags;
    result.active_request_count = info.active_request_count;
    result.default_max_output_tokens = info.default_max_output_tokens;
    result.generation = info.generation;
    result.history_message_count = info.history_message_count;
    result.kv_token_count = info.kv_token_count;
    result.tools_json = detail::Copy(info.tools_json);
    return result;
  }

  Result<std::string> ExportHistoryJson() const {
    if (!valid()) {
      return Status(LITERT_LM_STATUS_INVALID_STATE,
                    "Conversation is empty.");
    }
    auto copy = state_->engine->api.copy_conversation_history_json;
    if (copy == nullptr) {
      return Status(LITERT_LM_STATUS_NOT_SUPPORTED,
                    "The loaded ABI does not expose Conversation history.");
    }
    uint64_t size = 0;
    LiteRtLm_ErrorInfo error = detail::ErrorBuffer();
    LiteRtLm_Status code = copy(state_->handle, nullptr, &size, &error);
    if (code != LITERT_LM_STATUS_OK) return detail::CallStatus(code, error);
    if (size == 0 || size > static_cast<uint64_t>((
                         std::numeric_limits<size_t>::max)())) {
      return Status(LITERT_LM_STATUS_RESOURCE_EXHAUSTED,
                    "Conversation history size is invalid.");
    }
    std::vector<char> buffer(static_cast<size_t>(size));
    error = detail::ErrorBuffer();
    code = copy(state_->handle, buffer.data(), &size, &error);
    if (code != LITERT_LM_STATUS_OK) return detail::CallStatus(code, error);
    return std::string(buffer.data(), static_cast<size_t>(size - 1));
  }

  Result<Request> Send(std::string_view text) {
    return Send(text, ChatOptions());
  }

  Result<Request> Send(std::string_view text, const ChatOptions& options) {
    std::string json = detail::JsonUserMessage(text);
    return SendJson(json, options);
  }

  Result<Request> SendJson(std::string_view message_json) {
    return SendJson(message_json, ChatOptions());
  }

  Result<Request> SendJson(std::string_view message_json,
                           const ChatOptions& options) {
    if (!valid())
      return Status(LITERT_LM_STATUS_INVALID_STATE,
                    "Conversation is empty.");
    LiteRtLm_ChatRequest chat = LiteRtLm_DefaultChatRequest();
    chat.messages_json = detail::View(message_json);
    chat.context_mode = static_cast<LiteRtLm_ContextMode>(options.context_mode);
    chat.max_output_tokens = options.max_output_tokens;
    chat.constraint_kind =
        static_cast<LiteRtLm_ConstraintKind>(options.constraint_kind);
    chat.constraint = detail::View(options.constraint);
    if (options.sampling.has_value()) {
      if (state_->engine->api.abi_minor < 2 ||
          (state_->engine->api.capabilities &
           LITERT_LM_CAP_PER_REQUEST_SAMPLING) == 0) {
        return Status(LITERT_LM_STATUS_NOT_SUPPORTED,
                      "The loaded ABI does not support per-request sampling.");
      }
      chat.flags |= LITERT_LM_CHAT_OVERRIDE_SAMPLING;
      chat.sampling = detail::NativeSampling(*options.sampling);
    }
    if (options.tools_json.has_value()) {
      if (state_->engine->api.abi_minor < 3 ||
          (state_->engine->api.capabilities &
           LITERT_LM_CAP_PER_REQUEST_TOOLS) == 0) {
        return Status(LITERT_LM_STATUS_NOT_SUPPORTED,
                      "The loaded ABI does not support per-request tools.");
      }
      chat.flags |= LITERT_LM_CHAT_OVERRIDE_TOOLS;
      chat.tools_json = detail::View(*options.tools_json);
    }
    if (options.extra_context_json.has_value()) {
      if (state_->engine->api.abi_minor < 4 ||
          (state_->engine->api.capabilities &
           LITERT_LM_CAP_PER_REQUEST_EXTRA_CONTEXT) == 0) {
        return Status(
            LITERT_LM_STATUS_NOT_SUPPORTED,
            "The loaded ABI does not support per-request extra context.");
      }
      chat.flags |= LITERT_LM_CHAT_OVERRIDE_EXTRA_CONTEXT;
      chat.extra_context_json = detail::View(*options.extra_context_json);
    }
    chat.callback = options.callback;
    chat.user_data = options.user_data;

    // Allocate ownership before publishing a native handle so a later C++
    // allocation failure can never strand the Engine's retained Request slot.
    auto request_state = std::make_shared<detail::RequestState>();
    request_state->conversation = state_;
    LiteRtLm_ErrorInfo error = detail::ErrorBuffer();
    LiteRtLm_Status code = state_->engine->api.submit_chat(
        state_->handle, &chat, &request_state->handle, &error);
    if (code != LITERT_LM_STATUS_OK) return detail::CallStatus(code, error);
    return Request(std::move(request_state));
  }

  Result<Request> ReplaceJson(std::string_view complete_messages_json,
                              ChatOptions options = ChatOptions()) {
    options.context_mode = ContextMode::Replace;
    return SendJson(complete_messages_json, options);
  }

  Result<RequestResult> Chat(
      std::string_view text,
      std::chrono::milliseconds timeout = std::chrono::minutes(10)) {
    auto request = Send(text);
    if (!request) return request.status();
    auto result = request->Await(timeout);
    if (!result) return result.status();
    return result.TakeValue();
  }

 private:
  explicit Conversation(std::shared_ptr<detail::ConversationState> state)
      : state_(std::move(state)) {}

  std::shared_ptr<detail::ConversationState> state_;
  friend class Engine;
  friend class LiteRtLm;
};

class Engine {
 public:
  Engine() = default;
  Engine(Engine&&) noexcept = default;
  Engine& operator=(Engine&&) noexcept = default;
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  bool valid() const { return state_ && state_->handle != nullptr; }

  Result<Conversation> CreateConversation(
      std::string_view tools_json = std::string_view()) {
    ConversationOptions options;
    options.tools_json = tools_json;
    return CreateConversation(options);
  }

  Result<Conversation> CreateConversation(
      std::string_view system_prompt, std::string_view tools_json) {
    ConversationOptions options;
    options.system_prompt = system_prompt;
    options.tools_json = tools_json;
    return CreateConversation(options);
  }

  Result<Conversation> CreateConversation(
      const ConversationOptions& options) {
    if (!valid())
      return Status(LITERT_LM_STATUS_INVALID_STATE, "Engine is empty.");
    LiteRtLm_ConversationConfig config =
        LiteRtLm_DefaultConversationConfig();
    config.tools_json = detail::View(options.tools_json);
    config.flags = options.flags;
    config.default_max_output_tokens = options.default_max_output_tokens;
    config.system_message = detail::View(options.system_prompt);
    config.initial_messages_json = detail::View(options.initial_messages_json);
    config.extra_context_json = detail::View(options.extra_context_json);
    config.tool_call_channel = detail::View(options.tool_call_channel);
    config.lora_path = detail::View(options.lora_path);
    config.audio_lora_path = detail::View(options.audio_lora_path);
    // Allocate ownership before native creation so no successful handle can be
    // leaked by a subsequent std::make_shared failure.
    auto conversation_state =
        std::make_shared<detail::ConversationState>();
    conversation_state->engine = state_;
    LiteRtLm_ErrorInfo error = detail::ErrorBuffer();
    LiteRtLm_Status code = state_->api.create_conversation(
        state_->handle, &config, &conversation_state->handle, &error);
    if (code != LITERT_LM_STATUS_OK) return detail::CallStatus(code, error);
    return Conversation(std::move(conversation_state));
  }

  Result<std::vector<int32_t>> Tokenize(std::string_view text) const {
    if (!valid())
      return Status(LITERT_LM_STATUS_INVALID_STATE, "Engine is empty.");
    if (state_->api.tokenize == nullptr) {
      return Status(LITERT_LM_STATUS_NOT_SUPPORTED,
                    "The loaded ABI does not expose tokenization.");
    }
    uint64_t count = 0;
    LiteRtLm_ErrorInfo error = detail::ErrorBuffer();
    LiteRtLm_Status code = state_->api.tokenize(
        state_->handle, detail::View(text), nullptr, &count, &error);
    if (code != LITERT_LM_STATUS_OK) return detail::CallStatus(code, error);
    if (count > static_cast<uint64_t>((
                    std::numeric_limits<size_t>::max)())) {
      return Status(LITERT_LM_STATUS_RESOURCE_EXHAUSTED,
                    "Token count is too large.");
    }
    std::vector<int32_t> tokens(static_cast<size_t>(count));
    if (count == 0) return tokens;
    error = detail::ErrorBuffer();
    code = state_->api.tokenize(state_->handle, detail::View(text),
                                tokens.data(), &count, &error);
    if (code != LITERT_LM_STATUS_OK) return detail::CallStatus(code, error);
    tokens.resize(static_cast<size_t>(count));
    return tokens;
  }

  Result<std::string> Detokenize(const std::vector<int32_t>& tokens) const {
    return Detokenize(tokens.data(), tokens.size());
  }

  Result<std::string> Detokenize(const int32_t* tokens,
                                 size_t token_count) const {
    if (!valid())
      return Status(LITERT_LM_STATUS_INVALID_STATE, "Engine is empty.");
    if (state_->api.detokenize == nullptr) {
      return Status(LITERT_LM_STATUS_NOT_SUPPORTED,
                    "The loaded ABI does not expose detokenization.");
    }
    if (tokens == nullptr && token_count != 0) {
      return Status(LITERT_LM_STATUS_INVALID_ARGUMENT,
                    "Token pointer is null.");
    }
    uint64_t size = 0;
    LiteRtLm_ErrorInfo error = detail::ErrorBuffer();
    LiteRtLm_Status code = state_->api.detokenize(
        state_->handle, tokens, static_cast<uint64_t>(token_count), nullptr,
        &size, &error);
    if (code != LITERT_LM_STATUS_OK) return detail::CallStatus(code, error);
    if (size == 0 || size > static_cast<uint64_t>((
                         std::numeric_limits<size_t>::max)())) {
      return Status(LITERT_LM_STATUS_RESOURCE_EXHAUSTED,
                    "Detokenized text size is invalid.");
    }
    std::vector<char> buffer(static_cast<size_t>(size));
    error = detail::ErrorBuffer();
    code = state_->api.detokenize(
        state_->handle, tokens, static_cast<uint64_t>(token_count),
        buffer.data(), &size, &error);
    if (code != LITERT_LM_STATUS_OK) return detail::CallStatus(code, error);
    return std::string(buffer.data(), static_cast<size_t>(size - 1));
  }

  Result<EngineInfo> GetInfo() const {
    if (!valid())
      return Status(LITERT_LM_STATUS_INVALID_STATE, "Engine is empty.");
    LiteRtLm_EngineInfo info{};
    info.struct_size = static_cast<uint32_t>(sizeof(info));
    LiteRtLm_ErrorInfo error = detail::ErrorBuffer();
    LiteRtLm_Status code =
        state_->api.get_engine_info(state_->handle, &info, &error);
    if (code != LITERT_LM_STATUS_OK) return detail::CallStatus(code, error);
    EngineInfo result;
    result.configured_backend = static_cast<Backend>(info.configured_backend);
    result.resolved_backend = static_cast<Backend>(info.resolved_backend);
    result.configured_sampler_backend =
        static_cast<Backend>(info.configured_sampler_backend);
    result.backend_evidence = info.backend_evidence;
    result.enabled_modalities = info.enabled_modalities;
    result.max_context_tokens = info.max_context_tokens;
    result.conversation_count = info.conversation_count;
    result.active_request_count = info.active_request_count;
    result.retained_request_count = info.retained_request_count;
    result.applied_sampling = detail::CppSampling(info.applied_sampling);
    result.backend_description = detail::Copy(info.backend_description);
    return result;
  }

 private:
  explicit Engine(std::shared_ptr<detail::EngineState> state)
      : state_(std::move(state)) {}

  std::shared_ptr<detail::EngineState> state_;
  friend class Api;
  friend class LiteRtLm;
};

class Api {
 public:
  using GetApiFunction = LiteRtLm_Status(LITERT_LM_CALL*)(
      uint32_t requested_major, uint32_t out_api_size, LiteRtLm_Api* out_api);

  static Result<Api> Load() { return Load(&LiteRtLm_GetApi); }

  static Result<Api> Load(GetApiFunction get_api) {
    if (get_api == nullptr) {
      return Status(LITERT_LM_STATUS_INVALID_ARGUMENT,
                    "GetApi function is null.");
    }
    LiteRtLm_Api api{};
    LiteRtLm_Status code = get_api(LITERT_LM_ABI_MAJOR,
                                   static_cast<uint32_t>(sizeof(api)), &api);
    if (code != LITERT_LM_STATUS_OK) {
      return Status(code, "LiteRtLm_GetApi rejected the requested ABI.");
    }
    constexpr size_t kRequiredApiSize =
        offsetof(LiteRtLm_Api, clone_conversation);
    if (api.abi_major != LITERT_LM_ABI_MAJOR ||
        api.struct_size < kRequiredApiSize) {
      return Status(LITERT_LM_STATUS_ABI_MISMATCH,
                    "The loaded API table is incompatible.");
    }
    return Api(api);
  }

  Result<Engine> CreateEngine(
      std::string_view model_path, Backend backend = Backend::Gpu) const {
    EngineOptions options;
    options.model_path = model_path;
    options.backend = backend;
    return CreateEngine(options);
  }

  Result<Engine> CreateEngine(const EngineOptions& options) const {
    if (options.speculative_decoding &&
        (api_.capabilities & LITERT_LM_CAP_SPECULATIVE_DECODING) == 0) {
      return Status(LITERT_LM_STATUS_NOT_SUPPORTED,
                    "The loaded ABI does not expose speculative decoding.");
    }
    LiteRtLm_EngineConfig config = LiteRtLm_DefaultEngineConfig();
    config.model_path = detail::View(options.model_path);
    config.backend = static_cast<LiteRtLm_Backend>(options.backend);
    config.max_context_tokens = options.max_context_tokens;
    config.sampling = detail::NativeSampling(options.sampling);
    config.flags = 0;
    if (options.metrics) config.flags |= LITERT_LM_ENGINE_ENABLE_METRICS;
    if (!options.parallel_file_loading) {
      config.flags |= LITERT_LM_ENGINE_DISABLE_PARALLEL_FILE_LOADING;
    }
    if (options.speculative_decoding) {
      config.flags |= LITERT_LM_ENGINE_ENABLE_SPECULATIVE_DECODING;
    }
    config.num_threads = options.num_threads;
    config.prefill_chunk_size = options.prefill_chunk_size;
    config.cache_dir = detail::View(options.cache_dir);
    config.dispatch_library_dir = detail::View(options.dispatch_library_dir);

    // Allocate ownership before native creation so no successful handle can be
    // lost if the caller's allocator fails after CreateEngine returns.
    auto state = std::make_shared<detail::EngineState>();
    state->api = api_;
    LiteRtLm_ErrorInfo error = detail::ErrorBuffer();
    LiteRtLm_Status code =
        api_.create_engine(&config, &state->handle, &error);
    if (code != LITERT_LM_STATUS_OK) return detail::CallStatus(code, error);
    return Engine(std::move(state));
  }

  Status SetLogLevel(int32_t minimum_level) const {
    if (api_.set_log_level == nullptr) {
      return Status(LITERT_LM_STATUS_NOT_SUPPORTED,
                    "The loaded ABI does not expose log-level control.");
    }
    LiteRtLm_ErrorInfo error = detail::ErrorBuffer();
    LiteRtLm_Status code = api_.set_log_level(minimum_level, &error);
    return detail::CallStatus(code, error);
  }

  const LiteRtLm_Api& native() const { return api_; }

 private:
  explicit Api(const LiteRtLm_Api& api) : api_(api) {}
  LiteRtLm_Api api_{};
};

// Normal product code can use this single class. The first Create loads one
// model/Engine. NewContext returns the same class and shares that Engine, while
// owning an independent logical Conversation and canonical history. Requests
// from all contexts are serialized; same-context calls remain incremental and
// a cross-context call replays the target history into the physical runtime.
class LiteRtLm {
 public:
  LiteRtLm() = default;
  LiteRtLm(const LiteRtLm&) = delete;
  LiteRtLm& operator=(const LiteRtLm&) = delete;

  LiteRtLm(LiteRtLm&& other) noexcept { MoveFrom(std::move(other)); }

  LiteRtLm& operator=(LiteRtLm&& other) = delete;

  ~LiteRtLm() { Close(); }

  static Result<LiteRtLm> Create(std::string_view model_path) {
    LiteRtLmOptions options;
    options.model_path = model_path;
    return Create(options);
  }

  static Result<LiteRtLm> Create(std::string_view model_path,
                                 std::string_view system_prompt) {
    LiteRtLmOptions options;
    options.model_path = model_path;
    options.system_prompt = system_prompt;
    return Create(options);
  }

  static Result<LiteRtLm> Create(const LiteRtLmOptions& options) {
    return Create(options, &LiteRtLm_GetApi);
  }

  static Result<LiteRtLm> Create(const LiteRtLmOptions& options,
                                 Api::GetApiFunction get_api) {
    auto api = Api::Load(get_api);
    if (!api) return api.status();

    EngineOptions engine_options;
    engine_options.model_path = options.model_path;
    engine_options.backend = options.backend;
    engine_options.sampling = options.sampling;
    engine_options.max_context_tokens = options.max_context_tokens;
    engine_options.metrics = options.native_benchmark_metrics;
    engine_options.parallel_file_loading = options.parallel_file_loading;
    engine_options.speculative_decoding = options.speculative_decoding;
    engine_options.num_threads = options.num_threads;
    engine_options.prefill_chunk_size = options.prefill_chunk_size;
    engine_options.cache_dir = options.cache_dir;
    engine_options.dispatch_library_dir = options.dispatch_library_dir;

    auto engine = api->CreateEngine(engine_options);
    if (!engine) return engine.status();

    auto model = std::make_shared<detail::SharedModelState>();
    model->engine = engine->state_;
    ContextOptions context;
    context.system_prompt = options.system_prompt;
    context.tools_json = options.tools_json;
    context.initial_messages_json = options.initial_messages_json;
    context.extra_context_json = options.extra_context_json;
    context.tool_call_channel = options.tool_call_channel;
    context.lora_path = options.lora_path;
    context.audio_lora_path = options.audio_lora_path;
    context.flags = options.conversation_flags;
    context.default_max_output_tokens = options.default_max_output_tokens;
    return CreateContext(std::move(model), context);
  }

  bool valid() const {
    return context_id_ != 0 && model_ && model_->engine &&
           model_->engine->handle != nullptr && conversation_ &&
           conversation_->handle != nullptr;
  }

  Result<LiteRtLm> NewContext() const {
    return NewContext(ContextOptions());
  }

  Result<LiteRtLm> NewContext(std::string_view system_prompt) const {
    ContextOptions options;
    options.system_prompt = system_prompt;
    return NewContext(options);
  }

  Result<LiteRtLm> NewContext(const ContextOptions& options) const {
    if (!valid()) {
      return Status(LITERT_LM_STATUS_INVALID_STATE,
                    "LiteRtLm model is closed.");
    }
    return CreateContext(model_, options);
  }

  Result<EngineInfo> GetModelInfo() const {
    if (!valid()) {
      return Status(LITERT_LM_STATUS_INVALID_STATE,
                    "LiteRtLm model is closed.");
    }
    std::lock_guard<std::mutex> lock(model_->mutex);
    Engine engine(model_->engine);
    return engine.GetInfo();
  }

  // Ensures this context's native Conversation exists without generating model
  // output. Product wrappers may call this before reporting Ready.
  Status Prepare() {
    if (!valid()) {
      return Status(LITERT_LM_STATUS_INVALID_STATE,
                    "LiteRtLm context is closed.");
    }
    std::lock_guard<std::mutex> lock(model_->mutex);
    bool can_continue = false;
    return EnsureConversationLocked(&can_continue);
  }

  Result<RequestResult> Ask(
      std::string_view text,
      const AskOptions& options = AskOptions()) {
    auto result = AskRaw(text, options);
    if (!result) return result.status();
    if (result->status != LITERT_LM_STATUS_OK) {
      return Status(result->status, result->error_message,
                    result->error_phase, result->native_code,
                    result->retryable);
    }
    return result.TakeValue();
  }

  // Returns terminal diagnostics even when inference was cancelled or failed.
  Result<RequestResult> AskRaw(
      std::string_view text,
      const AskOptions& options = AskOptions()) {
    return AskJsonRaw(detail::JsonUserMessage(text), options);
  }

  // Advanced input for tool-result or multimodal message objects. The value may
  // be one message object or a non-empty array of messages.
  Result<RequestResult> AskJson(
      std::string_view messages_json,
      const AskOptions& options = AskOptions()) {
    auto result = AskJsonRaw(messages_json, options);
    if (!result) return result.status();
    if (result->status != LITERT_LM_STATUS_OK) {
      return Status(result->status, result->error_message,
                    result->error_phase, result->native_code,
                    result->retryable);
    }
    return result.TakeValue();
  }

  Result<RequestResult> AskJsonRaw(
      std::string_view messages_json,
      const AskOptions& options = AskOptions()) {
    if (!valid()) {
      return Status(LITERT_LM_STATUS_INVALID_STATE,
                    "LiteRtLm context is closed.");
    }

    std::vector<std::string> input_messages;
    Status parsed = detail::ParseMessages(messages_json, true, false,
                                          &input_messages);
    if (!parsed) return parsed;

    std::lock_guard<std::mutex> lock(model_->mutex);
    // Allocate the next canonical history while holding the shared operation
    // lock, before touching native state. Concurrent Ask calls are serialized
    // and can never commit a stale snapshot.
    std::vector<std::string> next_memory = memory_;
    next_memory.reserve(next_memory.size() + input_messages.size() + 1);
    next_memory.insert(next_memory.end(), input_messages.begin(),
                       input_messages.end());
    bool can_continue = false;
    Status ready = EnsureConversationLocked(&can_continue);
    if (!ready) {
      native_synced_ = false;
      return ready;
    }

    std::string payload =
        can_continue ? detail::JoinMessages(input_messages)
                     : detail::JoinMessages(memory_, input_messages);

    ChatOptions chat;
    // A dirty Conversation is rebuilt with canonical memory as its preface, so
    // only this turn's new messages are sent here.
    chat.context_mode = ContextMode::Continue;
    chat.max_output_tokens = options.max_output_tokens;
    chat.constraint_kind = options.constraint_kind;
    chat.constraint = options.constraint;
    chat.sampling = options.sampling;
    chat.tools_json = options.tools_json;
    chat.extra_context_json = options.extra_context_json;
    chat.callback = options.callback;
    chat.user_data = options.user_data;

    // From this point native state may advance. Keep the logical/native sync
    // bit pessimistically false until the complete assistant JSON is committed.
    native_synced_ = false;
    Conversation conversation(conversation_);
    auto request = conversation.SendJson(payload, chat);
    if (!request) {
      native_synced_ = false;
      return request.status();
    }

    auto terminal = AwaitRegisteredLocked(
        &*request, options.timeout, options.cancel_requested);
    if (!terminal) {
      native_synced_ = false;
      return terminal.status();
    }
    if (terminal->status != LITERT_LM_STATUS_OK) {
      native_synced_ = false;
      return terminal.TakeValue();
    }

    std::string assistant_message;
    Status assistant = detail::ParseSingleMessage(
        terminal->response_json, &assistant_message);
    if (!assistant) {
      native_synced_ = false;
      return Status(LITERT_LM_STATUS_INTERNAL_ERROR,
                    "Native inference succeeded but returned an invalid "
                    "assistant message JSON.",
                    LITERT_LM_PHASE_DECODE);
    }

    next_memory.push_back(std::move(assistant_message));
    memory_.swap(next_memory);
    // Native constraints are sticky in the current core. Preserve the logical
    // result, but force a clean REPLACE before the next free-form turn.
    native_synced_ = options.constraint_kind == ConstraintKind::None;
    return terminal.TakeValue();
  }

  // Thread-safe cancellation escape hatch for a streaming callback or another
  // controller thread. It targets the single currently running shared-model
  // request, regardless of which logical context started it.
  Status CancelCurrentInference() const {
    if (!model_ || !model_->engine || model_->engine->handle == nullptr) {
      return Status(LITERT_LM_STATUS_INVALID_STATE,
                    "LiteRtLm model is closed.");
    }
    std::lock_guard<std::mutex> lock(model_->cancel_mutex);
    if (model_->active_request == nullptr) {
      return Status(LITERT_LM_STATUS_NOT_READY,
                    "There is no active inference to cancel.");
    }
    LiteRtLm_ErrorInfo error = detail::ErrorBuffer();
    LiteRtLm_Status code = model_->engine->api.cancel_request(
        model_->active_request, &error);
    return detail::CallStatus(code, error);
  }

  // Completes cleanup for the rare case where a finite Ask timeout was
  // followed by a native callback-quiescence timeout. The retained handle is
  // never discarded, so callers can cancel and retry draining safely.
  Status DrainCurrentInference(
      std::chrono::milliseconds timeout = std::chrono::seconds(30)) {
    if (!model_ || !model_->engine || model_->engine->handle == nullptr) {
      return Status(LITERT_LM_STATUS_INVALID_STATE,
                    "LiteRtLm model is closed.");
    }
    std::lock_guard<std::mutex> operation_lock(model_->mutex);
    if (!model_->retained_request) {
      return Status(LITERT_LM_STATUS_NOT_READY,
                    "There is no retained inference to drain.");
    }

    Request retained(model_->retained_request);
    Status wait = retained.Wait(timeout);
    if (!wait) return wait;
    {
      std::lock_guard<std::mutex> cancel_lock(model_->cancel_mutex);
      model_->active_request = nullptr;
    }
    auto result = retained.GetResult();
    Status close = retained.Close();
    if (!close) return close;
    model_->retained_request.reset();
    if (!result) return result.status();
    return Status();
  }

  bool HasRetainedInference() const {
    if (!model_) return false;
    std::lock_guard<std::mutex> operation_lock(model_->mutex);
    return model_->retained_request != nullptr;
  }

  size_t MemorySize() const {
    if (!model_) return memory_.size();
    std::lock_guard<std::mutex> lock(model_->mutex);
    return memory_.size();
  }

  std::vector<std::string> Messages() const {
    if (!model_) return memory_;
    std::lock_guard<std::mutex> lock(model_->mutex);
    return memory_;
  }

  Result<std::string> Message(size_t index) const {
    std::unique_lock<std::mutex> lock;
    if (model_) lock = std::unique_lock<std::mutex>(model_->mutex);
    if (index >= memory_.size()) {
      return Status(LITERT_LM_STATUS_INVALID_ARGUMENT,
                    "Memory message index is out of range.");
    }
    return memory_[index];
  }

  std::string ExportMemoryJson() const {
    std::unique_lock<std::mutex> lock;
    if (model_) lock = std::unique_lock<std::mutex>(model_->mutex);
    return detail::JoinMessages(memory_);
  }

  Status ImportMemoryJson(std::string_view complete_messages_json) {
    std::vector<std::string> next;
    Status parsed = detail::ParseMessages(complete_messages_json, false, true,
                                          &next);
    if (!parsed) return parsed;
    return CommitMemory(std::move(next));
  }

  Status InsertMessage(size_t index, std::string_view message_json) {
    std::string message;
    Status parsed = detail::ParseSingleMessage(message_json, &message);
    if (!parsed) return parsed;
    if (!valid()) {
      return Status(LITERT_LM_STATUS_INVALID_STATE,
                    "LiteRtLm context is closed.");
    }
    std::lock_guard<std::mutex> lock(model_->mutex);
    if (index > memory_.size()) {
      return Status(LITERT_LM_STATUS_INVALID_ARGUMENT,
                    "Memory insertion index is out of range.");
    }
    std::vector<std::string> next = memory_;
    next.insert(next.begin() + static_cast<std::ptrdiff_t>(index),
                std::move(message));
    return CommitMemoryLocked(std::move(next));
  }

  Status ReplaceMessage(size_t index, std::string_view message_json) {
    std::string message;
    Status parsed = detail::ParseSingleMessage(message_json, &message);
    if (!parsed) return parsed;
    if (!valid()) {
      return Status(LITERT_LM_STATUS_INVALID_STATE,
                    "LiteRtLm context is closed.");
    }
    std::lock_guard<std::mutex> lock(model_->mutex);
    if (index >= memory_.size()) {
      return Status(LITERT_LM_STATUS_INVALID_ARGUMENT,
                    "Memory replacement index is out of range.");
    }
    std::vector<std::string> next = memory_;
    next[index] = std::move(message);
    return CommitMemoryLocked(std::move(next));
  }

  Status DeleteMessage(size_t index) {
    if (!valid()) {
      return Status(LITERT_LM_STATUS_INVALID_STATE,
                    "LiteRtLm context is closed.");
    }
    std::lock_guard<std::mutex> lock(model_->mutex);
    if (index >= memory_.size()) {
      return Status(LITERT_LM_STATUS_INVALID_ARGUMENT,
                    "Memory deletion index is out of range.");
    }
    std::vector<std::string> next = memory_;
    next.erase(next.begin() + static_cast<std::ptrdiff_t>(index));
    return CommitMemoryLocked(std::move(next));
  }

  Status ClearMemory() { return CommitMemory({}); }

  Status SaveMemory(const std::filesystem::path& path) const {
    if (path.empty()) {
      return Status(LITERT_LM_STATUS_INVALID_ARGUMENT,
                    "Memory path is empty.");
    }
    const std::string json = ExportMemoryJson();
    if (json.size() > kMaxMemoryFileBytes ||
        json.size() >
            static_cast<size_t>((std::numeric_limits<std::streamsize>::max)())) {
      return Status(LITERT_LM_STATUS_RESOURCE_EXHAUSTED,
                    "Memory snapshot is too large.");
    }

    std::filesystem::path temporary = path;
    temporary += ".tmp";
    std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
    if (!stream) {
      return Status(LITERT_LM_STATUS_IO_ERROR,
                    "Unable to open the memory file for writing.");
    }
    stream.write(json.data(), static_cast<std::streamsize>(json.size()));
    stream.flush();
    if (!stream) {
      return Status(LITERT_LM_STATUS_IO_ERROR,
                    "Unable to write the complete memory file.");
    }
    stream.close();
    if (!stream) {
      return Status(LITERT_LM_STATUS_IO_ERROR,
                    "Unable to close the memory file.");
    }

    std::error_code error;
    std::filesystem::rename(temporary, path, error);
    if (error) {
      // Windows rename does not replace an existing file. The previous file is
      // removed only after the complete temporary snapshot is safely closed.
      std::error_code remove_error;
      std::filesystem::remove(path, remove_error);
      if (remove_error) {
        return Status(LITERT_LM_STATUS_IO_ERROR,
                      "Unable to replace the previous memory file.");
      }
      error.clear();
      std::filesystem::rename(temporary, path, error);
      if (error) {
        return Status(LITERT_LM_STATUS_IO_ERROR,
                      "Unable to publish the memory file.");
      }
    }
    return Status();
  }

  Status LoadMemory(const std::filesystem::path& path) {
    if (path.empty()) {
      return Status(LITERT_LM_STATUS_INVALID_ARGUMENT,
                    "Memory path is empty.");
    }
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
      return Status(LITERT_LM_STATUS_IO_ERROR,
                    "Unable to open the memory file for reading.");
    }
    const std::streamoff length = stream.tellg();
    if (length < 0 || static_cast<uint64_t>(length) > kMaxMemoryFileBytes ||
        static_cast<uint64_t>(length) >
            static_cast<uint64_t>((std::numeric_limits<size_t>::max)()) ||
        static_cast<uint64_t>(length) > static_cast<uint64_t>((
                                            std::numeric_limits<
                                                std::streamsize>::max)())) {
      return Status(LITERT_LM_STATUS_RESOURCE_EXHAUSTED,
                    "Memory file is too large.");
    }
    std::string json(static_cast<size_t>(length), '\0');
    stream.seekg(0, std::ios::beg);
    if (!json.empty()) {
      stream.read(json.data(), static_cast<std::streamsize>(json.size()));
    }
    if (!stream) {
      return Status(LITERT_LM_STATUS_IO_ERROR,
                    "Unable to read the complete memory file.");
    }
    return ImportMemoryJson(json);
  }

  // Releases this context's native Conversation. The shared Engine is destroyed
  // automatically after the final context releases it. Repeated calls are safe.
  Status Close() {
    if (!model_ || context_id_ == 0) return Status();

    Status status;
    {
      std::lock_guard<std::mutex> lock(model_->mutex);
      if (model_->retained_request &&
          model_->retained_request->conversation == conversation_) {
        return Status(LITERT_LM_STATUS_BUSY,
                      "This context still owns a retained Request; drain it "
                      "before closing the Conversation.");
      }
      status = ReleaseConversationLocked();
      if (!status) return status;
      context_id_ = 0;
      native_synced_ = false;
    }
    model_.reset();
    return status;
  }

 private:
  static constexpr uint64_t kMaxMemoryFileBytes = 256ull * 1024ull * 1024ull;

  static Result<LiteRtLm> CreateContext(
      std::shared_ptr<detail::SharedModelState> model,
      const ContextOptions& options) {
    if (!model || !model->engine || model->engine->handle == nullptr) {
      return Status(LITERT_LM_STATUS_INVALID_STATE,
                    "Shared LiteRT-LM model is not available.");
    }

    LiteRtLm context;
    context.model_ = std::move(model);
    auto copy_view = [](std::string_view source, std::string* destination) {
      if (source.empty()) {
        destination->clear();
      } else {
        destination->assign(source.data(), source.size());
      }
    };
    copy_view(options.tools_json, &context.tools_json_);
    copy_view(options.extra_context_json, &context.extra_context_json_);
    copy_view(options.tool_call_channel, &context.tool_call_channel_);
    copy_view(options.lora_path, &context.lora_path_);
    copy_view(options.audio_lora_path, &context.audio_lora_path_);
    context.conversation_flags_ = options.flags;
    context.default_max_output_tokens_ = options.default_max_output_tokens;
    if (!options.system_prompt.empty()) {
      context.memory_.push_back(
          detail::JsonRoleMessage("system", options.system_prompt));
    }
    if (!options.initial_messages_json.empty()) {
      std::vector<std::string> initial_messages;
      Status parsed = detail::ParseMessages(options.initial_messages_json,
                                            false, true, &initial_messages);
      if (!parsed) return parsed;
      context.memory_.insert(context.memory_.end(),
                             std::make_move_iterator(initial_messages.begin()),
                             std::make_move_iterator(initial_messages.end()));
    }
    {
      std::lock_guard<std::mutex> lock(context.model_->mutex);
      context.context_id_ = context.model_->next_context_id++;
      if (context.context_id_ == 0) {
        context.context_id_ = context.model_->next_context_id++;
      }
      Engine engine(context.model_->engine);
      ConversationOptions native_options;
      native_options.system_prompt = options.system_prompt;
      native_options.tools_json = options.tools_json;
      native_options.initial_messages_json = options.initial_messages_json;
      native_options.extra_context_json = options.extra_context_json;
      native_options.tool_call_channel = options.tool_call_channel;
      native_options.lora_path = options.lora_path;
      native_options.audio_lora_path = options.audio_lora_path;
      native_options.flags = options.flags;
      native_options.default_max_output_tokens =
          options.default_max_output_tokens;
      auto conversation = engine.CreateConversation(native_options);
      if (!conversation) return conversation.status();
      context.conversation_ = conversation->state_;
      context.native_synced_ = true;
    }
    return context;
  }

  Result<RequestResult> AwaitRegisteredLocked(
      Request* request, std::chrono::milliseconds timeout,
      const std::atomic_bool* cancel_requested) {
    if (request == nullptr || !request->valid()) {
      return Status(LITERT_LM_STATUS_INVALID_ARGUMENT,
                    "Request is empty.");
    }
    {
      std::lock_guard<std::mutex> cancel_lock(model_->cancel_mutex);
      if (model_->active_request != nullptr) {
        return Status(LITERT_LM_STATUS_INTERNAL_ERROR,
                      "The shared model already has an active Request.");
      }
      model_->active_request = request->state_->handle;
    }

    if (cancel_requested != nullptr &&
        cancel_requested->load(std::memory_order_acquire)) {
      request->Cancel();
    }

    Status wait = request->Wait(timeout);
    if (!wait) {
      request->Cancel();
      Status quiescent = request->Wait(std::chrono::seconds(30));
      if (!quiescent) {
        // Transfer ownership out of the local Request. The Engine remains BUSY
        // by design until DrainCurrentInference succeeds; no handle is leaked.
        model_->retained_request = request->state_;
        request->state_.reset();
        return wait;
      }

      // The original timeout/error remains the public result, but cleanup is
      // now safe and deterministic.
      {
        std::lock_guard<std::mutex> cancel_lock(model_->cancel_mutex);
        model_->active_request = nullptr;
      }
      Status close = request->Close();
      if (!close) return close;
      return wait;
    }

    {
      std::lock_guard<std::mutex> cancel_lock(model_->cancel_mutex);
      model_->active_request = nullptr;
    }
    auto result = request->GetResult();
    Status close = request->Close();
    if (!close) return close;
    if (!result) return result.status();
    return result.TakeValue();
  }

  Status EnsureConversationLocked(bool* can_continue) {
    if (can_continue == nullptr) {
      return Status(LITERT_LM_STATUS_INVALID_ARGUMENT,
                    "Continuation output is null.");
    }
    if (model_->retained_request) {
      return Status(LITERT_LM_STATUS_BUSY,
                    "A timed-out Request is still retained; cancel and drain "
                    "it before the next Ask.");
    }
    *can_continue = native_synced_ && conversation_ &&
                    conversation_->handle != nullptr;
    if (*can_continue) return Status();

    auto conversation = CreateNativeConversationLocked(memory_);
    if (!conversation) return conversation.status();
    Status release = ReleaseConversationLocked();
    if (!release) return release;
    conversation_ = conversation->state_;
    native_synced_ = true;
    *can_continue = true;
    return Status();
  }

  Result<Conversation> CreateNativeConversationLocked(
      const std::vector<std::string>& history) {
    Engine engine(model_->engine);
    std::string history_json = detail::JoinMessages(history);
    ConversationOptions options;
    options.tools_json = tools_json_;
    options.initial_messages_json = history_json;
    options.extra_context_json = extra_context_json_;
    options.tool_call_channel = tool_call_channel_;
    options.lora_path = lora_path_;
    options.audio_lora_path = audio_lora_path_;
    options.flags = conversation_flags_;
    options.default_max_output_tokens = default_max_output_tokens_;
    return engine.CreateConversation(options);
  }

  Status ReleaseConversationLocked() {
    if (!conversation_ || conversation_->handle == nullptr) {
      conversation_.reset();
      return Status();
    }
    LiteRtLm_ErrorInfo error = detail::ErrorBuffer();
    LiteRtLm_Status code = conversation_->engine->api.destroy_conversation(
        conversation_->handle, &error);
    if (code != LITERT_LM_STATUS_OK) return detail::CallStatus(code, error);
    conversation_->handle = nullptr;
    conversation_.reset();
    return Status();
  }

  Status CommitMemory(std::vector<std::string> next) {
    if (!valid()) {
      return Status(LITERT_LM_STATUS_INVALID_STATE,
                    "LiteRtLm context is closed.");
    }
    std::lock_guard<std::mutex> lock(model_->mutex);
    return CommitMemoryLocked(std::move(next));
  }

  Status CommitMemoryLocked(std::vector<std::string> next) {
    auto replacement = CreateNativeConversationLocked(next);
    if (!replacement) return replacement.status();
    Status release = ReleaseConversationLocked();
    if (!release) return release;
    conversation_ = replacement->state_;
    memory_.swap(next);
    native_synced_ = true;
    return Status();
  }

  void MoveFrom(LiteRtLm&& other) noexcept {
    model_ = std::move(other.model_);
    conversation_ = std::move(other.conversation_);
    context_id_ = other.context_id_;
    tools_json_ = std::move(other.tools_json_);
    extra_context_json_ = std::move(other.extra_context_json_);
    tool_call_channel_ = std::move(other.tool_call_channel_);
    lora_path_ = std::move(other.lora_path_);
    audio_lora_path_ = std::move(other.audio_lora_path_);
    conversation_flags_ = other.conversation_flags_;
    default_max_output_tokens_ = other.default_max_output_tokens_;
    memory_ = std::move(other.memory_);
    native_synced_ = other.native_synced_;
    other.context_id_ = 0;
    other.native_synced_ = false;
  }

  std::shared_ptr<detail::SharedModelState> model_;
  std::shared_ptr<detail::ConversationState> conversation_;
  uint64_t context_id_ = 0;
  std::string tools_json_;
  std::string extra_context_json_;
  std::string tool_call_channel_;
  std::string lora_path_;
  std::string audio_lora_path_;
  uint32_t conversation_flags_ =
      LITERT_LM_CONVERSATION_RETURN_ERROR_ON_PARSE_FAILURE;
  uint32_t default_max_output_tokens_ = 0;
  std::vector<std::string> memory_;
  bool native_synced_ = false;
};

}  // namespace litertlm

#endif  // LITERT_LM_RUNTIME_ENGINE_LITERT_LM_HPP_
