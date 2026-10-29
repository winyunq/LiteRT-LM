// Copyright (c) 2025-2026 Winyunq. All rights reserved.
//
// Header-only, exception-free C++ ownership and overloads for LiteRT-LM's
// complete official C API. Every wrapper exposes get()/release(), so native
// consumers can call newly added upstream functions without waiting for this
// convenience layer to grow another method.
#ifndef LITERT_LM_RUNTIME_ENGINE_LITERT_LM_NATIVE_HPP_
#define LITERT_LM_RUNTIME_ENGINE_LITERT_LM_NATIVE_HPP_

#include "runtime/engine/litert_lm_native_api.h"

#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace litertlm::native {

enum class Backend { Cpu, Gpu, Npu };

inline const char* BackendName(Backend backend) {
  switch (backend) {
    case Backend::Cpu:
      return "cpu";
    case Backend::Gpu:
      return "gpu";
    case Backend::Npu:
      return "npu";
  }
  return "cpu";
}

namespace detail {

inline std::string Z(std::string_view value) {
  return std::string(value.data(), value.size());
}

template <typename T>
class Owner {
 public:
  Owner() = default;
  Owner(const Owner&) = delete;
  Owner& operator=(const Owner&) = delete;
  Owner(Owner&& other) noexcept
      : value_(std::exchange(other.value_, nullptr)),
        deleter_(std::exchange(other.deleter_, nullptr)) {}
  Owner& operator=(Owner&& other) noexcept {
    if (this == &other) return *this;
    reset();
    value_ = std::exchange(other.value_, nullptr);
    deleter_ = std::exchange(other.deleter_, nullptr);
    return *this;
  }
  ~Owner() { reset(); }

  T* get() const { return value_; }
  explicit operator bool() const { return value_ != nullptr; }
  T* release() {
    deleter_ = nullptr;
    return std::exchange(value_, nullptr);
  }
  void reset() {
    if (value_ != nullptr && deleter_ != nullptr) deleter_(value_);
    value_ = nullptr;
    deleter_ = nullptr;
  }

 protected:
  Owner(T* value, void (*deleter)(T*)) : value_(value), deleter_(deleter) {}

 private:
  T* value_ = nullptr;
  void (*deleter_)(T*) = nullptr;
};

}  // namespace detail

class SamplerParameters final : public detail::Owner<LiteRtLmSamplerParams> {
 public:
  SamplerParameters() = default;
  static SamplerParameters Create(
      LiteRtLmSamplerType type = kLiteRtLmSamplerTypeTopP) {
    return SamplerParameters(litert_lm_sampler_params_create(type));
  }
  SamplerParameters& TopK(int value) {
    if (get()) litert_lm_sampler_params_set_top_k(get(), value);
    return *this;
  }
  SamplerParameters& TopP(float value) {
    if (get()) litert_lm_sampler_params_set_top_p(get(), value);
    return *this;
  }
  SamplerParameters& Temperature(float value) {
    if (get()) litert_lm_sampler_params_set_temperature(get(), value);
    return *this;
  }
  SamplerParameters& Seed(int32_t value) {
    if (get()) litert_lm_sampler_params_set_seed(get(), value);
    return *this;
  }

 private:
  explicit SamplerParameters(LiteRtLmSamplerParams* value)
      : Owner(value, &litert_lm_sampler_params_delete) {}
};

class SessionConfig final : public detail::Owner<LiteRtLmSessionConfig> {
 public:
  SessionConfig() = default;
  static SessionConfig Create() {
    return SessionConfig(litert_lm_session_config_create());
  }
  SessionConfig& MaxOutputTokens(int value) {
    if (get()) litert_lm_session_config_set_max_output_tokens(get(), value);
    return *this;
  }
  SessionConfig& ApplyPromptTemplate(bool value) {
    if (get()) litert_lm_session_config_set_apply_prompt_template(get(), value);
    return *this;
  }
  SessionConfig& Sampler(const SamplerParameters& value) {
    if (get() && value.get()) {
      litert_lm_session_config_set_sampler_params(get(), value.get());
    }
    return *this;
  }
  bool TrySetLoraPath(std::string_view value) {
    const std::string text = detail::Z(value);
    return get() && litert_lm_session_config_set_lora_path(get(), text.c_str()) == 0;
  }
  bool TrySetAudioLoraPath(std::string_view value) {
    const std::string text = detail::Z(value);
    return get() &&
           litert_lm_session_config_set_audio_lora_path(get(), text.c_str()) == 0;
  }

 private:
  explicit SessionConfig(LiteRtLmSessionConfig* value)
      : Owner(value, &litert_lm_session_config_delete) {}
};

class ConversationConfig final
    : public detail::Owner<LiteRtLmConversationConfig> {
 public:
  ConversationConfig() = default;
  static ConversationConfig Create() {
    return ConversationConfig(litert_lm_conversation_config_create());
  }
  ConversationConfig& Session(const SessionConfig& value) {
    if (get() && value.get()) {
      litert_lm_conversation_config_set_session_config(get(), value.get());
    }
    return *this;
  }
  ConversationConfig& SystemMessage(std::string_view value) {
    const std::string text = detail::Z(value);
    if (get()) {
      litert_lm_conversation_config_set_system_message(get(), text.c_str());
    }
    return *this;
  }
  ConversationConfig& Tools(std::string_view value) {
    const std::string text = detail::Z(value);
    if (get()) litert_lm_conversation_config_set_tools(get(), text.c_str());
    return *this;
  }
  ConversationConfig& Messages(std::string_view value) {
    const std::string text = detail::Z(value);
    if (get()) litert_lm_conversation_config_set_messages(get(), text.c_str());
    return *this;
  }
  ConversationConfig& ExtraContext(std::string_view value) {
    const std::string text = detail::Z(value);
    if (get()) {
      litert_lm_conversation_config_set_extra_context(get(), text.c_str());
    }
    return *this;
  }
  ConversationConfig& ConstrainedDecoding(bool value) {
    if (get()) {
      litert_lm_conversation_config_set_enable_constrained_decoding(get(), value);
    }
    return *this;
  }
  ConversationConfig& FilterChannelContentFromKvCache(bool value) {
    if (get()) {
      litert_lm_conversation_config_set_filter_channel_content_from_kv_cache(
          get(), value);
    }
    return *this;
  }
  ConversationConfig& StreamToolCalls(
      bool value, std::string_view channel = "tool_call") {
    const std::string text = detail::Z(channel);
    if (get()) {
      litert_lm_conversation_config_set_stream_tool_calls(get(), value,
                                                          text.c_str());
    }
    return *this;
  }

 private:
  explicit ConversationConfig(LiteRtLmConversationConfig* value)
      : Owner(value, &litert_lm_conversation_config_delete) {}
};

class ConversationOptionalArgs final
    : public detail::Owner<LiteRtLmConversationOptionalArgs> {
 public:
  ConversationOptionalArgs() = default;
  static ConversationOptionalArgs Create() {
    return ConversationOptionalArgs(
        litert_lm_conversation_optional_args_create());
  }
  ConversationOptionalArgs& VisualTokenBudget(int value) {
    if (get()) {
      litert_lm_conversation_optional_args_set_visual_token_budget(get(), value);
    }
    return *this;
  }
  ConversationOptionalArgs& MaxOutputTokens(int value) {
    if (get()) {
      litert_lm_conversation_optional_args_set_max_output_tokens(get(), value);
    }
    return *this;
  }

 private:
  explicit ConversationOptionalArgs(LiteRtLmConversationOptionalArgs* value)
      : Owner(value, &litert_lm_conversation_optional_args_delete) {}
};

class EngineSettings final : public detail::Owner<LiteRtLmEngineSettings> {
 public:
  EngineSettings() = default;

  static EngineSettings Create(std::string_view model_path,
                               Backend backend = Backend::Gpu) {
    return Create(model_path, backend, std::nullopt, std::nullopt);
  }
  static EngineSettings Create(std::string_view model_path, Backend backend,
                               std::optional<Backend> vision_backend,
                               std::optional<Backend> audio_backend) {
    const std::string path = detail::Z(model_path);
    return EngineSettings(litert_lm_engine_settings_create(
        path.c_str(), BackendName(backend),
        vision_backend ? BackendName(*vision_backend) : nullptr,
        audio_backend ? BackendName(*audio_backend) : nullptr));
  }
  static EngineSettings CreateFromRawFileDescriptor(
      int descriptor, Backend backend = Backend::Gpu,
      std::optional<Backend> vision_backend = std::nullopt,
      std::optional<Backend> audio_backend = std::nullopt) {
    return EngineSettings(
        litert_lm_engine_settings_create_from_raw_file_descriptor(
            descriptor, BackendName(backend),
            vision_backend ? BackendName(*vision_backend) : nullptr,
            audio_backend ? BackendName(*audio_backend) : nullptr));
  }

  EngineSettings& MaxTokens(int value) {
    if (get()) litert_lm_engine_settings_set_max_num_tokens(get(), value);
    return *this;
  }
  EngineSettings& Threads(int value) {
    if (get()) litert_lm_engine_settings_set_num_threads(get(), value);
    return *this;
  }
  EngineSettings& AudioThreads(int value) {
    if (get()) litert_lm_engine_settings_set_audio_num_threads(get(), value);
    return *this;
  }
  EngineSettings& ParallelFileLoading(bool value) {
    if (get()) {
      litert_lm_engine_settings_set_parallel_file_section_loading(get(), value);
    }
    return *this;
  }
  EngineSettings& MaxImages(int value) {
    if (get()) litert_lm_engine_settings_set_max_num_images(get(), value);
    return *this;
  }
  EngineSettings& CacheDirectory(std::string_view value) {
    const std::string text = detail::Z(value);
    if (get()) litert_lm_engine_settings_set_cache_dir(get(), text.c_str());
    return *this;
  }
  EngineSettings& DispatchLibraryDirectory(std::string_view value) {
    const std::string text = detail::Z(value);
    if (get()) {
      litert_lm_engine_settings_set_litert_dispatch_lib_dir(get(), text.c_str());
    }
    return *this;
  }
  EngineSettings& ActivationDataType(int value) {
    if (get()) litert_lm_engine_settings_set_activation_data_type(get(), value);
    return *this;
  }
  EngineSettings& PrefillChunkSize(int value) {
    if (get()) litert_lm_engine_settings_set_prefill_chunk_size(get(), value);
    return *this;
  }
  EngineSettings& EnableBenchmark() {
    if (get()) litert_lm_engine_settings_enable_benchmark(get());
    return *this;
  }
  EngineSettings& BenchmarkPrefillTokens(int value) {
    if (get()) litert_lm_engine_settings_set_num_prefill_tokens(get(), value);
    return *this;
  }
  EngineSettings& BenchmarkDecodeTokens(int value) {
    if (get()) litert_lm_engine_settings_set_num_decode_tokens(get(), value);
    return *this;
  }
  EngineSettings& SpeculativeDecoding(bool value) {
    if (get()) {
      litert_lm_engine_settings_set_enable_speculative_decoding(get(), value);
    }
    return *this;
  }
  EngineSettings& LoraRank(int value) {
    if (get()) litert_lm_engine_settings_set_lora_rank(get(), value);
    return *this;
  }
  bool TrySetSupportedLoraRanks(std::span<const int> values) {
    return get() && litert_lm_engine_settings_set_supported_lora_ranks(
                        get(), values.data(), values.size()) == 0;
  }
  EngineSettings& AudioLoraRank(int value) {
    if (get()) litert_lm_engine_settings_set_audio_lora_rank(get(), value);
    return *this;
  }
  bool TrySetSupportedAudioLoraRanks(std::span<const int> values) {
    return get() && litert_lm_engine_settings_set_supported_audio_lora_ranks(
                        get(), values.data(), values.size()) == 0;
  }

 private:
  explicit EngineSettings(LiteRtLmEngineSettings* value)
      : Owner(value, &litert_lm_engine_settings_delete) {}
};

class InputData final : public detail::Owner<LiteRtLmInputData> {
 public:
  InputData() = default;
  static InputData Create(LiteRtLmInputDataType type, const void* data,
                          size_t size) {
    return InputData(litert_lm_input_data_create(type, data, size));
  }
  static InputData Text(std::string_view value) {
    return Create(kLiteRtLmInputDataTypeText, value.data(), value.size());
  }
  static InputData Image(std::span<const std::byte> value) {
    return Create(kLiteRtLmInputDataTypeImage, value.data(), value.size());
  }
  static InputData ImageEnd() {
    return Create(kLiteRtLmInputDataTypeImageEnd, nullptr, 0);
  }
  static InputData Audio(std::span<const std::byte> value) {
    return Create(kLiteRtLmInputDataTypeAudio, value.data(), value.size());
  }
  static InputData AudioEnd() {
    return Create(kLiteRtLmInputDataTypeAudioEnd, nullptr, 0);
  }

 private:
  explicit InputData(LiteRtLmInputData* value)
      : Owner(value, &litert_lm_input_data_delete) {}
};

class Responses final : public detail::Owner<LiteRtLmResponses> {
 public:
  Responses() = default;
  int size() const {
    return get() ? litert_lm_responses_get_num_candidates(get()) : 0;
  }
  std::string_view Text(int index = 0) const {
    const char* value =
        get() ? litert_lm_responses_get_response_text_at(get(), index) : nullptr;
    return value == nullptr ? std::string_view() : std::string_view(value);
  }
  std::optional<float> Score(int index) const {
    if (!get() || !litert_lm_responses_has_score_at(get(), index)) {
      return std::nullopt;
    }
    return litert_lm_responses_get_score_at(get(), index);
  }
  std::optional<int> TokenLength(int index) const {
    if (!get() || !litert_lm_responses_has_token_length_at(get(), index)) {
      return std::nullopt;
    }
    return litert_lm_responses_get_token_length_at(get(), index);
  }
  std::span<const float> TokenScores(int index) const {
    if (!get() || !litert_lm_responses_has_token_scores_at(get(), index)) {
      return {};
    }
    const int count =
        litert_lm_responses_get_num_token_scores_at(get(), index);
    const float* values = litert_lm_responses_get_token_scores_at(get(), index);
    return count <= 0 || values == nullptr
               ? std::span<const float>()
               : std::span<const float>(values, static_cast<size_t>(count));
  }

 private:
  friend class Session;
  explicit Responses(LiteRtLmResponses* value)
      : Owner(value, &litert_lm_responses_delete) {}
};

class BenchmarkInfo final : public detail::Owner<LiteRtLmBenchmarkInfo> {
 public:
  BenchmarkInfo() = default;
  double TimeToFirstTokenSeconds() const {
    return get() ? litert_lm_benchmark_info_get_time_to_first_token(get()) : 0.0;
  }
  double InitializationSeconds() const {
    return get()
               ? litert_lm_benchmark_info_get_total_init_time_in_second(get())
               : 0.0;
  }
  int PrefillTurns() const {
    return get() ? litert_lm_benchmark_info_get_num_prefill_turns(get()) : 0;
  }
  int DecodeTurns() const {
    return get() ? litert_lm_benchmark_info_get_num_decode_turns(get()) : 0;
  }
  int PrefillTokens(int index) const {
    return get()
               ? litert_lm_benchmark_info_get_prefill_token_count_at(get(), index)
               : 0;
  }
  int DecodeTokens(int index) const {
    return get()
               ? litert_lm_benchmark_info_get_decode_token_count_at(get(), index)
               : 0;
  }
  double PrefillTokensPerSecond(int index) const {
    return get() ? litert_lm_benchmark_info_get_prefill_tokens_per_sec_at(
                       get(), index)
                 : 0.0;
  }
  double DecodeTokensPerSecond(int index) const {
    return get() ? litert_lm_benchmark_info_get_decode_tokens_per_sec_at(
                       get(), index)
                 : 0.0;
  }

 private:
  friend class Session;
  friend class Conversation;
  explicit BenchmarkInfo(LiteRtLmBenchmarkInfo* value)
      : Owner(value, &litert_lm_benchmark_info_delete) {}
};

class JsonResponse final : public detail::Owner<LiteRtLmJsonResponse> {
 public:
  JsonResponse() = default;
  std::string_view json() const {
    const char* value = get() ? litert_lm_json_response_get_string(get()) : nullptr;
    return value == nullptr ? std::string_view() : std::string_view(value);
  }

 private:
  friend class Conversation;
  explicit JsonResponse(LiteRtLmJsonResponse* value)
      : Owner(value, &litert_lm_json_response_delete) {}
};

class TokenUnion final : public detail::Owner<LiteRtLmTokenUnion> {
 public:
  TokenUnion() = default;
  LiteRtLmTokenUnionType type() const {
    return get() ? litert_lm_token_union_get_type(get())
                 : kLiteRtLmTokenUnionTypeString;
  }
  std::string_view String() const {
    const char* value = get() ? litert_lm_token_union_get_string(get()) : nullptr;
    return value == nullptr ? std::string_view() : std::string_view(value);
  }
  std::span<const int> Ids() const {
    const int* values = nullptr;
    size_t size = 0;
    return get() && litert_lm_token_union_get_ids(get(), &values, &size) == 0 &&
                   values != nullptr
               ? std::span<const int>(values, size)
               : std::span<const int>();
  }

 private:
  friend class Engine;
  friend class TokenUnions;
  explicit TokenUnion(LiteRtLmTokenUnion* value)
      : Owner(value, &litert_lm_token_union_delete) {}
};

class TokenUnions final : public detail::Owner<LiteRtLmTokenUnions> {
 public:
  TokenUnions() = default;
  size_t size() const {
    return get() ? litert_lm_token_unions_get_num_tokens(get()) : 0;
  }
  TokenUnion At(size_t index) const {
    return TokenUnion(get() ? litert_lm_token_unions_get_token_at(get(), index)
                            : nullptr);
  }

 private:
  friend class Engine;
  explicit TokenUnions(LiteRtLmTokenUnions* value)
      : Owner(value, &litert_lm_token_unions_delete) {}
};

class Session final : public detail::Owner<LiteRtLmSession> {
 public:
  Session() = default;
  void Cancel() {
    if (get()) litert_lm_session_cancel_process(get());
  }
  int Prefill(std::span<const InputData> inputs) {
    const std::vector<const LiteRtLmInputData*> pointers = Pointers(inputs);
    return get() ? litert_lm_session_run_prefill(get(), pointers.data(),
                                                 pointers.size())
                 : -1;
  }
  Responses Decode() {
    return Responses(get() ? litert_lm_session_run_decode(get()) : nullptr);
  }
  int DecodeAsync(LiteRtLmStreamCallback callback, void* user_data = nullptr) {
    return get() ? litert_lm_session_run_decode_async(get(), callback, user_data)
                 : -1;
  }
  Responses Score(std::span<const std::string_view> targets,
                  bool store_token_lengths = false) {
    std::vector<std::string> storage;
    storage.reserve(targets.size());
    for (std::string_view target : targets) storage.emplace_back(target);
    std::vector<const char*> pointers;
    pointers.reserve(storage.size());
    for (const std::string& target : storage) pointers.push_back(target.c_str());
    return Responses(
        get() ? litert_lm_session_run_text_scoring(
                    get(), pointers.data(), pointers.size(), store_token_lengths)
              : nullptr);
  }
  Responses Generate(std::span<const InputData> inputs) {
    const std::vector<const LiteRtLmInputData*> pointers = Pointers(inputs);
    return Responses(get() ? litert_lm_session_generate_content(
                                 get(), pointers.data(), pointers.size())
                           : nullptr);
  }
  int Generate(std::span<const InputData> inputs,
               LiteRtLmStreamCallback callback, void* user_data = nullptr) {
    const std::vector<const LiteRtLmInputData*> pointers = Pointers(inputs);
    return get() ? litert_lm_session_generate_content_stream(
                       get(), pointers.data(), pointers.size(), callback,
                       user_data)
                 : -1;
  }
  BenchmarkInfo Benchmark() {
    return BenchmarkInfo(get() ? litert_lm_session_get_benchmark_info(get())
                               : nullptr);
  }

 private:
  friend class Engine;
  explicit Session(LiteRtLmSession* value)
      : Owner(value, &litert_lm_session_delete) {}
  static std::vector<const LiteRtLmInputData*> Pointers(
      std::span<const InputData> inputs) {
    std::vector<const LiteRtLmInputData*> result;
    result.reserve(inputs.size());
    for (const InputData& input : inputs) result.push_back(input.get());
    return result;
  }
};

class Conversation final : public detail::Owner<LiteRtLmConversation> {
 public:
  Conversation() = default;
  Conversation Clone() {
    return Conversation(get() ? litert_lm_conversation_clone(get()) : nullptr);
  }
  JsonResponse Send(std::string_view message_json) {
    return SendNative(message_json, {}, nullptr);
  }
  JsonResponse Send(std::string_view message_json,
                    const ConversationOptionalArgs& args) {
    return SendNative(message_json, {}, args.get());
  }
  JsonResponse Send(std::string_view message_json,
                    std::string_view extra_context_json,
                    const ConversationOptionalArgs* args = nullptr) {
    return SendNative(message_json, extra_context_json,
                      args ? args->get() : nullptr);
  }
  int Send(std::string_view message_json, LiteRtLmStreamCallback callback,
           void* user_data = nullptr) {
    return Send(message_json, {}, nullptr, callback, user_data);
  }
  int Send(std::string_view message_json,
           std::string_view extra_context_json,
           const ConversationOptionalArgs* args,
           LiteRtLmStreamCallback callback, void* user_data = nullptr) {
    const std::string message = detail::Z(message_json);
    const std::string context = detail::Z(extra_context_json);
    return get() ? litert_lm_conversation_send_message_stream(
                       get(), message.c_str(),
                       context.empty() ? nullptr : context.c_str(),
                       args ? args->get() : nullptr, callback, user_data)
                 : -1;
  }
  std::string_view RenderMessage(std::string_view message_json) {
    const std::string message = detail::Z(message_json);
    const char* value = get() ? litert_lm_conversation_render_message_to_string(
                                    get(), message.c_str())
                              : nullptr;
    return value == nullptr ? std::string_view() : std::string_view(value);
  }
  std::string_view RenderPreface() {
    const char* value =
        get() ? litert_lm_conversation_render_preface_to_string(get()) : nullptr;
    return value == nullptr ? std::string_view() : std::string_view(value);
  }
  void Cancel() {
    if (get()) litert_lm_conversation_cancel_process(get());
  }
  BenchmarkInfo Benchmark() {
    return BenchmarkInfo(
        get() ? litert_lm_conversation_get_benchmark_info(get()) : nullptr);
  }
  int TokenCount() const {
    return get() ? litert_lm_conversation_get_token_count(get()) : -1;
  }

 private:
  friend class Engine;
  explicit Conversation(LiteRtLmConversation* value)
      : Owner(value, &litert_lm_conversation_delete) {}
  JsonResponse SendNative(std::string_view message_json,
                          std::string_view extra_context_json,
                          const LiteRtLmConversationOptionalArgs* args) {
    const std::string message = detail::Z(message_json);
    const std::string context = detail::Z(extra_context_json);
    return JsonResponse(
        get() ? litert_lm_conversation_send_message(
                    get(), message.c_str(),
                    context.empty() ? nullptr : context.c_str(), args)
              : nullptr);
  }
};

class Engine final : public detail::Owner<LiteRtLmEngine> {
 public:
  Engine() = default;
  static Engine Create(const EngineSettings& settings) {
    return Engine(settings.get() ? litert_lm_engine_create(settings.get())
                                 : nullptr);
  }
  static Engine Create(std::string_view model_path,
                       Backend backend = Backend::Gpu) {
    EngineSettings settings = EngineSettings::Create(model_path, backend);
    return Create(settings);
  }
  Session NewSession() {
    return Session(get() ? litert_lm_engine_create_session(get(), nullptr)
                         : nullptr);
  }
  Session NewSession(SessionConfig& config) {
    return Session(get() && config.get()
                       ? litert_lm_engine_create_session(get(), config.get())
                       : nullptr);
  }
  Conversation NewConversation() {
    return Conversation(get() ? litert_lm_conversation_create(get(), nullptr)
                              : nullptr);
  }
  Conversation NewConversation(ConversationConfig& config) {
    return Conversation(get() && config.get()
                            ? litert_lm_conversation_create(get(), config.get())
                            : nullptr);
  }
  std::optional<std::vector<int>> Tokenize(std::string_view text) {
    const std::string input = detail::Z(text);
    LiteRtLmTokenizeResult* result =
        get() ? litert_lm_engine_tokenize(get(), input.c_str()) : nullptr;
    if (result == nullptr) return std::nullopt;
    const size_t size = litert_lm_tokenize_result_get_num_tokens(result);
    const int* values = litert_lm_tokenize_result_get_tokens(result);
    std::vector<int> output;
    if (size != 0 && values == nullptr) {
      litert_lm_tokenize_result_delete(result);
      return std::nullopt;
    }
    if (size != 0) output.assign(values, values + size);
    litert_lm_tokenize_result_delete(result);
    return output;
  }
  std::optional<std::string> Detokenize(std::span<const int> tokens) {
    LiteRtLmDetokenizeResult* result =
        get() ? litert_lm_engine_detokenize(get(), tokens.data(), tokens.size())
              : nullptr;
    if (result == nullptr) return std::nullopt;
    const char* value = litert_lm_detokenize_result_get_string(result);
    std::string output = value == nullptr ? "" : value;
    litert_lm_detokenize_result_delete(result);
    return output;
  }
  TokenUnion StartToken() {
    return TokenUnion(get() ? litert_lm_engine_get_start_token(get()) : nullptr);
  }
  TokenUnions StopTokens() {
    return TokenUnions(get() ? litert_lm_engine_get_stop_tokens(get()) : nullptr);
  }

 private:
  explicit Engine(LiteRtLmEngine* value)
      : Owner(value, &litert_lm_engine_delete) {}
};

inline void SetMinimumLogLevel(int level) {
  litert_lm_set_min_log_level(level);
}

}  // namespace litertlm::native

#endif  // LITERT_LM_RUNTIME_ENGINE_LITERT_LM_NATIVE_HPP_
