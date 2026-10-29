// Copyright (c) 2025-2026 Winyunq. All rights reserved.

#include "runtime/engine/litert_lm.hpp"

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#include "gtest/gtest.h"

namespace litertlm {
namespace {

struct FakeEngineState {
  uint32_t conversations = 0;
};

struct FakeConversationState {
  FakeEngineState* engine = nullptr;
};

LiteRtLm_Status LITERT_LM_CALL FakeCreateEngine(
    const LiteRtLm_EngineConfig*, LiteRtLm_Engine* out_engine,
    LiteRtLm_ErrorInfo*) {
  if (out_engine == nullptr) return LITERT_LM_STATUS_INVALID_ARGUMENT;
  *out_engine = reinterpret_cast<LiteRtLm_Engine>(new FakeEngineState());
  return LITERT_LM_STATUS_OK;
}

LiteRtLm_Status LITERT_LM_CALL FakeGetEngineInfo(
    LiteRtLm_Engine engine, LiteRtLm_EngineInfo* out_info,
    LiteRtLm_ErrorInfo*) {
  if (engine == nullptr || out_info == nullptr) {
    return LITERT_LM_STATUS_INVALID_ARGUMENT;
  }
  const uint32_t capacity = out_info->struct_size;
  LiteRtLm_EngineInfo value{};
  value.struct_size = sizeof(value);
  value.configured_backend = LITERT_LM_BACKEND_GPU;
  value.resolved_backend = LITERT_LM_BACKEND_GPU;
  value.configured_sampler_backend = LITERT_LM_BACKEND_GPU;
  value.conversation_count =
      reinterpret_cast<FakeEngineState*>(engine)->conversations;
  std::memcpy(out_info, &value, std::min<size_t>(capacity, sizeof(value)));
  return LITERT_LM_STATUS_OK;
}

LiteRtLm_Status LITERT_LM_CALL FakeDestroyEngine(
    LiteRtLm_Engine engine, LiteRtLm_ErrorInfo*) {
  auto* state = reinterpret_cast<FakeEngineState*>(engine);
  if (state == nullptr) return LITERT_LM_STATUS_INVALID_ARGUMENT;
  if (state->conversations != 0) return LITERT_LM_STATUS_BUSY;
  delete state;
  return LITERT_LM_STATUS_OK;
}

LiteRtLm_Status LITERT_LM_CALL FakeCreateConversation(
    LiteRtLm_Engine engine, const LiteRtLm_ConversationConfig*,
    LiteRtLm_Conversation* out_conversation, LiteRtLm_ErrorInfo*) {
  auto* state = reinterpret_cast<FakeEngineState*>(engine);
  if (state == nullptr || out_conversation == nullptr) {
    return LITERT_LM_STATUS_INVALID_ARGUMENT;
  }
  auto* conversation = new FakeConversationState();
  conversation->engine = state;
  ++state->conversations;
  *out_conversation =
      reinterpret_cast<LiteRtLm_Conversation>(conversation);
  return LITERT_LM_STATUS_OK;
}

LiteRtLm_Status LITERT_LM_CALL FakeResetConversation(
    LiteRtLm_Conversation conversation, LiteRtLm_ErrorInfo*) {
  return conversation == nullptr ? LITERT_LM_STATUS_INVALID_ARGUMENT
                                 : LITERT_LM_STATUS_OK;
}

LiteRtLm_Status LITERT_LM_CALL FakeDestroyConversation(
    LiteRtLm_Conversation conversation, LiteRtLm_ErrorInfo*) {
  auto* state = reinterpret_cast<FakeConversationState*>(conversation);
  if (state == nullptr) return LITERT_LM_STATUS_INVALID_ARGUMENT;
  --state->engine->conversations;
  delete state;
  return LITERT_LM_STATUS_OK;
}

LiteRtLm_Status LITERT_LM_CALL FakeGetApi(
    uint32_t requested_major, uint32_t out_api_size, LiteRtLm_Api* out_api) {
  if (requested_major != LITERT_LM_ABI_MAJOR || out_api == nullptr ||
      out_api_size < offsetof(LiteRtLm_Api, clone_conversation)) {
    return LITERT_LM_STATUS_ABI_MISMATCH;
  }
  LiteRtLm_Api api{};
  api.struct_size = sizeof(api);
  api.abi_major = LITERT_LM_ABI_MAJOR;
  api.abi_minor = LITERT_LM_ABI_MINOR;
  api.create_engine = &FakeCreateEngine;
  api.get_engine_info = &FakeGetEngineInfo;
  api.destroy_engine = &FakeDestroyEngine;
  api.create_conversation = &FakeCreateConversation;
  api.reset_conversation = &FakeResetConversation;
  api.destroy_conversation = &FakeDestroyConversation;
  std::memcpy(out_api, &api, std::min<size_t>(out_api_size, sizeof(api)));
  return LITERT_LM_STATUS_OK;
}

template <typename T>
concept HasVisionOption = requires(T value) { value.vision; };

template <typename T>
concept HasAudioOption = requires(T value) { value.audio; };

TEST(LiteRtLmCppFacadeTest, PublicEngineOptionsAreTextOnly) {
  static_assert(!HasVisionOption<EngineOptions>);
  static_assert(!HasAudioOption<EngineOptions>);
  static_assert(!HasVisionOption<LiteRtLmOptions>);
  static_assert(!HasAudioOption<LiteRtLmOptions>);
  SUCCEED();
}

TEST(LiteRtLmCppFacadeTest, EscapesRoleMessages) {
  EXPECT_EQ(detail::JsonRoleMessage("system", "line\n\"quoted\""),
            "{\"role\":\"system\",\"content\":\"line\\n\\\"quoted\\\"\"}");
}

TEST(LiteRtLmCppFacadeTest, SplitsNestedMessageArrayWithoutRewritingIt) {
  const std::string json =
      R"([{"role":"user","content":[{"type":"text","text":"hi"}]},{"role":"assistant","tool_calls":[{"function":{"name":"vote","arguments":"{\"seat\":7}"}}]}])";
  std::vector<std::string> messages;
  Status status = detail::ParseMessages(json, false, false, &messages);
  ASSERT_TRUE(status) << status.message();
  ASSERT_EQ(messages.size(), 2u);
  EXPECT_EQ(detail::JoinMessages(messages), json);
}

TEST(LiteRtLmCppFacadeTest, RejectsMalformedOrNonObjectMessages) {
  std::vector<std::string> messages;
  EXPECT_FALSE(detail::ParseMessages("[{\"role\":}]", false, false,
                                     &messages));
  EXPECT_FALSE(detail::ParseMessages("[1]", false, false, &messages));
  EXPECT_FALSE(detail::ParseMessages("{} trailing", true, false, &messages));
  EXPECT_FALSE(detail::ParseMessages("[]", false, false, &messages));
  EXPECT_TRUE(detail::ParseMessages("[]", false, true, &messages));
}

TEST(LiteRtLmCppFacadeTest, EmptyFacadeIsSafeAndReportsInvalidAsk) {
  LiteRtLm model;
  EXPECT_FALSE(model.valid());
  EXPECT_EQ(model.MemorySize(), 0u);
  EXPECT_EQ(model.ExportMemoryJson(), "[]");
  EXPECT_EQ(model.Prepare().code(), LITERT_LM_STATUS_INVALID_STATE);
  EXPECT_FALSE(model.HasRetainedInference());
  EXPECT_TRUE(model.Close());
  auto answer = model.Ask("hello");
  ASSERT_FALSE(answer);
  EXPECT_EQ(answer.status().code(), LITERT_LM_STATUS_INVALID_STATE);
}

TEST(LiteRtLmCppFacadeTest, StatusPreservesNativeDiagnostics) {
  Status status(LITERT_LM_STATUS_BACKEND_ERROR, "gpu failed",
                LITERT_LM_PHASE_ENGINE_CREATE, 17, true);
  EXPECT_EQ(status.code(), LITERT_LM_STATUS_BACKEND_ERROR);
  EXPECT_EQ(status.phase(), LITERT_LM_PHASE_ENGINE_CREATE);
  EXPECT_EQ(status.native_code(), 17);
  EXPECT_TRUE(status.retryable());
}

TEST(LiteRtLmCppFacadeTest, PreciseSamplingIsAnExplicitPerRequestOption) {
  const Sampling precise = Sampling::Precise(73);
  EXPECT_FLOAT_EQ(precise.temperature, 0.1f);
  EXPECT_FLOAT_EQ(precise.top_p, 0.8f);
  EXPECT_EQ(precise.top_k, 20);
  EXPECT_FALSE(precise.random_seed);
  EXPECT_EQ(precise.seed, 73);

  ChatOptions chat;
  AskOptions ask;
  EXPECT_FALSE(chat.sampling.has_value());
  EXPECT_FALSE(ask.sampling.has_value());
  ask.sampling = precise;
  ASSERT_TRUE(ask.sampling.has_value());
  EXPECT_EQ(ask.sampling->seed, 73);

  const LiteRtLm_ChatRequest native = LiteRtLm_DefaultChatRequest();
  EXPECT_EQ(native.struct_size, sizeof(LiteRtLm_ChatRequest));
  EXPECT_EQ(native.flags, 0u);
  EXPECT_FLOAT_EQ(native.sampling.temperature, 1.0f);
}

TEST(LiteRtLmCppFacadeTest, ReasonedDecisionHasAReproducibleThinkingBudget) {
  const Sampling sampling = Sampling::ReasonedDecision(29);
  EXPECT_FLOAT_EQ(sampling.temperature, 0.7f);
  EXPECT_FLOAT_EQ(sampling.top_p, 0.9f);
  EXPECT_EQ(sampling.top_k, 40);
  EXPECT_FALSE(sampling.random_seed);
  EXPECT_EQ(sampling.seed, 29);

  const ChatOptions chat = ChatOptions::ReasonedDecision(768, 29);
  ASSERT_TRUE(chat.sampling.has_value());
  EXPECT_EQ(chat.max_output_tokens, 768u);
  EXPECT_EQ(chat.sampling->seed, 29);

  const AskOptions ask = AskOptions::ReasonedDecision();
  ASSERT_TRUE(ask.sampling.has_value());
  EXPECT_EQ(ask.max_output_tokens, 1024u);
  EXPECT_FLOAT_EQ(ask.sampling->temperature, 0.7f);
}

TEST(LiteRtLmCppFacadeTest, ContextsOwnIndependentNativeConversations) {
  LiteRtLmOptions options;
  options.model_path = "fake.litertlm";
  options.system_prompt = "root";
  auto created = LiteRtLm::Create(options, &FakeGetApi);
  ASSERT_TRUE(created) << created.status().message();
  LiteRtLm root(created.TakeValue());

  ContextOptions second_options;
  second_options.system_prompt = "second";
  auto new_context = root.NewContext(second_options);
  ASSERT_TRUE(new_context) << new_context.status().message();
  LiteRtLm second(new_context.TakeValue());

  auto info = root.GetModelInfo();
  ASSERT_TRUE(info) << info.status().message();
  EXPECT_EQ(info->conversation_count, 2u);
  EXPECT_EQ(root.ExportMemoryJson(),
            R"([{"role":"system","content":"root"}])");
  EXPECT_EQ(second.ExportMemoryJson(),
            R"([{"role":"system","content":"second"}])");

  EXPECT_TRUE(second.Close());
  info = root.GetModelInfo();
  ASSERT_TRUE(info);
  EXPECT_EQ(info->conversation_count, 1u);
  EXPECT_TRUE(root.Close());
}

}  // namespace
}  // namespace litertlm
