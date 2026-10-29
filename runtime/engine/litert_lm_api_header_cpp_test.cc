// Compile-only C++20 convenience-layer smoke test.
#include "runtime/engine/litert_lm.hpp"

#include <string_view>

template <typename T>
concept HasVisionOption = requires(T value) { value.vision; };

template <typename T>
concept HasAudioOption = requires(T value) { value.audio; };

static_assert(!HasVisionOption<litertlm::EngineOptions>);
static_assert(!HasAudioOption<litertlm::EngineOptions>);
static_assert(!HasVisionOption<litertlm::LiteRtLmOptions>);
static_assert(!HasAudioOption<litertlm::LiteRtLmOptions>);

int LiteRtLm_CppHeaderSmokeTest() {
  litertlm::EngineOptions engine;
  engine.model_path = std::string_view("model.litertlm");
  engine.backend = litertlm::Backend::Gpu;
  engine.sampling = litertlm::Sampling::FixedSeed(42);

  litertlm::ChatOptions chat;
  chat.context_mode = litertlm::ContextMode::Replace;
  chat.constraint_kind = litertlm::ConstraintKind::JsonSchema;
  chat.sampling = litertlm::Sampling::Precise(17);
  return engine.sampling.seed == 42 &&
         chat.context_mode == litertlm::ContextMode::Replace &&
         chat.sampling.has_value() && chat.sampling->seed == 17;
}
