#include "runtime/engine/litert_lm_native.hpp"

#include <array>
#include <type_traits>

static_assert(!std::is_copy_constructible_v<litertlm::native::Engine>);
static_assert(std::is_move_constructible_v<litertlm::native::Engine>);
static_assert(!std::is_copy_constructible_v<litertlm::native::Conversation>);
static_assert(std::is_move_constructible_v<litertlm::native::Conversation>);

int main() {
  auto sampler = litertlm::native::SamplerParameters::Create();
  sampler.TopK(64).TopP(0.95f).Temperature(1.0f).Seed(7);
  auto session = litertlm::native::SessionConfig::Create();
  session.MaxOutputTokens(32).Sampler(sampler);
  auto conversation = litertlm::native::ConversationConfig::Create();
  conversation.Session(session).Tools("[]").StreamToolCalls(true);
  auto settings = litertlm::native::EngineSettings::Create(
      "model.litertlm", litertlm::native::Backend::Gpu);
  const std::array<int, 2> ranks = {4, 8};
  settings.MaxTokens(1024).SpeculativeDecoding(false);
  (void)settings.TrySetSupportedLoraRanks(ranks);
  return sampler && session && conversation && settings ? 0 : 1;
}
