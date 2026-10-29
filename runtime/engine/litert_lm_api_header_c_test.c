// Compile-only C11 ABI smoke test.
#include "runtime/engine/litert_lm_api.h"

int LiteRtLm_CHeaderSmokeTest(void) {
  LiteRtLm_EngineConfig engine = LiteRtLm_DefaultEngineConfig();
  LiteRtLm_ConversationConfig conversation =
      LiteRtLm_DefaultConversationConfig();
  LiteRtLm_ChatRequest chat = LiteRtLm_DefaultChatRequest();
  return engine.struct_size == sizeof(engine) &&
         conversation.struct_size == sizeof(conversation) &&
         chat.struct_size == sizeof(chat) &&
         engine.backend == LITERT_LM_BACKEND_GPU && chat.flags == 0 &&
         chat.sampling.temperature == 1.0f;
}
