// Copyright (c) 2025-2026 Winyunq. All rights reserved.
#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include "runtime/conversation/conversation.h"
#include "runtime/engine/engine_factory.h"
#include "runtime/engine/engine_settings.h"
#include "nlohmann/json.hpp"
#include "litert_lm_wrapper.h"

#include "runtime/components/constrained_decoding/llg_constraint_config.h"

using namespace litert::lm;
using json = nlohmann::json;

/**
 * 内部状态容器，确保 Wrapper 能像 CLI 一样管理对话
 */
struct LiteRtLm_ConversationContext {
    std::unique_ptr<Conversation> conversation;
    std::string last_user_text;
    std::mutex mtx;
};

extern "C" {

DLL_EXPORT const char* LiteRtLm_GetAvailableBackends() {
    return "cpu,gpu"; 
}

DLL_EXPORT void* LiteRtLm_CreateEngine(LiteRtLm_Config config) {
    auto model_assets_or = ModelAssets::Create(config.model_path);
    if (!model_assets_or.ok()) return nullptr;
    
    auto backend_or = GetBackendFromString(config.backend);
    if (!backend_or.ok()) return nullptr;

    auto settings_or = EngineSettings::CreateDefault(std::move(*model_assets_or), *backend_or);
    if (!settings_or.ok()) return nullptr;

    auto& main_settings = settings_or->GetMutableMainExecutorSettings();

    if (config.max_num_tokens > 0) {
        main_settings.SetMaxNumTokens(config.max_num_tokens);
    }
    if (config.num_threads > 0) {
        auto cpu_config_or = main_settings.MutableBackendConfig<CpuConfig>();
        if (cpu_config_or.ok()) {
            cpu_config_or->number_of_threads = config.num_threads;
        }
    }

    AdvancedSettings adv = main_settings.GetAdvancedSettings().value_or(AdvancedSettings());
    adv.optimize_shader_compilation = (config.bOptimizeShader != 0);
    adv.is_benchmark = (config.bEnableBenchmark != 0);
    main_settings.SetAdvancedSettings(adv);
    
    auto engine_or = EngineFactory::CreateAny(std::move(*settings_or));
    if (!engine_or.ok()) return nullptr;
    return static_cast<void*>(engine_or->release());
}

DLL_EXPORT void LiteRtLm_DestroyEngine(void* engine_ptr) {
    if (engine_ptr) delete static_cast<Engine*>(engine_ptr);
}

DLL_EXPORT void* LiteRtLm_CreateConversation(void* engine_ptr) {
    return LiteRtLm_CreateConversationWithConfig(engine_ptr, nullptr, 0);
}

DLL_EXPORT void* LiteRtLm_CreateConversationWithConfig(void* engine_ptr, const char* json_preface_str, int bEnableConstrainedDecoding) {
    if (!engine_ptr) return nullptr;
    Engine* engine = static_cast<Engine*>(engine_ptr);

    auto builder = ConversationConfig::Builder();
    builder.SetEnableConstrainedDecoding(bEnableConstrainedDecoding != 0);
    builder.SetConstraintProviderConfig(LlGuidanceConfig());

    if (json_preface_str) {
        try {
            auto j = json::parse(json_preface_str);
            JsonPreface jp;
            if (j.contains("messages")) jp.messages = j["messages"];
            if (j.contains("tools")) jp.tools = j["tools"];
            if (j.contains("extra_context")) jp.extra_context = j["extra_context"];
            builder.SetPreface(Preface(jp));

            if (j.contains("prompt_template") && j["prompt_template"].is_string()) {
                builder.SetOverwritePromptTemplate(PromptTemplate(j["prompt_template"].get<std::string>()));
            }
        } catch (...) {
            return nullptr;
        }
    }

    auto sess_cfg = SessionConfig::CreateDefault();
    auto conv_cfg_or = builder.SetSessionConfig(sess_cfg).Build(*engine);
    if (!conv_cfg_or.ok()) return nullptr;

    auto conv_or = Conversation::Create(*engine, *conv_cfg_or);
    if (!conv_or.ok()) return nullptr;

    auto* ctx = new LiteRtLm_ConversationContext();
    ctx->conversation = std::move(*conv_or);
    return static_cast<void*>(ctx);
}

DLL_EXPORT void LiteRtLm_DestroyConversation(void* conv_ptr) {
    if (conv_ptr) delete static_cast<LiteRtLm_ConversationContext*>(conv_ptr);
}

DLL_EXPORT void LiteRtLm_AppendUserMessage(void* conv_ptr, const char* text) {
    if (!conv_ptr || !text) return;
    auto* ctx = static_cast<LiteRtLm_ConversationContext*>(conv_ptr);
    std::lock_guard<std::mutex> lock(ctx->mtx);
    // 存入文本，由 RunInference 统一触发发送，逻辑对齐 CLI
    ctx->last_user_text = text;
}

DLL_EXPORT void LiteRtLm_AppendMessageJson(void* conv_ptr, const char* json_msg) {
    if (!conv_ptr || !json_msg) return;
    auto* ctx = static_cast<LiteRtLm_ConversationContext*>(conv_ptr);
    try {
        Message msg = json::parse(json_msg);
        OptionalArgs args;
        args.has_pending_message = true;
        ctx->conversation->SendMessage(msg, std::move(args));
    } catch (...) {}
}

DLL_EXPORT void LiteRtLm_AppendAssistantMessage(void* conv_ptr, const char* text) {
    if (!conv_ptr || !text) return;
    auto* ctx = static_cast<LiteRtLm_ConversationContext*>(conv_ptr);
    json msg = {{"role", "assistant"}, {"content", {{{"type", "text"}, {"text", text}}}}};
    OptionalArgs args;
    args.has_pending_message = true;
    ctx->conversation->SendMessage(msg, std::move(args));
}

DLL_EXPORT void LiteRtLm_RunInference(void* conv_ptr, LiteRtLm_SamplingParams params, LiteRtLmCallback callback, void* user_ptr) {
    if (!conv_ptr || !callback) return;
    auto* ctx = static_cast<LiteRtLm_ConversationContext*>(conv_ptr);

    // 构建触发消息
    json msg_to_send;
    {
        std::lock_guard<std::mutex> lock(ctx->mtx);
        if (!ctx->last_user_text.empty()) {
            // 对齐 CLI 消息格式
            msg_to_send = json::object({
                {"role", "user"}, 
                {"content", {{{"type", "text"}, {"text", ctx->last_user_text}}}}
            });
            ctx->last_user_text.clear();
        } else {
            // 如果没有挂起的消息，发送空对象触发生成
            msg_to_send = json::object();
        }
    }

    OptionalArgs args;
    args.max_output_tokens = params.max_tokens > 0 ? std::make_optional(params.max_tokens) : std::nullopt;
    
    if (params.constraint_type > 0 && params.constraint_string) {
        LlGuidanceConstraintArg llg_arg;
        switch (params.constraint_type) {
            case 1: llg_arg.constraint_type = LlgConstraintType::kRegex; break;
            case 2: llg_arg.constraint_type = LlgConstraintType::kJsonSchema; break;
            case 3: llg_arg.constraint_type = LlgConstraintType::kLark; break;
        }
        llg_arg.constraint_string = params.constraint_string;
        args.decoding_constraint = llg_arg;
    }

    // 内部 Lambda 包装回调，确保线程安全和字符串存活
    auto internal_cb = [callback, user_ptr](absl::StatusOr<Message> chunk) {
        LiteRtLm_Result res = {nullptr, nullptr, nullptr, 0, 0.0f};
        if (!chunk.ok()) {
            std::string err = std::string(chunk.status().message());
            res.error_msg = err.c_str(); res.bIsDone = 1;
            callback(res, user_ptr); return;
        }
        
        if (chunk->is_null() || chunk->empty()) {
            res.bIsDone = 1;
            callback(res, user_ptr); return;
        }

        std::string full_json_str = chunk->dump();
        res.full_json_chunk = full_json_str.c_str();

        std::string text_acc = "";
        if (chunk->contains("content")) {
            auto& content = (*chunk)["content"];
            if (content.is_array()) {
                for (const auto& part : content) {
                    if (part.is_object() && part.contains("text")) {
                        text_acc += part["text"].get<std::string>();
                    }
                }
            }
        }
        
        res.text_chunk = text_acc.empty() ? nullptr : text_acc.c_str();
        callback(res, user_ptr);
    };

    ctx->conversation->SendMessageAsync(msg_to_send, std::move(internal_cb), std::move(args));
}

DLL_EXPORT void LiteRtLm_StopMessage(void* conv_ptr) {
    if (conv_ptr) {
        auto* ctx = static_cast<LiteRtLm_ConversationContext*>(conv_ptr);
        ctx->conversation->CancelProcess();
    }
}

DLL_EXPORT int LiteRtLm_WaitUntilDone(void* engine_ptr, int timeout_sec) {
    if (!engine_ptr) return -1;
    Engine* engine = static_cast<Engine*>(engine_ptr);
    absl::Duration timeout = timeout_sec > 0 
        ? absl::Seconds(timeout_sec) 
        : Engine::kDefaultTimeout;
    auto status = engine->WaitUntilDone(timeout);
    if (status.ok()) return 0;
    if (absl::IsDeadlineExceeded(status)) return 1;
    return -1;
}

}
